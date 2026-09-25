"""Build derived/beam/tooth_selection_vNNNN.{json,csv}: which transmitter teeth the beam fit uses.

A thin driver on eigsep_data.beam_mapping (tx_background.tooth_background,
tx_teeth.select_teeth). The rule, unchanged from v0002 (generator kept as
debug/build_tooth_selection_legacy.py), uses no beam model:

  bg_error   max(control error, raster local residual rms). The control is the
             transmitter-off window 07-17 17:03-17:54 UTC, where the tooth
             channel holds the true background: median over rows of
             sqrt(bias^2 + scatter^2) of (D - B)/D. The raster residual is the
             median local fit residual (offsets 2-4 each side) over good rows.
  isolation  p90 over good raster rows of (D - B)/B, divided by bg_error.
Selected: within the HFSS frequency range, outside FM (87-108.5 MHz), good rows
>= 50%, isolation >= 10, and iterative neighbour coherence >= 0.9 with the two
nearest same-arm selected teeth within 48 channels.

--concentration-min sets the DPSS background mode cutoff ('legacy' is the
floor(2NW)+1 rule of v0001-v0002; the default 1e-6 represents a constant).
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / 'marjum-2026-07/data').is_dir())
CAMPAIGN = ROOT / 'marjum-2026-07'
POINT_TABLE = CAMPAIGN / 'curation/pointing_table.parquet'
RASTER = ('2026-07-17 20:26:00', '2026-07-17 21:28:40')
CONTROL = ('2026-07-17 17:03:00', '2026-07-17 17:54:00')
HFSS_RANGE_MHZ = (50.78125, 246.09375)
FM_MHZ = (87.0, 108.5)
LIMITS_MHZ = (40.0, 249.5)
DF = 250.0 / 1024
ISOLATION_MIN, GOOD_MIN, NEIGHBOUR_CHANNELS = 10.0, 0.5, 48


def load(window):
    import eigsep_data
    import pyarrow.parquet as pq
    from eigsep_data import AntennaResolutionPolicy, MetadataIndex
    eigsep_data.set_campaign_root(CAMPAIGN)
    t0, t1 = (pd.Timestamp(s, tz='UTC').timestamp() for s in window)
    policy = AntennaResolutionPolicy.load(CAMPAIGN / 'curation/antenna_resolution.json')
    bundle = MetadataIndex(CAMPAIGN / 'data').select(time=(t0, t1)).load_bundle(
        antenna='box-air', missing='skip', resolution_policy=policy)
    meta = bundle.meta.reset_index(drop=True).assign(t=lambda x: x.time_best.astype(float))
    point = pq.read_table(POINT_TABLE, columns=['t_utc_s', 'quality'],
                          filters=[('t_utc_s', '>=', t0 - 60), ('t_utc_s', '<=', t1 + 60)]).to_pandas()
    j = pd.merge_asof(meta.reset_index(names='r').sort_values('t'), point.sort_values('t_utc_s'),
                      left_on='t', right_on='t_utc_s', direction='nearest', tolerance=.35)
    j = j[(j.quality == 'ok') & (j.rfswitch == 'RFANT')]
    return np.asarray(bundle.data[j.r.to_numpy()], float), sorted(set(j.file.astype(str)))


def main(output, coherence_min=0.9, concentration_min=1e-6):
    from eigsep_data.beam_mapping import select_teeth, tooth_background

    freq = np.arange(1024) * DF
    fm = (freq > FM_MHZ[0]) & (freq < FM_MHZ[1])
    cand = np.flatnonzero((np.arange(1024) % 8 == 0) & (freq >= HFSS_RANGE_MHZ[0]) & (freq <= HFSS_RANGE_MHZ[1]))
    teeth = cand[~fm[cand]]
    background = dict(exclude=fm, limits_mhz=LIMITS_MHZ, concentration_min=concentration_min)

    Dc, control_files = load(CONTROL)
    Dc = Dc[::2]
    Bc = tooth_background(Dc, freq, teeth, **background)
    ec = (Dc[:, teeth] - Bc) / Dc[:, teeth]
    bias = np.nanmedian(ec, 0)
    scatter = 1.4826 * np.nanmedian(np.abs(ec - bias), 0)
    control_error = np.sqrt(bias ** 2 + scatter ** 2)

    Dr, raster_files = load(RASTER)
    Br, local = tooth_background(Dr, freq, teeth, return_residual_rms=True, **background)
    gap = 0.5 * (Dr[:, teeth - 4] + Dr[:, teeth + 4])
    ratio = gap / np.nanmedian(gap, 0)
    good = (ratio > 1 / 1.5) & (ratio < 1.5) & np.isfinite(Br)      # broadband dropouts, as in the beam fit
    raster_error = np.nanmedian(np.where(good, local, np.nan), 0)
    bg_error = np.fmax(control_error, raster_error)
    sel = select_teeth(teeth, Dr[:, teeth], Br, good, bg_error, isolation_min=ISOLATION_MIN,
                       good_fraction_min=GOOD_MIN, coherence_min=coherence_min,
                       neighbour_channels=NEIGHBOUR_CHANNELS)
    table = pd.DataFrame(dict(channel=teeth, freq_mhz=freq[teeth], arm=sel['arm'], control_bias=bias,
                              control_scatter=scatter, control_error=control_error,
                              raster_local_error=raster_error, bg_error=bg_error,
                              signal_p90_over_bg=sel['signal_p90_over_bg'], isolation=sel['isolation'],
                              good_fraction=sel['good_fraction'], selected=sel['selected'],
                              coherence_first_pass=sel['coherence_first_pass'],
                              coherence_at_drop=sel['coherence_at_drop'], reason=sel['reason']))
    table.loc[~np.isfinite(table.isolation) & ~table.selected, 'reason'] = 'no_background'
    fm_teeth = cand[fm[cand]]
    table = pd.concat([table, pd.DataFrame(dict(channel=fm_teeth, freq_mhz=freq[fm_teeth], arm=(fm_teeth // 8) % 2,
                                                selected=False, reason='fm'))])
    table = table.sort_values('channel').reset_index(drop=True)
    output = Path(output)
    table.to_csv(output.with_suffix('.csv'), index=False, float_format='%.6g')

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    git = lambda *a: subprocess.run(['git', '-C', str(SOURCE.parent), *a], capture_output=True, text=True).stdout.strip()
    import pyarrow.parquet as pq
    point_meta = json.loads(pq.read_schema(POINT_TABLE).metadata[b'eigsep_provenance'])
    selected = table[table.selected.astype(bool)]
    doc = dict(
        product='tooth_selection', version=output.name.rsplit('_', 1)[-1], campaign='marjum-2026-07',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        generator=str(SOURCE.relative_to(ROOT)), generator_commit=git('rev-parse', '--short', 'HEAD'),
        generator_dirty=bool(git('status', '--porcelain', '--', str(SOURCE.parent))),
        channels=[int(c) for c in selected.channel],
        freqs_mhz=[float(f) for f in selected.freq_mhz],
        counts=dict(selected=int(len(selected)), arm0=int((selected.arm == 0).sum()),
                    arm1=int((selected.arm == 1).sum()),
                    excluded=table[~table.selected.astype(bool)].reason.value_counts().to_dict()),
        rule=dict(candidates='channels %% 8 == 0 within the HFSS frequency range %s MHz' % list(HFSS_RANGE_MHZ),
                  fm_excluded_mhz=list(FM_MHZ), isolation_min=ISOLATION_MIN, good_fraction_min=GOOD_MIN,
                  coherence_min=coherence_min, coherence_neighbour_channels=NEIGHBOUR_CHANNELS,
                  implementation='eigsep_data.beam_mapping.tx_teeth.select_teeth',
                  isolation='p90 over good raster rows of (D-B)/B, divided by max(control total error, '
                            'raster local residual rms)',
                  good='gap-ratio dropout mask (gap/median gap in (1/1.5, 1.5)) and a finite background'),
        background=dict(method='local DPSS per spectrum (eigsep_data.beam_mapping.tx_background.tooth_background)',
                        smooth_halfwidth_ns=150.0, window_mhz=48.0, step_mhz=24.0, window_limits_mhz=list(LIMITS_MHZ),
                        masked='tooth and +/-1 channels; FM band %g-%g MHz' % FM_MHZ,
                        clipping='symmetric, 4 sigma (1.4826 MAD)',
                        concentration_min=concentration_min if concentration_min is not None
                        else 'legacy floor(2NW)+1 modes'),
        raster_utc=list(RASTER), control_utc=list(CONTROL), pointing=point_meta['compact'],
        keys='channel: correlator channel (freq = channel*250/1024 MHz); arm = (channel//8) % 2',
        table=output.with_suffix('.csv').name,
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                      [SOURCE, POINT_TABLE] + [CAMPAIGN / 'data' / f for f in raster_files + control_files]},
        caveats=['The control is at 87.5 m, points at the horizon and has no transmitter noise; the raster '
                 'local residual covers transmitter-on structure only at offsets 2-4, not under the tooth.',
                 'Isolation uses the p90 signal: teeth are kept if their strong samples are well above the '
                 'background error, even though their beam nulls are not.'])
    output.with_suffix('.json').write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps(doc['counts'], indent=1))
    print('wrote', output.with_suffix('.json'), output.with_suffix('.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True,
                        help='output stem, e.g. marjum-2026-07/derived/beam/tooth_selection_v0003; '
                             '.json and .csv are written; the version is its suffix')
    parser.add_argument('--coherence-min', type=float, default=0.9,
                        help='neighbour-coherence threshold (0 disables it)')
    parser.add_argument('--concentration-min', default='1e-6',
                        help="DPSS background mode cutoff; 'legacy' for floor(2NW)+1 modes")
    args = parser.parse_args()
    main(args.output, args.coherence_min,
         None if args.concentration_min == 'legacy' else float(args.concentration_min))
