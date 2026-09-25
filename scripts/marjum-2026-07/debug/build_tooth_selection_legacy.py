"""Build derived/beam/tooth_selection_vNNNN.{json,csv}: which transmitter teeth the beam fit uses.

Exploratory, single-study code. It replaces the hand-picked band (145-235 MHz)
and suspect list (720, 816, 912, 960) of beam fits v0001-v0008. Those came from
per-frequency fits on potentiometer azimuth, which was wrong. The rule here
uses no beam model, so it cannot be tuned toward good fit residuals.

Background: tooth_background.py (local DPSS, 150 ns). Per candidate tooth:
  bg_error   the larger of
             - control: the median over rows of sqrt(bias^2 + scatter^2) of
               (D - B)/D at the tooth channel, on the transmitter-off control
               (07-17 17:03-17:54 UTC), where the tooth channel holds the true
               background;
             - raster: the median over rows of the local fit residual rms
               (offsets 2-4 each side), a fraction of the background. It catches
               transmitter-on structure and RFI the control lacks.
  signal     the 90th percentile over good raster rows of (D - B)/B, the tooth's
             strong-sample amplitude relative to the background.
  isolation  signal / bg_error.
Selected: within the HFSS frequency range, outside FM (87-108.5 MHz), good rows
>= 50%, and isolation >= 10, so the background error is at most 10% of the
tooth's strong-sample signal. Threshold declared before scoring.

v0002 (2026-09-25) adds neighbour coherence (--coherence-min 0.9; 0 gives v0001).
Isolation uses the 90th-percentile signal, which intermittent RFI inflates, so
v0001 kept five teeth dominated by it (aeronautical VHF, ORBCOMM, the band
edge) that fail completely in beam fits v0009-v0010. A transmitter tooth sees
the beam, which changes smoothly over 3.9 MHz, so its time series should track
its nearest same-arm teeth. Among the isolation-selected teeth, iteratively: the
correlation over good rows of log(tooth signal) with log(mean of the two nearest
same-arm selected teeth within 48 channels) is computed, and the least coherent
tooth below the threshold is dropped, then the rest are recomputed, so one bad
tooth cannot drag its neighbours down. A tooth left with no neighbour cannot be
checked and is dropped too. The test was chosen in
data-analysis/notebooks/arp/marjum-2026-07/debug/tooth_selection_debug.ipynb,
where fit scores were used only to check it, and 0.9 was compared with 0.85 and
0.95 there.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / 'marjum-2026-07/data').is_dir())
CAMPAIGN = ROOT / 'marjum-2026-07'
BG_SOURCE = SOURCE.with_name('tooth_background.py')
bg = runpy.run_path(str(BG_SOURCE))
DF = bg['DF']
POINT_TABLE = CAMPAIGN / 'curation/pointing_table.parquet'
RASTER = ('2026-07-17 20:26:00', '2026-07-17 21:28:40')
CONTROL = ('2026-07-17 17:03:00', '2026-07-17 17:54:00')
HFSS_RANGE_MHZ = (50.78125, 246.09375)
ISOLATION_MIN, GOOD_MIN = 10.0, 0.5
NEIGHBOUR_CHANNELS = 48


def coherence_drops(signal, good, channels, keep, threshold):
    """Iterative neighbour coherence over the teeth in `keep`. signal, good: (rows, teeth).
    Returns the dropped teeth as (channel, coherence at drop) in drop order, and the first-pass
    coherence of every kept tooth."""
    keep = keep.copy()
    arms = (channels // 8) % 2

    def coherence(i):
        same = [j for j in np.flatnonzero(keep) if j != i and arms[j] == arms[i]
                and abs(int(channels[j]) - int(channels[i])) <= NEIGHBOUR_CHANNELS]
        same = sorted(same, key=lambda j: abs(int(channels[j]) - int(channels[i])))[:2]
        if not same:
            return np.nan
        ok = good[:, i] & np.all(good[:, same], axis=1)
        floor = 1e-3 * np.nanmax(signal[:, i])
        x = np.log(np.clip(signal[ok, i], floor, None))
        y = np.log(np.clip(np.mean(signal[ok][:, same], axis=1), floor, None))
        return np.corrcoef(x, y)[0, 1]

    first = {i: coherence(i) for i in np.flatnonzero(keep)}
    dropped = []
    while threshold > 0:
        c = {i: coherence(i) for i in np.flatnonzero(keep)}
        worst = min(c, key=lambda i: c[i] if np.isfinite(c[i]) else -1)
        if np.isfinite(c[worst]) and c[worst] >= threshold:
            break
        keep[worst] = False
        dropped.append((int(channels[worst]), float(c[worst])))
    return keep, dropped, first


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


def main(output, coherence_min=0.9):
    freq = bg['FREQ']
    cand = np.flatnonzero((np.arange(1024) % 8 == 0) & (freq >= HFSS_RANGE_MHZ[0]) & (freq <= HFSS_RANGE_MHZ[1]))
    in_fm = bg['FM'][cand]
    teeth = cand[~in_fm]

    Dc, control_files = load(CONTROL)
    Dc = Dc[::2]
    Bc = bg['tooth_background'](Dc, teeth)
    ec = (Dc[:, teeth] - Bc) / Dc[:, teeth]
    bias = np.nanmedian(ec, 0)
    scatter = 1.4826 * np.nanmedian(np.abs(ec - bias), 0)
    control_error = np.sqrt(bias ** 2 + scatter ** 2)

    Dr, raster_files = load(RASTER)
    Br, local = bg['tooth_background'](Dr, teeth, return_residual_rms=True)
    gap = 0.5 * (Dr[:, teeth - 4] + Dr[:, teeth + 4])
    ratio = gap / np.nanmedian(gap, 0)
    good = (ratio > 1 / 1.5) & (ratio < 1.5) & np.isfinite(Br)      # broadband dropouts, as in the beam fit
    sb = np.where(good, (Dr[:, teeth] - Br) / Br, np.nan)
    signal = np.nanpercentile(sb, 90, axis=0)
    raster_error = np.nanmedian(np.where(good, local, np.nan), 0)
    bg_error = np.fmax(control_error, raster_error)
    isolation = signal / bg_error
    good_frac = good.mean(0)

    table = pd.DataFrame(dict(channel=teeth, freq_mhz=freq[teeth], arm=(teeth // 8) % 2,
                              control_bias=bias, control_scatter=scatter, control_error=control_error,
                              raster_local_error=raster_error, bg_error=bg_error,
                              signal_p90_over_bg=signal, isolation=isolation, good_fraction=good_frac))
    table['selected'] = (table.isolation >= ISOLATION_MIN) & (table.good_fraction >= GOOD_MIN)
    signal = Dr[:, teeth] - Br
    keep, dropped, first = coherence_drops(signal, good, teeth, table.selected.to_numpy(), coherence_min)
    table['coherence_first_pass'] = [first.get(i, np.nan) for i in range(len(teeth))]
    table['coherence_at_drop'] = np.nan
    for c, v in dropped:
        table.loc[table.channel == c, 'coherence_at_drop'] = v
    coherent_fail = table.selected & ~keep
    table['selected'] = keep
    print('neighbour coherence (min %g) drops %d teeth:' % (coherence_min, len(dropped)),
          [(c, round(float(freq[c]), 1), round(v, 3)) for c, v in dropped])
    table['reason'] = np.where(coherent_fail, 'coherence', np.where(table.selected, 'selected',
                               np.where(table.good_fraction < GOOD_MIN, 'good_fraction',
                                        np.where(np.isfinite(table.isolation), 'isolation', 'no_background'))))
    fm_rows = pd.DataFrame(dict(channel=cand[in_fm], freq_mhz=freq[cand[in_fm]], arm=(cand[in_fm] // 8) % 2,
                                selected=False, reason='fm'))
    table = pd.concat([table, fm_rows]).sort_values('channel').reset_index(drop=True)
    output = Path(output)
    table.to_csv(output.with_suffix('.csv'), index=False, float_format='%.6g')

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    commit = subprocess.run(['git', '-C', str(SOURCE.parent), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', '-C', str(SOURCE.parent), 'status', '--porcelain', '--', str(SOURCE.parent)],
                                capture_output=True, text=True).stdout.strip())
    point_meta = json.loads(__import__('pyarrow.parquet', fromlist=['x']).read_schema(POINT_TABLE)
                            .metadata[b'eigsep_provenance'])
    selected = table[table.selected]
    doc = dict(
        product='tooth_selection', version=Path(output).name.rsplit('_', 1)[-1], campaign='marjum-2026-07',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        generator=str(SOURCE.relative_to(ROOT)), generator_commit=commit, generator_dirty=dirty,
        channels=[int(c) for c in selected.channel],
        freqs_mhz=[float(f) for f in selected.freq_mhz],
        counts=dict(selected=int(len(selected)), arm0=int((selected.arm == 0).sum()),
                    arm1=int((selected.arm == 1).sum()),
                    excluded=table[~table.selected].reason.value_counts().to_dict()),
        rule=dict(candidates='channels %% 8 == 0 within the HFSS frequency range %s MHz' % list(HFSS_RANGE_MHZ),
                  fm_excluded_mhz=list(bg['FM_MHZ']), isolation_min=ISOLATION_MIN, good_fraction_min=GOOD_MIN,
                  coherence_min=coherence_min, coherence_neighbour_channels=NEIGHBOUR_CHANNELS,
                  coherence='iterative: correlation over good rows of log(tooth signal) with log(mean of the '
                            'two nearest same-arm selected teeth within 48 channels); the least coherent tooth '
                            'below the minimum is dropped and the rest recomputed; a tooth with no neighbour '
                            'is dropped',
                  isolation='p90 over good raster rows of (D-B)/B, divided by max(control total error, '
                            'raster local residual rms)',
                  good='gap-ratio dropout mask (gap/median gap in (1/1.5, 1.5)) and a finite background'),
        background=bg['DESCRIPTION'],
        raster_utc=list(RASTER), control_utc=list(CONTROL),
        pointing=point_meta['compact'],
        keys='channel: correlator channel (freq = channel*250/1024 MHz); arm = (channel//8) % 2',
        table=output.with_suffix('.csv').name,
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                      [SOURCE, BG_SOURCE, POINT_TABLE] + [CAMPAIGN / 'data' / f for f in raster_files + control_files]},
        caveats=['The control is at 87.5 m, points at the horizon and has no transmitter noise; the raster '
                 'local residual covers transmitter-on structure only at offsets 2-4, not under the tooth.',
                 'Isolation uses the p90 signal: teeth are kept if their strong samples are well above the '
                 'background error, even though their beam nulls are not.',
                 'Replaces the hand-picked 145-235 MHz band and suspect list of beam fits v0001-v0008.'])
    output.with_suffix('.json').write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps(doc['counts'], indent=1))
    print('wrote', output.with_suffix('.json'), output.with_suffix('.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CAMPAIGN / 'derived/beam/tooth_selection_v0002',
                        help='output stem; .json and .csv are written; the version is its suffix')
    parser.add_argument('--coherence-min', type=float, default=0.9,
                        help='neighbour-coherence threshold (0 disables it, reproducing v0001)')
    args = parser.parse_args()
    main(args.output, args.coherence_min)
