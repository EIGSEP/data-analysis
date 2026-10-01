"""HFSS, DPSS and PCA beam fits to the Marjum 07-17 transmitter raster.

A thin driver on eigsep_data.beam_mapping. The steps and model live in the
package; this script holds the campaign choices (raster window, pointing,
flags, tooth selection, site geometry) and writes the product in the layout of
derived/beam/empirical_raster_v0008-v0011:

1. Data: the elevation-sweep raster (07-17 20:26:00-21:28:40 UTC), box-air,
   RFANT rows with pointing_table v2.x quality 'ok' (commanded azimuth, IMU
   elevation), with isolated elevation glitches dropped.
2. Background under each tooth: tx_background.tooth_background (local DPSS,
   150 ns, FM excluded). Broadband dropouts are masked by the gap-ratio test;
   raster flags (sample flags and tooth spikes) are applied.
3. Geometry: highline pointing convention (psi = 142.164 deg), antenna from the
   v0001 geometry release, transmitter from curation. A joint scan finds the az
   offset and polarization centre, then fit_geometry refines the corrections.
4. Beams: ell_max and a spectral basis (PCA or DPSS) from the normalized HFSS
   fields; JointBeamFit at prior weights 1e-2, 1e-3, 1e-4, chosen on the
   validation stripes, then refitted on all samples and exported.

--concentration-min legacy uses the floor(2NW)+1 DPSS background of
v0009-v0011. The default (1e-6) represents smooth backgrounds correctly. The
generator of v0008-v0011 itself is kept as debug/fit_beam_legacy.py.
"""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / 'marjum-2026-07/data').is_dir())
CAMPAIGN = ROOT / 'marjum-2026-07'
POINT_TABLE = CAMPAIGN / 'curation/pointing_table.parquet'
RELEASE = CAMPAIGN / 'imgs/fits/v0001_marjum_geometry/shared.json'
TRANSMITTER = CAMPAIGN / 'curation/transmitter_position.json'
RASTER = ('2026-07-17 20:26:00', '2026-07-17 21:28:40')
PSI_DEG = 142.164                 # highline direction, deg ccw from East (known_quantities v0001)
DF = 250.0 / 1024
FM_MHZ = (87.0, 108.5)
BACKGROUND_LIMITS_MHZ = (40.0, 249.5)
PENALTIES = (1e-2, 1e-3, 1e-4)
log = logging.getLogger('fit_beam')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(repo):
    git = lambda *a: subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git('rev-parse', '--short', 'HEAD'), branch=git('branch', '--show-current'),
                dirty=bool(git('status', '--porcelain', '--untracked-files=no')))


def reversal_glitches(el_deg, tol=15.0):
    """Isolated solved-elevation faults: a sample is good if at least 3 of its 4
    neighbours (+/-1, +/-2) lie within `tol` deg across the +/-180 wrap."""
    el = np.asarray(el_deg, float)
    close = np.zeros(el.size, int)
    for k in (-2, -1, 1, 2):
        d = np.abs((el - np.roll(el, k) + 180.0) % 360.0 - 180.0)
        d[slice(0, -k) if k < 0 else slice(-k, None)] = 0.0
        close += d <= tol
    return close < 3


def load_data(teeth, concentration_min):
    """Raster spectra, pointing and the background-subtracted teeth."""
    import eigsep_data
    import pyarrow.parquet as pq
    from eigsep_data import AntennaResolutionPolicy, MetadataIndex
    from eigsep_data.beam_mapping import tooth_background

    eigsep_data.set_campaign_root(CAMPAIGN)
    t0, t1 = [pd.Timestamp(s, tz='UTC').timestamp() for s in RASTER]
    policy = AntennaResolutionPolicy.load(CAMPAIGN / 'curation/antenna_resolution.json')
    bundle = MetadataIndex(CAMPAIGN / 'data').select(time=(t0, t1)).load_bundle(
        antenna='box-air', missing='skip', resolution_policy=policy)
    freq = np.asarray(bundle.freqs_mhz, float)
    assert np.allclose(freq, np.arange(1024) * DF)
    point_prov = json.loads(pq.read_schema(POINT_TABLE).metadata[b'eigsep_provenance'])
    assert point_prov['version'].startswith('v2.'), point_prov['version']
    pointing = pq.read_table(POINT_TABLE, columns=['t_utc_s', 'az_deg', 'el_deg', 'quality', 'az_fused_deg'],
                             filters=[('t_utc_s', '>=', t0 - 60), ('t_utc_s', '<=', t1 + 60)]).to_pandas()
    meta = bundle.meta.reset_index(drop=True).assign(t=lambda x: x.time_best.astype(float))
    j = pd.merge_asof(meta.sort_values('t').reset_index(names='bundle_row'), pointing.sort_values('t_utc_s'),
                      left_on='t', right_on='t_utc_s', direction='nearest', tolerance=.35)
    j = j[(j.quality == 'ok') & (j.rfswitch == 'RFANT')].reset_index(drop=True)
    glitch = reversal_glitches(j.el_deg.to_numpy())
    j = j[~glitch].reset_index(drop=True)
    raw = np.asarray(bundle.data[j.bundle_row.to_numpy()], float)
    channels = np.asarray(sorted(teeth))
    fm = (freq > FM_MHZ[0]) & (freq < FM_MHZ[1])
    background = tooth_background(raw, freq, channels, exclude=fm, limits_mhz=BACKGROUND_LIMITS_MHZ,
                                  concentration_min=concentration_min).T
    tooth = raw[:, channels].T
    gap = 0.5 * (raw[:, channels - 4] + raw[:, channels + 4]).T
    ratio = gap / np.nanmedian(gap, axis=1)[:, None]
    good = (ratio > 1 / 1.5) & (ratio < 1.5) & np.isfinite(tooth - background)
    return dict(t=j.t.to_numpy(), az=j.az_deg.to_numpy(), el=j.el_deg.to_numpy(),
                az_v1=j.az_fused_deg.to_numpy(), data=tooth - background, good=good, channels=channels,
                files=sorted(set(j.file.astype(str))), n_glitches=int(glitch.sum()), point_prov=point_prov)


def apply_flags(raw, flags):
    """Sample flags and tooth spikes from a raster_flags npz, aligned on time and channel."""
    with np.load(flags) as f:
        order = np.argsort(f['t'])
        pos = np.clip(np.searchsorted(f['t'][order], raw['t']), 0, len(order) - 1)
        near = order[np.where(abs(f['t'][order][pos] - raw['t']) <= 1e-5, pos, 0)]
        assert np.all(abs(f['t'][near] - raw['t']) <= 1e-5), 'samples missing from flags'
        sample_flag = f['sample_flag'][near].astype(bool)
        tooth_flag = np.zeros_like(raw['good'])
        fcol = {int(c): i for i, c in enumerate(f['channels'])}
        for i, c in enumerate(raw['channels']):
            if int(c) in fcol:
                tooth_flag[i] = f['tooth_flag'][fcol[int(c)], near].astype(bool)
        return sample_flag, tooth_flag, json.loads(str(f['provenance']))


def main(output, teeth_path, flags, bases, maxiter, concentration_min):
    import healpy as hp

    import eigsep_data.beam_mapping as bm
    from eigsep_data.beam_sim import DEFAULT_BEAM_PATH, read_beam

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(teeth_path).read_text())
    raw = load_data(selection['channels'], concentration_min)
    sample_flag, tooth_flag, flag_prov = apply_flags(raw, flags)
    good = raw['good'] & ~sample_flag[None, :] & ~tooth_flag
    freqs = raw['channels'] * DF
    data = bm.ToothData(t=raw['t'], az_deg=raw['az'], el_deg=raw['el'], data=raw['data'], good=good,
                        channels=raw['channels'], freqs_mhz=freqs, split=bm.stripe_split(raw['az']))
    log.info('data %s teeth x %s samples; flags: %d samples, %d tooth spikes', *data.data.shape,
             sample_flag.sum(), tooth_flag.sum())

    ant = np.asarray(json.loads(RELEASE.read_text())['antenna_91m_era']['position_enu_m'], float)
    tx = np.asarray(json.loads(TRANSMITTER.read_text())['best_estimate_enu_m'], float)
    hfields, _, hfreqs = read_beam(drop_last=False)
    hnorm, _, _ = bm.normalize_fields(hfields, hfreqs)
    fields_teeth = bm.interpolate_fields(hnorm, hfreqs, freqs)

    model, scan = bm.coarse_offset_alpha(data, fields_teeth, bm.TxGeometryModel(PSI_DEG, ant, tx))
    pd.DataFrame(scan).to_csv(output / 'az_offset_alpha_scan.csv', index=False)
    at_zero = min(r['normalized_rms'] for r in scan if r['az_offset_deg'] == 0.0)
    log.info('coarse az offset %+.0f, alpha %.0f (best at offset 0: %.4f)',
             model.az_offset_deg, model.alpha0_deg, at_zero)
    params, best_geometry, geometry_table, _ = bm.fit_geometry(data, fields_teeth, model)
    pd.DataFrame(geometry_table).to_csv(output / 'geometry_comparison.csv', index=False)

    band = (float(freqs.min()) - DF, float(freqs.max()) + DF)
    first = max(0, np.searchsorted(hfreqs, band[0]) - 1)
    last = min(len(hfreqs), np.searchsorted(hfreqs, band[1]) + 1)
    native_f, native_fields = hfreqs[first:last], hnorm[first:last]
    lmax, sh, lmax_records = bm.select_lmax(native_fields)
    pd.DataFrame(lmax_records).to_csv(output / 'spatial_order.csv', index=False)
    train = data.split == 0
    baseline_power = bm.hfss_power(fields_teeth, data, model, params)
    baseline = bm.tooth_gains(baseline_power, data.data, data.good, train)[:, None] * baseline_power
    all_metrics = bm.score(data, baseline, 'hfss')
    descriptions, optimizations = {}, []
    theta, phi, _ = model.frame(data, params)
    coverage = np.bincount(hp.ang2pix(hp.npix2nside(hfields.shape[-1]), theta, phi),
                           minlength=hfields.shape[-1])
    for kind in bases:
        directory = output / kind
        directory.mkdir(exist_ok=True)
        if kind == 'pca':
            basis = bm.pca_basis(native_f, native_fields)
            np.savez_compressed(directory / 'spectral_basis.npz', kind=np.array('pca'), native_freqs=native_f,
                                native_modes=basis.A, singular_values=basis.singular_values)
            description = dict(kind='pca', modes=basis.nmodes,
                               construction='uncentered complex SVD; left singular vectors across frequency')
        else:
            basis = bm.DPSSSpectralBasis.from_fields(native_f, native_fields, freqs)
            basis.save(directory / 'spectral_basis.npz')
            description = dict(kind='dpss', modes=basis.nmodes, delay_half_width_ns=basis.delay_halfwidth_ns,
                               grid_limits_mhz=[float(basis.grid[0]), float(basis.grid[-1])])
        pd.DataFrame(basis.records or []).to_csv(directory / 'basis_selection.csv', index=False)
        c0 = bm.initial_coefficients(basis, sh)
        fit = bm.JointBeamFit(data, model, params, lmax, basis, c0)
        trials = []
        for penalty in PENALTIES:
            coeff, gain, prediction, info = fit.fit(train, penalty, maxiter=maxiter)
            records = bm.score(data, prediction, f'{kind}_lambda_{penalty:g}')
            validation = float(np.median([r['fractional_rms'] for r in records
                                          if r['split'] == 'validation' and r['used_for_beam_fit']]))
            info.update(basis=kind, stage='train', validation_median_rms=validation)
            trials.append((validation, coeff, gain, prediction, info))
            all_metrics.extend(records)
            optimizations.append(info)
            log.info('%s penalty %g: validation %.4f', kind, penalty, validation)
        _, coeff, gain, heldout, selected = min(trials, key=lambda entry: entry[0])
        all_metrics.extend(bm.score(data, heldout, kind + '_heldout'))
        final_coeff, final_gain, final_prediction, final_info = fit.fit(
            np.ones(len(data.t), bool), selected['penalty'], initial=coeff, initial_gain=gain, maxiter=maxiter)
        final_info.update(basis=kind, stage='all_sample_refit')
        optimizations.append(final_info)
        all_metrics.extend(bm.score(data, final_prediction, kind + '_refit'))
        export_gain, export_fields, export_freqs = bm.export_beam(
            directory / 'empirical_beam.npz', hfreqs, hnorm, basis, final_coeff, lmax, band, freqs, final_gain,
            sample_count=coverage,
            metadata=dict(geometry_parameters=params, geometry_parameter_names=np.array(bm.PARAM_NAMES),
                          status='exploratory; power-only phase ambiguity; unsampled directions prior-dependent'))
        idx = np.searchsorted(export_freqs, freqs)
        exported = export_gain[:, None] * bm.hfss_power(export_fields[idx], data, model, params)
        all_metrics.extend(bm.score(data, exported, kind + '_exported'))
        export_error = np.sqrt(np.sum(np.where(data.good, (exported - final_prediction) ** 2, 0), axis=1)
                               / np.sum(np.where(data.good, data.data ** 2, 0), axis=1))
        assert np.max(export_error[data.fit_channels]) < .01, export_error
        description.update(selected_penalty=selected['penalty'], lmax=lmax,
                           max_export_prediction_error_relative_to_data=float(export_error.max()))
        descriptions[kind] = description
        np.savez_compressed(
            directory / 'diagnostics.npz', az=data.az_deg + model.az_offset_deg, az_table=data.az_deg,
            az_pointing_table_v1=raw['az_v1'], el=data.el_deg, t=data.t, data=data.data, good=data.good,
            split=data.split, channels=data.channels, freqs=freqs, arms=data.arms,
            fit_channels=data.fit_channels, el0_slew=np.zeros(len(data.t), bool), sample_flag=sample_flag,
            tooth_flag=tooth_flag, az_offset_deg=model.az_offset_deg, alpha0_deg=model.alpha0_deg,
            convention=np.array('highline'), psi_deg=PSI_DEG, hfss=baseline, empirical_heldout=heldout,
            empirical_refit=final_prediction, empirical_exported=exported, initial_coeff=c0, heldout_coeff=coeff,
            coeff=final_coeff, heldout_gain=gain, final_gain=final_gain, exported_gain=export_gain,
            params=params, spectral_at_teeth=basis.evaluate(freqs), spectral_at_hfss=basis.A,
            native_hfss_freqs=native_f)
        pd.DataFrame(all_metrics).to_csv(output / 'metrics.csv', index=False)
        (output / 'optimization.json').write_text(json.dumps(optimizations, indent=2) + '\n')

    import eigsep_base
    import eigsep_data
    inputs = [SOURCE, POINT_TABLE, DEFAULT_BEAM_PATH, RELEASE, TRANSMITTER, Path(teeth_path).resolve(),
              Path(flags).resolve(), CAMPAIGN / 'curation/antenna_resolution.json']
    inputs += [CAMPAIGN / 'data' / Path(f).name for f in raw['files']]
    artifacts = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                 if p.is_file() and p.name not in ('provenance.json', 'README.md')}
    geometry = dict(zip(bm.PARAM_NAMES, map(float, params)))
    provenance = dict(
        status='exploratory', generator=str(SOURCE.relative_to(ROOT)),
        pipeline='eigsep_data.beam_mapping (tx_background, tx_fit, beam_basis, tx_export)',
        pointing=raw['point_prov']['compact'],
        pointing_rule='az_deg: commanded azimuth (motor counts plus one pot offset per anchor segment); el_deg: IMU',
        convention='highline',
        rotation='v_ENU = Rz(psi) Rx(el + el_zero_delta) Rz(az + az_offset + az_zero_delta) v_body, psi = %.3f deg '
                 '(eigsep_base.rotations.mount_rotation)' % PSI_DEG,
        az_offset_deg=model.az_offset_deg, alpha0_deg=model.alpha0_deg,
        table_az_zero_from_highline_deg=model.az_offset_deg + geometry['az_zero_delta_deg'],
        geometry=geometry, geometry_model=best_geometry,
        background=dict(method='local DPSS (eigsep_data.beam_mapping.tx_background)', halfwidth_ns=150.0,
                        window_mhz=48.0, excluded_mhz=list(FM_MHZ), limits_mhz=list(BACKGROUND_LIMITS_MHZ),
                        concentration_min=concentration_min if concentration_min is not None
                        else 'legacy floor(2NW)+1 modes (v0009-v0011)'),
        tooth_selection=dict(path=str(Path(teeth_path).resolve().relative_to(ROOT)), version=selection['version'],
                             channels=len(selection['channels'])),
        flags=dict(path=str(Path(flags).resolve().relative_to(ROOT)), version=flag_prov['version'],
                   samples=int(sample_flag.sum()), tooth_spikes=int(tooth_flag.sum())),
        spectral_bases=descriptions, lmax=lmax, band_mhz=list(band), raster_utc=list(RASTER),
        split='12 deg stripes of pointing-table azimuth (tx_fit.stripe_split)', maxiter=maxiter,
        n_glitches=raw['n_glitches'],
        repositories={name: revision(ROOT / name) for name in ('data-analysis', 'eigsep_data', 'eigsep_base')},
        package_versions=dict(eigsep_data=getattr(eigsep_data, '__version__', None),
                              eigsep_base=eigsep_base.__version__),
        input_sha256={str(Path(p).resolve().relative_to(ROOT)): digest(p) for p in inputs},
        artifact_sha256=artifacts,
        caveats=['Power-only phase and circular-polarization ambiguities remain; coefficients are HFSS-regularized.',
                 'Commanded azimuth assumes the platform followed the motor; see curation/pointing_table.README.md.',
                 'Validation/test stripes are recomputed from commanded azimuth; not a fresh blind test.',
                 'Only %.1f-%.1f MHz is fitted. Other exported slices are normalized HFSS.' % band,
                 'All-sample refit/export scores are not independent validation.'])
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    log.info('completed %s', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True,
                        help='new product directory, e.g. marjum-2026-07/derived/beam/empirical_raster_v0012')
    parser.add_argument('--teeth', default=str(CAMPAIGN / 'derived/beam/tooth_selection_v0002.json'))
    parser.add_argument('--flags', default=str(CAMPAIGN / 'derived/beam/raster_flags_v0001.npz'))
    parser.add_argument('--bases', nargs='+', choices=['dpss', 'pca'], default=['dpss', 'pca'])
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--concentration-min', default='1e-6',
                        help="DPSS background mode cutoff; 'legacy' for the floor(2NW)+1 modes of v0009-v0011")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    cmin = None if args.concentration_min == 'legacy' else float(args.concentration_min)
    main(args.output, args.teeth, args.flags, tuple(args.bases), args.maxiter, cmin)
