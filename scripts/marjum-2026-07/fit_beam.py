"""HFSS, DPSS and PCA beam fits to the 07-17 transmitter raster, on commanded pointing.

Exploratory, single-study code. It repeats the v0001 geometry fit and the v0003
DPSS/PCA joint fits with one change of input. Pointing comes from the campaign
pointing table, marjum-2026-07/curation/pointing_table.parquet at v2.x, whose
az_deg is commanded azimuth and el_deg the IMU elevation. pointing_table@v1.2
azimuth carried a potentiometer slip.

Until 2026-09-24 this script was empirical_beam_commanded.py and read the
raster-only point_table_v2-beta.npz; that version, which generated
derived/beam/empirical_raster_v0006 and v0007, is kept in debug/. On the raster
the campaign table's az_deg is v2-beta's minus a constant 0.313 deg, which the
fitted az offset absorbs. The 12 deg stripes are cut on az_deg, so a few
samples at stripe edges change split.

Consequences:
- The 12 deg train/validation/test stripes are recomputed from commanded
  azimuth, so held-out sets differ from v0003's.
- Azimuth zero is not tied to the beam frame, so a coarse az offset is chosen
  first by the HFSS beam on training samples; the v0001 geometry fit then
  refines polarization, az/el zero and transmitter position within its bounds.
With --flags, raster_flags sample flags and tooth spikes are also excluded.
Everything else follows v0003: the no-slew mask, 42 fitting teeth with
explicit joint gains, the HFSS-selected lmax and spectral bases, penalty
selection on validation stripes, all-sample refit and export.
"""
from pathlib import Path
import argparse
import json
import runpy

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve()
STUDY_SOURCE = SOURCE.with_name('empirical_beam_spectral_no_el0.py')
study = runpy.run_path(str(STUDY_SOURCE))
legacy = study['legacy']
ROOT, CAMPAIGN, BAND = study['ROOT'], study['CAMPAIGN'], study['BAND']
POINT_TABLE = CAMPAIGN / 'curation/pointing_table.parquet'
DF = legacy['DF']
log, digest, metrics = legacy['log'], legacy['digest'], legacy['metrics']


def load_data():
    """legacy load_data with commanded pointing and commanded-az stripes."""
    import eigsep_data
    from eigsep_data import AntennaResolutionPolicy, MetadataIndex
    eigsep_data.set_campaign_root(CAMPAIGN)
    t0, t1 = [pd.Timestamp(s, tz='UTC').timestamp() for s in legacy['RASTER']]
    policy = AntennaResolutionPolicy.load(CAMPAIGN / 'curation/antenna_resolution.json')
    bundle = MetadataIndex(CAMPAIGN / 'data').select(time=(t0, t1)).load_bundle(
        antenna='box-air', missing='skip', resolution_policy=policy)
    import pyarrow.parquet as pq
    table = pq.read_table(POINT_TABLE, columns=['t_utc_s', 'az_deg', 'el_deg', 'quality', 'az_fused_deg'],
                          filters=[('t_utc_s', '>=', t0 - 60), ('t_utc_s', '<=', t1 + 60)])
    point_provenance = json.loads(pq.read_schema(POINT_TABLE).metadata[b'eigsep_provenance'])
    # v2.x is where az_deg became commanded azimuth; v1.x az_deg is the pot-fused value.
    assert point_provenance['version'].startswith('v2.'), point_provenance['version']
    pointing = table.to_pandas().rename(columns={'az_fused_deg': 'az_deg_v1'})
    meta = bundle.meta.reset_index(drop=True).assign(t=lambda x: x.time_best.astype(float))
    j = pd.merge_asof(meta.sort_values('t').reset_index(names='bundle_row'),
                      pointing.sort_values('t_utc_s'),
                      left_on='t', right_on='t_utc_s', direction='nearest', tolerance=.35)
    j = j[(j.quality == 'ok') & (j.rfswitch == 'RFANT')].reset_index(drop=True)
    glitch = legacy['glitches'](j.el_deg.to_numpy())
    j = j[~glitch].reset_index(drop=True)
    channels = np.arange(8, 1020, 8)
    channels = channels[(channels*DF >= BAND[0]) & (channels*DF <= BAND[1])]
    col = {round(f/DF): i for i, f in enumerate(bundle.freqs_mhz)}
    raw = np.asarray(bundle.data[j.bundle_row.to_numpy()], float)
    tooth = raw[:, [col[c] for c in channels]].T
    gap = .5*(raw[:, [col[c-4] for c in channels]] + raw[:, [col[c+4] for c in channels]]).T
    ratio = gap / np.nanmedian(gap, axis=1)[:, None]
    az = j.az_deg.to_numpy()
    assert np.isfinite(az).all()
    # Entire 12-degree stripes of commanded azimuth are held out.
    azbin = np.floor(((az+180) % 360) / 12).astype(int)
    split = np.where(azbin % 5 == 1, 1, np.where(azbin % 5 == 3, 2, 0))
    return dict(data=tooth-gap, good=(ratio > 1/1.5) & (ratio < 1.5) & np.isfinite(tooth-gap),
                gap=gap, channels=channels, freqs=channels*DF, arms=(channels//8) % 2,
                az=az, az_table=az.copy(), az_pointing_table_v1=j.az_deg_v1.to_numpy(),
                el=j.el_deg.to_numpy(), t=j.t.to_numpy(), split=split,
                files=sorted(set(j.file.astype(str))), n_glitches=int(glitch.sum()),
                fit_channels=~np.isin(channels, legacy['SUSPECT_CHANNELS']),
                point_provenance=point_provenance)


def coarse_az_offset(d, h, offsets=np.arange(-60., 60.01, 1.)):
    """HFSS-only, zero-geometry scan of a constant azimuth offset on training samples."""
    idx = np.flatnonzero(d['split'] == 0)[::4]
    usef = np.flatnonzero(d['fit_channels'])
    scale = np.sqrt(np.mean(d['data'][:, idx]**2, axis=1))
    base = d['az'].copy()
    scores = []
    for offset in offsets:
        d['az'] = base + offset
        p = legacy['sample_hfss'](d, h, np.zeros(5), idx)
        g = legacy['gains'](p, d['data'][:, idx], d['good'][:, idx], np.ones(len(idx), bool))
        r = ((g[:, None]*p - d['data'][:, idx])/scale[:, None])*d['good'][:, idx]
        scores.append(float(np.sqrt(np.mean(r[usef]**2))))
    d['az'] = base
    best = float(offsets[int(np.argmin(scores))])
    return best, pd.DataFrame(dict(az_offset_deg=offsets, normalized_rms=scores))


def main(output, maxiter=3000, bases=('dpss', 'pca'), flags=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    d, h = load_data(), legacy['load_hfss']()
    # The opening near-zero-elevation azimuth slew (56 samples, masked here through
    # v0007) is quality='suspect' (AZ_SLEW) in pointing_table v2.0, so load_data never
    # returns it. The rule is kept as a check that none of it came back.
    el0_slew = (d['t']-d['t'].min() <= 30.) & (np.abs(d['el']) < 3.)
    assert not el0_slew.any(), el0_slew.sum()
    d['el0_slew'] = el0_slew
    d['good'][:, el0_slew] = False
    sample_flag = np.zeros(len(d['t']), bool)
    tooth_flag = np.zeros_like(d['good'])
    flag_provenance = None
    if flags is not None:
        with np.load(flags) as f:
            # Align on time: the flags may cover samples this table drops (e.g. the v2-beta-era
            # flags include the 56-sample opening slew), but every loaded sample must be flagged.
            assert np.array_equal(f['channels'], d['channels'])
            order = np.argsort(f['t'])
            pos = np.clip(np.searchsorted(f['t'][order], d['t']), 0, len(order) - 1)
            near = order[np.where(abs(f['t'][order][pos] - d['t']) <= 1e-5, pos, 0)]
            assert np.all(abs(f['t'][near] - d['t']) <= 1e-5), 'samples missing from flags'
            sample_flag, tooth_flag = f['sample_flag'][near].astype(bool), f['tooth_flag'][:, near].astype(bool)
            flag_provenance = json.loads(str(f['provenance']))
        d['good'] &= ~sample_flag[None, :] & ~tooth_flag
        log('flags', flags, 'samples', int(sample_flag.sum()), 'tooth spikes', int(tooth_flag.sum()))
    log('data', d['data'].shape, 'split counts', np.bincount(d['split']))

    az_offset, scan = coarse_az_offset(d, h)
    scan.to_csv(output/'az_offset_scan.csv', index=False)
    d['az'] = d['az_table'] + az_offset
    log('coarse az offset', az_offset)
    params, best_geometry, geometry, geometry_records, _ = legacy['geometry_fits'](d, h)
    geometry.to_csv(output/'geometry_comparison.csv', index=False)
    log('geometry', best_geometry, dict(zip(legacy['PARAM_NAMES'], params)))

    native_f, native_fields = study['hfss_samples'](h)
    lmax, angular_coefficients, angular_errors = study['spatial_expansion'](h, native_fields)
    pd.DataFrame(angular_errors).to_csv(output/'spatial_order.csv', index=False)
    train = d['split'] == 0
    baseline_power = legacy['sample_hfss'](d, h, params)
    baseline = legacy['gains'](baseline_power, d['data'], d['good'], train)[:, None]*baseline_power
    all_metrics = metrics(d, baseline, 'hfss')
    descriptions, optimizations = {}, []
    for kind in bases:
        directory = output/kind
        directory.mkdir(exist_ok=True)
        basis = study['SpectralBasis'](kind, native_f, native_fields, d['freqs'])
        descriptions[kind] = basis.description
        pd.DataFrame(basis.records).to_csv(directory/'basis_selection.csv', index=False)
        basis.save(directory/'spectral_basis.npz')
        c0 = np.linalg.lstsq(basis.native_modes, angular_coefficients, rcond=None)[0]
        fit = study['JointFit'](d, params, lmax, basis, c0)
        trials = []
        for penalty in (1e-2, 1e-3, 1e-4):
            coeff, gain, prediction, info = fit.fit(train, penalty, maxiter=maxiter)
            records = metrics(d, prediction, f'{kind}_lambda_{penalty:g}')
            scores = pd.DataFrame(records)
            validation = float(scores.loc[(scores.split == 'validation') & scores.used_for_beam_fit,
                                          'fractional_rms'].median())
            info.update(stage='train', validation_median_rms=validation)
            trials.append((validation, coeff, gain, prediction, info))
            all_metrics.extend(records)
            optimizations.append(info)
            log(kind, 'validation', penalty, validation)
        _, coeff, gain, heldout, selected = min(trials, key=lambda entry: entry[0])
        all_metrics.extend(metrics(d, heldout, kind+'_heldout'))
        final_coeff, final_gain, final_prediction, final_info = fit.fit(
            np.ones(len(d['t']), bool), selected['penalty'], initial=coeff,
            initial_gain=gain, maxiter=maxiter)
        final_info.update(stage='all_sample_refit')
        optimizations.append(final_info)
        all_metrics.extend(metrics(d, final_prediction, kind+'_refit'))
        export_gain, exported = study['export_beam'](directory/'empirical_beam.npz', h, fit,
                                                     final_coeff, final_gain)
        all_metrics.extend(metrics(d, exported, kind+'_exported'))
        export_error = np.sqrt(np.sum((exported-final_prediction)**2*d['good'], axis=1) /
                               np.sum(d['data']**2*d['good'], axis=1))
        assert np.max(export_error[d['fit_channels']]) < .01, export_error
        descriptions[kind].update(selected_penalty=selected['penalty'], lmax=lmax,
            max_export_prediction_error_relative_to_data=float(export_error[d['fit_channels']].max()))
        np.savez_compressed(directory/'diagnostics.npz',
            **{k: d[k] for k in ('az', 'az_table', 'az_pointing_table_v1', 'el', 't', 'data', 'good',
                                 'split', 'channels', 'freqs', 'arms', 'fit_channels')},
            el0_slew=el0_slew, sample_flag=sample_flag, tooth_flag=tooth_flag, az_offset_deg=az_offset,
            hfss=baseline, empirical_heldout=heldout, empirical_refit=final_prediction,
            empirical_exported=exported, initial_coeff=c0, heldout_coeff=coeff,
            coeff=final_coeff, heldout_gain=gain, final_gain=final_gain,
            exported_gain=export_gain, params=params,
            spectral_at_teeth=basis.evaluate(d['freqs']),
            spectral_at_hfss=basis.native_modes, native_hfss_freqs=native_f)
        pd.DataFrame(all_metrics).to_csv(output/'metrics.csv', index=False)
        (output/'optimization.json').write_text(json.dumps(optimizations, indent=2)+'\n')
    pd.DataFrame(geometry_records).to_csv(output/'geometry_metrics.csv', index=False)

    inputs = [SOURCE, STUDY_SOURCE, study['PREVIOUS_SOURCE'], POINT_TABLE, h['path']]
    inputs += [CAMPAIGN/'curation'/f for f in ('antenna_resolution.json', 'horizon_profiles.json',
                                               'transmitter_position.json')]
    inputs += [CAMPAIGN/'data'/Path(f).name for f in d['files']]
    inputs += [Path(flags).resolve()] if flags is not None else []
    artifacts = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                 if p.is_file() and p.name not in ('provenance.json', 'README.md')}
    provenance = dict(
        status='exploratory', pointing=d['point_provenance']['compact'],
        pointing_rule='az_deg: commanded azimuth (motor counts plus one pot offset per anchor segment); el_deg: IMU',
        az_offset_deg=az_offset,
        az_frame='fit az = pointing_table az_deg + az_offset_deg; the geometry az_zero_delta_deg is added on top',
        geometry=dict(zip(legacy['PARAM_NAMES'], params)), geometry_model=best_geometry,
        geometry_source='v0001 HFSS geometry fit, repeated on commanded pointing',
        spectral_bases=descriptions, lmax=lmax, band_mhz=BAND, raster_utc=legacy['RASTER'],
        split='12 deg stripes of commanded azimuth (differs from v0001-v0005)',
        flags=None if flags is None else dict(path=str(Path(flags).resolve().relative_to(ROOT)), version=flag_provenance['version'],
                                              summary=flag_provenance['summary'], bits=flag_provenance['bits']),
        excluded_trajectory='opening near-zero-elevation azimuth slew (56 samples): quality=suspect (AZ_SLEW) in pointing_table v2.0, not loaded',
        maxiter=maxiter, excluded_fit_channels=legacy['SUSPECT_CHANNELS'],
        repositories={name: legacy['revision'](ROOT/name) for name in ('data-analysis', 'eigsep_data')},
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs}, artifact_sha256=artifacts,
        caveats=['Power-only phase and circular-polarization ambiguities remain; coefficients are HFSS-regularized.',
                 'Commanded azimuth assumes the platform followed the motor; see curation/pointing_table.README.md (v2.0).',
                 'The pot/motor offset also jumped by about 10 deg during the opening slew, which is excluded.',
                 'Validation/test stripes are recomputed from commanded azimuth; not a new blind test.',
                 'Only 145-235 MHz is fitted. Other exported slices are normalized HFSS.',
                 'All-sample refit/export scores are not independent validation.'])
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    log('completed', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # No default: the old default named an existing product and would have overwritten it.
    parser.add_argument('--output', type=Path, required=True,
                        help='new product directory, e.g. marjum-2026-07/derived/beam/empirical_raster_v0008')
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--bases', nargs='+', choices=['dpss', 'pca'], default=['dpss', 'pca'])
    parser.add_argument('--flags', type=Path, default=None,
                        help='raster_flags npz; flagged samples and tooth spikes are excluded')
    args = parser.parse_args()
    main(args.output, args.maxiter, tuple(args.bases), args.flags)
