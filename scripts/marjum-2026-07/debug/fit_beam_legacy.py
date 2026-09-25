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

Defaults (2026-09-25) reproduce the reference, v0011: --convention highline
--background dpss --teeth tooth_selection_v0002 --flags raster_flags_v0001.
v0010: --teeth tooth_selection_v0001. v0009: that plus --convention legacy.
v0008: --convention legacy --background gap --teeth legacy (flags as the default). The background and teeth options, added 2026-09-24:
- --background dpss: the background under each tooth is the local DPSS fit of
  tooth_background.py, instead of the mean of the channels +/-4 away (gap
  differencing). Chosen in debug/tooth_background_debug.ipynb and
  debug/tooth_background_basis_debug.ipynb.
- --teeth: a tooth_selection JSON (build_tooth_selection.py). Its channels are
  the fitting teeth, and the fitted band becomes their frequency span, replacing
  the fixed 145-235 MHz band and suspect list. That band is a module global of
  empirical_beam_spectral_no_el0.py and empirical_beam.py, which v0006-v0008
  hash as inputs, so it is overridden here at run time rather than edited there.

Pointing convention, changed 2026-09-25 (Aaron: the earlier one was wrong).
--convention highline (the default) maps the HFSS body frame (boresight +z,
dipole arm +x) to ENU as

    v_ENU = Rz(psi) Rx(el + d_el) Rz(az + d_az) v_body,   psi = 142.164 deg,

where az and el are the pointing table's az_deg and el_deg. az rolls the
antenna about its boresight (right-handed), with the arm along the highline at
az = 0; el tips the boresight about the highline axis: 0 = zenith, +90 = the
horizon toward compass 37.836 deg, 180 = nadir. psi is the GPS highline
direction, counter-clockwise from East (known_quantities v0001, +/-0.6 deg).
In this frame az = 0 should put the arm along the highline. The transmitter
data instead want the table's az zero about 16 deg away (v0009 inputs, HFSS
only), so a constant az offset is fitted as a calibration (Aaron, 2026-09-25)
and reported as the measured offset of the table's az zero from the highline.
The transmitter polarization alpha is an ENU angle: the dipole axis is
(-sin a, cos a, 0) for arm 0, and arm 1 is 90 deg from it. The az offset and
alpha centre come from a joint coarse scan with HFSS alone (az offset
-60..60 deg, alpha 0..180 deg), and the geometry fit refines both.
--convention legacy is the v0001-v0009 convention: Rx(-el + d_el) Rz(-(az + offset)
+ 177 + d_az) in bare ENU, with a scanned coarse az offset and alpha centred at
45 deg. It is kept only to reproduce those products.
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
BG_SOURCE = SOURCE.with_name('tooth_background.py')
background_module = runpy.run_path(str(BG_SOURCE))


PSI_DEG = 142.164          # highline direction, deg ccw from East (known_quantities v0001)
RELEASE = CAMPAIGN / 'imgs/fits/v0001_marjum_geometry/shared.json'
ALPHA0 = 45.0             # transmitter polarization centre; set by coarse_offset_alpha under --convention highline


def rotation_highline(az, el):
    """Rz(psi) Rx(el) Rz(az) for arrays of az, el in degrees: (n, 3, 3), body -> ENU."""
    p = np.deg2rad(PSI_DEG)
    rz = np.array([[np.cos(p), -np.sin(p), 0.], [np.sin(p), np.cos(p), 0.], [0., 0., 1.]])
    return np.einsum('ij,njk->nik', rz, legacy['rotation'](az, el))


def coordinates_highline(d, params, indices=None):
    """Drop-in for empirical_beam.coordinates in the highline convention: the transmitter
    direction in body (theta, phi) and each arm's field unit vector in the body frame."""
    ant = np.asarray(json.loads(RELEASE.read_text())['antenna_91m_era']['position_enu_m'], float)
    tx = np.array(json.loads((CAMPAIGN/'curation/transmitter_position.json').read_text())['best_estimate_enu_m'])
    tx = tx + np.array([params[3], params[4], 0.])
    direction = (tx - ant) / np.linalg.norm(tx - ant)
    idx = slice(None) if indices is None else indices
    r = rotation_highline(d['az'][idx] + params[1], d['el'][idx] + params[2])
    body = np.einsum('nji,j->ni', r, direction)                # R^T: ENU -> body
    theta = np.arccos(np.clip(body[:, 2], -1, 1))
    phi = np.mod(np.arctan2(body[:, 1], body[:, 0]), 2*np.pi)
    fields = []
    for arm in (0, 1):
        alpha = np.deg2rad(ALPHA0 + params[0] + 90*arm)
        e = np.einsum('nji,j->ni', r, [-np.sin(alpha), np.cos(alpha), 0.])
        e -= (e*body).sum(axis=1)[:, None]*body
        fields.append(e)
    return theta, phi, np.array(fields)


def use_convention(name):
    """Route every coordinates() call in the imported fit chain to the chosen convention."""
    fn = coordinates_highline if name == 'highline' else LEGACY_COORDINATES
    legacy['coordinates'] = fn                                  # study code looks it up in this dict
    legacy['sample_hfss'].__globals__['coordinates'] = fn       # legacy code uses its own globals


LEGACY_COORDINATES = legacy['coordinates']


def set_band(band):
    """Point the imported study and legacy modules at a new fitted band. runpy returns a
    copy of each module's globals, so the functions' own __globals__ are updated."""
    global BAND
    BAND = tuple(band)
    for fn in (study['hfss_samples'], study['export_beam'], legacy['sample_hfss']):
        fn.__globals__['BAND'] = BAND
DF = legacy['DF']
log, digest, metrics = legacy['log'], legacy['digest'], legacy['metrics']


def load_data(background='gap', teeth=None):
    """legacy load_data with commanded pointing and commanded-az stripes.

    background: 'gap' (tooth minus the mean of the channels +/-4 away) or 'dpss'
    (tooth_background.py). teeth: fitting channels; None keeps the legacy band and
    suspect list. The gap-ratio dropout mask is used with either background.
    """
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
    if teeth is None:
        channels = np.arange(8, 1020, 8)
        channels = channels[(channels*DF >= BAND[0]) & (channels*DF <= BAND[1])]
        fit_channels = ~np.isin(channels, legacy['SUSPECT_CHANNELS'])
    else:
        channels = np.asarray(sorted(teeth))
        fit_channels = np.ones(len(channels), bool)
    col = {round(f/DF): i for i, f in enumerate(bundle.freqs_mhz)}
    raw = np.asarray(bundle.data[j.bundle_row.to_numpy()], float)
    tooth = raw[:, [col[c] for c in channels]].T
    gap = .5*(raw[:, [col[c-4] for c in channels]] + raw[:, [col[c+4] for c in channels]]).T
    ratio = gap / np.nanmedian(gap, axis=1)[:, None]
    if background == 'dpss':
        assert np.array_equal([col[c] for c in range(1024)], np.arange(1024))
        under = background_module['tooth_background'](raw, channels).T
    else:
        under = gap
    az = j.az_deg.to_numpy()
    assert np.isfinite(az).all()
    # Entire 12-degree stripes of commanded azimuth are held out.
    azbin = np.floor(((az+180) % 360) / 12).astype(int)
    split = np.where(azbin % 5 == 1, 1, np.where(azbin % 5 == 3, 2, 0))
    return dict(data=tooth-under, good=(ratio > 1/1.5) & (ratio < 1.5) & np.isfinite(tooth-under),
                gap=gap, background=under, channels=channels, freqs=channels*DF, arms=(channels//8) % 2,
                az=az, az_table=az.copy(), az_pointing_table_v1=j.az_deg_v1.to_numpy(),
                el=j.el_deg.to_numpy(), t=j.t.to_numpy(), split=split,
                files=sorted(set(j.file.astype(str))), n_glitches=int(glitch.sum()),
                fit_channels=fit_channels, point_provenance=point_provenance)


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


def coarse_offset_alpha(d, h):
    """HFSS-only, zero-geometry joint scan of the az offset and the transmitter polarization
    centre on training samples: a coarse grid (3 deg in offset, 6 deg in alpha), then 1 deg
    around its minimum. Sets ALPHA0; returns (offset, alpha, table of every point scored)."""
    global ALPHA0
    idx = np.flatnonzero(d['split'] == 0)[::4]
    usef = np.flatnonzero(d['fit_channels'])
    scale = np.sqrt(np.mean(d['data'][:, idx]**2, axis=1))
    base = d['az'].copy()
    rows = []

    def score(offset, alpha):
        global ALPHA0
        d['az'], ALPHA0 = base + offset, float(alpha)
        p = legacy['sample_hfss'](d, h, np.zeros(5), idx)
        g = legacy['gains'](p, d['data'][:, idx], d['good'][:, idx], np.ones(len(idx), bool))
        r = ((g[:, None]*p - d['data'][:, idx])/scale[:, None])*d['good'][:, idx]
        rows.append(dict(az_offset_deg=float(offset), alpha_deg=float(alpha),
                         normalized_rms=float(np.sqrt(np.mean(r[usef]**2)))))

    for offset in np.arange(-60., 60.01, 3.):
        for alpha in np.arange(0., 180., 6.):
            score(offset, alpha)
    best = min(rows, key=lambda x: x['normalized_rms'])
    for offset in np.arange(best['az_offset_deg'] - 3, best['az_offset_deg'] + 3.01, 1.):
        for alpha in np.arange(best['alpha_deg'] - 6, best['alpha_deg'] + 6.01, 1.):
            score(offset, alpha % 180)
    best = min(rows, key=lambda x: x['normalized_rms'])
    d['az'], ALPHA0 = base, best['alpha_deg']
    return best['az_offset_deg'], best['alpha_deg'], pd.DataFrame(rows)


def main(output, maxiter=3000, bases=('dpss', 'pca'), flags=None, background='gap', teeth=None,
         convention='highline'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    use_convention(convention)
    log('pointing convention', convention)
    selection = None
    if teeth is not None:
        selection = json.loads(Path(teeth).read_text())
        freqs = np.asarray(selection['channels']) * DF
        # One channel of margin: the band is used as a closed interval on tooth frequencies.
        set_band((float(freqs.min()) - DF, float(freqs.max()) + DF))
        log('tooth selection', teeth, len(selection['channels']), 'teeth; band', BAND)
    d, h = load_data(background, None if selection is None else selection['channels']), legacy['load_hfss']()
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
            order = np.argsort(f['t'])
            pos = np.clip(np.searchsorted(f['t'][order], d['t']), 0, len(order) - 1)
            near = order[np.where(abs(f['t'][order][pos] - d['t']) <= 1e-5, pos, 0)]
            assert np.all(abs(f['t'][near] - d['t']) <= 1e-5), 'samples missing from flags'
            sample_flag = f['sample_flag'][near].astype(bool)
            # Tooth spikes exist only for the flags' own channels (the v0006 teeth, 145-235 MHz);
            # other teeth get sample flags only.
            fcol = {int(c): i for i, c in enumerate(f['channels'])}
            for i, c in enumerate(d['channels']):
                if int(c) in fcol:
                    tooth_flag[i] = f['tooth_flag'][fcol[int(c)], near].astype(bool)
            flag_provenance = json.loads(str(f['provenance']))
        d['good'] &= ~sample_flag[None, :] & ~tooth_flag
        log('flags', flags, 'samples', int(sample_flag.sum()), 'tooth spikes', int(tooth_flag.sum()))
    log('data', d['data'].shape, 'split counts', np.bincount(d['split']))

    if convention == 'highline':
        az_offset, alpha0, scan = coarse_offset_alpha(d, h)
        scan.to_csv(output/'az_offset_alpha_scan.csv', index=False)
        at_zero = scan[scan.az_offset_deg == 0.0].normalized_rms.min()
        log('coarse az offset %+.0f deg, alpha %.0f deg; rms %.4f (best at offset 0: %.4f)'
            % (az_offset, alpha0, scan.normalized_rms.min(), at_zero))
    else:
        alpha0 = 45.0
        az_offset, scan = coarse_az_offset(d, h)
        scan.to_csv(output/'az_offset_scan.csv', index=False)
        log('coarse az offset', az_offset)
    d['az'] = d['az_table'] + az_offset
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
            alpha0_deg=alpha0, convention=np.array(convention), psi_deg=PSI_DEG,
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
    inputs += [RELEASE] if convention == 'highline' else []
    inputs += [CAMPAIGN/'curation'/f for f in ('antenna_resolution.json', 'horizon_profiles.json',
                                               'transmitter_position.json')]
    inputs += [CAMPAIGN/'data'/Path(f).name for f in d['files']]
    inputs += [Path(flags).resolve()] if flags is not None else []
    inputs += [BG_SOURCE] if background == 'dpss' else []
    inputs += [Path(teeth).resolve()] if teeth is not None else []
    artifacts = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                 if p.is_file() and p.name not in ('provenance.json', 'README.md')}
    provenance = dict(
        status='exploratory', pointing=d['point_provenance']['compact'],
        pointing_rule='az_deg: commanded azimuth (motor counts plus one pot offset per anchor segment); el_deg: IMU',
        az_offset_deg=az_offset,
        convention=convention,
        rotation=('v_ENU = Rz(psi) Rx(el + el_zero_delta) Rz(az + az_zero_delta) v_body, psi = %.3f deg; '
                  'body: boresight +z, dipole arm +x; el 0 = zenith, +90 = horizon toward compass 37.836 deg'
                  % PSI_DEG if convention == 'highline' else
                  'legacy: Rx(-el + el_zero_delta) Rz(-(az + az_offset) + 177 + az_zero_delta) in bare ENU'),
        alpha0_deg=alpha0,
        polarization='arm-0 transmitter dipole axis (-sin a, cos a, 0) in ENU, a = alpha0 + polarization_delta; '
                     'arm 1 at a + 90' if convention == 'highline' else 'legacy: a = 45 + polarization_delta in the legacy frame',
        table_az_zero_from_highline_deg=(az_offset + dict(zip(legacy['PARAM_NAMES'], params))['az_zero_delta_deg']
                                         if convention == 'highline' else None),
        az_frame=('fit az = pointing_table az_deg + az_offset_deg (a fitted calibration: the highline convention '
                  'expects 0); az_zero_delta_deg is fitted on top'
                  if convention == 'highline' else
                  'fit az = pointing_table az_deg + az_offset_deg; the geometry az_zero_delta_deg is added on top'),
        geometry=dict(zip(legacy['PARAM_NAMES'], params)), geometry_model=best_geometry,
        geometry_source='v0001 HFSS geometry fit, repeated on commanded pointing',
        spectral_bases=descriptions, lmax=lmax, band_mhz=BAND, raster_utc=legacy['RASTER'],
        split='12 deg stripes of commanded azimuth (differs from v0001-v0005)',
        flags=None if flags is None else dict(path=str(Path(flags).resolve().relative_to(ROOT)), version=flag_provenance['version'],
                                              summary=flag_provenance['summary'], bits=flag_provenance['bits']),
        excluded_trajectory='opening near-zero-elevation azimuth slew (56 samples): quality=suspect (AZ_SLEW) in pointing_table v2.0, not loaded',
        maxiter=maxiter,
        excluded_fit_channels=legacy['SUSPECT_CHANNELS'] if selection is None else 'not loaded: see tooth_selection',
        background=('gap differencing: tooth minus the mean of channels +/-4' if background == 'gap'
                    else background_module['DESCRIPTION']),
        tooth_selection=None if selection is None else dict(
            path=str(Path(teeth).resolve().relative_to(ROOT)), version=selection['version'],
            rule=selection['rule'], counts=selection['counts']),
        repositories={name: legacy['revision'](ROOT/name) for name in ('data-analysis', 'eigsep_data')},
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs}, artifact_sha256=artifacts,
        caveats=['Power-only phase and circular-polarization ambiguities remain; coefficients are HFSS-regularized.',
                 'Commanded azimuth assumes the platform followed the motor; see curation/pointing_table.README.md (v2.0).',
                 'The pot/motor offset also jumped by about 10 deg during the opening slew, which is excluded.',
                 'Validation/test stripes are recomputed from commanded azimuth; not a new blind test.',
                 'Only %.1f-%.1f MHz is fitted. Other exported slices are normalized HFSS.' % BAND,
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
    parser.add_argument('--flags', default=str(CAMPAIGN/'derived/beam/raster_flags_v0001.npz'),
                        help="raster_flags npz; flagged samples and tooth spikes are excluded ('none' for no flags)")
    parser.add_argument('--background', choices=['gap', 'dpss'], default='dpss',
                        help='background under each tooth (default dpss, as v0009-v0010; gap as v0001-v0008)')
    parser.add_argument('--teeth', default=str(CAMPAIGN/'derived/beam/tooth_selection_v0002.json'),
                        help="tooth_selection JSON ('legacy' for the 145-235 MHz band and suspect list of v0001-v0008)")
    parser.add_argument('--convention', choices=['highline', 'legacy'], default='highline',
                        help='pointing convention (default highline; legacy reproduces v0006-v0009)')
    args = parser.parse_args()
    flags = None if args.flags == 'none' else Path(args.flags)
    teeth = None if args.teeth == 'legacy' else Path(args.teeth)
    main(args.output, args.maxiter, tuple(args.bases), flags, args.background, teeth, args.convention)
