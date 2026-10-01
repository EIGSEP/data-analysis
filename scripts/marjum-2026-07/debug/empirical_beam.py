"""Exploratory raster beam fit; execute from channel_vs_hfss.ipynb or the CLI.

This is a single-study script, not a package or a public interface.
Outputs use the existing HFSS NPZ contract and include independent diagnostics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
import numpy as np
import pandas as pd
import healpy as hp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, minimize
from scipy.special import sph_harm_y
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / 'marjum-2026-07'
RASTER = ('2026-07-17 20:26:00', '2026-07-17 21:28:40')
BAND = (145., 235.)
DF = 250. / 1024
# Declared before fitting, from beam_explorer section 5, not selected by this fit.
SUSPECT_CHANNELS = (720, 816, 912, 960)
PARAM_NAMES = ['polarization_delta_deg', 'az_zero_delta_deg',
               'el_zero_delta_deg', 'tx_east_delta_m', 'tx_north_delta_m']


def log(*args):
    print(*args, flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def revision(path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                dirty=bool(git('status', '--porcelain')))


def rotation(az, el):
    a, e = np.deg2rad(az), np.deg2rad(el)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    r = np.empty((np.size(a), 3, 3))
    r[:, 0] = np.array([ca, -sa, np.zeros_like(a)]).T
    r[:, 1] = np.array([ce * sa, ce * ca, -se]).T
    r[:, 2] = np.array([se * sa, se * ca, ce]).T
    return r


def glitches(el, tol=15.):
    close = np.zeros(len(el), int)
    for k in (-2, -1, 1, 2):
        d = np.abs((el - np.roll(el, k) + 180) % 360 - 180)
        d[slice(0, -k) if k < 0 else slice(-k, None)] = 0
        close += d <= tol
    return close < 3


def load_data():
    import eigsep_data
    from eigsep_data import AntennaResolutionPolicy, MetadataIndex
    eigsep_data.set_campaign_root(CAMPAIGN)
    t0, t1 = [pd.Timestamp(s, tz='UTC').timestamp() for s in RASTER]
    policy = AntennaResolutionPolicy.load(CAMPAIGN / 'curation/antenna_resolution.json')
    bundle = MetadataIndex(CAMPAIGN / 'data').select(time=(t0, t1)).load_bundle(
        antenna='box-air', missing='skip', resolution_policy=policy)
    pointing = pq.read_table(CAMPAIGN / 'curation/pointing_table.parquet').to_pandas()
    pointing = pointing[(pointing.t_utc_s >= t0-60) & (pointing.t_utc_s <= t1+60)]
    meta = bundle.meta.reset_index(drop=True).assign(t=lambda x: x.time_best.astype(float))
    j = pd.merge_asof(meta.sort_values('t').reset_index(names='bundle_row'),
                      pointing.sort_values('t_utc_s')[['t_utc_s', 'az_deg', 'el_deg', 'quality']],
                      left_on='t', right_on='t_utc_s', direction='nearest', tolerance=.35)
    j = j[(j.quality == 'ok') & (j.rfswitch == 'RFANT')].reset_index(drop=True)
    n_glitches = int(glitches(j.el_deg.to_numpy()).sum())
    j = j[~glitches(j.el_deg.to_numpy())].reset_index(drop=True)
    channels = np.arange(8, 1020, 8)
    channels = channels[(channels*DF >= BAND[0]) & (channels*DF <= BAND[1])]
    col = {round(f/DF): i for i, f in enumerate(bundle.freqs_mhz)}
    rows = j.bundle_row.to_numpy()
    raw = np.asarray(bundle.data[rows], float)
    tooth = raw[:, [col[c] for c in channels]].T
    gap = .5*(raw[:, [col[c-4] for c in channels]] + raw[:, [col[c+4] for c in channels]]).T
    data = tooth-gap
    ratio = gap / np.nanmedian(gap, axis=1)[:, None]
    good = (ratio > 1/1.5) & (ratio < 1.5) & np.isfinite(data)
    # Entire 12-degree azimuth stripes are held out: no random adjacent samples.
    az = j.az_deg.to_numpy()
    azbin = np.floor(((az+180) % 360) / 12).astype(int)
    split = np.where(azbin % 5 == 1, 1, np.where(azbin % 5 == 3, 2, 0))
    return dict(data=data, good=good, gap=gap, channels=channels, freqs=channels*DF,
                arms=(channels//8) % 2, az=az, el=j.el_deg.to_numpy(),
                t=j.t.to_numpy(), split=split, files=sorted(set(j.file.astype(str))),
                n_glitches=n_glitches, fit_channels=~np.isin(channels, SUSPECT_CHANNELS),
                metadata_columns=list(j.columns))


def load_hfss():
    from eigsep_data.beam_sim import DEFAULT_BEAM_PATH
    with np.load(DEFAULT_BEAM_PATH) as z:
        fields = z['beam_cart'].copy()
        freqs = z['freqs'].copy()
        nside = int(z['nside'])
    # Only the shape is identifiable with a free gain per tooth. Remove overall
    # amplitude and unobservable common phase before interpolating in frequency.
    amp = np.sqrt(np.mean(np.sum(abs(fields)**2, axis=1), axis=1))
    ref = fields[np.argmin(abs(freqs-190))].ravel()
    phase = np.angle(fields.reshape(len(freqs), -1) @ ref.conj())
    fields /= amp[:, None, None]*np.exp(1j*phase[:, None, None])
    return dict(fields=fields, freqs=freqs, nside=nside, path=Path(DEFAULT_BEAM_PATH),
                amplitude=amp, phase=phase)


def coordinates(d, params, indices=None):
    ant = np.array(json.loads((CAMPAIGN/'curation/horizon_profiles.json').read_text())['antenna_enu_m'])
    tx = np.array(json.loads((CAMPAIGN/'curation/transmitter_position.json').read_text())['best_estimate_enu_m'])
    tx = tx + np.array([params[3], params[4], 0.])
    direction = (tx-ant)/np.linalg.norm(tx-ant)
    idx = slice(None) if indices is None else indices
    r = rotation(-d['az'][idx]+177+params[1], -d['el'][idx]+params[2])
    body = np.einsum('nji,j->ni', r, direction)
    theta = np.arccos(np.clip(body[:, 2], -1, 1))
    phi = np.mod(np.arctan2(body[:, 1], body[:, 0]), 2*np.pi)
    fields = []
    for arm in (0, 1):
        alpha = np.deg2rad(45+params[0]+90*arm)
        e = np.einsum('nji,j->ni', r, [-np.sin(alpha), np.cos(alpha), 0.])
        e -= (e*body).sum(axis=1)[:, None]*body
        fields.append(e)
    return theta, phi, np.array(fields)


def sample_hfss(d, h, params, indices=None, nearest=False):
    th, ph, e = coordinates(d, params, indices)
    beam = interp1d(h['freqs'], h['fields'], axis=0)(d['freqs'])
    if nearest:
        w = beam[:, :, hp.ang2pix(h['nside'], th, ph)]
    else:
        pix, weight = hp.get_interp_weights(h['nside'], th, ph)
        w = sum(beam[:, :, pix[k]]*weight[k] for k in range(4))
    coupling = np.einsum('fcn,fnc->fn', w, e[d['arms']])
    return abs(coupling)**2


def gains(power, data, good, train):
    w = good*train[None, :]
    return np.maximum(np.sum(w*power*data, axis=1) /
                      np.maximum(np.sum(w*power**2, axis=1), 1e-30), 0)


def metrics(d, prediction, label):
    records = []
    for f, freq in enumerate(d['freqs']):
        for s, name in ((0, 'train'), (1, 'validation'), (2, 'test'), (-1, 'all')):
            use = d['good'][f] & ((d['split']==s) if s >= 0 else True)
            y, pred = d['data'][f, use], prediction[f, use]
            err = y-pred
            records.append(dict(model=label, freq_mhz=float(freq), channel=int(d['channels'][f]),
                                arm=int(d['arms'][f]), split=name, n=int(use.sum()),
                                used_for_beam_fit=bool(d['fit_channels'][f]),
                                fractional_rms=float(np.sqrt(np.sum(err**2)/np.sum(y**2))),
                                r_squared=float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),
                                mean_residual=float(err.mean())))
    return records


def geometry_fits(d, h):
    train = d['split']==0
    # Fixed subsampling only for the cheap geometry search; score on full stripes.
    idx = np.flatnonzero(train)[::8]
    usef = np.flatnonzero(d['fit_channels'])
    scale = np.sqrt(np.mean(d['data'][:, idx]**2, axis=1))
    def residual(params):
        p = sample_hfss(d, h, params, idx)
        g = gains(p, d['data'][:, idx], d['good'][:, idx], np.ones(len(idx), bool))
        res = ((g[:, None]*p-d['data'][:, idx])/scale[:, None])
        return (res*d['good'][:, idx])[usef].ravel()/np.sqrt(len(idx)*len(usef))
    bound = np.array([30., 10., 10., 1.5/np.sqrt(2), 1.5/np.sqrt(2)])
    candidates = [('fixed', []), ('polarization', [0]), ('az_el', [1, 2]),
                  ('tx_position', [3, 4]), ('polarization_az_el', [0, 1, 2]),
                  ('all', [0, 1, 2, 3, 4])]
    summary, all_metrics, predictions, parameters = [], [], {}, {}
    for name, free in candidates:
        x = np.zeros(5)
        if free:
            def unpack(v):
                p = np.zeros(5); p[free] = v; return p
            opt = least_squares(lambda v: residual(unpack(v)), x[free],
                                bounds=(-bound[free], bound[free]), diff_step=1e-3,
                                max_nfev=100, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            x = unpack(opt.x)
            singular = np.linalg.svd(opt.jac, compute_uv=False)
            condition = float(singular[0]/max(singular[-1], 1e-30))
        else:
            condition = 1.
        p = sample_hfss(d, h, x)
        g = gains(p, d['data'], d['good'], train)
        prediction = g[:, None]*p
        m = metrics(d, prediction, name)
        scores = pd.DataFrame(m)
        row = dict(model=name, **dict(zip(PARAM_NAMES, x)), jacobian_condition=condition)
        for split in ('train', 'validation', 'test'):
            row[split+'_median_rms'] = float(scores.loc[(scores.split==split)&scores.used_for_beam_fit, 'fractional_rms'].median())
        summary.append(row); all_metrics.extend(m)
        predictions[name] = prediction; parameters[name] = x
        log('geometry', row)
    table = pd.DataFrame(summary)
    best = table.sort_values('validation_median_rms').iloc[0].model
    return parameters[best], best, table, all_metrics, predictions


def harmonics(theta, phi, lmax):
    columns = []
    for ell in range(lmax+1):
        for m in range(-ell, ell+1):
            y = sph_harm_y(ell, abs(m), theta, phi)
            columns.append(y.real if m==0 else np.sqrt(2)*(y.imag if m<0 else y.real))
    return np.column_stack(columns)


def choose_basis(h):
    support = (h['freqs'] >= BAND[0]) & (h['freqs'] <= BAND[1])
    freqs, beam = h['freqs'][support], h['fields'][support]
    th, ph = hp.pix2ang(h['nside'], np.arange(beam.shape[-1]))
    all_y = harmonics(th, ph, 12)
    target = beam.transpose(2, 0, 1).reshape(len(th), -1)
    spatial, coefficients = [], {}
    for ell in (2, 4, 6, 8, 10, 12):
        y = all_y[:, :(ell+1)**2]
        coef = np.linalg.solve(y.T@y, y.T@target)
        prediction = (y@coef).reshape(len(th), len(freqs), 3).transpose(1, 2, 0)
        error = np.linalg.norm((beam-prediction).reshape(len(freqs), -1), axis=1)/np.linalg.norm(beam.reshape(len(freqs), -1), axis=1)
        spatial.append(dict(lmax=ell, worst_relative_field_rms=float(error.max()), median_relative_field_rms=float(np.median(error))))
        coefficients[ell] = coef.reshape((ell+1)**2, len(freqs), 3).transpose(1, 2, 0)
    # Smallest tested order with <=1% field RMS at every HFSS training slice.
    eligible = [r['lmax'] for r in spatial if r['worst_relative_field_rms'] <= .01]
    if not eligible:
        raise RuntimeError('HFSS spatial threshold unmet; inspect rather than silently increase complexity')
    lmax = min(eligible)
    coef = coefficients[lmax]
    x = 2*(freqs-BAND[0])/(BAND[1]-BAND[0])-1
    spectral = []
    for degree in range(1, 9):
        v = np.polynomial.legendre.legvander(x, degree)
        fit = np.linalg.lstsq(v, beam.reshape(len(freqs), -1), rcond=None)[0]
        err = np.linalg.norm(beam.reshape(len(freqs), -1)-v@fit, axis=1)/np.linalg.norm(beam.reshape(len(freqs), -1), axis=1)
        spectral.append(dict(degree=degree, worst_relative_field_rms=float(err.max()), median_relative_field_rms=float(np.median(err))))
    degree = min(r['degree'] for r in spectral if r['worst_relative_field_rms'] <= .01)
    v = np.polynomial.legendre.legvander(x, degree)
    c0 = np.linalg.lstsq(v, coef.reshape(len(freqs), -1), rcond=None)[0]
    log('basis selected', lmax, degree)
    return lmax, degree, c0, spatial, spectral


class EmpiricalFit:
    def __init__(self, d, params, lmax, degree, c0):
        self.d, self.c0 = d, c0
        self.lmax, self.degree = lmax, degree
        th, ph, e = coordinates(d, params)
        y = harmonics(th, ph, lmax)
        self.design = np.array([np.concatenate([y*arm[:, c, None] for c in range(3)], axis=1) for arm in e])
        x = 2*(d['freqs']-BAND[0])/(BAND[1]-BAND[0])-1
        self.spectral = np.polynomial.legendre.legvander(x, degree)
        self.shape = c0.shape

    def power(self, coeff):
        q = np.empty_like(self.d['data'], dtype=complex)
        for arm in (0, 1):
            f = self.d['arms']==arm
            q[f] = self.spectral[f] @ coeff @ self.design[arm].T
        return abs(q)**2

    def fit(self, train, penalty, initial=None, maxiter=800):
        d = self.d
        valid = d['good'] & train[None, :] & d['fit_channels'][:, None]
        keep = np.flatnonzero(train)
        y = d['data'][:, keep]
        valid = valid[:, keep]
        scale = np.sqrt(np.sum(valid*y*y, axis=1)/np.maximum(valid.sum(axis=1), 1))
        scale = np.maximum(scale, 1.)
        target = y/scale[:, None]
        weight = valid/np.maximum(valid.sum(axis=1), 1)[:, None]/d['fit_channels'].sum()
        designs = self.design[:, keep]
        norm = np.sum(abs(self.c0)**2)
        n = self.c0.size
        def unpack(x):
            return (x[:n]+1j*x[n:]).reshape(self.shape)
        def objective(x):
            c = unpack(x)
            q = np.empty_like(target, dtype=complex)
            for arm in (0, 1):
                f = d['arms']==arm
                q[f] = self.spectral[f] @ c @ designs[arm].T
            power = abs(q)**2
            g = np.maximum(np.sum(weight*power*target, axis=1)/np.maximum(np.sum(weight*power**2, axis=1), 1e-30), 0.)
            residual = g[:, None]*power-target
            delta = c-self.c0
            value = np.sum(weight*residual**2)+penalty*np.sum(abs(delta)**2)/norm
            dq = 4*weight*residual*g[:, None]*q
            grad = np.zeros_like(c)
            for arm in (0, 1):
                f = d['arms']==arm
                grad += self.spectral[f].T @ (dq[f] @ designs[arm])
            grad += 2*penalty*delta/norm
            return value, np.r_[grad.real.ravel(), grad.imag.ravel()]
        start = self.c0 if initial is None else initial
        x0 = np.r_[start.real.ravel(), start.imag.ravel()]
        # A directional derivative check tests the profiled-gain gradient.
        rng = np.random.default_rng(42)
        direction = rng.normal(size=len(x0)); direction /= np.linalg.norm(direction)
        eps = 1e-5
        value, grad = objective(x0)
        finite_difference = (objective(x0+eps*direction)[0]-objective(x0-eps*direction)[0])/(2*eps)
        analytic = grad@direction
        assert np.isclose(finite_difference, analytic, rtol=1e-3, atol=1e-7), (finite_difference, analytic)
        started = time.monotonic()
        opt = minimize(objective, x0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxcor=20))
        c = unpack(opt.x)
        p = self.power(c)
        g = gains(p, d['data'], d['good'], train)
        info = dict(penalty=penalty, iterations=int(opt.nit), success=bool(opt.success),
                    message=str(opt.message), loss=float(opt.fun), seconds=time.monotonic()-started,
                    gradient_directional_error=float(abs(finite_difference-analytic)))
        log('empirical optimization', info)
        return c, g, g[:, None]*p, info


def save_beam(path, h, model, coeff, params, d):
    # All original HFSS frequencies and keys are retained. Outside BAND the
    # map is explicitly HFSS-only, so the reader's default drop_last is safe.
    export_freqs = np.unique(np.r_[h['freqs'], d['freqs']])
    fields = interp1d(h['freqs'], h['fields'], axis=0)(export_freqs)
    th, ph = hp.pix2ang(h['nside'], np.arange(fields.shape[-1]))
    y = harmonics(th, ph, model.lmax)
    support = (export_freqs>=BAND[0]) & (export_freqs<=BAND[1])
    x = 2*(export_freqs[support]-BAND[0])/(BAND[1]-BAND[0])-1
    s = np.polynomial.legendre.legvander(x, model.degree)
    fields[support] = (s@coeff).reshape(s.shape[0], 3, y.shape[1]) @ y.T
    direction = np.array(hp.pix2vec(h['nside'], np.arange(fields.shape[-1])))
    fields -= np.sum(fields*direction[None], axis=1)[:, None]*direction[None]
    # Unit spherical-average total power per frequency: arbitrary channel
    # gain remains a separate nuisance; no measured absolute gain is claimed.
    fields /= np.sqrt(np.mean(np.sum(abs(fields)**2, axis=1), axis=1))[:, None, None]
    eth = np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
    eph = np.array([-np.sin(ph), np.cos(ph), np.zeros_like(ph)])
    gain_th = abs(np.sum(fields*eth[None], axis=1))**2
    gain_ph = abs(np.sum(fields*eph[None], axis=1))**2
    coverage = np.zeros(fields.shape[-1], int)
    td, pd_, _ = coordinates(d, params)
    np.add.at(coverage, hp.ang2pix(h['nside'], td, pd_), 1)
    np.savez_compressed(path, beam_cart=fields, gain_th=gain_th, gain_ph=gain_ph,
                        freqs=export_freqs, nside=h['nside'], empirical_frequency_mask=support,
                        sample_count=coverage, fitted_frequency_limits_mhz=np.array(BAND),
                        geometry_parameters=params, geometry_parameter_names=np.array(PARAM_NAMES),
                        normalization=np.array('unit spherical-average total power; arbitrary per-channel gains'),
                        status=np.array('exploratory; HFSS phase prior; unsampled directions not validated'))
    from eigsep_data.beam_sim import read_beam
    reloaded, _, freq = read_beam(path, drop_last=False)
    assert np.array_equal(freq, export_freqs) and np.array_equal(reloaded, fields)
    assert np.isfinite(fields).all()
    assert np.allclose(gain_th+gain_ph, np.sum(abs(fields)**2, axis=1), atol=1e-10)
    assert np.max(abs(np.sum(fields*direction[None], axis=1))) < 1e-10
    return fields, export_freqs


def main(output, maxiter=800):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    d, h = load_data(), load_hfss()
    log('data', d['data'].shape, 'files', len(d['files']), 'split counts', np.bincount(d['split']))
    params, best, geometry, records, predictions = geometry_fits(d, h)
    geometry.to_csv(output/'geometry_comparison.csv', index=False)
    lmax, degree, c0, spatial, spectral = choose_basis(h)
    pd.DataFrame(spatial).to_csv(output/'spatial_order.csv', index=False)
    pd.DataFrame(spectral).to_csv(output/'spectral_order.csv', index=False)
    model = EmpiricalFit(d, params, lmax, degree, c0)
    train = d['split']==0
    fits = []
    for penalty in (1e-2, 1e-3, 1e-4):
        coeff, g, pred, info = model.fit(train, penalty, maxiter=maxiter)
        label = f'empirical_lambda_{penalty:g}'
        m = metrics(d, pred, label); records.extend(m)
        score = pd.DataFrame(m)
        val = float(score.loc[(score.split=='validation')&score.used_for_beam_fit, 'fractional_rms'].median())
        info['validation_median_rms'] = val
        fits.append((val, coeff, g, pred, info))
        log('validation', penalty, val)
    fits.sort(key=lambda item: item[0])
    _, coeff, gain, pred, chosen = fits[0]
    records.extend(metrics(d, pred, 'empirical_heldout'))
    # Refit on all samples only AFTER saving independent held-out predictions.
    final_coeff, final_gain, final_pred, final_info = model.fit(np.ones(len(d['t']), bool),
        chosen['penalty'], initial=coeff, maxiter=maxiter)
    records.extend(metrics(d, final_pred, 'empirical_refit'))
    beam_path = output/'empirical_beam.npz'
    exported, export_freqs = save_beam(beam_path, h, model, final_coeff, params, d)
    # Compare the serialized frequency-sampled model through the same HFSS
    # sampling path. This includes pixel and frequency interpolation errors.
    eh = dict(h, fields=exported, freqs=export_freqs)
    exported_power = sample_hfss(d, eh, params)
    exported_gain = gains(exported_power, d['data'], d['good'], np.ones(len(d['t']), bool))
    exported_prediction = exported_gain[:, None]*exported_power
    records.extend(metrics(d, exported_prediction, 'empirical_exported'))
    pd.DataFrame(records).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame([f[-1] for f in fits]+[dict(final_info, stage='all_sample_refit')]).to_csv(output/'optimization.csv', index=False)
    np.savez_compressed(output/'diagnostics.npz', az=d['az'], el=d['el'], t=d['t'],
        data=d['data'], good=d['good'], split=d['split'], channels=d['channels'],
        freqs=d['freqs'], arms=d['arms'], fit_channels=d['fit_channels'],
        fixed=predictions['fixed'], hfss=predictions[best], empirical_heldout=pred,
        empirical_refit=final_pred, empirical_exported=exported_prediction,
        final_gain=final_gain, exported_gain=exported_gain, params=params,
        coeff=final_coeff, initial_coeff=c0)
    inputs = [h['path'], CAMPAIGN/'curation/pointing_table.parquet',
              CAMPAIGN/'curation/antenna_resolution.json', CAMPAIGN/'curation/horizon_profiles.json',
              CAMPAIGN/'curation/transmitter_position.json', Path(__file__)]
    inputs += [CAMPAIGN/'data'/Path(f).name for f in d['files']]
    provenance = dict(status='exploratory', raster_utc=RASTER, band_mhz=BAND,
        normalization='unit spherical-average total power; independent nuisance gain per channel',
        geometry=dict(zip(PARAM_NAMES, params)), selected_geometry=best,
        lmax=lmax, spectral_degree=degree, penalty=chosen['penalty'],
        excluded_fit_channels=SUSPECT_CHANNELS, n_pointing_glitches=d['n_glitches'],
        repositories={name: revision(ROOT/name) for name in ('data-analysis','eigsep_data')},
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
        artifact_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()},
        caveats=['Power-only measurements do not identify complex phase or circular-polarization response.',
                 'HFSS-centered coefficient regularization chooses a phase-compatible solution.',
                 'Unsampled directions are prior-dominated, not measured.',
                 'Outside 145-235 MHz the exported map is HFSS-only.',
                 'Four predeclared suspect frequencies are excluded from fitting but reported.',
                 'The 8-channel comb is treated as the transmitter following Aaron and beam_explorer, superseding conflicting archival labels.',
                 'Geometry parameters are nuisance alignment corrections, not a new surveyed position.'])
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    log('finished', output)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'scratch/empirical-beam')
    parser.add_argument('--maxiter', type=int, default=800)
    args = parser.parse_args()
    main(args.output, args.maxiter)
