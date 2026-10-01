"""DPSS/PCA joint fit excluding the initial near-zero-elevation azimuth slew.

Execute this script, not an importable package. The previous study supplies
unchanged data selection, geometry, coordinate and metric helpers via runpy;
its polynomial basis/fit/export routines are never invoked.
"""
from pathlib import Path
import argparse
import json
import os
import runpy
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import healpy as hp
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal.windows import dpss

jax.config.update('jax_enable_x64', True)
SOURCE = Path(__file__).resolve()
PREVIOUS_SOURCE = SOURCE.with_name('empirical_beam.py')
legacy = runpy.run_path(str(PREVIOUS_SOURCE))
ROOT, CAMPAIGN, BAND = (legacy[k] for k in ('ROOT', 'CAMPAIGN', 'BAND'))
PREVIOUS_PRODUCT = CAMPAIGN/'derived/beam/empirical_raster_v0001'
log, digest, metrics = (legacy[k] for k in ('log', 'digest', 'metrics'))
THRESHOLD = .01


def hfss_samples(h):
    # Include one native HFSS sample beyond each band edge: interpolation at
    # the first odd tooth must not extrapolate the PCA modes.
    native = h['freqs']
    first = max(0, np.searchsorted(native, BAND[0])-1)
    last = min(len(native), np.searchsorted(native, BAND[1])+1)
    return native[first:last], h['fields'][first:last]


def spatial_expansion(h, fields):
    th, ph = hp.pix2ang(h['nside'], np.arange(fields.shape[-1]))
    all_y = legacy['harmonics'](th, ph, 12)
    target = fields.transpose(2, 0, 1).reshape(len(th), -1)
    records, coefficients = [], {}
    for ell in (2, 4, 6, 8, 10, 12):
        y = all_y[:, :(ell+1)**2]
        c = np.linalg.solve(y.T@y, y.T@target)
        pred = (y@c).reshape(len(th), len(fields), 3).transpose(1, 2, 0)
        error = np.linalg.norm((fields-pred).reshape(len(fields), -1), axis=1)/np.linalg.norm(fields.reshape(len(fields), -1), axis=1)
        records.append(dict(lmax=ell, worst_relative_field_rms=float(error.max()),
                            median_relative_field_rms=float(np.median(error))))
        coefficients[ell] = c.reshape((ell+1)**2, len(fields), 3).transpose(1, 2, 0)
    ell = min(r['lmax'] for r in records if r['worst_relative_field_rms'] <= THRESHOLD)
    return ell, coefficients[ell].reshape(len(fields), -1), records


class SpectralBasis:
    def __init__(self, kind, frequencies, fields, tooth_frequencies):
        self.kind, self.native_freqs = kind, frequencies
        target = fields.reshape(len(frequencies), -1)
        denominator = np.linalg.norm(target, axis=1)
        records = []
        if kind == 'pca':
            # Left singular vectors are spectral modes. Angular eigen-beams
            # are discarded; each spectral mode gets free angular coefficients.
            u, singular, _ = np.linalg.svd(target, full_matrices=False)
            self.singular_values = singular
            for k in range(1, min(10, len(u))+1):
                s = u[:, :k]
                error = np.linalg.norm(target-s@(s.conj().T@target), axis=1)/denominator
                records.append(dict(modes=k, worst_relative_field_rms=float(error.max()),
                                    median_relative_field_rms=float(np.median(error)),
                                    cumulative_field_energy=float(np.sum(singular[:k]**2)/np.sum(singular**2))))
            self.count = min(r['modes'] for r in records if r['worst_relative_field_rms'] <= THRESHOLD)
            self.native_modes = u[:, :self.count]
            self.description = dict(kind=kind, modes=self.count,
                construction='uncentered complex SVD; left singular vectors across frequency',
                interpolation='linear resampling of fixed learned spectral functions; no polynomial regressors')
        else:
            # One data-band of padding on either side avoids forcing a DPSS
            # taper at the measured band edges. Only >=90%-concentrated modes
            # are used. The smallest tested delay bandwidth passing HFSS sets
            # the spectral scale; this is not a measured physical path delay.
            self.df = float(np.diff(tooth_frequencies).min())
            pad = len(tooth_frequencies)
            self.grid = tooth_frequencies[0]+np.arange(-pad, len(tooth_frequencies)+pad)*self.df
            selected = None
            for delay_ns in (1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20):
                w = delay_ns*1e-3*self.df
                nw = len(self.grid)*w
                modes, eigen = dpss(len(self.grid), nw, Kmax=min(len(self.grid), int(np.ceil(2*nw))+4), return_ratios=True)
                keep = eigen >= .9
                if not np.any(keep):
                    continue
                raw = (2*w*np.sinc(2*w*(frequencies[:, None]-self.grid[None, :])/self.df))@(modes[keep].T/eigen[keep])
                q, r = np.linalg.qr(raw)
                error = np.linalg.norm(target-q@(q.T@target), axis=1)/denominator
                records.append(dict(delay_half_width_ns=delay_ns, time_bandwidth=nw,
                    modes=int(keep.sum()), minimum_concentration=float(eigen[keep].min()),
                    worst_relative_field_rms=float(error.max()), median_relative_field_rms=float(np.median(error))))
                if selected is None and error.max() <= THRESHOLD:
                    selected = (delay_ns, w, nw, modes[keep], eigen[keep], r)
            if selected is None:
                raise RuntimeError('No HFSS-supported DPSS bandwidth passed the threshold')
            delay, self.w, nw, self.modes, self.eigen, self.r = selected
            self.count = len(self.eigen)
            self.native_modes = self.evaluate(frequencies)
            self.description = dict(kind=kind, modes=self.count, delay_half_width_ns=delay,
                time_bandwidth=nw, grid_size=len(self.grid), grid_spacing_mhz=self.df,
                grid_limits_mhz=[float(self.grid[0]), float(self.grid[-1])],
                concentration_threshold=.9, minimum_concentration=float(self.eigen.min()),
                padding_samples_each_side=pad,
                interpolation='bandlimited Nystrom sinc continuation; QR changes coordinates within the same span')
        self.records = records
        assert np.allclose(self.native_modes.conj().T@self.native_modes, np.eye(self.count), atol=1e-10)
        log('basis', self.description)

    def evaluate(self, frequencies):
        f = np.asarray(frequencies)
        if self.kind == 'pca':
            return interp1d(self.native_freqs, self.native_modes, axis=0, bounds_error=True)(f)
        raw = (2*self.w*np.sinc(2*self.w*(f[:, None]-self.grid[None, :])/self.df))@(self.modes.T/self.eigen)
        return np.linalg.solve(self.r.T, raw.T).T

    def save(self, path):
        arrays = dict(kind=np.array(self.kind), native_freqs=self.native_freqs,
                      native_modes=self.native_modes)
        if self.kind == 'pca':
            arrays['singular_values'] = self.singular_values
        else:
            arrays.update(grid_mhz=self.grid, dpss_modes=self.modes, concentrations=self.eigen,
                          half_bandwidth_cycles_per_sample=self.w, qr_r=self.r)
        np.savez_compressed(path, **arrays)


class JointFit:
    def __init__(self, d, params, lmax, basis, c0):
        self.d, self.params, self.lmax, self.basis, self.c0 = d, params, lmax, basis, c0
        th, ph, e = legacy['coordinates'](d, params)
        y = legacy['harmonics'](th, ph, lmax)
        self.design = np.array([np.concatenate([y*arm[:, c, None] for c in range(3)], axis=1) for arm in e])
        self.spectral = basis.evaluate(d['freqs'])

    def power(self, coeff):
        q = np.empty_like(self.d['data'], dtype=complex)
        for arm in (0, 1):
            use = self.d['arms']==arm
            q[use] = self.spectral[use]@coeff@self.design[arm].T
        return abs(q)**2

    def fit(self, train, penalty, initial=None, initial_gain=None, maxiter=3000):
        d = self.d
        rows = np.flatnonzero(train)
        usef = np.flatnonzero(d['fit_channels'])
        valid = d['good'][usef][:, rows]
        observations = d['data'][usef][:, rows]
        scale = np.sqrt(np.sum(np.where(valid, observations**2, 0), axis=1)/valid.sum(axis=1))
        target = jnp.asarray(np.where(valid, observations/scale[:, None], 0))
        weight = jnp.asarray(valid/valid.sum(axis=1)[:, None]/len(usef))
        spectral = jnp.asarray(self.spectral[usef])
        design = jnp.asarray(self.design[:, rows])
        groups = [np.flatnonzero(d['arms'][usef]==arm) for arm in (0, 1)]
        shape, n = self.c0.shape, self.c0.size
        c0 = jnp.asarray(self.c0)
        norm = jnp.sum(jnp.abs(c0)**2)

        def objective(x):
            coeff = (x[:n]+1j*x[n:2*n]).reshape(shape)
            power = jnp.zeros_like(target)
            for arm, idx in enumerate(groups):
                coupling = spectral[idx]@coeff@design[arm].T
                power = power.at[idx].set(jnp.abs(coupling)**2)
            gain = jnp.exp(x[2*n:])
            residual = gain[:, None]*power-target
            return jnp.sum(weight*residual**2)+penalty*jnp.sum(jnp.abs(coeff-c0)**2)/norm

        value_gradient = jax.jit(jax.value_and_grad(objective))
        start = self.c0 if initial is None else initial
        initial_power = self.power(start)
        if initial_gain is None:
            initial_gain = legacy['gains'](initial_power, d['data'], d['good'], train)
        eta = np.log(np.maximum(initial_gain[usef]/scale, 1e-12))
        x0 = np.r_[start.real.ravel(), start.imag.ravel(), eta]

        def scipy_objective(x):
            value, gradient = value_gradient(jnp.asarray(x))
            return float(value), np.asarray(gradient, float)

        # Independent finite differences exercise both field and gain blocks.
        rng = np.random.default_rng(20260923)
        checks = []
        value, gradient = scipy_objective(x0)
        for block in ('coefficients', 'log_gains'):
            direction = rng.normal(size=len(x0))
            direction[2*n:] = 0 if block=='coefficients' else direction[2*n:]
            if block=='log_gains':
                direction[:2*n] = 0
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            fd = (scipy_objective(x0+eps*direction)[0]-scipy_objective(x0-eps*direction)[0])/(2*eps)
            analytic = float(gradient@direction)
            assert np.isclose(fd, analytic, rtol=1e-3, atol=1e-7), (block, fd, analytic)
            checks.append(dict(block=block, finite_difference=fd, autodiff=analytic, absolute_error=abs(fd-analytic)))
        history = []
        started = time.monotonic()
        iterations = 0
        def callback(x):
            nonlocal iterations
            iterations += 1
            if iterations % 250 == 0:
                value, gradient = scipy_objective(x)
                history.append(dict(iteration=iterations, objective=value, gradient_max=float(np.max(abs(gradient)))))
                log(self.basis.kind, 'lambda', penalty, 'iteration', iterations, 'loss', value)
        opt = minimize(scipy_objective, x0, jac=True, method='L-BFGS-B', callback=callback,
                       bounds=[(None, None)]*(2*n)+[(-25., 25.)]*len(usef),
                       options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-6, maxcor=30, maxls=40))
        coeff = (opt.x[:n]+1j*opt.x[n:2*n]).reshape(shape)
        power = self.power(coeff)
        # Excluded teeth do not update the beam; fit their diagnostic gains
        # conditionally. All 42 fitting teeth have explicit joint gain variables.
        gain = legacy['gains'](power, d['data'], d['good'], train)
        conditional = gain.copy()
        gain[usef] = np.exp(opt.x[2*n:])*scale
        gain_stationarity = float(np.max(abs(gain[usef]/conditional[usef]-1)))
        info = dict(basis=self.basis.kind, penalty=penalty, iterations=int(opt.nit),
            success=bool(opt.success), message=str(opt.message), loss=float(opt.fun),
            gradient_max=float(np.max(abs(opt.jac))), seconds=time.monotonic()-started,
            max_relative_gain_difference_from_conditional_optimum=gain_stationarity,
            gain_bound_hit=bool(np.any(abs(opt.x[2*n:])>=24.99)),
            coefficient_real_parameters=2*n, joint_channel_gains=len(usef), gradient_checks=checks,
            history=history)
        log('finished fit', {k:v for k,v in info.items() if k not in ('history','gradient_checks')})
        return coeff, gain, gain[:, None]*power, info


def export_beam(path, h, fit, coeff, gain):
    d = fit.d
    frequencies = np.unique(np.r_[h['freqs'], d['freqs']])
    fields = interp1d(h['freqs'], h['fields'], axis=0)(frequencies)
    empirical = (frequencies>=BAND[0]) & (frequencies<=BAND[1])
    th, ph = hp.pix2ang(h['nside'], np.arange(fields.shape[-1]))
    y = legacy['harmonics'](th, ph, fit.lmax)
    fields[empirical] = (fit.basis.evaluate(frequencies[empirical])@coeff).reshape(empirical.sum(), 3, y.shape[1])@y.T
    direction = np.array(hp.pix2vec(h['nside'], np.arange(fields.shape[-1])))
    fields -= np.sum(fields*direction[None], axis=1)[:, None]*direction[None]
    normalization_power = np.mean(np.sum(abs(fields)**2, axis=1), axis=1)
    fields /= np.sqrt(normalization_power)[:, None, None]
    eth = np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
    eph = np.array([-np.sin(ph), np.cos(ph), np.zeros_like(ph)])
    gt = abs(np.sum(fields*eth[None], axis=1))**2
    gp = abs(np.sum(fields*eph[None], axis=1))**2
    keep = ~d['el0_slew']
    td, pd_, _ = legacy['coordinates'](d, fit.params, np.flatnonzero(keep))
    coverage = np.bincount(hp.ang2pix(h['nside'], td, pd_), minlength=fields.shape[-1])
    # Preserve jointly fitted gains under export's per-frequency normalization.
    export_gain = gain*normalization_power[np.searchsorted(frequencies, d['freqs'])]
    np.savez_compressed(path, beam_cart=fields, gain_th=gt, gain_ph=gp, freqs=frequencies,
        nside=h['nside'], empirical_frequency_mask=empirical, sample_count=coverage,
        fitted_frequency_limits_mhz=np.array(BAND), spectral_basis=np.array(fit.basis.kind),
        geometry_parameters=fit.params, geometry_parameter_names=np.array(legacy['PARAM_NAMES']),
        normalization=np.array('unit spherical-average total power; independent tooth gains'),
        status=np.array('exploratory; power-only phase ambiguity; unsampled directions prior-dependent'))
    from eigsep_data.beam_sim import read_beam
    loaded, _, f = read_beam(path, drop_last=False)
    assert np.array_equal(f, frequencies) and np.array_equal(loaded, fields)
    assert np.isfinite(fields).all()
    assert np.allclose(gt+gp, np.sum(abs(fields)**2, axis=1), atol=1e-10)
    assert np.max(abs(np.sum(fields*direction[None], axis=1))) < 1e-10
    assert np.allclose(np.mean(gt+gp, axis=1), 1.)
    power = legacy['sample_hfss'](d, dict(h, fields=fields, freqs=frequencies), fit.params)
    prediction = export_gain[:, None]*power
    return export_gain, prediction


def main(output, maxiter=3000, bases=('dpss','pca')):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    d, h = legacy['load_data'](), legacy['load_hfss']()
    with np.load(PREVIOUS_PRODUCT/'diagnostics.npz') as prior:
        params = prior['params'].copy()
        assert np.array_equal(d['t'], prior['t'])
        assert np.array_equal(d['good'], prior['good'])
    # The plotted el~0 horizontal line is the opening azimuth slew: the first
    # 30 seconds remain between -3 and +3 degrees while spanning >150 degrees
    # in azimuth. Remove only this trajectory; retain every later elevation
    # sweep crossing through zero.
    el0_slew = ((d['t']-d['t'].min() <= 30.) & (np.abs(d['el']) < 3.))
    assert el0_slew.sum() == 56
    assert np.ptp(np.unwrap(np.deg2rad(d['az'][el0_slew]))) > np.deg2rad(150.)
    d['el0_slew'] = el0_slew
    d['good'][:, el0_slew] = False
    native_f, native_fields = hfss_samples(h)
    lmax, angular_coefficients, angular_errors = spatial_expansion(h, native_fields)
    pd.DataFrame(angular_errors).to_csv(output/'spatial_order.csv', index=False)
    train = d['split']==0
    baseline_power = legacy['sample_hfss'](d, h, params)
    baseline_gain = legacy['gains'](baseline_power, d['data'], d['good'], train)
    baseline = baseline_gain[:, None]*baseline_power
    all_metrics = metrics(d, baseline, 'hfss')
    descriptions, optimizations = {}, []
    log('data', d['data'].shape, 'lmax', lmax, 'jax', jax.__version__, jax.devices())
    for kind in bases:
        directory = output/kind; directory.mkdir(exist_ok=True)
        basis = SpectralBasis(kind, native_f, native_fields, d['freqs'])
        descriptions[kind] = basis.description
        pd.DataFrame(basis.records).to_csv(directory/'basis_selection.csv', index=False)
        basis.save(directory/'spectral_basis.npz')
        c0 = np.linalg.lstsq(basis.native_modes, angular_coefficients, rcond=None)[0]
        fit = JointFit(d, params, lmax, basis, c0)
        trials = []
        for penalty in (1e-2, 1e-3, 1e-4):
            coeff, gain, prediction, info = fit.fit(train, penalty, maxiter=maxiter)
            records = metrics(d, prediction, f'{kind}_lambda_{penalty:g}')
            scores = pd.DataFrame(records)
            val = float(scores.loc[(scores.split=='validation')&scores.used_for_beam_fit, 'fractional_rms'].median())
            info.update(stage='train', validation_median_rms=val)
            trials.append((val, coeff, gain, prediction, info))
            all_metrics.extend(records); optimizations.append(info)
            log(kind, 'validation', penalty, val)
        _, coeff, gain, heldout, selected = min(trials, key=lambda entry:entry[0])
        all_metrics.extend(metrics(d, heldout, kind+'_heldout'))
        # Gains from the training fit initialize explicit joint all-sample fit.
        final_coeff, final_gain, final_prediction, final_info = fit.fit(
            np.ones(len(d['t']), bool), selected['penalty'], initial=coeff,
            initial_gain=gain, maxiter=maxiter)
        final_info.update(stage='all_sample_refit')
        optimizations.append(final_info)
        all_metrics.extend(metrics(d, final_prediction, kind+'_refit'))
        export_gain, exported = export_beam(directory/'empirical_beam.npz', h, fit, final_coeff, final_gain)
        all_metrics.extend(metrics(d, exported, kind+'_exported'))
        export_error = np.sqrt(np.sum((exported-final_prediction)**2*d['good'], axis=1)/np.sum(d['data']**2*d['good'], axis=1))
        assert np.max(export_error[d['fit_channels']]) < .01, export_error
        descriptions[kind].update(selected_penalty=selected['penalty'], lmax=lmax,
            max_export_prediction_error_relative_to_data=float(export_error[d['fit_channels']].max()))
        np.savez_compressed(directory/'diagnostics.npz', **{k:d[k] for k in
            ('az','el','t','data','good','split','channels','freqs','arms','fit_channels')},
            el0_slew=el0_slew,
            hfss=baseline, empirical_heldout=heldout, empirical_refit=final_prediction,
            empirical_exported=exported, initial_coeff=c0, heldout_coeff=coeff,
            coeff=final_coeff, heldout_gain=gain, final_gain=final_gain,
            exported_gain=export_gain, params=params,
            spectral_at_teeth=basis.evaluate(d['freqs']),
            spectral_at_hfss=basis.native_modes, native_hfss_freqs=native_f)
        # Persist partial completed-basis evidence even if another run fails.
        pd.DataFrame(all_metrics).to_csv(output/'metrics.csv', index=False)
        (output/'optimization.json').write_text(json.dumps(optimizations, indent=2)+'\n')
    inputs = [SOURCE, PREVIOUS_SOURCE, PREVIOUS_PRODUCT/'diagnostics.npz', h['path']]
    inputs += [CAMPAIGN/'curation'/f for f in ('pointing_table.parquet','antenna_resolution.json','horizon_profiles.json','transmitter_position.json')]
    inputs += [CAMPAIGN/'data'/Path(f).name for f in d['files']]
    artifacts = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                 if p.is_file() and p.name not in ('provenance.json','README.md')}
    provenance = dict(status='exploratory', spectral_bases=descriptions, lmax=lmax,
        band_mhz=BAND, raster_utc=legacy['RASTER'], hfss_reconstruction_threshold=THRESHOLD,
        geometry=dict(zip(legacy['PARAM_NAMES'], params)),
        geometry_source='previous HFSS alignment, fixed for the spectral-basis comparison',
        excluded_trajectory=dict(name='initial near-zero-elevation azimuth slew',
            rule='t - first selected sample <= 30 s AND abs(el_deg) < 3 deg',
            samples=int(el0_slew.sum()), start_utc='2026-07-17 20:26:00 UTC',
            stop_utc='2026-07-17 20:26:29.5 UTC',
            elevation_range_deg=[float(d['el'][el0_slew].min()), float(d['el'][el0_slew].max())],
            unwrapped_azimuth_span_deg=float(np.rad2deg(np.ptp(np.unwrap(np.deg2rad(d['az'][el0_slew]))))),
            later_elevation_sweep_zero_crossings_retained=True),
        normalization='unit spherical-average total power; export gains transformed algebraically, not refitted',
        optimizer='SciPy L-BFGS-B with JAX float64 value_and_grad; explicit log gains and complex angular coefficients',
        maxiter=maxiter, excluded_fit_channels=legacy['SUSPECT_CHANNELS'],
        versions=dict(jax=jax.__version__, numpy=np.__version__),
        repositories={name:legacy['revision'](ROOT/name) for name in ('data-analysis','eigsep_data')},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs}, artifact_sha256=artifacts,
        caveats=['Power-only phase and circular-polarization ambiguities remain; coefficients are HFSS-regularized.',
                 'The 56-sample opening near-zero-elevation azimuth slew is excluded from fitting and all reported scores.',
                 'Both spectral bases give every mode freely fitted angular coefficients.',
                 '42 fitting teeth have explicit jointly optimized positive gains; four predeclared suspect teeth have diagnostic conditional gains only.',
                 'Geometry and validation/test stripes are inherited; these are comparisons on existing held-out data, not a new blind test.',
                 'Only 145-235 MHz is fitted. Other exported slices are normalized HFSS.',
                 'The DPSS delay width is an HFSS-compatible modeling scale, not a measured propagation delay.',
                 'All-sample refit/export scores are not independent validation.'])
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    log('completed', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'scratch/empirical-dpss-pca')
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--bases', nargs='+', choices=['dpss','pca'], default=['dpss','pca'])
    args = parser.parse_args()
    main(args.output, args.maxiter, args.bases)
