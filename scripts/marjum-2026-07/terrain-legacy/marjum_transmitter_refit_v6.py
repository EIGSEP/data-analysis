"""Joint transmitter/camera refit with a re-initializable or droppable free camera.

Same objective as `marjum_transmitter_ray_polish_v4.py` (picked-ray/range
parameterization, strict horizon guards), generalized so that the set of free
cameras is not hard-coded at six. Two operations are supported:

  --reinit KEY=DONOR   replace KEY's starting pose (position + orientation)
                       with DONOR's, keeping KEY's own focal length and
                       distortion, which are lens properties.
  --drop KEY           remove KEY from the fit entirely: it stops being a free
                       camera, its transmitter pick stops contributing a
                       reprojection term, and its feature pairs are dropped, so
                       it constrains nothing. Its stale pose remains in the
                       saved state but is marked excluded in camera_provenance.

Motivation (2026-09-14): camera 2159 sits in a false terrain minimum -- flipped
along the quasi-symmetric canyon axis and pushed ~5x too far from the dish. The
v4 objective bounds orientation perturbations to +/-0.3 rad, so it cannot
recover from that basin on its own; the bad pose is inherited from the source
NPZ and must be replaced at initialization instead.

This script does NOT modify marjum_transmitter_ray_polish_v4.py, whose hash is
pinned by the v0001_marjum_geometry release manifest.

Usage:
  marjum_transmitter_refit_v6.py --output DIR [--reinit 2159=2203] [--drop 2159]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import marjum_transmitter_joint_polish as jp
from marjum_camera import rays, project, radial_support, epipolar_error

BASE_FREE = ('2159', '2171', '2172', '2198', '2199', '2203')
DEFAULT_SOURCE = 'cv_transmitter_joint_v1/fit_transmitter.npz'


def run(output, source=DEFAULT_SOURCE, reinit=None, drop=(), max_nfev=200, fixed_sigma=15.,
        keep_2159_2199=False):
    drop = tuple(drop)
    free = tuple(k for k in BASE_FREE if k not in drop)
    n = len(free)
    NP = 9 * n           # camera block width
    TOT = NP + 3         # + transmitter offset

    # Polish reads the module-level FREE, so set it before constructing.
    jp.FREE = free

    # Polish.unpack hardcodes a six-camera layout (`q[:54].reshape(6, 9)`), and
    # Polish.__init__ calls metrics(np.zeros(54)) to record a baseline before
    # this function can override anything on the instance. With any other free
    # camera count that baseline call raises on the (n,6)/(6,6) broadcast, so
    # the class method is made generic first. The instance-level unpack
    # installed below (picked-ray/range parameterization) still takes over for
    # the fit itself; this only fixes the baseline and save/metrics paths.
    def _generic_unpack(self, q):
        m = len(self.base)
        q = np.asarray(q[:9 * m]).reshape(m, 9)
        return (np.c_[self.base[:, :6] + q[:, :6], self.base[:, 6] * np.exp(q[:, 6])],
                self.k0 + q[:, 7:9])

    jp.Polish.unpack = _generic_unpack
    fit = jp.Polish(source, output)

    # v4 rejects the 2159-2199 pair as having failed a preceding geometric
    # audit. That audit was run against 2159's false-minimum pose, so the
    # rejection may be a consequence of the bad pose rather than a bad pair --
    # and with it rejected, 2159 has NO feature pairs at all, leaving only its
    # own horizon and its own (jointly-fitted) transmitter ray to constrain it.
    # That is precisely how a camera drifts into a false minimum, so the
    # rejection is made switchable and tested both ways.
    if not keep_2159_2199:
        fit.pairs = [p for p in fit.pairs if {p['a'], p['b']} != {'2159', '2199'}]

    # A dropped camera must constrain nothing at all: no transmitter
    # reprojection term, no feature pairs.
    for key in drop:
        fit.txkeys = [k for k in fit.txkeys if k != key]
        fit.pairs = [p for p in fit.pairs if key not in (p['a'], p['b'])]

    # Re-initialize a free camera's starting pose from a donor camera. Position
    # and orientation are replaced; focal length and distortion are kept,
    # because they belong to the lens, not the vantage point.
    reinit_note = None
    if reinit:
        key, donor = reinit
        i = free.index(key)
        j = fit.keys.index(donor)
        donor_cam = fit.state['cameras'][j]
        old = fit.base[i].copy()
        fit.base[i, :6] = donor_cam[:6]
        reinit_note = dict(key=key, donor=donor, old_camera=old.tolist(),
                           new_start=fit.base[i].tolist())
        print(f'reinit {key} from {donor}: {np.round(old[:6], 3)} -> {np.round(fit.base[i, :6], 3)}',
              flush=True)

    ranges = np.linalg.norm(fit.base[:, :3] - fit.tx0, axis=1)
    observations = np.array([fit.meta[k]['transmitter_px'] for k in free])

    def unpack(x):
        q = x[:NP].reshape(n, 9)
        tx = fit.tx0 + x[NP:TOT]
        ks = fit.k0 + q[:, 5:7]
        cams = []
        for i in range(n):
            p = np.r_[fit.base[i, :3], fit.base[i, 3:6] + q[i, 1:4], fit.base[i, 6] * np.exp(q[i, 4])]
            d = rays(p, fit.shapes[i], [observations[i] + q[i, 7:9]], ks[i])[0]
            p[:3] = tx - ranges[i] * np.exp(q[i, 0]) * d
            cams.append(p)
        return np.array(cams), ks

    fit.unpack = unpack

    terrain_report = json.loads(Path('cv_transmitter_joint_v2/report.json').read_text())['terrain_only']
    limits = np.array([terrain_report['horizons'][k]['train_rms_px'] * 1.15 + 4 for k in free])

    exact_rows = []

    def residual(x, structure=False):
        cams, ks = unpack(x)
        q = x[:NP].reshape(n, 9)
        tx = fit.tx0 + x[NP:TOT]
        values, deps, linear = [], [], []

        def add(v, cols, strict=False):
            v = np.atleast_1d(v)
            values.extend(v)
            deps.extend([cols] * len(v))
            linear.extend([strict] * len(v))

        for i, key in enumerate(free):
            p = cams[i]
            cols = list(range(9 * i, 9 * i + 9)) + list(range(NP, TOT))
            h = fit.horizon(i, tuple(p), tuple(ks[i]))[fit.train[i]]
            add(h / 8, cols)
            add((p[:3] - fit.base[i, :3]) / [8, 8, 5], cols, True)
            add(q[i, 1:4] / .10, cols, True)
            add([q[i, 4] / .10], cols, True)
            add(q[i, 5:7] / .025, cols, True)
            add(q[i, 7:9] / 3, cols, True)
            ground = max(fit.terrain.height(*p[:2]),
                         float(np.asarray(fit.terrain.dem.interp_alt(np.array([p[0]]), np.array([p[1]])))[0]))
            add([min(p[2] - ground - .6, 0) / .1], cols, True)
            add(np.minimum(radial_support(p, fit.shapes[i], ks[i]) - .3, 0) * 100, cols, True)
            add([max(np.sqrt(np.mean(h * h)) - limits[i], 0) / .5], cols, True)

        for key in fit.txkeys:
            if key in free:
                continue
            p, s, k = fit.camera(key, cams, ks)
            pred, depth = project(p, s, tx, k)
            add((pred[0] - fit.meta[key]['transmitter_px']) / fixed_sigma, list(range(NP, TOT)))
            add([min(depth[0] - .5, 0) / .1], list(range(NP, TOT)), True)

        for pair in fit.pairs:
            a, b = pair['a'], pair['b']
            p, s, k = fit.camera(a, cams, ks)
            p2, s2, k2 = fit.camera(b, cams, ks)
            cols = [j for key in [a, b] if key in free
                    for j in range(9 * fit.active[key], 9 * fit.active[key] + 9)] + list(range(NP, TOT))
            e = epipolar_error(p, s, k, p2, s2, k2, pair['x'], pair['y'])
            add(np.nan_to_num(e, nan=1000.) / 3, cols)
            add([min(np.linalg.norm(p[:3] - p2[:3]) - .03, 0) / .01], cols, True)

        if structure:
            exact_rows.extend(linear)
            mat = lil_matrix((len(values), TOT), dtype=int)
            for i, cols in enumerate(deps):
                mat[i, cols] = 1
            return mat.tocsr()
        return np.array(values)

    # Source camera rays supply a consistent initial intersection, independent
    # of the stale transmitter array in the supplied NPZ.
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(n):
        d = rays(fit.base[i], fit.shapes[i], [observations[i]], fit.k0[i])[0]
        nn = np.eye(3) - np.outer(d, d)
        A += nn
        b += nn @ fit.base[i, :3]
    initial_tx = np.linalg.solve(A, b)
    x0 = np.zeros(TOT)
    x0[NP:] = initial_tx - fit.tx0
    for i in range(n):
        x0[9 * i] = np.log(np.linalg.norm(initial_tx - fit.base[i, :3]) / ranges[i])

    sparsity = residual(x0, True)
    strict = np.array(exact_rows)

    def loss(z):
        t = 1 + z
        rho = np.array([2 * (np.sqrt(t) - 1), 1 / np.sqrt(t), -.5 * t ** (-1.5)])
        rho[:, strict] = [z[strict], np.ones(strict.sum()), np.zeros(strict.sum())]
        return rho

    bound = np.r_[np.tile([1.2, .3, .3, .25, .25, .08, .08, 15, 15], n), [6, 6, 6]]
    scale = np.r_[np.tile([.08, .02, .02, .02, .03, .01, .01, 1, 1], n), [1, 1, 1]]
    result = least_squares(residual, x0, bounds=(-bound, bound), jac_sparsity=sparsity, x_scale=scale,
                           loss=loss, f_scale=2, max_nfev=max_nfev, ftol=1e-6, verbose=2)

    tx = fit.tx0 + result.x[NP:]
    fit.save('fit_transmitter.npz', result.x, tx)

    # Mark dropped cameras in the saved provenance, so no downstream consumer
    # mistakes the retained stale pose for a fitted one.
    if drop:
        path = fit.out / 'fit_transmitter.npz'
        state = dict(np.load(path))
        prov = list(state['camera_provenance'].astype(str))
        for key in drop:
            prov[fit.keys.index(key)] = 'EXCLUDED: false terrain minimum, pose is stale and unfitted'
        state['camera_provenance'] = np.array(prov)
        np.savez_compressed(path, **state)

    report = fit.metrics(result.x, tx)
    report.update(source=str(fit.source), input_sha256=fit.report['input_sha256'],
                  initializer=initial_tx.tolist(),
                  optimizer=dict(success=bool(result.success), nfev=result.nfev, cost=float(result.cost)),
                  horizon_limits_train_px=dict(zip(free, limits.tolist())),
                  excluded_pair=None if keep_2159_2199 else ['2159', '2199'],
                  feature_pairs=[(q['a'], q['b'], len(q['x'])) for q in fit.pairs],
                  free_cameras=list(free), dropped_cameras=list(drop), reinit=reinit_note,
                  range_bound_hit={k: bool(abs(abs(result.x[9 * i]) - 1.2) < 1e-3) for i, k in enumerate(free)},
                  note='Refit of the v4 objective with a re-initializable/droppable free camera. '
                       'Deterministic fit, not a posterior. Free camera rays are jointly constrained '
                       'and must not be counted again as independent fixed-camera measurements.')
    (fit.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(fit.out / 'optimizer.npz', x=result.x, jac=result.jac.toarray(), residual=result.fun)
    print('transmitter', tx, flush=True)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--source', default=DEFAULT_SOURCE)
    ap.add_argument('--reinit', default=None, help='KEY=DONOR, e.g. 2159=2203')
    ap.add_argument('--drop', default=[], nargs='*')
    ap.add_argument('--max-nfev', type=int, default=200)
    ap.add_argument('--keep-2159-2199', action='store_true',
                    help="retain the 2159-2199 feature pair that v4 rejected")
    a = ap.parse_args()
    ri = tuple(a.reinit.split('=')) if a.reinit else None
    run(a.output, source=a.source, reinit=ri, drop=a.drop, max_nfev=a.max_nfev,
        keep_2159_2199=a.keep_2159_2199)
