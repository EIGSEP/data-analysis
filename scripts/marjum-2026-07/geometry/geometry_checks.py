"""Held-out checks of a geometry_posterior product (first used on v0004).

Each check removes one kind of evidence from the v0004 joint model, refits only
what that evidence touched, and scores the removed evidence against the refit.
The v0004 chains did not converge, so there is no posterior to marginalize;
these are conditional maximum-density refits started from the v0004 state,
using the model's own exact log-density terms (``Posterior`` in
``eigsep_terrain/marjum_geometry/marjum_mcmc_b21.py``). Everything not refitted
stays at its v0004 value, which is the stated scope of each result.

Checks
  loo_antenna   For each of the 23 antenna-labelled cameras: drop its antenna
                label, refit its pose from its remaining terms (horizon, tie
                points, GPS, priors), refit the antenna from the other 22 labels
                at their v0004 poses, and report how far the dropped label's ray
                passes from the refit antenna.
  tx_pair       Drop the transmitter labels of IMG_2210/2211 (the two views not
                conditioned on the transmitter), refit those two poses, refit the
                transmitter and the six transmitter-era poses together from the
                six remaining labels, then triangulate the pair's two rays and
                compare with the six-camera transmitter (the historical ~1 m
                vertical disagreement).
  gps_withheld  Drop every HEIC GPS term (horizontal fixes, the common bias,
                EXIF altitude), fit a rigid translation of the whole network
                (cameras, antenna, transmitter, landmarks), then refit each pose,
                the antenna and the transmitter. Reports how far the antenna and
                transmitter move, and the camera-to-HEIC distances out of sample.

Usage: EIGSEP_CAMPAIGN_ROOT=... geometry_checks.py --out DIR [--checks ...]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import make_product as mp

# Scales that make one optimizer unit comparable across pose parameters.
CAM_SCALE = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-2])   # m, m, m, rad, rad, rad, log f
OPT = dict(method='Powell', options=dict(xtol=1e-3, ftol=1e-7, maxfev=4000))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_commit(path):
    run = lambda *a: subprocess.run(['git', '-C', str(path), *a], capture_output=True,
                                    text=True).stdout.strip()
    return run('rev-parse', 'HEAD') + ('-dirty' if run('status', '--porcelain',
                                                        '--untracked-files=no') else '')


class Fit:
    """Mutable copy of a model state with conditional refits."""

    def __init__(self, model, state):
        self.m = model
        self.cam, self.ant, self.tx, self.bias, self.extra, self.txe, self.pts = model.unpack(state)
        self.cam = self.cam.copy()

    def cam_logp(self, i, cam_i):
        return self.m.camera_logp(i, cam_i, self.ant, self.pts, self.bias, self.extra,
                                  tx=self.tx, tx_extra=self.txe)

    def refit_camera(self, i):
        c0 = self.cam[i].copy()

        def nll(x):
            ci = c0.copy()
            ci[:6] = c0[:6] + x[:6] * CAM_SCALE[:6]
            ci[6] = c0[6] * np.exp(x[6] * CAM_SCALE[6])
            v = self.cam_logp(i, ci)
            return -v if np.isfinite(v) else 1e12

        r = minimize(nll, np.zeros(7), **OPT)
        ci = c0.copy()
        ci[:6] = c0[:6] + r.x[:6] * CAM_SCALE[:6]
        ci[6] = c0[6] * np.exp(r.x[6] * CAM_SCALE[6])
        if -r.fun >= self.cam_logp(i, c0):
            self.cam[i] = ci
        return int(r.nfev)

    def refit_antenna(self):
        a0 = self.ant.copy()
        nll = lambda x: -self.m.antenna_logp(self.cam, a0 + x, self.extra)
        r = minimize(nll, np.zeros(3), **OPT)
        self.ant = a0 + r.x

    def refit_transmitter(self):
        t0 = self.tx.copy()
        nll = lambda x: -self.m.transmitter_logp(self.cam, t0 + x, self.txe)
        r = minimize(nll, np.zeros(3), **OPT)
        self.tx = t0 + r.x


def ray(model, cam_i, i, xy):
    """Unit ENU ray through fit-convention pixel xy of camera i (model's projection)."""
    from marjum_camera import rays
    d = rays(cam_i, model.shapes[i], np.atleast_2d(np.asarray(xy, float)), model.distortion[i])
    return d[0] / np.linalg.norm(d[0])


def miss(p0, d, X):
    v = X - p0
    return float(np.linalg.norm(v - (v @ d) * d))


def pixel_miss(model, cam_i, i, X, xy):
    from marjum_camera import project
    p, _ = project(cam_i, model.shapes[i], np.atleast_2d(X), model.distortion[i])
    return float(np.hypot(*(p[0] - np.asarray(xy))))


def closest_point(rays_):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p0, d in rays_:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ p0
    return np.linalg.solve(A, b)


def loo_antenna(model, state, log):
    rows = []
    base = Fit(model, state)
    for i in np.flatnonzero(model.has_ant_label):
        t = time.time()
        m = copy.copy(model)
        m.has_ant_label = model.has_ant_label.copy()
        m.has_ant_label[i] = False
        m.n_ant_label = int(m.has_ant_label.sum())
        f = Fit(m, state)
        nfev = f.refit_camera(i)
        f.refit_antenna()
        xy = model.axy[i]
        d_in = ray(model, base.cam[i], i, xy)
        d_out = ray(model, f.cam[i], i, xy)
        row = dict(image=model.keys[i],
                   in_sample_miss_m=miss(base.cam[i, :3], d_in, base.ant),
                   held_out_miss_m=miss(f.cam[i, :3], d_out, f.ant),
                   held_out_miss_px=pixel_miss(model, f.cam[i], i, f.ant, xy),
                   pose_shift_m=float(np.linalg.norm(f.cam[i, :3] - base.cam[i, :3])),
                   antenna_shift_m=(f.ant - base.ant).tolist(),
                   range_m=float(np.linalg.norm(f.ant - f.cam[i, :3])), nfev=nfev)
        rows.append(row)
        log(f"loo {row['image']}: held-out miss {row['held_out_miss_m']:.2f} m "
            f"({row['held_out_miss_px']:.0f} px), in-sample {row['in_sample_miss_m']:.2f} m, "
            f"pose moved {row['pose_shift_m']:.2f} m [{time.time()-t:.0f} s]")
    return rows


def tx_pair(model, state, log, rounds=3):
    pair = [model.keys.index(k) for k in ('2210', '2211')]
    m = copy.copy(model)
    m.has_tx_label = model.has_tx_label.copy()
    m.has_tx_label[pair] = False
    m.n_tx_label = int(m.has_tx_label.sum())
    f = Fit(m, state)
    for i in pair:
        f.refit_camera(i)
    six = [i for i in np.flatnonzero(m.has_tx_label)]
    for r in range(rounds):
        f.refit_transmitter()
        for i in six:
            f.refit_camera(i)
        log(f"tx_pair round {r}: tx {np.round(f.tx, 3).tolist()}")
    f.refit_transmitter()
    rays_ = [(f.cam[i, :3], ray(model, f.cam[i], i, model.txy[i])) for i in pair]
    p = closest_point(rays_)
    sep = np.cross(rays_[0][1], rays_[1][1])
    gap = abs((rays_[1][0] - rays_[0][0]) @ sep) / np.linalg.norm(sep)
    base = Fit(model, state)
    out = dict(tx_six=f.tx.tolist(), tx_v0004=base.tx.tolist(), tx_pair=p.tolist(),
               pair_minus_six_m=(p - f.tx).tolist(),
               pair_minus_six_vertical_m=float(p[2] - f.tx[2]),
               pair_minus_six_horizontal_m=float(np.hypot(*(p[:2] - f.tx[:2]))),
               pair_ray_gap_m=float(gap),
               pair_misses_of_six_m={model.keys[i]: miss(f.cam[i, :3], d, f.tx)
                                     for i, (_, d) in zip(pair, rays_)},
               six_minus_v0004_m=(f.tx - base.tx).tolist(),
               pair_pose_shift_m={model.keys[i]: float(np.linalg.norm(f.cam[i, :3] - base.cam[i, :3]))
                                  for i in pair})
    log(f"tx_pair: pair - six = {np.round(p - f.tx, 2).tolist()} m, ray gap {gap:.2f} m")
    return out


def gps_withheld(model, state, log, rounds=2):
    m = copy.copy(model)
    m.has_gps = np.zeros_like(model.has_gps)
    m.has_alt = np.zeros_like(model.has_alt)
    base = Fit(model, state)

    def translated(t):
        f = Fit(m, state)
        f.cam[:, :3] += t
        f.ant = f.ant + t
        f.tx = f.tx + t
        f.pts = f.pts + t
        f.bias = np.zeros(2)
        return f

    def nll(t):
        f = translated(t)
        v = m.logp(m.pack(f.cam, f.ant, f.tx, f.bias, f.extra, f.txe, f.pts))
        return -v if np.isfinite(v) else 1e12

    r = minimize(nll, np.zeros(3), method='Nelder-Mead',
                 options=dict(xatol=0.02, fatol=0.01, initial_simplex=np.array(
                     [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 2]], float)))
    shift = r.x
    log(f"gps_withheld: network translation {np.round(shift, 2).tolist()} m "
        f"(logp {-r.fun:.2f}, {r.nfev} evals)")
    f = translated(shift)
    for k in range(rounds):
        for i in range(m.nc):
            f.refit_camera(i)
        f.refit_antenna()
        f.refit_transmitter()
        log(f"gps_withheld round {k}: ant {np.round(f.ant, 2).tolist()} tx {np.round(f.tx, 2).tolist()}")
    has = model.has_gps
    sep_in = np.hypot(*(base.cam[has, :2] - model.gps[has]).T)
    sep_out = np.hypot(*(f.cam[has, :2] - model.gps[has]).T)
    off = np.mean(model.gps[has] - f.cam[has, :2], axis=0)
    sep_out_c = np.hypot(*(f.cam[has, :2] + off - model.gps[has]).T)
    return dict(network_translation_m=shift.tolist(),
                antenna=f.ant.tolist(), transmitter=f.tx.tolist(),
                antenna_minus_v0004_m=(f.ant - base.ant).tolist(),
                transmitter_minus_v0004_m=(f.tx - base.tx).tolist(),
                gps_n=int(has.sum()),
                gps_rms_in_sample_m=float(np.sqrt(np.mean(sep_in ** 2))),
                gps_rms_withheld_m=float(np.sqrt(np.mean(sep_out ** 2))),
                gps_common_offset_withheld_m=off.tolist(),
                gps_rms_withheld_after_common_offset_m=float(np.sqrt(np.mean(sep_out_c ** 2))),
                gps_max_withheld_m=float(sep_out.max()),
                gps_worst_withheld=model.keys[int(np.flatnonzero(has)[np.argmax(sep_out)])],
                camera_shift_m={model.keys[i]: float(np.linalg.norm(f.cam[i, :3] - base.cam[i, :3]))
                                for i in range(m.nc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--checks', nargs='+', default=['loo_antenna', 'tx_pair', 'gps_withheld'])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'checks.log', 'a')

    def log(msg):
        line = f"{datetime.now(timezone.utc):%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    _, model = mp.load_model()
    geom = mp.HERE / 'provisional_geometry.npz'
    with np.load(geom) as z:
        state = np.asarray(z['state'], float)
    lp = model.logp(state)
    log(f"v0004 state logp {lp:.3f}")
    results = dict(v0004_logp=lp)
    for name in a.checks:
        t = time.time()
        results[name] = globals()[name](model, state, log)
        log(f"{name} done in {time.time()-t:.0f} s")
        (out / f'{name}.json').write_text(json.dumps(results[name], indent=1) + '\n')
    here = Path(__file__).resolve()
    import eigsep_terrain
    manifest = dict(
        product='geometry_checks', subject='geometry_posterior/v0004',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        method='conditional maximum-density refits from the v0004 state with one kind of '
               'evidence removed; scipy Powell (Nelder-Mead for the network translation)',
        code=dict(path='data-analysis/scripts/marjum-2026-07/geometry/' + here.name,
                  data_analysis_commit=git_commit(here.parent),
                  eigsep_terrain_commit=git_commit(Path(eigsep_terrain.__file__).resolve().parents[2])),
        inputs=dict(geometry=dict(path=str(geom), sha256=sha256(geom)),
                    model_inputs=str(mp.INPUT), dem=dict(path=str(mp.DEM), sha256=sha256(mp.DEM))),
        checks=a.checks)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    log('wrote ' + str(out))


if __name__ == '__main__':
    main()
