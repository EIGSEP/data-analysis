"""Box-air LIDAR ranges against the DEM: an independent check of a geometry release.

Ported from the retired ``terrain/marjum_lidar_constraint.py`` (terrain
``0acb255``, archived as ``terrain-legacy/marjum_lidar_constraint.py``) and its
free-azimuth companion ``_q13_inversion.py`` (terrain ``2a57ac1``), which made
``derived/lidar_constraint/v0001`` at the v0001 antenna.

The ray model is unchanged: mount ``R = R_highline(el) . R_mast(az)``, LIDAR
ray bearing ``daz - az_pot``, elevation about the GPS-measured highline axis.
What changes is the frame. Bearings (``daz`` and the highline axis) are true
bearings; the DEM is a UTM raster grid whose north is grid north. The legacy
script applied true bearings directly on the raster axes. With
``--frame utm`` each true bearing ``B`` is applied as the grid bearing
``B - gamma``, where ``gamma`` is the DEM's ``grid_convergence`` (true minus
grid, -1.52 deg at the antenna). ``--frame legacy`` reproduces v0001.

Inputs are the frozen ``pointing_table_v0.npz`` of lidar_constraint v0001 (kept
so that only the geometry changes), a DEM product and an antenna position.

Outputs, in ``--out``: ``results.npz`` (returns, splits, predicted ranges and
residuals at the fixed-azimuth optimum, the height-offset curve, the free-azimuth
profile and bootstrap) and ``summary.json``.

Usage:
  lidar_constraint.py --out DIR --antenna E N U [--frame utm|legacy]
                      [--dem PATH] [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GOOD_MASK = 0x17B           # every flag except the two provenance flags
NEARFIELD_MAX = 10.0        # m; returns from the antenna's own structure
AZ_WEST_BEARING_TRUE = 307.836   # highline axis, true bearing, GPS anchors
DAZ_GPS = 37.836            # az-zero from the GPS anchors, true
SIG_GPS = 0.6               # deg, for +/-3 m per anchor
STEP, RMAX = 0.5, 255.0
DU_GRID = np.arange(-3.0, 1.01, 0.25)
DAZ_GRID = np.arange(33.0, 42.01, 0.1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_returns(pointing):
    z = np.load(pointing, allow_pickle=True)
    el, az, d, flags, t = (z['el_deg'], z['az_deg'], z['lidar_dist_m'],
                           z['flags'], z['time_utc'])
    good = (flags & GOOD_MASK) == 0
    ret = good & np.isfinite(d) & (d < 249.0) & (d > 0.5)
    el, az, d, t = el[ret], az[ret], d[ret], t[ret]
    nearfield = d < NEARFIELD_MAX
    plumb = nearfield & (el < -90)
    dwell = (d > 85) & (d < 99) & (el > 80) & (el < 90)
    sweep = ~nearfield & ~dwell
    return dict(el=el, az=az, d=d, t=t, plateau=dwell, sweep=sweep,
                nearfield=nearfield, plumb=plumb)


class Model:
    """Ray-march LIDAR returns through a DEM from an antenna position."""

    def __init__(self, dem, gamma_deg):
        self.dem = dem
        self.gamma = gamma_deg          # true minus grid bearing; 0 for legacy

    def ray(self, el, az, daz, delta=0.0):
        el = np.radians(np.asarray(el, float) + delta)
        a = np.radians(daz - np.asarray(az, float) - self.gamma)
        bw = np.radians(AZ_WEST_BEARING_TRUE - self.gamma)
        h = np.stack([np.sin(a), np.cos(a), np.zeros_like(a)], axis=-1)
        west = np.array([np.sin(bw), np.cos(bw), 0.0])
        wxh = np.cross(west, h)
        wdh = h @ west
        c, s = np.cos(el), np.sin(el)
        return h * c[..., None] + wxh * s[..., None] + west[None, :] * (wdh * (1.0 - c))[..., None]

    def march(self, origin, el, az, daz, delta=0.0):
        v = self.ray(el, az, daz, delta)
        r = np.arange(STEP, RMAX, STEP)
        out = np.full(len(v), np.nan)
        for k in range(0, len(v), 128):
            vv = v[k:k + 128]
            p = origin[None, None, :] + vv[:, None, :] * r[None, :, None]
            g = self.dem.interp_alt(p[..., 0].ravel(), p[..., 1].ravel()).reshape(p.shape[:2])
            below = p[..., 2] <= g
            out[k:k + 128] = np.where(below.any(1), r[np.argmax(below, 1)], np.nan)
        return out

    def score(self, origin, el, az, d, daz, min_frac=0.5):
        p = self.march(origin, el, az, daz)
        ok = np.isfinite(p)
        if ok.sum() < max(15, min_frac * len(el)):
            return np.nan, np.nan, int(ok.sum())
        r = d[ok] - p[ok]
        return float(np.median(np.abs(r))), float(np.median(r)), int(ok.sum())


def fit_du(model, ret, antenna, daz):
    sw, pl = ret['sweep'], ret['plateau']
    el, az, d = ret['el'], ret['az'], ret['d']
    mad = np.full(len(DU_GRID), np.nan)
    plat = np.full_like(mad, np.nan)
    nhit = np.zeros_like(mad)
    for j, du in enumerate(DU_GRID):
        o = antenna + np.array([0.0, 0.0, du])
        mad[j], _, nhit[j] = model.score(o, el[sw], az[sw], d[sw], daz)
        _, plat[j], _ = model.score(o, el[pl], az[pl], d[pl], daz, min_frac=0.1)
    j = int(np.nanargmin(mad))
    return dict(daz_deg=float(daz), dU_m=float(DU_GRID[j]), mad_m=float(mad[j]),
                n_hit=int(nhit[j]), n_total=int(sw.sum())), mad, plat, nhit


def residual_cube(model, ret, antenna):
    sw = ret['sweep']
    el, az, d = ret['el'][sw], ret['az'][sw], ret['d'][sw]
    cube = np.full((len(DAZ_GRID), len(DU_GRID), len(el)), np.nan)
    for i, daz in enumerate(DAZ_GRID):
        for j, du in enumerate(DU_GRID):
            cube[i, j] = d - model.march(antenna + np.array([0., 0., du]), el, az, daz)
    return cube


def daz_from_cube(cube, idx=None):
    """Profiled-over-dU MAD minimum in daz; parabola over +/-1 deg (as q13)."""
    c = cube if idx is None else cube[:, :, idx]
    with np.errstate(invalid='ignore'):
        mad = np.nanmedian(np.abs(c), axis=2)
    hits = np.isfinite(c).sum(axis=2)
    mad[hits < max(15, 0.5 * c.shape[2])] = np.nan
    prof = np.nanmin(mad, axis=1)
    if not np.isfinite(prof).any():
        return np.nan, np.nan, prof
    i = int(np.nanargmin(prof))
    best = float(prof[i])
    w = int(round(1.0 / (DAZ_GRID[1] - DAZ_GRID[0])))
    m = slice(max(0, i - w), min(len(prof), i + w + 1))
    x, y = DAZ_GRID[m], prof[m]
    good = np.isfinite(y)
    if good.sum() >= 5:
        a, b, _ = np.polyfit(x[good], y[good], 2)
        if a > 0:
            v = -b / (2 * a)
            if x[good].min() <= v <= x[good].max():
                return float(v), best, prof
    return float(DAZ_GRID[i]), best, prof


def git_commit(path):
    run = lambda *a: subprocess.run(['git', '-C', str(path), *a], capture_output=True,
                                    text=True).stdout.strip()
    sha = run('rev-parse', 'HEAD')
    return sha + ('-dirty' if run('status', '--porcelain', '--untracked-files=no') else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--antenna', nargs=3, type=float, required=True)
    ap.add_argument('--frame', choices=('utm', 'legacy'), default='utm')
    ap.add_argument('--dem', required=True)
    ap.add_argument('--pointing', required=True)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    import eigsep_terrain.dem as tdem
    from eigsep_terrain.dem import DEM

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dem = DEM(cache_file=a.dem)
    antenna = np.array(a.antenna, float)
    gamma = float(np.degrees(dem.grid_convergence(*antenna[:2]))) if a.frame == 'utm' else 0.0
    model = Model(dem, gamma)
    ret = load_returns(a.pointing)
    sw = ret['sweep']
    print(f"returns {len(ret['el'])} dwell {ret['plateau'].sum()} sweep {sw.sum()} "
          f"frame {a.frame} gamma {gamma:+.4f} deg", flush=True)

    best, mad_du, plat_du, nhit_du = fit_du(model, ret, antenna, DAZ_GPS)
    best['pass_geometry_chain_check'] = bool(best['mad_m'] < 3.0
                                              and best['n_hit'] > 0.9 * best['n_total'])
    o = antenna + np.array([0., 0., best['dU_m']])
    sw_pred = model.march(o, ret['el'][sw], ret['az'][sw], DAZ_GPS)
    dw_pred = model.march(o, ret['el'][ret['plateau']], ret['az'][ret['plateau']], DAZ_GPS)
    ok = np.isfinite(sw_pred)
    resid = ret['d'][sw][ok] - sw_pred[ok]
    print(f"fixed az-zero: dU {best['dU_m']:+.2f} m, MAD {best['mad_m']:.3f} m, "
          f"{best['n_hit']}/{best['n_total']} hit", flush=True)

    cube = residual_cube(model, ret, antenna)
    daz_fit, mad_fit, prof = daz_from_cube(cube)
    rng = np.random.default_rng(20260918)
    n = int(sw.sum())
    boot = np.array([daz_from_cube(cube, rng.integers(0, n, n))[0] for _ in range(a.n_boot)])
    boot = boot[np.isfinite(boot)]
    sig_fit = float(np.std(boot, ddof=1))
    ddaz = DAZ_GPS - daz_fit
    sig = float(np.hypot(SIG_GPS, sig_fit))
    free = dict(free_daz_deg=daz_fit, free_daz_mad_m=mad_fit, sigma_fit_bootstrap_deg=sig_fit,
                n_boot=int(len(boot)), delta_daz_deg=float(ddaz), sigma_delta_daz_deg=sig,
                tension_sigma=float(abs(ddaz) / sig),
                cost_of_fixing_daz_m=best['mad_m'] - mad_fit)
    print(f"free az-zero: {daz_fit:.3f} +/- {sig_fit:.3f} deg (MAD {mad_fit:.3f} m); "
          f"GPS - free = {ddaz:+.3f} +/- {sig:.3f} deg, {abs(ddaz)/sig:.2f} sigma", flush=True)

    dwell = ret['d'][ret['plateau']]
    np.savez(out / 'results.npz', du_grid=DU_GRID, sweep_pred=sw_pred, dwell_pred=dw_pred,
             resid=resid, mad_du=mad_du, plateau_resid_du=plat_du, nhit_du=nhit_du,
             daz_grid=DAZ_GRID, daz_profile=prof, daz_boot=boot,
             **{k: ret[k] for k in ('el', 'az', 'd', 't', 'plateau', 'sweep', 'nearfield', 'plumb')})
    here = Path(__file__).resolve()
    summary = dict(
        label=a.label,
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        code=dict(path='data-analysis/scripts/marjum-2026-07/geometry/' + here.name,
                  data_analysis_commit=git_commit(here.parent),
                  eigsep_terrain_dem_sha256=sha256(tdem.__file__)),
        inputs=dict(pointing=dict(path=a.pointing, sha256=sha256(a.pointing)),
                    dem=dict(path=a.dem, sha256=sha256(a.dem)),
                    antenna_m=antenna.tolist(), frame=a.frame, grid_convergence_deg=gamma,
                    good_flag_mask=hex(GOOD_MASK), nearfield_mask_max_m=NEARFIELD_MAX,
                    az_west_bearing_true_deg=AZ_WEST_BEARING_TRUE, daz_gps_deg=DAZ_GPS,
                    sigma_gps_deg=SIG_GPS, march_step_m=STEP, march_max_m=RMAX),
        counts=dict(returns=int(len(ret['el'])), dwell=int(ret['plateau'].sum()),
                    sweep=int(sw.sum()), nearfield_masked=int(ret['nearfield'].sum())),
        result=dict(best_fit=best, free_daz=free,
                    dwell=dict(n=int(len(dwell)), measured_median_m=float(np.median(dwell)),
                               predicted_median_m=float(np.nanmedian(dw_pred)),
                               residual_m=float(np.median(dwell) - np.nanmedian(dw_pred))),
                    residuals_at_optimum=dict(
                        n=int(ok.sum()), median_m=float(np.median(resid)),
                        mad_m=float(np.median(np.abs(resid))),
                        n_far_10m=int(np.sum(~(np.abs(ret['d'][sw] - sw_pred) <= 10))))))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
