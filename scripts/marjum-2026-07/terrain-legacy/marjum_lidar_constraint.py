"""Use the platform LIDAR ranges as an independent constraint on the antenna.

For every good LIDAR return we know the platform elevation `el`, the (unregistered)
potentiometer azimuth `az`, and a measured range. Given a candidate antenna
position we can ray-trace the same look direction into the DEM and ask what the
range *should* have been. The disagreement constrains the antenna.

Two populations, with very different information content, and they must not be
pooled:

  plateau  the 01:32-01:41 near-nadir returns. This is a DWELL, not a sweep --
           439 of 488 samples sit in a single 0.5 deg elevation bin -- so it is
           one very well-repeated range measurement, not a profile. It pins
           height at ~1:1 and says almost nothing about azimuth.
  sweep    the remaining returns, el 30-150 deg, 84-207 m, striking the canyon
           walls at long slant range. Few samples, but these are what carry
           azimuth and horizontal-position information.

Pointing model (corrected 2026-09-18, per geometer / MEMO-012 and Aaron's
mount-composition confirmation; see `_lidar_ray`). The mount is
`R = R_highline(el) . R_mast(az)`: azimuth is an *inner* rotation about the
antenna's own mast axis (bearing shift `daz - az`), elevation is an *outer*
rotation about the fixed, externally measured highline axis
(`AZ_WEST_BEARING_DEG`). This is NOT the separable alt-az model this script
used before -- azimuth and elevation only decouple at `az_pot = 0`, which is
why the earlier bug reproduced the dwell plateau exactly and still looked
right. Nadir at `el = +90` and zenith at `el = 0` (for the LIDAR; the
boresight, 90 deg away, is the reverse) now fall straight out of this
geometry -- there is no separate `EL_NADIR` constant to plug in any more.

Ray azimuth is `daz - az`, where `daz` is the offset between the
potentiometer azimuth and true north (sign per geometer's 2026-09-18
handedness finding: az_pot increases by RHR about +U, so compass bearing
DECREASES as az_pot increases -- not `az + daz`). `daz` is now FIXED at
`DAZ_DEG` = 37.836 deg for the primary result (Aaron, B36, 2026-09-18):
this script is a geometry *test* against the GPS-measured az-zero, not an
azimuth fit -- 665 LIDAR returns all within 1.30 deg of one azimuth cannot
out-constrain that measurement. `scan`'s daz sweep is kept only as a
diagnostic (see `main`); the primary fit is `fit_du`.

Caveat carried through every number below: the DEM is int32 on a 0.5 m grid and
`interp_alt` is nearest-neighbour, so ground elevation is quantised to 1 m. That
is comparable to the disagreements being measured.

Usage: marjum_lidar_constraint.py [--out lidar_constraint_v1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# This script lives at the repo root; marjum-2026-07/ is a fixed sibling
# directory in the same checkout (not a separate/movable one -- confirmed
# with data-archivist, 2026-09-16).
WORKSPACE = Path(__file__).resolve().parent
POINTING = WORKSPACE.parent / 'marjum-2026-07' / 'pointing' / 'pointing_table_v0.npz'
# Everything except the two flags that are provenance rather than defect.
GOOD_MASK = 0x17B
# Near-field returns are the LIDAR striking the structure that carries the
# antenna, not terrain. They are separated from every terrain return by a clean
# 23 m gap (the largest near-field return is 6.37 m, the smallest terrain return
# 29.34 m), so the cut is set by the data rather than chosen.
NEARFIELD_MAX = 10.0
# There used to be two competing "EL_NADIR" constants here (92.25 from a
# plumb-line fit, 87.68 from the dwell pointing). Both are RETRACTED: the
# plumb fit assumed the LIDAR sits on the elevation axis (it does not -- the
# fit has no power to constrain a lever arm it never modelled), and the
# dwell value was never a nadir measurement, just where the platform was
# parked. Nadir at el=+90 now falls straight out of the corrected mount
# geometry in `_lidar_ray` -- it is derived, not a free constant, so there
# is nothing to plug in here any more. `plumb_fit` below is kept purely as
# a retracted-context diagnostic; its output no longer feeds the ray model.
#
# The one thing genuinely worth a nuisance parameter is `delta`, a small
# pointing-table elevation zero-offset (what the old EL_NADIR grid scan was
# actually measuring -- flat MAD basin, <~2 deg, DEM-quantisation limited).
# Don't let that reappear mislabelled as a nadir measurement (geometer,
# 2026-09-18).
#
# Highline axis compass bearing, measured directly from Aaron's ground
# anchors by GPS (geometer, 2026-09-18) -- not derived from `daz`, so using
# it to fit `daz` below is not circular. Working-grid ENU; the geodesic
# figure agrees to 0.013 deg, the spherical great-circle alternative is an
# outlier at 0.125 deg off and is not used.
AZ_WEST_BEARING_DEG = 307.836
# = compass(AZ_WEST_BEARING_DEG) + 90, mod 360. FIXED for the primary result
# (Aaron, B36, 2026-09-18): this script is a geometry test against the GPS
# az-zero, not an azimuth fit -- 665 LIDAR returns all within 1.30 deg of one
# azimuth is not enough to out-constrain the GPS measurement. `scan`'s
# `daz_grid` sweep survives only as an explicitly diagnostic-only check
# (see `main`); `fit_du` is the primary fit.
DAZ_DEG = 37.836
# Deterministic photogrammetric antenna position, v0001 (m, ENU).
ANTENNA = np.array([1655.868, 2030.923, 1777.397])

STEP = 0.5
RMAX = 255.0


def load_returns():
    """Good, finite, in-range LIDAR returns, split into near-field, dwell, sweep.

    The near-field split is not cosmetic. Those samples are the LIDAR hitting
    the antenna's own support structure, and they sit at ~1 m where terrain is
    at 84-207 m. Left in a terrain fit they are 26 maximally-wrong points out of
    203, and they were in the sweep set of the first pass.
    """
    z = np.load(POINTING, allow_pickle=True)
    el, az, d, flags, t = (z['el_deg'], z['az_deg'], z['lidar_dist_m'],
                           z['flags'], z['time_utc'])
    good = (flags & GOOD_MASK) == 0
    # 250.00 m exactly is the out-of-range sentinel, not a measurement.
    ret = good & np.isfinite(d) & (d < 249.0) & (d > 0.5)
    el, az, d, t = el[ret], az[ret], d[ret], t[ret]

    nearfield = d < NEARFIELD_MAX
    # The up-looking near-field cluster is the one that behaves like a vertical
    # line; the down-looking one does not (see plumb_fit).
    plumb = nearfield & (el < -90)
    dwell = (d > 85) & (d < 99) & (el > 80) & (el < 90)
    sweep = ~nearfield & ~dwell
    return dict(el=el, az=az, d=d, t=t, plateau=dwell, sweep=sweep,
                nearfield=nearfield, plumb=plumb)


def plumb_fit(ret, grid=np.arange(-98., -78., 0.25)):
    """Elevation registration from the LIDAR striking the vertical frame member.

    The frame hangs under gravity, so it is a plumb line: a ray at angle `chi`
    to it strikes it at range `x / sin(chi)`, where `x` is the perpendicular
    offset from the LIDAR. Fitting `x` and the elevation at which the ray is
    parallel to the frame measures where the LIDAR points straight up, and hence
    where it points straight down -- independently of any range-vs-height
    assumption.

    Caveat that must travel with the number: `x / sin(chi)` and
    `h / cos(chi - 90)` are the same function, so this fit cannot distinguish a
    vertical line from a plane perpendicular to it. The plumb-line reading comes
    from the hardware, not from the data.
    """
    from scipy.optimize import least_squares
    e, r = ret['el'][ret['plumb']], ret['d'][ret['plumb']]

    def rms_at(ev):
        s = least_squares(lambda p: p[0] / np.abs(np.sin(np.radians(e - ev))) - r, [0.2])
        res = s.x[0] / np.abs(np.sin(np.radians(e - ev))) - r
        return float(np.sqrt((res ** 2).mean())), float(s.x[0])

    vals = np.array([rms_at(v)[0] for v in grid])
    i = int(np.argmin(vals))
    # Inflate the best rms by the usual (n / (n - p)) factor for a ~1-sigma band.
    thr = vals[i] * np.sqrt(1 + 1.0 / (len(e) - 2))
    inside = grid[vals <= thr]
    rms, x = rms_at(grid[i])
    return dict(n=int(len(e)), el_vertical_deg=float(grid[i]), offset_x_m=x,
                rms_resid_m=rms,
                el_nadir_deg=float(grid[i]) + 180.0,
                el_nadir_lo=float(inside.min()) + 180.0,
                el_nadir_hi=float(inside.max()) + 180.0,
                rms_at_dwell_assumption=rms_at(87.68 - 180.0)[0],  # historical, RETRACTED
                profile_grid=grid.tolist(), profile_rms=vals.tolist())


def _lidar_ray(el, az, daz=DAZ_DEG, delta=0.0, west_bearing=AZ_WEST_BEARING_DEG):
    """LIDAR ray direction in topocentric ENU, from the corrected
    `R_highline(el) . R_mast(az)` mount composition (see module docstring).

    Azimuth rotates the LIDAR arm about the antenna's own mast axis first
    (bearing `daz - az`, defect-1 sign), then elevation tips the whole
    assembly about the fixed, externally measured highline axis
    (`west_bearing`). Unlike the old alt-az parametrization this replaces,
    azimuth and elevation are NOT separable except at `az_pot = 0`, where the
    two formulas agree exactly for every `el` -- confirmed against Aaron's
    quoted divergence figures (2026-09-18): this formula reproduces 120 deg
    of divergence from the old one at `el=120, az_pot=90` exactly, and the
    ~1.5x-az_pot small-angle slope quoted at `el=120` for az_pot=1/5/15 deg.

    `delta` is the pointing-table elevation zero-offset nuisance parameter
    (what the old `el_nadir` grid scan was actually measuring); it is not a
    nadir angle. Nadir at `el=+90` and zenith at `el=0` are structural here,
    not fitted.
    """
    el = np.radians(np.asarray(el, dtype=float) + delta)
    a = np.radians(daz - np.asarray(az, dtype=float))
    bw = np.radians(west_bearing)
    h = np.stack([np.sin(a), np.cos(a), np.zeros_like(a)], axis=-1)
    west = np.array([np.sin(bw), np.cos(bw), 0.0])
    west_cross_h = np.cross(west, h)
    west_dot_h = h @ west
    c, s = np.cos(el), np.sin(el)
    return (h * c[..., None] + west_cross_h * s[..., None]
            + west[None, :] * (west_dot_h * (1.0 - c))[..., None])


def march(dem, origin, el, az, daz=DAZ_DEG, delta=0.0):
    """First DEM intersection range along each LIDAR ray; NaN where none."""
    v = _lidar_ray(el, az, daz, delta)
    r = np.arange(STEP, RMAX, STEP)
    out = np.full(len(v), np.nan)
    # Chunked: the full (ray, step, 3) array is too large to hold at once.
    for k in range(0, len(v), 128):
        vv = v[k:k + 128]
        p = origin[None, None, :] + vv[:, None, :] * r[None, :, None]
        g = dem.interp_alt(p[..., 0].ravel(), p[..., 1].ravel()).reshape(p.shape[:2])
        below = p[..., 2] <= g
        out[k:k + 128] = np.where(below.any(1), r[np.argmax(below, 1)], np.nan)
    return out


def ground_points(dem, origin, el, az, daz=DAZ_DEG, delta=0.0):
    """Where each ray meets the DEM, for plotting the illuminated footprint."""
    rng = march(dem, origin, el, az, daz, delta)
    v = _lidar_ray(el, az, daz, delta)
    return origin[None, :] + v * rng[:, None], rng


def _score(dem, origin, el, az, d, daz, delta=0.0, min_frac=0.5):
    """Median absolute range residual; NaN if too few rays even hit terrain."""
    p = march(dem, origin, el, az, daz, delta)
    ok = np.isfinite(p)
    if ok.sum() < max(15, min_frac * len(el)):
        return np.nan, np.nan, int(ok.sum())
    r = d[ok] - p[ok]
    return float(np.median(np.abs(r))), float(np.median(r)), int(ok.sum())


def scan(dem, ret, daz_grid, du_grid, antenna=ANTENNA, delta=0.0, verbose=True):
    """(daz, dU) grid: sweep MAD, and the plateau's signed median residual.

    DIAGNOSTIC ONLY (Aaron, B36, 2026-09-18): `daz` is fixed at `DAZ_DEG` for
    the primary result -- 665 LIDAR returns all within 1.30 deg of one
    azimuth is not enough to out-constrain the GPS measurement of
    `AZ_WEST_BEARING_DEG`. This function still sweeps `daz_grid` for the
    documentation plot showing where the data's own (weak) preference sits
    relative to the fixed value; see `fit_du` for the primary fit.
    """
    sw, pl = ret['sweep'], ret['plateau']
    el, az, d = ret['el'], ret['az'], ret['d']
    mad = np.full((len(daz_grid), len(du_grid)), np.nan)
    plat = np.full_like(mad, np.nan)
    nhit = np.zeros_like(mad)
    for i, daz in enumerate(daz_grid):
        for j, du in enumerate(du_grid):
            o = antenna + np.array([0.0, 0.0, du])
            mad[i, j], _, n = _score(dem, o, el[sw], az[sw], d[sw], daz, delta)
            nhit[i, j] = n
            _, plat[i, j], _ = _score(dem, o, el[pl], az[pl], d[pl], daz, delta, min_frac=0.1)
        if verbose:
            print(f'  daz {daz:5.1f} done', flush=True)
    return mad, plat, nhit


def fit_du(dem, ret, du_grid, daz=DAZ_DEG, delta=0.0, antenna=ANTENNA):
    """Primary fit: dU only, at daz fixed to `DAZ_DEG` (Aaron, B36, 2026-09-18).

    This is what makes the script a geometry *test* rather than an azimuth
    fit: with `daz` no longer free, `mad_m` at the returned `dU_m` is exactly
    the model's agreement with the LIDAR ranges under the GPS az-zero.
    """
    sw, pl = ret['sweep'], ret['plateau']
    el, az, d = ret['el'], ret['az'], ret['d']
    mad = np.full(len(du_grid), np.nan)
    plat = np.full_like(mad, np.nan)
    nhit = np.zeros_like(mad)
    for j, du in enumerate(du_grid):
        o = antenna + np.array([0.0, 0.0, du])
        mad[j], _, nhit[j] = _score(dem, o, el[sw], az[sw], d[sw], daz, delta)
        _, plat[j], _ = _score(dem, o, el[pl], az[pl], d[pl], daz, delta, min_frac=0.1)
    j = int(np.nanargmin(mad))
    return dict(daz_deg=float(daz), dU_m=float(du_grid[j]), mad_m=float(mad[j])), mad, plat, nhit


def sensitivities(dem, ret, antenna=ANTENNA, delta=0.0):
    """How much the plateau residual moves per unit of each nuisance parameter.

    The plateau is the height constraint, so what matters is what else can
    masquerade as height. Horizontal position and the el-zero offset both can.
    """
    pl = ret['plateau']
    el, az, d = ret['el'][pl], ret['az'][pl], ret['d'][pl]
    base = _score(dem, antenna, el, az, d, DAZ_DEG, delta, min_frac=0.1)[1]
    out = {'baseline_median_residual_m': base}
    for lbl, dv in (('antenna_E_plus_1m', [1., 0, 0]),
                    ('antenna_N_plus_1m', [0, 1., 0]),
                    ('antenna_U_plus_1m', [0, 0, 1.])):
        r = _score(dem, antenna + np.array(dv), el, az, d, DAZ_DEG, delta, min_frac=0.1)[1]
        out[lbl] = dict(residual_m=r, d_residual_per_m=r - base)
    for dn in (-2., -1., 1., 2.):
        r = _score(dem, antenna, el, az, d, DAZ_DEG, delta=delta + dn, min_frac=0.1)[1]
        out[f'el_delta_{dn:+.0f}deg'] = dict(residual_m=r, d_residual=r - base)
    for daz in (40., 180.):
        r = _score(dem, antenna, el, az, d, daz, delta, min_frac=0.1)[1]
        out[f'daz_{daz:+.0f}deg'] = dict(residual_m=r, d_residual=r - base)
    return out


def robustness(dem, ret, du_grid, daz=DAZ_DEG, antenna=ANTENNA, delta=0.0, subsets=True):
    """Independent subsets and horizontal perturbations, fitted separately.

    `daz` fixed (Aaron, B36, 2026-09-18) -- this now checks dU stability
    across subsets, not daz stability. A dU that is real reappears in every
    subset; that's still the check, just on the one parameter left to fit.
    """
    sw = np.where(ret['sweep'])[0]
    el, az, d, t = ret['el'], ret['az'], ret['d'], ret['t']

    def fit(idx, origin=antenna):
        best = (np.nan, np.inf)
        for du in du_grid:
            m, _, _ = _score(dem, origin + np.array([0., 0., du]),
                             el[idx], az[idx], d[idx], daz, delta)
            if np.isfinite(m) and m < best[1]:
                best = (float(du), float(m))
        return dict(n=int(len(idx)), daz_deg=float(daz), dU_m=best[0], mad_m=best[1])

    by_t = sw[np.argsort(t[sw])]
    by_r = sw[np.argsort(d[sw])]
    out = {'full_sweep': fit(sw)}
    if not subsets:
        return out
    out.update({
        'first_half_time': fit(by_t[:len(by_t) // 2]),
        'second_half_time': fit(by_t[len(by_t) // 2:]),
        'near_half_range': fit(by_r[:len(by_r) // 2]),
        'far_half_range': fit(by_r[len(by_r) // 2:]),
    })
    for lbl, dv in (('antenna_E_plus_2m', [2., 0, 0]), ('antenna_E_minus_2m', [-2., 0, 0]),
                    ('antenna_N_plus_2m', [0, 2., 0]), ('antenna_N_minus_2m', [0, -2., 0])):
        out[lbl] = fit(sw, antenna + np.array(dv))
    return out


def _dem_sample_sha256(dem):
    """Cheap, cache-swap-sensitive DEM content fingerprint.

    Must match `_q13_inversion.py`'s `provenance()` exactly (same sparse
    sample, same hash) so the two are directly comparable -- this is the
    consumer side of that file's staleness check (geometer, 2026-09-18,
    `terrain`@`2a57ac1`): nothing else enforces that a stored `mad_fit` was
    produced under the same DEM/antenna/daz as the MAD `main()` computes
    live, and a stale JSON would otherwise mix a fresh number with an old
    one silently -- the same failure class as the DEM double-shift.
    """
    import hashlib
    sample = np.ascontiguousarray(dem.data[::500, ::500])
    return hashlib.sha256(sample.tobytes()).hexdigest()[:16]


def main(out_dir='lidar_constraint_v1'):
    from eigsep_terrain.marjum_dem import MarjumDEM as DEM
    from marjum_bundle import working_grid
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    # `marjum_dem_sw.npz` + `working_grid` double-shifts as of the 2026-09-17
    # eigsep_terrain merge (MEMO-012 Sec 3 "Reproduction warning"): the antenna
    # lands ~400 m under the reported ground, so every ray fails at the first
    # step regardless of azimuth (confirmed 2026-09-18: flat MAD=91.3 m across
    # the whole daz/dU grid). `marjum_dem.npz` (shift 0,0) is the memo's
    # confirmed-unaffected file and the one ANTENNA's coordinates and every
    # number in MEMO-012 are expressed against -- use it until the
    # eigsep_terrain double-shift is fixed upstream.
    dem = working_grid(DEM(cache_file='marjum_dem.npz'))

    ret = load_returns()
    sw = ret['sweep']
    print(f"returns {len(ret['el'])}  dwell {ret['plateau'].sum()}  sweep {sw.sum()}  "
          f"near-field MASKED {ret['nearfield'].sum()}", flush=True)

    plumb = plumb_fit(ret)
    print('plumb line (RETRACTED, diagnostic only): el_nadir %.2f (%.2f..%.2f), '
          'offset %.3f m, rms %.3f m'
          % (plumb['el_nadir_deg'], plumb['el_nadir_lo'], plumb['el_nadir_hi'],
             plumb['offset_x_m'], plumb['rms_resid_m']), flush=True)

    du_grid = np.arange(-3., 1.01, 0.25)

    # daz FIXED at DAZ_DEG (Aaron, B36, 2026-09-18) -- this is a geometry
    # test now, not an azimuth fit. 665 LIDAR returns all within 1.30 deg of
    # one azimuth cannot out-constrain the GPS measurement of
    # AZ_WEST_BEARING_DEG. No free-daz grid scan here any more -- the
    # honest "cost of imposing the GPS value" instead cites
    # `_q13_inversion.py`'s own fine-grid (0.1 deg), bootstrapped,
    # parabola-refined free-daz fit (`q13_inversion.json`) rather than
    # recomputing a coarser version of the same number.
    sens = sensitivities(dem, ret)
    best, mad_du, plat_du, nhit_du = fit_du(dem, ret, du_grid)
    rob = robustness(dem, ret, du_grid)
    best_j = int(np.argmin(mad_du))
    best['n_hit'] = int(nhit_du[best_j])
    best['n_total'] = int(sw.sum())
    # Pass criterion (geometer, B36, 2026-09-18): MAD <~1.5 m with most rays
    # hitting terrain is a healthy geometry chain; >~3 m means something is
    # broken (wrong DEM, wrong mount composition, wrong az-zero), not just
    # imprecise. This isn't a tolerance on the physical answer -- it's a
    # check that the chain didn't silently break the way the pre-fix DEM
    # double-shift did (flat MAD=91.3 m, a hard failure, not "imprecise").
    best['pass_geometry_chain_check'] = bool(best['mad_m'] < 3.0
                                              and best['n_hit'] > 0.9 * best['n_total'])
    print(f"geometry chain check: MAD={best['mad_m']:.3f} m, "
          f"{best['n_hit']}/{best['n_total']} rays hit terrain -> "
          f"{'PASS' if best['pass_geometry_chain_check'] else 'FAIL'} "
          f"(pass: MAD <~1.5 m healthy, >~3 m broken)", flush=True)

    q13_path = WORKSPACE / 'q13_inversion.json'
    if q13_path.exists():
        q13 = json.loads(q13_path.read_text())
        prov = q13.get('provenance')
        if prov is None:
            cost = dict(warning='q13_inversion.json has no provenance block (pre-'
                                 '`terrain`@2a57ac1) -- cannot verify it matches this '
                                 'run\'s DEM/antenna/daz; not reporting a number.')
        else:
            live = dict(dem_sample_sha256=_dem_sample_sha256(dem),
                        antenna_enu_m=[float(x) for x in ANTENNA],
                        daz_reference_deg=float(DAZ_DEG))
            mismatches = [k for k in live if live[k] != prov.get(k)]
            if mismatches:
                cost = dict(warning=f'q13_inversion.json provenance mismatch on {mismatches} '
                                     '-- stale relative to this run (different DEM, antenna, '
                                     'or daz reference); not reporting a number.',
                            live=live, stored={k: prov.get(k) for k in live})
            else:
                cost = dict(free_daz_deg=q13['daz_fit'], free_daz_mad_m=q13['mad_fit'],
                            cost_of_fixing_daz_m=best['mad_m'] - q13['mad_fit'],
                            cost_of_fixing_daz_frac=(best['mad_m'] - q13['mad_fit']) / q13['mad_fit'],
                            tension_sigma=q13['significance_sigma'], source=str(q13_path),
                            provenance_verified=True)
        if 'warning' in cost:
            print(f"WARNING: {cost['warning']}", flush=True)
        else:
            print(f"cost of fixing daz at {DAZ_DEG} deg vs the free optimum "
                  f"({q13['daz_fit']:.3f} deg, MAD {q13['mad_fit']:.3f} m): "
                  f"+{cost['cost_of_fixing_daz_m']:.3f} m "
                  f"({cost['cost_of_fixing_daz_frac']*100:.0f}%), "
                  f"same {q13['significance_sigma']:.2f} sigma tension as MEMO-012 Sec 4.12 "
                  "(provenance verified)", flush=True)
    else:
        cost = dict(warning='q13_inversion.json not found -- run _q13_inversion.py for the '
                             'honest free-daz reference; not recomputed here.')
        print(f"WARNING: {cost['warning']}", flush=True)

    o = ANTENNA + np.array([0., 0., best['dU_m']])
    # Overlay arrays: measured vs best-fit model prediction, both populations.
    sw_pred = march(dem, o, ret['el'][sw], ret['az'][sw], best['daz_deg'])
    dw_pred = march(dem, o, ret['el'][ret['plateau']], ret['az'][ret['plateau']],
                    best['daz_deg'])
    pts, rng = ground_points(dem, o, ret['el'][sw], ret['az'][sw], best['daz_deg'])
    ok = np.isfinite(rng)
    resid = ret['d'][sw][ok] - rng[ok]
    overlay = dict(sweep_pred=sw_pred, dwell_pred=dw_pred, ground_pts=pts, resid=resid,
                   mad_du=mad_du, plateau_resid_du=plat_du, nhit_du=nhit_du)
    result = dict(
        best_fit=best, sensitivities=sens, robustness=rob,
        cost_of_fixing_daz=cost,
        residuals_at_optimum=dict(
            n=int(ok.sum()), median_m=float(np.median(resid)),
            mad_m=float(np.median(np.abs(resid))), sd_m=float(resid.std()),
            frac_within_1m=float(np.mean(np.abs(resid) < 1)),
            frac_within_2m=float(np.mean(np.abs(resid) < 2)),
            frac_within_5m=float(np.mean(np.abs(resid) < 5)),
            frac_beyond_20m=float(np.mean(np.abs(resid) > 20))))
    print(json.dumps(dict(best_fit=best, cost_of_fixing_daz=cost), indent=2), flush=True)

    np.savez(out_dir / 'results.npz', du_grid=du_grid,
             **overlay, **{k: ret[k] for k in ('el', 'az', 'd', 't', 'plateau',
                                               'sweep', 'nearfield', 'plumb')})
    summary = dict(
        inputs=dict(pointing_table=str(POINTING), good_flag_mask=hex(GOOD_MASK),
                    dem='marjum_dem.npz (int32, 0.5 m grid, 1 m vertical quantisation)',
                    antenna_prior_enu_m=ANTENNA.tolist(),
                    nearfield_mask_max_m=NEARFIELD_MAX,
                    az_west_bearing_deg=AZ_WEST_BEARING_DEG,
                    daz_reference_deg=DAZ_DEG,
                    march_step_m=STEP, march_max_m=RMAX),
        counts=dict(returns=int(len(ret['el'])), dwell=int(ret['plateau'].sum()),
                    sweep=int(sw.sum()), nearfield_masked=int(ret['nearfield'].sum()),
                    plumb_cluster=int(ret['plumb'].sum())),
        plumb=plumb, result=result)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('wrote', out_dir)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='lidar_constraint_v1')
    a = ap.parse_args()
    main(a.out)
