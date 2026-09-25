#!/usr/bin/env python3
"""Q13 inversion: use the GPS-measured azimuth zero to constrain the antenna position.

The LIDAR terrain-range fit determines `daz` *relative to an assumed antenna
position*. Now that az-zero is independently measured (GPS highline anchors,
37.836 +/- 0.6 deg), the fit's preferred `daz` becomes a constraint on the
antenna position instead.

This script:
  1. reproduces the fit's preferred daz at the nominal antenna position, on a
     fine grid with parabolic sub-grid refinement, profiling over dU;
  2. bootstraps over the 151 sweep rays to get the fit's OWN uncertainty on daz;
  3. measures the gradient d(daz)/d(antenna E, N) on a 5x5 perturbation grid;
  4. inverts the observed daz difference into an antenna-position correction,
     and reports which direction is constrained and which is null.

Analysis only. Writes exactly one file, and note it writes it BACK INTO
terrain/: `terrain/q13_inversion.json`. That is deliberate --
`marjum_lidar_constraint.main()` reads it via a WORKSPACE-relative path,
and the JSON's provenance block is a snapshot of terrain state (DEM hash,
ANTENNA, DAZ_DEG, terrain git rev), so the artifact belongs beside its
consumer even though this producer does not. Modifies nothing else.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Moved out of terrain/ into data-analysis 2026-09-18 (TERRAIN_REORG_PLAN).
# The inputs -- DEM caches, marjum_bundle, marjum_lidar_constraint -- and the
# one output all still live in terrain/, so point there explicitly rather than
# at this script's own directory.
def _terrain_root() -> Path:
    """Locate the terrain checkout, matching eigsep_data.geometry_release._terrain.

    An explicit ``EIGSEP_TERRAIN_ROOT`` wins; otherwise ``terrain/`` beside the
    campaign checkout, which is where it has always been. Resolved from this
    file's own location so the script does not depend on the caller's cwd.
    """
    env = os.environ.get("EIGSEP_TERRAIN_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parents[4] / "terrain"
    if not (root / "marjum_bundle.py").exists():
        raise SystemExit(
            f"terrain checkout not found at {root}. Set EIGSEP_TERRAIN_ROOT."
        )
    return root


TERRAIN = _terrain_root()
sys.path.insert(0, str(TERRAIN))
import marjum_bundle as mb                                   # noqa: E402
from eigsep_terrain.marjum_dem import MarjumDEM              # noqa: E402
import marjum_lidar_constraint as mlc                        # noqa: E402

DAZ_GPS = 37.836          # GPS highline anchors, working-grid
SIG_GPS = 0.6             # provisional on receiver accuracy
DAZ_GRID = np.arange(33.0, 42.01, 0.1)
DU_GRID = np.arange(-3.0, 1.01, 0.25)
PERT = np.array([-4., -2., 0., 2., 4.])
N_BOOT = 1000
RNG = np.random.default_rng(20260918)

DEM_FILE = 'marjum_dem.npz'
dem = mb.working_grid(MarjumDEM(cache_file=str(TERRAIN / DEM_FILE)))
ret = mlc.load_returns()


def provenance():
    """Everything whose change would invalidate this run's `mad_fit`.

    `marjum_lidar_constraint.main()` subtracts our stored `mad_fit` from a MAD
    it computes live, so the two must have been produced under the same DEM,
    ray model and antenna. Nothing enforced that, and a stale JSON would have
    silently mixed a fresh number with an old one -- the same silent-failure
    class as the DEM double-shift. Record the inputs so the consumer can check.
    """
    import hashlib
    import subprocess
    # Sparse content sample: cheap, and sensitive to a cache swap (the
    # int32->float32 rebuild changes every value) without re-reading 192 MB.
    sample = np.ascontiguousarray(dem.data[::500, ::500])
    try:
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(TERRAIN),
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = 'unknown'
    return dict(
        dem_file=DEM_FILE,
        dem_shape=list(dem.data.shape),
        dem_dtype=str(dem.data.dtype),
        dem_sample_sha256=hashlib.sha256(sample.tobytes()).hexdigest()[:16],
        antenna_enu_m=[float(x) for x in mlc.ANTENNA],
        daz_reference_deg=float(mlc.DAZ_DEG),
        az_west_bearing_deg=float(mlc.AZ_WEST_BEARING_DEG),
        n_sweep_rays=int(np.count_nonzero(ret['sweep'])),
        terrain_git_rev=rev,
    )
sw = ret['sweep']
el, az, d = ret['el'][sw], ret['az'][sw], ret['d'][sw]
n_ray = len(el)
print(f'{n_ray} sweep rays; daz grid {DAZ_GRID[0]}..{DAZ_GRID[-1]} @ {DAZ_GRID[1]-DAZ_GRID[0]:.2f}, '
      f'dU {DU_GRID[0]}..{DU_GRID[-1]} @ {DU_GRID[1]-DU_GRID[0]:.2f}')


def residual_cube(origin):
    """Per-ray signed range residual over the (daz, dU) grid. NaN where no hit."""
    cube = np.full((len(DAZ_GRID), len(DU_GRID), n_ray), np.nan)
    for i, daz in enumerate(DAZ_GRID):
        for j, du in enumerate(DU_GRID):
            p = mlc.march(dem, origin + np.array([0., 0., du]), el, az, daz)
            cube[i, j] = d - p
    return cube


def daz_from_cube(cube, idx=None):
    """Profiled-over-dU MAD minimum in daz, with parabolic sub-grid refinement."""
    c = cube if idx is None else cube[:, :, idx]
    with np.errstate(invalid='ignore'):
        mad = np.nanmedian(np.abs(c), axis=2)
    hits = np.isfinite(c).sum(axis=2)
    mad[hits < max(15, 0.5 * c.shape[2])] = np.nan
    prof = np.nanmin(mad, axis=1)
    if not np.isfinite(prof).any():
        return np.nan, np.nan
    i = int(np.nanargmin(prof))
    best = float(prof[i])
    # The MAD profile is jagged at the 0.1 deg scale -- DEM quantisation puts
    # multiple local minima within ~0.4 deg of each other. A 3-point parabola
    # latches onto that jitter, so fit a parabola over a +/-1 deg window
    # instead; that is the estimator whose value is quoted.
    w = int(round(1.0 / (DAZ_GRID[1] - DAZ_GRID[0])))
    m = slice(max(0, i - w), min(len(prof), i + w + 1))
    x, y = DAZ_GRID[m], prof[m]
    good = np.isfinite(y)
    if good.sum() >= 5:
        a, b, _ = np.polyfit(x[good], y[good], 2)
        if a > 0:
            v = -b / (2 * a)
            if x[good].min() <= v <= x[good].max():
                return float(v), best
    return float(DAZ_GRID[i]), best


# ---- 1/2. nominal position, and the fit's own uncertainty ----------------
t0 = time.time()
cube0 = residual_cube(mlc.ANTENNA)
daz_fit, mad_fit = daz_from_cube(cube0)
print(f'\nnominal antenna: daz_fit = {daz_fit:.3f} deg, profiled MAD = {mad_fit:.4f} m '
      f'({time.time()-t0:.0f} s)')

boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    boot[b] = daz_from_cube(cube0, RNG.integers(0, n_ray, n_ray))[0]
boot = boot[np.isfinite(boot)]
sig_fit = float(np.std(boot, ddof=1))
lo, hi = np.percentile(boot, [16, 84])
print(f'bootstrap over {n_ray} rays ({len(boot)} good of {N_BOOT}): '
      f'daz = {np.mean(boot):.3f} +/- {sig_fit:.3f} deg, 16-84% [{lo:.3f}, {hi:.3f}]')

# ---- 3. gradient d(daz)/d(antenna E, N) ---------------------------------
print('\nperturbation grid (daz_fit at each offset antenna position):')
rows = []
for dE in PERT:
    line = []
    for dN in PERT:
        a, _ = daz_from_cube(residual_cube(mlc.ANTENNA + np.array([dE, dN, 0.])))
        rows.append((dE, dN, a))
        line.append(a)
    print('  dE %+5.1f : ' % dE + ' '.join(f'{v:7.3f}' for v in line))
rows = np.array(rows)
A = np.column_stack([rows[:, 0], rows[:, 1], np.ones(len(rows))])
coef, *_ = np.linalg.lstsq(A, rows[:, 2], rcond=None)
gE, gN, c0 = coef
resid_plane = rows[:, 2] - A @ coef
g = np.array([gE, gN])
gmag = float(np.hypot(gE, gN))
ghat = g / gmag
print(f'\ngradient  d(daz)/dE = {gE:+.4f} deg/m   d(daz)/dN = {gN:+.4f} deg/m')
print(f'  |g| = {gmag:.4f} deg/m  -> 1 deg of daz == {1/gmag:.2f} m of antenna motion')
print(f'  sensitive direction (compass) = {np.degrees(np.arctan2(ghat[0], ghat[1]))%360:.1f} deg')
print(f'  NULL direction   (compass) = {np.degrees(np.arctan2(-ghat[1], ghat[0]))%360:.1f} deg')
print(f'  plane-fit residual rms over the 5x5 grid = {resid_plane.std(ddof=1):.3f} deg (linearity check)')

# ---- 4. invert ----------------------------------------------------------
ddaz = DAZ_GPS - daz_fit
sig_ddaz = float(np.hypot(SIG_GPS, sig_fit))
shift = ddaz / gmag
sig_shift = sig_ddaz / gmag
print(f'\n=== INVERSION ===')
print(f'  daz_GPS - daz_fit = {DAZ_GPS:.3f} - {daz_fit:.3f} = {ddaz:+.3f} +/- {sig_ddaz:.3f} deg')
print(f'    (GPS {SIG_GPS:.2f}, fit {sig_fit:.2f}); significance {abs(ddaz)/sig_ddaz:.2f} sigma')
print(f'  implied antenna shift ALONG the sensitive direction:')
print(f'    {shift:+.2f} +/- {sig_shift:.2f} m   (95%: {shift-1.96*sig_shift:+.2f} .. {shift+1.96*sig_shift:+.2f} m)')
print(f'  |shift| 95% upper bound = {abs(shift)+1.96*sig_shift:.2f} m')
print(f'  PERPENDICULAR component: UNCONSTRAINED by this inversion.')

out = dict(
    daz_gps=DAZ_GPS, sigma_gps=SIG_GPS,
    daz_fit=daz_fit, mad_fit=mad_fit, sigma_fit_bootstrap=sig_fit, n_boot=int(len(boot)),
    grad_daz_dE_deg_per_m=float(gE), grad_daz_dN_deg_per_m=float(gN),
    grad_magnitude_deg_per_m=gmag, metres_per_degree=float(1 / gmag),
    sensitive_direction_compass_deg=float(np.degrees(np.arctan2(ghat[0], ghat[1])) % 360),
    null_direction_compass_deg=float(np.degrees(np.arctan2(-ghat[1], ghat[0])) % 360),
    plane_fit_residual_rms_deg=float(resid_plane.std(ddof=1)),
    delta_daz_deg=float(ddaz), sigma_delta_daz_deg=sig_ddaz,
    significance_sigma=float(abs(ddaz) / sig_ddaz),
    implied_shift_along_sensitive_m=float(shift),
    sigma_implied_shift_m=float(sig_shift),
    upper_bound_95_m=float(abs(shift) + 1.96 * sig_shift),
    perturbation_grid=[[float(x) for x in r] for r in rows],
    note='Analysis only. daz measured relative to a FIXED antenna prior; '
         'E/N were not free parameters in this fit.',
    provenance=provenance(),
    consumer_note='marjum_lidar_constraint.main() reads mad_fit from this file to '
                  'report cost_of_fixing_daz. Re-run this script after ANY change to '
                  'the DEM, the ray model or ANTENNA, and compare `provenance` '
                  'against the live values before trusting that cost.',
)
Path(TERRAIN / 'q13_inversion.json').write_text(json.dumps(out, indent=2))
print('\nwrote q13_inversion.json')
