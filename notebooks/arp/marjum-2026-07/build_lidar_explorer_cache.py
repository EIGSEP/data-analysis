"""Build a self-contained cache for lidar_explorer.ipynb.

Everything expensive or bulky is done once here: the 320 MB DEM mosaic is cut
down to the ~2.5 MB subtile the rays can actually reach, and the pointing table
is reduced to the 665 good returns with their population masks. The interactive
notebook then needs only numpy / matplotlib / ipywidgets -- no eigsep_terrain
install, no DEM mosaic, no pointing table.

The ray march is reimplemented inside the notebook in ~15 lines of numpy. This
script verifies that reimplementation reproduces `marjum_lidar_constraint.march`
exactly before writing the cache, so the explorer cannot silently drift away
from the memo it is meant to let you interrogate.

Run:  PYTHONPATH=. python3 build_lidar_explorer_cache.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Moved out of terrain/ into data-analysis 2026-09-18 (TERRAIN_REORG_PLAN).
# The modules and DEM caches it reads still live in terrain/, so reach them
# explicitly rather than relying on being run from that directory.
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

import marjum_lidar_constraint as M

HERE = Path(__file__).resolve().parent
OUT = HERE / 'lidar_explorer_cache.npz'

# Slider travel for antenna E/N, plus the maximum ray reach, plus a margin.
PAD_M = M.RMAX + 25.0
# The working-grid shift for marjum_dem_sw.npz: interp_alt(e, n) indexes the
# raw array at round((e - se)/res), round((n - sn)/res).
SE, SN = -1000.0, -1000.0
RES = 0.5


def main():
    from eigsep_terrain.marjum_dem import MarjumDEM as DEM
    from marjum_bundle import working_grid

    dem = working_grid(DEM(cache_file=str(TERRAIN / 'marjum_dem_sw.npz')))
    raw = np.load(TERRAIN / 'marjum_dem_sw.npz')['dem']

    ret = M.load_returns()
    ant = M.ANTENNA

    # Subtile bounds in working-grid metres, converted to raw array indices
    # with exactly the same rounding MarjumDEM.m2px uses.
    e_lo, e_hi = ant[0] - PAD_M, ant[0] + PAD_M
    n_lo, n_hi = ant[1] - PAD_M, ant[1] + PAD_M
    col0 = int(np.round((e_lo - SE) / RES))
    col1 = int(np.round((e_hi - SE) / RES)) + 1
    row0 = int(np.round((n_lo - SN) / RES))
    row1 = int(np.round((n_hi - SN) / RES)) + 1
    sub = raw[row0:row1, col0:col1]
    assert sub.min() > -32000 and sub.max() < 32000, 'subtile does not fit int16'
    sub = sub.astype(np.int16)
    print('subtile %s from rows %d:%d cols %d:%d  -> %.1f MB'
          % (sub.shape, row0, row1, col0, col1, sub.nbytes / 1e6))
    print('elevation range %d..%d m' % (sub.min(), sub.max()))

    # --- the reimplementation the notebook will use --------------------------
    def alt(e, n):
        """Nearest-neighbour ground elevation from the subtile, NaN outside."""
        c = np.round((np.asarray(e) - SE) / RES).astype(np.int64) - col0
        r = np.round((np.asarray(n) - SN) / RES).astype(np.int64) - row0
        ok = (c >= 0) & (c < sub.shape[1]) & (r >= 0) & (r < sub.shape[0])
        out = np.full(np.shape(e), np.nan)
        out[ok] = sub[r[ok], c[ok]]
        return out

    def march(origin, el, az_true, el_nadir, step=M.STEP, rmax=M.RMAX, bias=0.0):
        chi = np.radians(el - el_nadir + 180.0)
        a = np.radians(az_true)
        v = np.stack([np.sin(chi) * np.sin(a), np.sin(chi) * np.cos(a), np.cos(chi)], -1)
        r = np.arange(step, rmax, step)
        p = np.asarray(origin)[None, None, :] + v[:, None, :] * r[None, :, None]
        g = alt(p[..., 0], p[..., 1]) + bias
        below = p[..., 2] <= g
        return np.where(below.any(1), r[np.argmax(below, 1)], np.nan)

    # --- verification: must match the reference march exactly ----------------
    sw = ret['sweep']
    checks = [(43.0, 0.75, 92.25, 0.0, 0.0), (36.0, -2.0, 92.25, 1.0, -6.0),
              (34.0, -1.25, 87.68, 0.0, 0.0), (50.0, 2.0, 90.0, -8.0, 8.0)]
    worst = 0.0
    for daz, du, eln, de, dn in checks:
        o = M.ANTENNA + np.array([de, dn, du])
        ref = M.march(dem, o, ret['el'][sw], ret['az'][sw] + daz, eln)
        new = march(o, ret['el'][sw], ret['az'][sw] + daz, eln)
        both_nan = np.isnan(ref) & np.isnan(new)
        diff = np.abs(np.where(both_nan, 0.0, np.nan_to_num(ref) - np.nan_to_num(new)))
        assert np.isnan(ref).sum() == np.isnan(new).sum(), 'NaN pattern differs'
        worst = max(worst, float(np.nanmax(diff)))
        print('  check daz=%5.1f dU=%+5.2f el_nadir=%5.2f dE=%+.0f dN=%+.0f : max |ref-new| = %.3g m'
              % (daz, du, eln, de, dn, np.nanmax(diff)))
    assert worst == 0.0, 'reimplementation does not reproduce the reference march'
    print('VERIFIED: notebook march reproduces marjum_lidar_constraint.march exactly')

    S = json.loads((HERE / 'lidar_constraint_v2' / 'summary.json').read_text())
    np.savez_compressed(
        OUT,
        dem=sub, dem_row0=row0, dem_col0=col0, dem_res=RES, dem_se=SE, dem_sn=SN,
        el=ret['el'], az=ret['az'], d=ret['d'], t=ret['t'],
        dwell=ret['plateau'], sweep=ret['sweep'],
        nearfield=ret['nearfield'], plumb=ret['plumb'],
        antenna=M.ANTENNA, march_step=M.STEP, march_rmax=M.RMAX,
        el_nadir_plumb=M.EL_NADIR_PLUMB, el_nadir_dwell=M.EL_NADIR_DWELL,
        nearfield_max=M.NEARFIELD_MAX,
        published=json.dumps(S['by_el_nadir']['plumb']['best_fit']),
        plumb_fit=json.dumps({k: v for k, v in S['plumb'].items()
                              if not k.startswith('profile')}),
        en_configs=json.dumps(S['en_exploration']['configs']),
        windows=json.dumps(S['en_exploration']['window_definitions']),
    )
    print('wrote %s (%.1f MB)' % (OUT, OUT.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
