#!/usr/bin/env python3
"""Independent recheck of every load-bearing number in MEMO-012 revision 2.

DEM-derived elevations updated 2026-09-18 for the float32 cache rebuild (B38);
tolerances tightened accordingly, since the 1 m quantisation floor is gone.

Recomputes from primary inputs (GPS coordinates, DEM, bundle poses, curation
JSON) with no reuse of session state. Run with the arp python.
"""
import json
import os
import sys
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
import marjum_bundle as mb                                    # noqa: E402
from eigsep_terrain.marjum_dem import MarjumDEM               # noqa: E402

# ---- primary inputs, typed in from their sources -------------------------
EAST = (39.24672, -113.40102)        # highline east anchor (Aaron, GPS/CalTopo)
WEST = (39.24904, -113.40486)        # highline west anchor
TIE = (39.24789, -113.40271)         # pulley-plate tiedown, CalTopo-revised
TX_GM = (39.2477654, -113.4028198)   # Google Maps TX identification (rejected)
TX_CT = (39.24779, -113.40284)       # CalTopo TX identification (rejected)
THEO = (39.247783, -113.4027710)     # theodolite app phone fix
ANT = np.array([1655.868, 2030.923, 1777.397])   # antenna product, +/-1.7 m
TXP = np.array([1652.198, 2025.104, 1684.384])   # transmitter product, +/-1.5 m

ok = []


def check(label, got, expect, tol):
    good = abs(got - expect) <= tol
    ok.append(good)
    print(f'  [{"OK " if good else "FAIL"}] {label}: {got:.4f} (memo {expect}, tol {tol})')


def gc_bearing(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    dlo = lo2 - lo1
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return float(np.degrees(np.arctan2(y, x)) % 360.)


dem = mb.working_grid(MarjumDEM(cache_file=str(TERRAIN / 'marjum_dem.npz')))
en = lambda ll: np.array(dem.latlon_to_enu(*ll)[:2], float)
alt = lambda p: float(dem.interp_alt(np.array([p[0]]), np.array([p[1]]))[0])

pe, pw, pt = en(EAST), en(WEST), en(TIE)
hl = pw - pe
u = hl / np.linalg.norm(hl)
span = float(np.linalg.norm(hl))
grid_brg = float(np.degrees(np.arctan2(hl[0], hl[1])) % 360.)
perp = lambda p: float((p - pe)[0] * u[1] - (p - pe)[1] * u[0])
along = lambda p: float((p - pe) @ u)

print('== 4.8 anchors, bearing, az-zero ==')
check('east anchor E', pe[0], 1789.164, 0.01); check('east anchor N', pe[1], 1914.729, 0.01)
check('west anchor E', pw[0], 1457.629, 0.01); check('west anchor N', pw[1], 2172.228, 0.01)
check('east anchor ground', alt(pe), 1842.95, 0.05); check('west anchor ground', alt(pw), 1875.22, 0.05)
check('grid bearing', grid_brg, 307.836, 0.005)
check('spherical bearing', gc_bearing(EAST, WEST), 307.961, 0.005)
try:
    from pyproj import Geod
    az, _, d = Geod(ellps='WGS84').inv(EAST[1], EAST[0], WEST[1], WEST[0])
    check('geodesic bearing', az % 360, 307.849, 0.005)
    check('geodesic span', d, 419.79, 0.05)
except Exception as e:
    print('  [warn] pyproj unavailable:', e)
check('grid span', span, 419.79, 0.05)
check('daz = grid + 90', (grid_brg + 90) % 360, 37.836, 0.005)
for sig, exp in ((3., 0.58), (5., 0.97), (8., 1.54)):
    check(f'bearing sd @ +/-{sig:.0f} m', np.degrees(np.arctan(sig * np.sqrt(2) / span)), exp, 0.01)

print('== 4.8 tiedown / offsets ==')
check('tiedown E', pt[0], 1643.25, 0.01); check('tiedown N', pt[1], 2044.59, 0.01)
check('tiedown ground', alt(pt), 1685.06, 0.05)
check('tiedown perp', perp(pt), 13.06, 0.02); check('tiedown along', along(pt), 194.9, 0.1)
check('antenna perp', perp(ANT[:2]), 10.00, 0.02); check('antenna along', along(ANT[:2]), 176.5, 0.1)
check('TX product perp', perp(TXP[:2]), 3.16, 0.02)
check('station separation', abs(along(pt) - along(ANT[:2])), 18.4, 0.1)
check('antenna fraction to tiedown (%)', 100 * perp(ANT[:2]) / perp(pt), 77., 1.)
check('antenna above tiedown ground', ANT[2] - alt(pt), 92.33, 0.05)
check('tether off vertical', np.degrees(np.arctan2(np.hypot(*(ANT[:2] - pt)), ANT[2] - alt(pt))), 11.39, 0.02)
check('ground under antenna', alt(ANT[:2]), 1684.05, 0.05)
check('TX product above its ground', TXP[2] - alt(TXP[:2]), 1.29, 0.02)

print('== 4.5 transmitter challenges ==')
gm, ct, th = en(TX_GM), en(TX_CT), en(THEO)
check('Maps E', gm[0], 1633.77, 0.01); check('Maps N', gm[1], 2030.76, 0.01)
check('CalTopo E', ct[0], 1632.03, 0.01); check('CalTopo N', ct[1], 2033.49, 0.01)
check('Maps-CalTopo separation', np.hypot(*(ct - gm)), 3.24, 0.02)
check('Maps to product', np.hypot(*(gm - TXP[:2])), 19.27, 0.02)
check('CalTopo to product', np.hypot(*(ct - TXP[:2])), 21.84, 0.02)
check('theodolite to product', np.hypot(*(th - TXP[:2])), 16.1, 0.1)
check('CalTopo to theodolite', np.hypot(*(ct - th)), 6.01, 0.05)
check('CalTopo to tiedown (revised)', np.hypot(*(ct - pt)), 15.79, 0.05)
check('Maps perp', perp(gm), -3.68, 0.02); check('CalTopo perp', perp(ct), -2.60, 0.02)
check('ground at CalTopo', alt(ct), 1679.40, 0.05)
check('product-u above CalTopo ground', TXP[2] - alt(ct), 4.98, 0.02)
for tag, p, exp_off, exp_nad, exp_brg in (('Maps', gm, 22.10, 13.36, 269.57),
                                          ('CalTopo', ct, 23.98, 14.46, 276.14)):
    d = p - ANT[:2]
    check(f'{tag} horiz from antenna', np.hypot(*d), exp_off, 0.02)
    check(f'{tag} deg from nadir', np.degrees(np.arctan2(np.hypot(*d), ANT[2] - TXP[2])), exp_nad, 0.02)
    check(f'{tag} bearing from antenna', np.degrees(np.arctan2(d[0], d[1])) % 360, exp_brg, 0.02)

print('== 4.5 non-circular pair (2210/2211) ==')
B = np.load(TERRAIN / 'archive/2026-09-14_joint_posterior_v1_NONCONVERGED/fit_transmitter.npz',
            allow_pickle=True)
keys = [str(k) for k in B['keys']]
cams = {k: B['cameras'][keys.index(k)][:3] for k in ('2210', '2211')}
cond = [str(k) for k in B['transmitter_conditioned_keys']]
assert '2210' not in cond and '2211' not in cond, 'pair is conditioned?!'
J = json.load(open('/mnt/data02/eigsep/marjum-2026-07/curation/transmitter_position.json'))
P = np.array(J['independent_pair_solution']['solution_enu_m'])
check('pair mutual ray gap (json)', J['independent_pair_solution']['mutual_ray_gap_m'], 0.715, 0.001)
check('pair-to-PRODUCT horizontal', np.hypot(*(P[:2] - TXP[:2])), 0.251, 0.005)
check('pair-to-v4-reference horizontal (json)',
      J['independent_pair_solution']['offset_from_reference_m']['horizontal'], 0.455, 0.001)
v1, v2 = (TXP - cams['2210']), (TXP - cams['2211'])
check('range 2210', np.linalg.norm(v1), 34.9, 0.1)
check('range 2211', np.linalg.norm(v2), 33.5, 0.1)
theta = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
check('subtended angle', theta, 43.74, 0.05)
# per-camera mispointing needed for the CalTopo position to be the truth
ct3 = np.array([ct[0], ct[1], TXP[2]])
for k in ('2210', '2211'):
    a, b = TXP - cams[k], ct3 - cams[k]
    ang = np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    print(f'  [info] {k}: ray to product vs ray to CalTopo point = {ang:.1f} deg '
          f'(observed pair consistency ~ {np.degrees(np.arctan2(0.715, 34.)):.1f} deg)')

print('== 4.9 transmitter vs boresight ==')
U = np.array([0., 0., 1.])
b = np.radians(grid_brg)
a_hat = np.array([np.sin(b), np.cos(b), 0.])
D = TXP - ANT
check('dE', D[0], -3.670, 0.001); check('dN', D[1], -5.819, 0.001); check('dU', D[2], -93.013, 0.001)
check('bearing antenna->TX', np.degrees(np.arctan2(D[0], D[1])) % 360, 212.239, 0.01)
check('deg from nadir', 180 - np.degrees(np.arccos(D[2] / np.linalg.norm(D))), 4.230, 0.005)
dh = D / np.linalg.norm(D)
check('out of boresight plane (deg)', np.degrees(np.arcsin(abs(dh @ a_hat))), 0.412, 0.005)
check('perpendicular miss (m)', abs(D @ a_hat), 0.671, 0.005)


def R(axis, deg):
    a = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


els = np.arange(180., 190., 0.01)
seps = [np.degrees(np.arccos(np.clip((R(a_hat, e) @ U) @ dh, -1, 1))) for e in els]
i = int(np.argmin(seps))
check('closest boresight el', els[i], 184.21, 0.05)
check('closest boresight separation', seps[i], 0.412, 0.01)
check('azimuth lever sin(4.23)', np.sin(np.radians(4.230)), 0.0738, 0.0005)

print('== 4.9/4.10 mount model ==')
l0 = np.cross(a_hat, U); l0 /= np.linalg.norm(l0)
check('LIDAR at az=0,el=0 bearing', np.degrees(np.arctan2(l0[0], l0[1])) % 360, 37.836, 0.005)
hw = lambda az, el: R(a_hat, el) @ (R(U, -az) @ l0)
def sph(az, el):
    z, brg = np.radians(el + 90.), np.radians((37.836 - az) % 360.)
    return np.array([np.sin(z) * np.sin(brg), np.sin(z) * np.cos(brg), np.cos(z)])
for az, exp in ((0., 0.), (1., 1.500), (5., 7.498), (15., 22.451), (45., 66.088), (90., 120.000)):
    s = np.degrees(np.arccos(np.clip(hw(az, 60.) @ sph(az, 60.), -1, 1)))
    check(f'model separation az={az:.0f} el=60', s, exp, 0.01)
# "exact" is quoted in the memo at the 1e-4 deg level; the residual here is
# rounding of the 37.836 constant in this script's sph(), not model error.
for el in (0., 30., 90., 150.):
    s = np.degrees(np.arccos(np.clip(hw(0., el) @ sph(0., el), -1, 1)))
    ok.append(s < 1e-4)
    print(f'  [{"OK " if s < 1e-4 else "FAIL"}] az=0 agreement at el={el:.0f}: {s:.2e} deg')
# boresight invariance and epsilon cone
for eps, exp in ((1., 2.0), (5., 10.0)):
    bb = R(np.array([1., 0., 0.]), eps) @ U
    m = max(np.degrees(np.arccos(np.clip((R(U, az) @ bb) @ bb, -1, 1))) for az in range(0, 360, 5))
    check(f'cone 2*eps at eps={eps:.0f}', m, exp, 0.01)

print('== 4.8 comparison-table signs (GPS minus route) ==')
GPS = 37.836
for route, val in (('published', 38.86), ('sign-only', 36.68), ('postfix coarse', 37.0),
                   ('postfix refit', 36.25), ('retracted', 50.45)):
    print(f'  [info] GPS - {route}: {GPS - val:+.2f}')

print('== 4.8 Q13 mapping ==')
for lat_err, exps in ((2.0, (1.24, 0.76, 0.55)), (4.6, (2.85, 1.76, 1.27))):
    for rng, exp in zip((92.3, 150., 207.), exps):
        check(f'{lat_err} m at {rng:.0f} m', np.degrees(np.arctan2(lat_err, rng)), exp, 0.01)

print('== 4.4 / 4.11 polarization arithmetic ==')
check('arm1', 164 + 2.71, 166.71, 1e-9); check('arm2', 74 + 2.71, 76.71, 1e-9)
check('2202 axis vs arm1', 132.5 - 166.71, -34.21, 1e-9)
check('2202 axis vs arm2', 132.5 - 76.71, 55.79, 1e-9)
check('sum', 34.21 + 55.79, 90.0, 1e-9)
check('EXIF vs theodolite raw', 174.00 - 127.02, 46.98, 1e-9)
check('mils', 174 / 360 * 6400, 3093.3, 0.05)

n_fail = ok.count(False)
print(f'\n{len(ok)} checks, {n_fail} failures')
sys.exit(1 if n_fail else 0)
