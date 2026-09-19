#!/usr/bin/env python3
"""Generate geometry_explorer.ipynb.

Viewer only: it plots existing products and archived draws. It fits nothing and
derives no new geometry. Re-run this script to regenerate the notebook, then
execute the notebook to refresh its outputs.
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent

MD = []
CODE = []
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip('\n')))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip('\n')))


md(r'''
# Marjum 2026-07 — geometry explorer

A map view of where things are and where the face-value headings point. This
notebook **plots existing products**; it fits nothing and derives no new
geometry.

Contents:

1. Hillshade of the working-grid DEM.
2. East/north draws from the joint MCMC for the **antenna**, the
   **transmitter**, and **all 29 cameras**.
3. Three heading rays, each drawn *at face value* from its own anchor:
   - **2237/2238** — the highline bearing implied by the recorded plane strike.
   - **2204** — the theodolite app reading.
   - **2202** — that camera's own heading read as the polarization axis.

## Read these warnings before quoting anything off this map

| item | status |
|---|---|
| `joint_posterior_v1` draws | **NOT CONVERGED** (212/213 coordinates miss both Rhat and ESS; camera 2159 is bimodal). Reference only — the scatter is *not* a credible region. |
| highline bearing 320.45 deg | **RETRACTED 2026-09-17** as an artefact. Real uncertainty is +/-59 deg, not +/-0.58 deg. Drawn here only because "at face value" is the question. |
| 2202 position | **unusable** (focal/depth degeneracy, basin +/-4-6 m). Its *heading* is sound (spread 1.10 deg across 10 m of positional freedom). |
| 2204 theodolite fix | consumer phone GPS, `GPSHPositioningError` 5.37 m. Sits 16.1 m from the photogrammetric transmitter. Sanity check only. |

The deterministic **brackets** remain the citable products, not these draws:
antenna `[1655.868, 2030.923, 1777.397]` +/-1.7 m, transmitter
`[1652.198, 2025.104, 1684.384]` +/-1.5 m. Both are hard brackets, not sigmas.
''')

code(r'''
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# terrain checkout, matching eigsep_data.geometry_release: an explicit
# EIGSEP_TERRAIN_ROOT wins, else terrain/ beside the campaign checkout.
# Resolved from the notebook's own directory, not the caller's cwd.
_env = os.environ.get('EIGSEP_TERRAIN_ROOT')
TERRAIN = Path(_env) if _env else Path.cwd().resolve().parents[3] / 'terrain'
if not (TERRAIN / 'marjum_bundle.py').exists():
    raise SystemExit(f'terrain checkout not found at {TERRAIN}. Set EIGSEP_TERRAIN_ROOT.')
sys.path.insert(0, str(TERRAIN))

ARCHIVE = TERRAIN / 'archive' / '2026-09-14_joint_posterior_v1_NONCONVERGED'
POSTERIOR = ARCHIVE / 'joint_posterior_v1' / 'combined.npz'
BUNDLE = ARCHIVE / 'fit_transmitter.npz'

# marjum_dem.npz has working-grid shift (0, 0). The _south/_sw caches are
# double-shifted by working_grid() against the current eigsep_terrain and put
# the camera ~490 m underground; do not substitute them here.
DEM_FILE = 'marjum_dem.npz'

# Map window (working-grid metres) and how far to project each heading ray.
EXTENT = (1430., 1830., 1880., 2210.)   # e0, e1, n0, n1
RAY_LENGTH_M = 260.
DRAWS_PER_CHAIN = 400                   # thinning for the scatter only

plt.rcParams.update({'figure.dpi': 120, 'font.size': 9})
print('terrain root:', TERRAIN)
''')

md(r'''
## 1. Inputs

`combined.npz` holds 8 chains x 7000 draws x 213 global coordinates. The
coordinate names are used directly, so nothing depends on column order.
''')

code(r'''
post = np.load(POSTERIOR, allow_pickle=True)
names = [str(s) for s in post['names']]
cam_keys = [str(s) for s in post['keys']]
draws = post['draws']                       # (chain, draw, coord)
idx = {n: i for i, n in enumerate(names)}

bundle = np.load(BUNDLE, allow_pickle=True)
b_keys = [str(s) for s in bundle['keys']]
b_cams = bundle['cameras']                  # (29, 7) e n u th ph ti f
b_shapes = bundle['shapes']

print(f'{draws.shape[0]} chains x {draws.shape[1]} draws x {draws.shape[2]} coords')
print(f'{len(cam_keys)} cameras: {" ".join(cam_keys)}')

rhat = post['rhat']
bad = int(((rhat > 1.01) | (post['ess_bulk'] < 400)).sum())
print(f'convergence: {bad}/{len(rhat)} coordinates fail Rhat<=1.01 or ESS>=400; '
      f'max Rhat {rhat.max():.2f} at {names[int(np.argmax(rhat))]}')
''')

code(r'''
# Thin the chains for plotting. Chains are kept separate so that a separated
# camera (2159 especially) is visible as separated, not blurred into one cloud.
step = max(1, draws.shape[1] // DRAWS_PER_CHAIN)


def en_draws(prefix):
    # (chain, draw_thinned, 2) east/north draws for one named target.
    e = draws[:, ::step, idx[f'{prefix}_e']]
    n = draws[:, ::step, idx[f'{prefix}_n']]
    return np.stack([e, n], axis=-1)


targets = {'antenna': en_draws('antenna'), 'transmitter': en_draws('transmitter')}
cameras = {k: en_draws(f'cam{k}') for k in cam_keys}

for name, d in targets.items():
    med = np.median(d.reshape(-1, 2), axis=0)
    spread = d.reshape(-1, 2).std(axis=0)
    print(f'{name:12s} median E/N = {med[0]:9.3f} {med[1]:9.3f}   sd = {spread[0]:.3f} {spread[1]:.3f}')
print(f'\nthinned to {targets["antenna"].shape[1]} draws/chain '
      f'({targets["antenna"].shape[0] * targets["antenna"].shape[1]} per target)')
''')

md(r'''
## 2. DEM and hillshade

Standard Lambertian hillshade, sun from the northwest at 45 deg altitude. The
DEM is int32 at 0.5 m with nearest-neighbour lookup, so it quantises
vertically at 1 m — fine for a basemap, and the reason terrain-fit residuals
have a floor.
''')

code(r'''
import marjum_bundle as mb
from eigsep_terrain.marjum_dem import MarjumDEM

dem = MarjumDEM(cache_file=str(TERRAIN / DEM_FILE))
mb.working_grid(dem)                        # no-op for marjum_dem.npz (shift 0,0)
e_axis, n_axis = dem.get_en()
print(f'DEM {dem.data.shape} at {dem.res} m; E {e_axis.min()}..{e_axis.max()}, '
      f'N {n_axis.min()}..{n_axis.max()}')

e0, e1, n0, n1 = EXTENT
ce = slice(int(np.searchsorted(e_axis, e0)), int(np.searchsorted(e_axis, e1)))
rn = slice(int(np.searchsorted(n_axis, n0)), int(np.searchsorted(n_axis, n1)))
tile = dem.data[rn, ce].astype(float)
tile_extent = (e_axis[ce][0], e_axis[ce][-1], n_axis[rn][0], n_axis[rn][-1])


def hillshade(z, res, azimuth_deg=315., altitude_deg=45., smooth_px=2.):
    # Rows increase northward, columns eastward. The DEM is int32 with 1 m
    # vertical quantisation, which puts visible moire in a raw gradient
    # shading, so smooth before differencing. Cosmetic only: the smoothed
    # surface is never used for geometry, just for the basemap.
    from scipy.ndimage import gaussian_filter
    if smooth_px:
        z = gaussian_filter(z, smooth_px)
    dz_dn, dz_de = np.gradient(z, res, res)
    slope = np.arctan(np.hypot(dz_de, dz_dn))
    aspect = np.arctan2(-dz_de, dz_dn)       # 0 = uphill to the north
    az = np.radians(azimuth_deg)
    alt = np.radians(altitude_deg)
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0., 1.)


shade = hillshade(tile, dem.res)
print(f'tile {tile.shape}, elevation {tile.min():.0f}..{tile.max():.0f} m')
''')

md(r'''
## 3. Heading anchors

Each ray is drawn from its own anchor with its own face-value bearing. Nothing
here is fitted in this notebook; the bearings are the recorded readings and the
anchors are the recorded positions.

| ray | anchor | bearing (compass, deg) | provenance |
|---|---|---|---|
| highline / 2237 | camera 2237, bundle pose | 320.89 | plane strike, **retracted** |
| highline / 2238 | camera 2238, bundle pose | 320.05 | plane strike, **retracted** |
| theodolite / 2204 | app GPS fix, lat 39.247783 lon -113.4027710 | 174.00 | `IMG_2204.PNG` |
| polarization / 2202 | camera 2202 manual retune pose | 132.82 | optical axis, recomputed below |

The 2202 bearing is recomputed from its stored pose rather than typed in, since
the pose file is the authority for it. The theodolite anchor is converted from
lat/lon through the same DEM used for the basemap.
''')

code(r'''
def rotation(p):
    # Body-to-ENU, identical to marjum_bundle.rotation / HorizonImage.get_rays.
    th, ph, ti = p[3:6]

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    c, s = np.cos(th), np.sin(th)
    return rz(ph) @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ rz(ti)


def axis_bearing(p):
    # Compass bearing of the optical axis (body +z) projected to horizontal.
    a = rotation(p) @ np.array([0., 0., 1.])
    return float(np.degrees(np.arctan2(a[0], a[1])) % 360.)


# --- 2202: position unusable, heading sound -------------------------------
pose_2202 = np.load(TERRAIN / '_tx2202' / 'fit_result.npz')['pose']
brg_2202 = axis_bearing(pose_2202)

# --- 2204: theodolite app fix --------------------------------------------
THEO_LATLON = (39.247783, -113.4027710)
THEO_BEARING = 174.00
theo_en = dem.latlon_to_enu(*THEO_LATLON)[:2]

# --- 2237/2238: recorded plane strike ------------------------------------
HIGHLINE = {'2237': 320.89, '2238': 320.05}

# --- highline GPS anchors (Aaron, 2026-09-18) ----------------------------
# Ground tie-off points on the two canyon rims. Independent of photogrammetry,
# so immune to the camera-baseline degeneracy that killed the strike estimate.
HL_EAST = (39.24672, -113.40102)
HL_WEST = (39.24904, -113.40486)
# Pulley-plate tiedown (one of two). Same GPS receiver, via CalTopo.
HL_TIE = (39.24789, -113.40271)
hl_e = np.asarray(dem.latlon_to_enu(*HL_EAST)[:2], float)
hl_w = np.asarray(dem.latlon_to_enu(*HL_WEST)[:2], float)
hl_tie = np.asarray(dem.latlon_to_enu(*HL_TIE)[:2], float)
hl_vec = hl_w - hl_e
hl_bearing = float(np.degrees(np.arctan2(hl_vec[0], hl_vec[1])) % 360.)

anchors = {}
for k, brg in HIGHLINE.items():
    p = b_cams[b_keys.index(k)]
    anchors[f'highline {k}'] = (p[:2].copy(), brg, 'highline')
anchors['theodolite 2204'] = (np.asarray(theo_en, float), THEO_BEARING, 'theodolite')
anchors['polarization 2202'] = (pose_2202[:2].copy(), brg_2202, 'polarization')

for name, (xy, brg, _) in anchors.items():
    print(f'{name:20s} anchor E/N = {xy[0]:9.3f} {xy[1]:9.3f}   bearing = {brg:7.2f} deg')
print(f'\n2202 recomputed bearing {brg_2202:.2f} deg vs recorded 132.5 +/- 2 deg')
print(f'2204 anchor sits {np.hypot(*(theo_en - [1652.198, 2025.104])):.1f} m '
      f'from the transmitter product')
''')

code(r'''
def ray(xy, bearing_deg, length=RAY_LENGTH_M, both=False):
    # Compass bearing -> (east, north) unit step. Returns a 2-point polyline.
    b = np.radians(bearing_deg)
    d = np.array([np.sin(b), np.cos(b)])
    start = xy - d * length if both else xy
    return np.array([start, xy + d * length])
''')

md(r'''
## 4. The map

Draws are plotted per chain so that non-convergence stays visible. The
polarization ray is drawn **double-ended** because a polarization arm is an
axis, not a vector — the two directions are the same physical statement. The
highline and theodolite rays are single-ended from their anchors.
''')

code(r'''
CLR = {'highline': '#d95f02', 'theodolite': '#7570b3', 'polarization': '#1b9e77',
       'gps': '#e7298a'}
fig, ax = plt.subplots(figsize=(9.2, 9.2))

ax.imshow(shade, extent=tile_extent, origin='lower', cmap='gray',
          vmin=.15, vmax=1., interpolation='bilinear', zorder=0, alpha=.85)
from scipy.ndimage import gaussian_filter
cs = ax.contour(np.linspace(tile_extent[0], tile_extent[1], tile.shape[1]),
                np.linspace(tile_extent[2], tile_extent[3], tile.shape[0]),
                gaussian_filter(tile, 3.), levels=np.arange(1600, 2100, 50),
                colors='k', linewidths=.4, alpha=.30, zorder=1)
ax.clabel(cs, inline=True, fontsize=5, fmt='%.0f')

# --- camera draws ---------------------------------------------------------
for k, d in cameras.items():
    flat = d.reshape(-1, 2)
    ax.plot(flat[:, 0], flat[:, 1], '.', ms=.7, alpha=.10,
            color='#2166ac', zorder=2, rasterized=True)
for k, d in cameras.items():
    med = np.median(d.reshape(-1, 2), axis=0)
    ax.plot(*med, 'o', ms=3.2, mfc='#2166ac', mec='w', mew=.5, zorder=5)
    ax.annotate(k, med, textcoords='offset points', xytext=(4, 3),
                fontsize=5.5, color='#0b3d6b', zorder=6)

# --- antenna and transmitter ---------------------------------------------
for name, colour, marker in (('antenna', '#b2182b', '*'),
                             ('transmitter', '#000000', 'D')):
    d = targets[name]
    flat = d.reshape(-1, 2)
    ax.plot(flat[:, 0], flat[:, 1], '.', ms=1.0, alpha=.13, color=colour,
            zorder=3, rasterized=True)
    med = np.median(flat, axis=0)
    ax.plot(*med, marker, ms=12 if marker == '*' else 6, mfc=colour,
            mec='w', mew=1.0, zorder=7, label=f'{name} (posterior median)')

# --- the citable brackets, for scale ------------------------------------
for name, xy, half in (('antenna bracket', (1655.868, 2030.923), 1.7),
                       ('transmitter bracket', (1652.198, 2025.104), 1.5)):
    ax.add_patch(plt.Rectangle((xy[0] - half, xy[1] - half), 2 * half, 2 * half,
                               fill=False, ec='#b2182b' if 'antenna' in name else 'k',
                               lw=.9, ls=':', zorder=8))

# --- GPS highline span: the measured anchors, drawn as a span not a ray ---
ax.plot([hl_e[0], hl_w[0]], [hl_e[1], hl_w[1]], '-', lw=2.2, color=CLR['gps'],
        alpha=.95, zorder=9)
ax.plot([hl_e[0], hl_w[0]], [hl_e[1], hl_w[1]], 's', ms=6, mfc=CLR['gps'],
        mec='w', mew=.8, zorder=10)
ax.plot(*hl_tie, 'v', ms=7, mfc=CLR['gps'], mec='w', mew=.8, zorder=10)
ax.plot([hl_tie[0], 1655.868], [hl_tie[1], 2030.923], '--', lw=1.0,
        color=CLR['gps'], alpha=.75, zorder=9)
for xy, lab in ((hl_e, 'E anchor'), (hl_w, 'W anchor'), (hl_tie, 'tiedown')):
    ax.annotate(lab, xy, textcoords='offset points', xytext=(5, 5), fontsize=6.5,
                color=CLR['gps'], zorder=11,
                bbox=dict(fc='w', ec='none', alpha=.65, pad=.8))

# --- heading rays --------------------------------------------------------
for name, (xy, brg, kind) in anchors.items():
    seg = ray(xy, brg, both=(kind == 'polarization'))
    ax.plot(seg[:, 0], seg[:, 1], '-', lw=1.7, color=CLR[kind], alpha=.95, zorder=9)
    ax.plot(*xy, 's', ms=4.5, mfc=CLR[kind], mec='w', mew=.6, zorder=10)
    tip = seg[-1]
    ax.annotate(f'{name}\n{brg:.2f} deg', tip, textcoords='offset points',
                xytext=(5, -2), fontsize=6.5, color=CLR[kind], zorder=11,
                bbox=dict(fc='w', ec='none', alpha=.6, pad=.8))

handles = [Line2D([], [], marker='*', ls='', ms=11, mfc='#b2182b', mec='w',
                  label='antenna (posterior median)'),
           Line2D([], [], marker='D', ls='', ms=6, mfc='k', mec='w',
                  label='transmitter (posterior median)'),
           Line2D([], [], marker='o', ls='', ms=5, mfc='#2166ac', mec='w',
                  label='camera (posterior median)'),
           Line2D([], [], ls=':', c='k', label='deterministic bracket'),
           Line2D([], [], c=CLR['gps'], lw=2.2, marker='s', ms=5, mfc=CLR['gps'],
                  mec='w', label='highline GPS anchors (measured span)'),
           Line2D([], [], c=CLR['highline'], lw=1.7, label='highline 2237/2238 (RETRACTED)'),
           Line2D([], [], c=CLR['theodolite'], lw=1.7, label='theodolite 2204 (174 deg)'),
           Line2D([], [], c=CLR['polarization'], lw=1.7, label='polarization 2202 (axis)')]
ax.legend(handles=handles, loc='lower left', fontsize=6.8, framealpha=.92)

ax.set_xlim(EXTENT[0], EXTENT[1])
ax.set_ylim(EXTENT[2], EXTENT[3])
ax.set_aspect('equal')
ax.set_xlabel('east (working grid, m)')
ax.set_ylabel('north (working grid, m)')
ax.set_title('Marjum 2026-07 geometry — MCMC draws (NOT CONVERGED) and '
             'face-value headings', fontsize=9.5)
fig.tight_layout()
fig.savefig('nb_fig_geometry_explorer.png', dpi=170, bbox_inches='tight')
plt.show()
''')

md(r'''
## 5. Zoom on the antenna / transmitter pair

At this scale the two brackets and the posterior clouds separate. Note that
`antenna_N` and `transmitter_N` carry posterior correlation **0.562** — the two
positions do not have independent error bars, and their *difference* is better
determined than either alone.
''')

code(r'''
ZE, ZN, PAD = 1653.5, 2028.5, 16.
fig, ax = plt.subplots(figsize=(6.4, 6.4))

ze = slice(int(np.searchsorted(e_axis, ZE - PAD)), int(np.searchsorted(e_axis, ZE + PAD)))
zn = slice(int(np.searchsorted(n_axis, ZN - PAD)), int(np.searchsorted(n_axis, ZN + PAD)))
ztile = dem.data[zn, ze].astype(float)
zext = (e_axis[ze][0], e_axis[ze][-1], n_axis[zn][0], n_axis[zn][-1])
ax.imshow(hillshade(ztile, dem.res), extent=zext, origin='lower', cmap='gray',
          vmin=0., vmax=1., interpolation='bilinear', zorder=0)

for name, colour, marker in (('antenna', '#b2182b', '*'),
                             ('transmitter', '#000000', 'D')):
    flat = targets[name].reshape(-1, 2)
    ax.plot(flat[:, 0], flat[:, 1], '.', ms=1.6, alpha=.10, color=colour,
            zorder=2, rasterized=True)
    med = np.median(flat, axis=0)
    ax.plot(*med, marker, ms=13 if marker == '*' else 7, mfc=colour, mec='w',
            mew=1.1, zorder=5, label=f'{name} median')

for name, xy, half, colour in (('antenna', (1655.868, 2030.923), 1.7, '#b2182b'),
                               ('transmitter', (1652.198, 2025.104), 1.5, 'k')):
    ax.add_patch(plt.Rectangle((xy[0] - half, xy[1] - half), 2 * half, 2 * half,
                               fill=False, ec=colour, lw=1.2, ls=':', zorder=6))
    ax.plot(*xy, '+', ms=9, mec=colour, mew=1.4, zorder=7)

for name, (xy, brg, kind) in anchors.items():
    seg = ray(xy, brg, length=40., both=(kind == 'polarization'))
    ax.plot(seg[:, 0], seg[:, 1], '-', lw=1.6, color=CLR[kind], alpha=.9, zorder=8)

ax.set_xlim(ZE - PAD, ZE + PAD)
ax.set_ylim(ZN - PAD, ZN + PAD)
ax.set_aspect('equal')
ax.set_xlabel('east (m)')
ax.set_ylabel('north (m)')
ax.set_title('Antenna / transmitter detail: draws, brackets (dotted), rays', fontsize=9)
ax.legend(loc='upper left', fontsize=7)
fig.tight_layout()
fig.savefig('nb_fig_geometry_explorer_zoom.png', dpi=170, bbox_inches='tight')
plt.show()

d_en = (np.median(targets['antenna'].reshape(-1, 2), axis=0)
        - np.median(targets['transmitter'].reshape(-1, 2), axis=0))
print(f'antenna - transmitter (posterior medians): dE {d_en[0]:+.3f}  dN {d_en[1]:+.3f} m')
print('bracketed products give                  : dE +3.670  dN +5.819 m')
''')

md(r'''
## 6. What the rays do and do not say

Where the three face-value rays point, relative to each other and to the
targets. Printed rather than eyeballed, because a 10 deg difference is hard to
judge off a map.
''')

code(r'''
ant_med = np.median(targets['antenna'].reshape(-1, 2), axis=0)
tx_med = np.median(targets['transmitter'].reshape(-1, 2), axis=0)


def bearing_to(xy, target):
    d = np.asarray(target, float) - np.asarray(xy, float)
    return float(np.degrees(np.arctan2(d[0], d[1])) % 360.)


def sep180(a, b):
    # Axis separation: polarization arms are axes, so compare mod 180.
    return abs((a - b + 90.) % 180. - 90.)


print('ray                    bearing   ->antenna   ->transmitter   miss@antenna')
for name, (xy, brg, kind) in anchors.items():
    b_ant = bearing_to(xy, ant_med)
    b_tx = bearing_to(xy, tx_med)
    rng = np.hypot(*(ant_med - xy))
    miss = rng * np.sin(np.radians((brg - b_ant + 180.) % 360. - 180.))
    print(f'{name:22s} {brg:7.2f}   {b_ant:8.2f}   {b_tx:12.2f}   {miss:+9.2f} m')

print()
print('Axis separations (mod 180, since an arm is an axis, not a vector):')
print(f'  2202 polarization vs SUPERSEDED theodolite arm 1 (166.71): '
      f'{sep180(brg_2202, 166.71):6.2f} deg')
print(f'  2202 polarization vs SUPERSEDED theodolite arm 2 ( 76.71): '
      f'{sep180(brg_2202, 76.71):6.2f} deg')
print(f'  2202 polarization vs theodolite reading (174.00): '
      f'{sep180(brg_2202, 174.00):6.2f} deg')
print()
print('RESOLVED 2026-09-18 -- the theodolite reading is the outlier, not a')
print('live conflict (MEMO-012 section 4.11). This camera heading is corroborated')
print('by its OWN EXIF compass to 2.77 deg (0.59 sigma of the 4.69 deg single-')
print('reading scatter); the theodolite is a single uncorroborated phone reading')
print('and lands ~45 deg from BOTH candidate arms, the least informative place.')
print()
print('  ADOPTED arms (axes, mod 180): 132.5 / 42.5 deg')
print('  SUPERSEDED theodolite arms  : 166.71 / 76.71 deg')
print()
print('Caveat that still stands: azimuth rotates polarization (section 4.9), so a')
print('sky-frame polarization axis is only meaningful with the az_pot it applies')
print('at -- and IMG_2202 predates the pointing table, so that az_pot is unrecorded.')
''')

md(r'''
## 7. Az-zero from the GPS highline anchors (2026-09-18)

The anchors are ground tie-off points measured directly, so the bearing between
them is **immune to the camera-baseline degeneracy** that retracted the 320.45
deg strike estimate. Any translation of the pair leaves the bearing unchanged,
so this number survives arbitrary common-mode GPS offset.

Applying the hardware relation `daz = compass(axis_westward) + 90` (elevation
axis in line with the highline; at el=0 the LIDAR is 90 deg from it by RHR
about up; pot zero referenced to that frame).
''')

code(r'''
def gc_bearing(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    dlo = lo2 - lo1
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return float(np.degrees(np.arctan2(y, x)) % 360.)


b_sphere = gc_bearing(HL_EAST, HL_WEST)
span = float(np.hypot(*hl_vec))
print(f'east->west bearing: spherical {b_sphere:.3f} deg, working-grid {hl_bearing:.3f} deg')
print(f'grid convergence (grid - true): {hl_bearing - b_sphere:+.3f} deg')
print(f'span {span:.1f} m')
print()
print('These are TRUE-north bearings by construction (computed from coordinates).')
print('No magnetic declination applies. The phone compass was already shown to be')
print("true-north (GPSImgDirectionRef='T'), so the chain is internally consistent;")
print('local declination is about +11 deg E and would be a 11 deg error if applied.')
print()
for sigma in (3., 5., 8.):
    print(f'  anchor sigma {sigma:.0f} m -> bearing sd '
          f'{np.degrees(np.arctan(sigma * np.sqrt(2) / span)):.2f} deg')

daz = (hl_bearing + 90.) % 360.
print(f'\naz-zero / daz = {hl_bearing:.2f} + 90 = {daz:.2f} deg  (+/- ~1.0 deg)')
print()
for tag, val in (('terrain LIDAR, eln pinned at 90', 38.86),
                 ('terrain LIDAR, E/N free', 38.5),
                 ('camera strike recomputed from picks', 38.03),
                 ('RETRACTED strike product (320.45+90)', 50.45),
                 ('terrain fit under RHR pot sense', 36.68)):
    print(f'  vs {tag:38s} {val:6.2f} -> {daz - val:+7.2f} deg')
''')

code(r'''
# Falsifier: the antenna hangs from the highline, so it should sit on the span.
u_hl = hl_vec / np.linalg.norm(hl_vec)
print('perpendicular offset from the GPS span (translation-sensitive, '
      'bearing-independent):')
for name, p in (('antenna', np.array([1655.868, 2030.923])),
                ('transmitter', np.array([1652.198, 2025.104]))):
    v = p - hl_e
    perp = v[0] * u_hl[1] - v[1] * u_hl[0]
    along = v @ u_hl
    print(f'  {name:12s} {perp:+7.2f} m   ({along:6.1f} m along a {span:.0f} m span)')

print()
print('TIEDOWN TEST -- the antenna should be pulled to the tiedown side:')
v = hl_tie - hl_e
tie_perp = v[0] * u_hl[1] - v[1] * u_hl[0]
tie_along = v @ u_hl
va = np.array([1655.868, 2030.923]) - hl_e
ant_perp = va[0] * u_hl[1] - va[1] * u_hl[0]
print(f'  tiedown  {tie_perp:+7.2f} m perp   ({tie_along:6.1f} m along)')
print(f'  antenna  {ant_perp:+7.2f} m perp   ({va @ u_hl:6.1f} m along)')
print(f'  same side: {"YES" if np.sign(tie_perp) == np.sign(ant_perp) else "NO"}'
      f'   |  stations agree to {abs(tie_along - va @ u_hl):.1f} m')
tie_alt = float(dem.interp_alt(np.array([hl_tie[0]]), np.array([hl_tie[1]]))[0])
hdist = float(np.hypot(*(np.array([1655.868, 2030.923]) - hl_tie)))
print(f'  tiedown ground {tie_alt:.1f} m; antenna {1777.4 - tie_alt:.1f} m above it; '
      f'tether {np.degrees(np.arctan2(hdist, 1777.4 - tie_alt)):.1f} deg off vertical')
print('  (this is one of two tiedowns, so it need not account for the full offset)')

print()
print('Anchor ground elevations vs the suspended hardware:')
for tag, ll in (('east anchor', HL_EAST), ('west anchor', HL_WEST)):
    en = dem.latlon_to_enu(*ll)[:2]
    alt = float(dem.interp_alt(np.array([en[0]]), np.array([en[1]]))[0])
    print(f'  {tag:12s} ground {alt:7.1f} m')
floor = float(dem.interp_alt(np.array([1655.868]), np.array([2030.923]))[0])
print(f'  antenna      {1777.4:7.1f} m   (hangs below both rims)')
print(f'  transmitter  {1684.4:7.1f} m   (canyon floor here is {floor:.1f} m)')
''')

md(r'''
### Reading of section 7

The span is **rim to rim**: both anchors sit well above the antenna, which hangs
65-98 m below them, and the transmitter sits essentially on the canyon floor.
That is a coherent highline, so the anchors are what they claim to be.

**az-zero = 37.8 deg +/- ~1.0 deg**, against the terrain LIDAR fit's 38.86 deg.
The two independent routes agree to **1.0 deg**, so the old 7-12 deg
highline/LIDAR gap **collapses** — it was an artefact of the retracted 320.45
deg input, not a real disagreement.

The antenna sits **10 m** to one side of the straight line between the anchors,
and that is **expected, not a discrepancy** (Aaron, 2026-09-18): the antenna
hangs from the highline on a **pulley plate tethered to the ground in two
places**, and those tethers pull it off the span axis.

This is now **confirmed, not just asserted**. The measured tiedown sits
**+6.57 m** off the span on the **same side** as the antenna's +10.00 m, at the
**same station** along the span (188.8 m vs 176.5 m of 419.8 m). A tiedown on
the opposite side, or hundreds of metres away, would have falsified the
explanation. It does not — and since this is one of *two* tiedowns, it need not
account for the full displacement on its own.

So the anchors were never a positional check on the antenna, only a bearing
measurement. Also consistent with the +11.7 deg cable lean and the tiedown lines
seen dropping below the antenna in 2237/2238.

The coordinates come from a **GPS receiver uploaded to CalTopo**, not the
iPhone, so the fitted common GPS bias does not apply. At a handheld receiver's
typical few-metre accuracy the bearing error bar is **+/-0.6 deg** (3 m per
anchor) rather than the +/-1.0 deg assumed before.

### Azimuth handedness

**CONFIRMED by Aaron 2026-09-18:** azimuth is measured by **right-hand rule
about the up vector**, so the compass bearing **decreases** as `az_pot`
increases. Handedness is closed.

Consequence: `marjum_lidar_constraint.py:26` uses the **opposite** sense
(`ray azimuth = az + daz`; it should be `daz - az`). With the terrain returns
sitting only 1.09 deg from pot zero, the two senses move the fit's implied
az-zero by 2.18 deg in total — **36.68 deg** under the correct sense against
38.86 deg as published. Near pot zero that is a small correction; far from zero
the sign **inverts the entire excursion**, so the pointing table needs it.
Flagged for the pointing-table owner; not changed here, since it is shared code.
''')

md(r'''
## 8. Provenance and one flagged discrepancy

### Inputs

| file | role |
|---|---|
| `archive/2026-09-14_joint_posterior_v1_NONCONVERGED/joint_posterior_v1/combined.npz` | MCMC draws, Rhat/ESS, coordinate names |
| `archive/2026-09-14_joint_posterior_v1_NONCONVERGED/fit_transmitter.npz` | bundle poses used as ray anchors for 2237/2238 |
| `marjum_dem.npz` | basemap; working-grid shift (0, 0) |
| `_tx2202/fit_result.npz` | 2202 pose (heading only; position unusable) |
| `marjum-2026-07/tx/IMG_2204.PNG` | theodolite app reading, transcribed |

### Discrepancy found while building this notebook

The recorded per-camera highline strikes (320.89 and 320.05) **do not
reproduce** from the archived cable picks in `_highline/cable_223[78].npy` with
the bundle poses. Recomputing the camera-plus-cable plane gives **308.0 deg
(2237)** and **308.1 deg (2238)** — within about 1 deg of each camera's own
bearing to the antenna (307.55 and 306.97), which is exactly the degeneracy
that got the 320.45 deg product retracted.

**Resolved 2026-09-18 by the GPS anchors (section 7):** the true bearing is
**307.85 deg**, so the recomputed 308.0/308.1 are *right* and the recorded
320.89/320.05 are wrong by about 13 deg. The estimator still has no resolving
power — the forward test stands — but its answer happened to be correct,
because the photographer sighted along the highline, which puts the
camera-to-antenna bearing on the true highline bearing by construction. An
estimator with no resolving power that lands on the truth for a structural
reason is still not evidence; the GPS anchors are.

The pick convention was pinned before drawing that conclusion: the arrays are
`(row, col, count)`, not `(x, y, count)`, and only that reading puts the fitted
plane through the antenna (0.09 m and 0.56 m, against 7-50 m for the
alternatives).

So the 320.45 deg number is not reconstructible from the archived picks by the
obvious route. This does **not** change its status — it was already retracted,
and this notebook draws it only because "at face value" was the request. It
does mean that anyone re-deriving a highline bearing should start from the
picks and a documented estimator, not from the recorded value. Logged as a
deferred finding; not chased here.

### Deferred findings

- Recorded strike 320.89/320.05 not reconstructible from archived picks (above).
- The rays are drawn from posterior/bundle anchors, so each inherits its
  anchor's positional error. For 2202 that error is metres, but the ray's
  *bearing* is degeneracy-immune, so only the ray's origin moves, not its
  direction.

### Decision requested

None. This is a viewer built to a direct request, not an analysis milestone. It
asserts no new geometry and supersedes nothing.
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python',
                          'name': 'python3'}
nb.metadata.language_info = {'name': 'python'}
out = HERE / 'geometry_explorer.ipynb'
nbf.write(nb, str(out))
print(f'wrote {out} ({len(cells)} cells)')
