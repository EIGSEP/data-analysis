"""Build the 23-camera MCMC diagnostic notebook.

Deliverable for reviewing the `marjum_mcmc_b21` antenna posterior:
  * per-image trace plots -- one set of axes per camera, never all cameras
    overlaid on a single plot;
  * a hillshade DEM map carrying the posterior E/N draws for every camera,
    the antenna draws, and the transmitter.

Usage: _build_mcmc_notebook.py RUN_DIR [OUT.ipynb]
"""
import sys
from pathlib import Path

import nbformat as nbf

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else 'b21_posterior_v1'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('Marjum 2026-07 Antenna MCMC Review.ipynb')
WORKSPACE = Path(__file__).resolve().parent  # this script lives at the repo root

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(r"""# Marjum 2026-07 — Joint Antenna + Transmitter Posterior: MCMC Review

**Milestone:** a **joint** posterior over the transmitter, the antenna, and
every camera that constrains either — plus the per-image trace and map
diagnostics Aaron asked for.

**Why this run exists.** Every position quoted so far has been a *deterministic*
fit with a hard bracket around it: ±1.5 m for the transmitter, ±1.7 m for the
antenna, both explicitly *not* posteriors. This run produces actual posteriors,
and for the transmitter it is the first one ever computed.

**Scope, and the count.** v0001 records 37 labelled images, of which **31**
carry an antenna pixel label and **8** a transmitter pixel label. Only **29**
have a usable camera pose: the other 8 antenna-labelled images were rejected at
registration (7 of them for entering a false terrain minimum) and have
`camera: null`, so they constrain nothing. This run therefore uses all **29**
poses — 23 antenna-labelled + 6 transmitter-only — which is every image that
actually constrains either target.

**What is sampled:** 29 camera poses (7 coordinates each), the antenna (3), the
transmitter (3), a common GPS bias (2), and an excess label-scatter term for
each target (2) — 213 global coordinates — plus 3,200 landmarks against 11,261
shared feature observations.

**What this notebook is not.** It does not re-run any fit. It reads the chains
and reports them.""")

code(r"""import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

WORKSPACE = Path('/home/aparsons/projects/eigsep/terrain')
RUN = WORKSPACE / %r

manifest = json.loads((RUN / 'manifest.json').read_text())
conv = json.loads((RUN / 'convergence.json').read_text())
with np.load(RUN / 'combined.npz', allow_pickle=True) as z:
    draws = np.asarray(z['draws'], float)          # (chain, draw, param)
    rhat = np.asarray(z['rhat'], float)
    ess_bulk = np.asarray(z['ess_bulk'], float)
    names = [str(s) for s in z['names']]
    keys = [str(s) for s in z['keys']]

nchain, ndraw, nparam = draws.shape
PRM = ('e', 'n', 'u', 'th', 'ph', 'ti', 'logf')
JOINT = bool(manifest.get('joint'))
print(f'{nchain} chains x {ndraw} retained draws x {nparam} coordinates, {len(keys)} cameras')
print('joint (transmitter + antenna):', JOINT)
print('antenna-labelled views:', manifest.get('antenna_labelled_views_n'),
      ' transmitter-labelled views:', manifest.get('transmitter_labelled_views_n'))
print('verdict:', conv['verdict'])""" % RUN_DIR)

md(r"""## 1. Provenance and configuration

The run refuses to start unless the state file it samples matches the SHA-256
that `v0001_marjum_geometry` pins for it, so the posterior is traceable to a
released geometry rather than to a working copy. Artifacts that the release
pins and that *have* since changed are recorded rather than hidden.""")

code(r"""g = manifest['geometry_release']
print('release        :', g['release'], '  status:', g.get('release_status'), '  cut', g.get('release_generated_utc'))
print('state file     :', g['state_file'])
print('  pinned sha256:', g['pinned_sha256'])
print('  actual sha256:', g['state_sha256'])
print('  VERIFIED     :', g['verified'])
print()
print('Other artifacts pinned by the release that now differ (expected, see caveats):')
for path, d in g.get('other_pinned_artifacts_that_differ', {}).items():
    print(f'  {path:24s} pinned {d["pinned"][:12]}  actual {d["actual"][:12]}')
print()
cfg = manifest['config']
print(f"cameras {manifest['cameras']}  landmarks {manifest['landmarks']}  observations {manifest['observations']}")
print(f"cameras without GPS/heading prior: {manifest['cameras_without_gps_or_heading_prior']}")
print(f"skyline_samples {cfg['skyline_samples']}  horizon_sigma {cfg['horizon_sigma_angular_px']} px  "
      f"antenna_label_sigma {cfg['antenna_label_sigma_px']} px  student_df {cfg['student_df']}")
print(f"tune {manifest['args']['tune']}  draws {manifest['args']['draws']}  seed {manifest['args']['seed']}")""")

md(r"""## 2. Convergence

Split-Rhat and effective sample size per coordinate, reduced to the worst-mixing
direction — a posterior is only as converged as its worst coordinate. Thresholds
are Rhat ≤ 1.01 (Vehtari et al. 2021, not the older 1.1) and bulk ESS ≥ 400.""")

code(r"""print(conv['verdict'])
print()
print(f"max Rhat      {conv['rhat_max']:.4f}  at {conv['rhat_max_at']}")
print(f"min bulk ESS  {conv['ess_bulk_min']:.0f}  at {conv['ess_bulk_min_at']}")
print(f"min tail ESS  {conv['ess_tail_min']:.0f}")
print(f"coordinates over Rhat threshold : {conv['n_rhat_above_threshold']} / {nparam}")
print(f"coordinates under ESS threshold : {conv['n_ess_below_threshold']} / {nparam}")
print()
print('Worst 15 coordinates by Rhat:')
print(f"{'coordinate':22s} {'Rhat':>8s} {'ESS bulk':>10s} {'ESS tail':>10s}")
for r in conv['worst_coordinates']:
    print(f"{r['name']:22s} {r['rhat']:8.4f} {r['ess_bulk']:10.0f} {r['ess_tail']:10.0f}")""")

code(r"""fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(rhat, bins=40, color='tab:blue')
axs[0].axvline(conv['thresholds']['rhat_max'], color='r', ls='--', label=f"threshold {conv['thresholds']['rhat_max']}")
axs[0].set_xlabel('split-Rhat'); axs[0].set_ylabel('coordinates'); axs[0].legend()
axs[0].set_title(f'Rhat over all {nparam} coordinates')
axs[1].hist(ess_bulk, bins=40, color='tab:green')
axs[1].axvline(conv['thresholds']['ess_min'], color='r', ls='--', label=f"threshold {conv['thresholds']['ess_min']:.0f}")
axs[1].set_xlabel('bulk ESS'); axs[1].legend(); axs[1].set_title('Effective sample size')
plt.tight_layout(); plt.show()""")

md(r"""## 3. Antenna posterior

The quantity this run exists to produce. Compare against the deterministic
products: the camera-fixed fit at `[1655.868, 2030.923, 1777.397]` m and its
**±1.7 m hard bracket** (`antenna_position_bracket.json`). The bracket is a
deterministic envelope with no coverage guarantee; the interval below is a
genuine posterior, but conditioned on a fixed camera network, fixed distortion
and a sampled-ray skyline surrogate, so it is *not* a total error budget and
should not be expected to reproduce the bracket.""")

code(r"""a = conv['antenna']
det = np.array([1655.868, 2030.923, 1777.397])
print(f"{'':12s}{'E':>12s}{'N':>12s}{'U':>12s}")
print(f"{'mean':12s}" + ''.join(f'{v:12.3f}' for v in a['mean']))
print(f"{'sd':12s}" + ''.join(f'{v:12.3f}' for v in a['sd']))
print(f"{'q05':12s}" + ''.join(f'{v:12.3f}' for v in a['q05']))
print(f"{'q50':12s}" + ''.join(f'{v:12.3f}' for v in a['q50']))
print(f"{'q95':12s}" + ''.join(f'{v:12.3f}' for v in a['q95']))
print(f"{'Rhat':12s}" + ''.join(f'{v:12.4f}' for v in a['rhat']))
print(f"{'ESS bulk':12s}" + ''.join(f'{v:12.0f}' for v in a['ess_bulk']))
print()
print(f"{'determ. fit':12s}" + ''.join(f'{v:12.3f}' for v in det))
off = np.array(a['mean']) - det
print(f"{'offset':12s}" + ''.join(f'{v:12.3f}' for v in off) + f'   |offset| = {np.linalg.norm(off):.3f} m')
print(f"posterior 90% widths (m): " + ', '.join(f'{hi-lo:.3f}' for lo, hi in zip(a['q05'], a['q95'])))
print()
print('Per-chain antenna means (dispersed starts must have met, not merely settled):')
for i, m in enumerate(a['per_chain_mean']):
    print(f'  chain {i}: {m}')
print('between-chain sd of those means (m):', a['between_chain_sd_of_means'])""")

code(r"""ai = names.index('antenna_e')
ant = draws[:, :, ai:ai+3]
fig, axs = plt.subplots(3, 2, figsize=(13, 8), gridspec_kw=dict(width_ratios=[3, 1]))
for j, lab in enumerate('ENU'):
    for c in range(nchain):
        axs[j, 0].plot(ant[c, :, j], lw=.4, alpha=.7, rasterized=True)
    axs[j, 0].set_ylabel(f'antenna {lab} (m)')
    axs[j, 1].hist(ant[:, :, j].ravel(), bins=60, orientation='horizontal',
                   color='tab:gray', rasterized=True)
    axs[j, 1].axhline(det[j], color='r', ls='--', lw=1, label='deterministic fit')
    axs[j, 1].set_yticklabels([])
    if j == 0:
        axs[j, 1].legend(fontsize=7)
axs[2, 0].set_xlabel('retained draw')
axs[0, 0].set_title(f'Antenna position: {nchain} chains overlaid')
axs[0, 1].set_title('marginal')
plt.tight_layout(); plt.show()""")

md(r"""## 3b. Transmitter posterior — the first one computed for this target

Compare against the deterministic product: `[1652.198, 2025.104, 1684.384]` m
with a **±1.5 m hard bracket**, and against the independent-pair solution
`[1652.410, 2025.238, 1684.742]` m obtained from cameras 2210 and 2211 alone —
the only two transmitter views whose poses are *not* fitted against the
transmitter.

**Read the width of this posterior with care.** Six of the eight
transmitter-labelled views are conditioned on the transmitter, and two of them
(2199, 2203) sit only ~3 m from it, so they pin it with enormous angular
leverage while contributing no independent information. A narrow posterior here
is therefore *not* evidence of a well-determined transmitter; it is what
conditioning looks like. The ±1.5 m bracket spans several fit generations and
captures methodological spread this posterior structurally cannot see.""")

code(r"""if JOINT and 'transmitter' in conv:
    t = conv['transmitter']
    det_tx = np.array([1652.198, 2025.104, 1684.384])     # bracketed product
    ind_tx = np.array([1652.410, 2025.238, 1684.742])     # 2210+2211 alone
    print(f"{'':14s}{'E':>12s}{'N':>12s}{'U':>12s}")
    for lab in ('mean', 'sd', 'q05', 'q50', 'q95'):
        print(f'{lab:14s}' + ''.join(f'{v:12.3f}' for v in t[lab]))
    print(f"{'Rhat':14s}" + ''.join(f'{v:12.4f}' for v in t['rhat']))
    print(f"{'ESS bulk':14s}" + ''.join(f'{v:12.0f}' for v in t['ess_bulk']))
    print()
    print(f"{'bracketed':14s}" + ''.join(f'{v:12.3f}' for v in det_tx) + '   (+/-1.5 m hard bracket)')
    print(f"{'indep pair':14s}" + ''.join(f'{v:12.3f}' for v in ind_tx) + '   (2210+2211 alone)')
    m = np.array(t['mean'])
    print(f"{'offset vs bracketed':20s} {np.linalg.norm(m-det_tx):.3f} m"
          f"   vs independent pair {np.linalg.norm(m-ind_tx):.3f} m")
    print(f"posterior 90% widths (m): " + ', '.join(f'{hi-lo:.3f}' for lo, hi in zip(t['q05'], t['q95'])))
    print()
    print('Per-chain transmitter means:')
    for i, mm in enumerate(t['per_chain_mean']):
        print(f'  chain {i}: {mm}')
else:
    print('antenna-only run: no transmitter sampled')""")

md(r"""### 3c. How inconsistent are the manual picks?

Both targets carry an excess-scatter parameter: the extra pixel scatter, beyond
the assumed 3 px labelling precision, needed to reconcile all the picks of that
target. It is estimated, not assumed. A large value is the model reporting that
the picks disagree with each other by more than the stated precision — which for
the transmitter is exactly the known 2210/2211-vs-conditioned-six group
disagreement, now measured rather than argued about.""")

code(r"""for key, label in (('antenna_scatter_px', 'antenna'), ('transmitter_scatter_px', 'transmitter')):
    if key in conv:
        s = conv[key]
        print(f"{label:12s} excess scatter {s['mean']:6.2f} px  (sd {s['sd']:5.2f}, "
              f"90% {s['q05']:5.2f}-{s['q95']:5.2f})  Rhat {s['rhat']:.3f}  ESS {s['ess_bulk']:.0f}")
print()
print('Assumed base labelling precision: 3 px. Excess is added in quadrature.')""")

md(r"""## 4. Per-image trace plots — one set of axes per camera

Each camera gets its own row of axes: its seven sampled coordinates
(`e, n, u` in metres, `th, ph, ti` in radians, `log f`), with all chains
overlaid *within* each panel. Cameras are never overlaid on one another.
Panel titles carry that coordinate's Rhat, so a badly-mixing camera is visible
without cross-referencing the table. Traces are thinned and rasterized to keep
the review artifacts a sane size; thinning affects only the rendering.""")

code(r"""THIN = max(1, ndraw // 700)   # ~700 plotted points per chain per panel

def trace_block(cam_keys, thin=THIN):
    nrow = len(cam_keys)
    fig, axs = plt.subplots(nrow, 7, figsize=(17, 1.45 * nrow), squeeze=False)
    for r, key in enumerate(cam_keys):
        base = keys.index(key) * 7
        for j, prm in enumerate(PRM):
            ax = axs[r, j]
            for c in range(nchain):
                ax.plot(np.arange(0, ndraw, thin), draws[c, ::thin, base + j],
                        lw=.35, alpha=.75, rasterized=True)
            ax.set_xticks([])
            ax.tick_params(labelsize=6)
            ax.set_title(f'{prm}  R={rhat[base+j]:.2f}', fontsize=7,
                         color=('firebrick' if rhat[base + j] > conv['thresholds']['rhat_max'] else 'black'))
            if j == 0:
                ax.set_ylabel(f'IMG_{key}', fontsize=8, fontweight='bold')
    fig.suptitle(f'Per-camera parameter traces ({nchain} chains overlaid per panel), '
                 f'cameras {cam_keys[0]}-{cam_keys[-1]}', fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()

trace_block(keys[:8])""")

code(r"""trace_block(keys[8:16])""")

code(r"""trace_block(keys[16:24])""")

code(r"""if len(keys) > 24:
    trace_block(keys[24:])""")

md(r"""## 5. Hillshade map with posterior E/N draws

Every camera's posterior draws in east-north space, on the hillshade DEM, with
the antenna draws and the transmitter. This is the plot that shows whether the
posterior is a tight cloud per camera or a smear, and how the camera network
sits relative to the two targets.

Both targets are sampled here, so both carry draw clouds. The bracketed
transmitter product (±1.5 m) is drawn as a circle for comparison — note how much
tighter the posterior cloud is than that bracket, and see §3b and the caveats
for why that tightness is not the same thing as accuracy.""")

code(r"""from eigsep_terrain.marjum_dem import MarjumDEM as DEM
from marjum_bundle import working_grid
from terrain_hillshade_cmap import plot_terrain_hillshade
from matplotlib.patches import Circle

dem = working_grid(DEM(cache_file='marjum_dem_sw.npz'))
se, sn = dem._working_shift
dem_raw = DEM(cache_file='marjum_dem_sw.npz')

ant_mean = np.array(conv['antenna']['mean'])
shared = json.loads((WORKSPACE / 'v0001_marjum_geometry_snapshot' / 'shared.json').read_text())
tx_rec = shared['transmitter']['recommended_for_propagation']
tx_pos, tx_bound = np.array(tx_rec['position_enu_m']), tx_rec['bound_m']
tx_i = names.index('transmitter_e') if 'transmitter_e' in names else None
tx_mean = np.array(conv['transmitter']['mean']) if JOINT and 'transmitter' in conv else None

# Frame the map on everything we intend to draw.
cam_en = np.array([[draws[:, :, keys.index(k)*7].mean(), draws[:, :, keys.index(k)*7+1].mean()]
                   for k in keys])
allpts = np.vstack([cam_en, ant_mean[:2][None, :], tx_pos[:2][None, :]])
lo, hi = allpts.min(0) - 60, allpts.max(0) + 60
e0, n0 = (lo + hi) / 2
rng = float(max(hi - lo) / 2 + 40)

E_raw, N_raw, U_map = dem_raw.get_tile(erng_m=(e0 - se - rng, e0 - se + rng),
                                       nrng_m=(n0 - sn - rng, n0 - sn + rng), mesh=False)
E_map, N_map = E_raw + se, N_raw + sn

fig, ax = plt.subplots(figsize=(12, 11))
im = plot_terrain_hillshade(ax, U_map, res=float(dem.res),
                            extent=(E_map[0], E_map[-1], N_map[0], N_map[-1]))
fig.colorbar(im, ax=ax, label='Elevation [m]')

TH = max(1, ndraw // 300)
colors = plt.cm.tab20(np.linspace(0, 1, len(keys)))
for k, col in zip(keys, colors):
    b = keys.index(k) * 7
    e = draws[:, ::TH, b].ravel(); n = draws[:, ::TH, b + 1].ravel()
    ax.scatter(e, n, s=1.2, color=col, alpha=.35, linewidths=0, rasterized=True)
    ax.annotate(k, (e.mean(), n.mean()), fontsize=6.5, color='k',
                xytext=(3, 3), textcoords='offset points')

ax.scatter(draws[:, ::TH, ai].ravel(), draws[:, ::TH, ai + 1].ravel(),
           s=2.5, color='k', alpha=.5, linewidths=0, rasterized=True, label='antenna draws')
ax.plot(*ant_mean[:2], 'w*', ms=20, mec='k', mew=1.2, label='antenna posterior mean')
if tx_i is not None:
    ax.scatter(draws[:, ::TH, tx_i].ravel(), draws[:, ::TH, tx_i + 1].ravel(),
               s=2.5, color='tab:red', alpha=.5, linewidths=0, rasterized=True,
               label='transmitter draws')
    ax.plot(*tx_mean[:2], 'r*', ms=18, mec='k', mew=1.2, label='transmitter posterior mean')
ax.plot(*tx_pos[:2], 'cX', ms=13, mec='k', mew=.8, label=f'transmitter bracketed product (±{tx_bound} m)')
ax.add_patch(Circle(tx_pos[:2], tx_bound, fill=False, color='c', ls='--'))

ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]'); ax.set_aspect('equal')
ax.set_title('Posterior E/N draws per camera, antenna draws, and the transmitter')
ax.legend(loc='best', fontsize=8)
plt.tight_layout(); plt.show()""")

code(r"""# Zoom on the antenna cloud alone: at canyon scale the camera spread dominates
# and the antenna posterior is a dot, so its shape needs its own axes.
fig, ax = plt.subplots(figsize=(7, 6.5))
e = draws[:, :, ai].ravel(); n = draws[:, :, ai + 1].ravel()
ax.scatter(e, n, s=2, alpha=.25, color='tab:blue', linewidths=0, rasterized=True)
ax.plot(ant_mean[0], ant_mean[1], 'k*', ms=18, label='posterior mean')
ax.plot(det[0], det[1], 'rP', ms=12, label='deterministic fit')
ax.add_patch(Circle(det[:2], 1.7, fill=False, color='r', ls='--', label='±1.7 m bracket'))
ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]'); ax.set_aspect('equal')
ax.set_title('Antenna posterior in E/N, against the deterministic fit and its bracket')
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

code(r"""if tx_i is not None:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    e = draws[:, :, tx_i].ravel(); n = draws[:, :, tx_i + 1].ravel()
    ax.scatter(e, n, s=2, alpha=.25, color='tab:red', linewidths=0, rasterized=True)
    ax.plot(tx_mean[0], tx_mean[1], 'k*', ms=18, label='posterior mean')
    ax.plot(tx_pos[0], tx_pos[1], 'cP', ms=12, label='bracketed product')
    ax.plot(1652.410, 2025.238, 'mD', ms=9, label='2210+2211 alone (independent)')
    ax.add_patch(Circle(tx_pos[:2], tx_bound, fill=False, color='c', ls='--', label=f'±{tx_bound} m bracket'))
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]'); ax.set_aspect('equal')
    ax.set_title('Transmitter posterior in E/N, against the bracket and the independent pair')
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()""")

md(r"""## 6. Sampler behaviour

Acceptance rates per block, per chain. Targets are 0.234 (camera blocks), 0.3
(antenna, shift), 0.35 (landmarks), 0.44 (scatter term).""")

code(r"""cols = [c for c in ('camera', 'antenna', 'transmitter', 'point', 'extra', 'tx_extra', 'shift')
        if c in conv['acceptance'][0]]
print(f"{'chain':>6s}" + ''.join(f'{c:>13s}' for c in cols))
for i, acc in enumerate(conv['acceptance']):
    print(f"{i:6d}" + ''.join(f"{acc[c]:13.3f}" for c in cols))
print()
print('Targets: camera 0.234, antenna/transmitter/shift 0.300, point 0.350, scatter terms 0.440')
print()
print('Note: the per-100-iteration acceptance printed in the chain logs reads 0.00 during '
      'the tuning phase. That is a reporting artifact -- adapt() zeroes the counters every '
      '25 iterations and the 100-iteration print lands on a reset. The values above are '
      'accumulated over retained draws only, with the kernel frozen.')""")

md(r"""## 7. Caveats

1. **These are posteriors, but not total error budgets.** They are conditioned
   on a fixed camera-network topology, fixed per-lens-group distortion, and a
   sampled-ray skyline surrogate (4096 rays) rather than a raster horizon. They
   do not include the methodological spread between fit generations that the
   ±1.5 m and ±1.7 m deterministic brackets capture. Quoting a posterior sd as
   *the* uncertainty would understate it for either target.
1b. **The transmitter posterior is the sharper trap.** Six of its eight views
   are conditioned on it and two sit ~3 m away, so it is pinned by cameras that
   were fitted to it. Its narrowness measures that conditioning, not accuracy.
   The independent evidence — 2210 and 2211, the only unconditioned views —
   places the transmitter ~1 m higher than every joint fit in this lineage, and
   that disagreement is **not** resolved by this run. It survives removing
   camera 2159 from the fit entirely (control run
   `cv_transmitter_refit_2159drop`), so it is a property of the remaining
   ultra-close conditioned cameras, not of any one view.
2. **Two cameras have no GPS or heading prior** (2209, 2210) and are two of the
   four opposite-side views carrying most of the horizontal information. Their
   priors were dropped rather than invented; this is recorded in the manifest.
3. **`marjum_bundle.py` and `meta.json` differ from the hashes v0001 pins.**
   `marjum_bundle.py`: `working_grid` was restored to it on 2026-09-14 —
   this module *requires* that function; without it every camera lands ~255 m
   underground. `meta.json`: 2211's transmitter pixel was re-picked after the
   release, which does not touch this antenna-era posterior (it uses `ant_px`).
   The state file itself — the antenna fit being sampled — matches v0001 exactly.
4. **The landmark coordinates are sampled but thinned to every 100th draw** in
   the saved chains, so landmark diagnostics are coarse by construction. The
   167 global coordinates reported here are retained at full rate.
5. **A bug in the sampler's step-size adaptation was fixed to make this run
   possible** — `adapt()` compared an array of per-camera proposal counts with
   a scalar and raised, which aborted every chain at iteration 25. That is why
   the earlier attempt left a manifest and no chain file.
6. **The state sampled is not the released v0001 fit.** It is
   `cv_transmitter_refit_2159B`, the re-fit that recovered camera 2159 from a
   false terrain minimum (158° from its own compass bearing → 1.3°, and 48.8 m
   from camera 2203 → 2.0 m, matching Aaron's recollection of "a couple of
   metres"). The release check records this honestly rather than claiming a
   match. The provenance chain is: v0001 → 2159 re-initialised from 2203 →
   this state.
7. **The six transmitter-era cameras have GPS/heading priors for the first
   time.** Their EXIF was never in the cached prior file; it was extended from
   the HEICs (`marjum_exif_extend.py`). Positions are reconstructed through
   EPSG:6341 minus an offset calibrated on the already-cached images, good to
   ~1.8 m, which is charged to those entries' stated GPS error and sits inside
   the 10 m prior floor. Headings need no projection and are exact.""")

md(r"""## 8. Decision requested

1. **Do the posteriors supersede, complement, or sit alongside the ±1.5 m and
   ±1.7 m deterministic brackets** for downstream consumers? They answer
   different questions — a posterior is conditioned on one network, a bracket
   spans several — and quoting either alone misleads in a different direction.
   This matters most for the transmitter, where the posterior is roughly an
   order of magnitude tighter than the bracket and, for the reasons in caveat
   1b, that tightness should **not** be read as an improvement.
2. **Is this run's convergence sufficient to publish**, or should it be
   extended / re-run with more draws? See §2 for the worst-mixing coordinates.
3. **Should the posterior be handed to data-archivist as a curation product**
   alongside the transmitter and antenna bracket products, and if so under what
   version/stamp?
4. **2209 and 2210 carry no GPS/heading prior** and dominate the horizontal
   information. Is dropping those priors still the right call, or should a
   weak, explicitly-labelled prior be supplied instead?
5. **Should the eight rejected images be re-tried?** Seven were excluded for
   "entering a false terrain minimum" — the exact failure we just proved
   recoverable on 2159 by re-initialising from a neighbouring pose. Recovering
   any of them would widen the constraining set beyond the current 29, and
   2210/2211-style unconditioned views are what the transmitter most needs.
6. **Should the compass-versus-fitted-boresight check become a standing
   guard?** It is nearly free and it isolates exactly this failure: across the
   six transmitter-era cameras it flagged 2159 at 157.6° while every other
   camera sat within 0.4-4.1° of its own tag.""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'arp', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}
nbf.write(nb, OUT)
print('wrote', OUT)
