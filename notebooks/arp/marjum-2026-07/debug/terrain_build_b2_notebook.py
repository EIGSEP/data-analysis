"""Build the B2 review notebook: Tx/antenna position with defensible uncertainties.

Generates `Marjum 2026-07 B2 Position Review.ipynb` from scratch (not by
editing the large historical antenna-pick / fit-inspector notebooks, which
carry megabytes of embedded image output). Run, then execute with nbconvert.
"""
import json
import nbformat as nbf
from pathlib import Path as _Path

_WORKSPACE = _Path(__file__).resolve().parent  # this script lives at the repo root
# marjum-2026-07/ is a fixed sibling directory in the same checkout (not a
# separate/movable one -- confirmed with data-archivist, 2026-09-16).
tx = json.loads((_WORKSPACE.parent / 'marjum-2026-07' / 'curation' / 'transmitter_position.json').read_text())
ant = json.loads((_WORKSPACE / 'cv_antenna_repick_v1' / 'report.json').read_text())

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(r"""# Marjum 2026-07 — B2: Transmitter & Antenna Position, with Defensible Uncertainties

**Milestone:** B2 (Tx/antenna position with defensible uncertainties).
**Prepared by:** geometer, 2026-09-13. **Requested by:** experimental-strategist,
under Aaron-approved fleet freeze (2026-09-13).

**Question.** What are the best-supported ENU coordinates of the transmitter
and the antenna on the working DEM grid, and what uncertainty can we
defensibly attach to each — as opposed to a single fit's own training
residual?

**Approach.** For both targets we do not trust any single optimizer fit's
residual as an uncertainty. Instead:
- **Transmitter:** bound the position by the envelope over independently
  defensible fit generations, checked against the *only* two cameras
  (2210, 2211) that are never refit against the transmitter ray and are
  therefore not circular evidence.
- **Antenna:** report the camera-fixed robust triangulation together with its
  full per-image reprojection residual table and leave-one-out (LOO)
  sensitivity, and state plainly where the residual is too large to treat as
  Gaussian.

This notebook is read-only over already-produced products; it runs no new
optimizer fits.""")

code(r"""import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

WORKSPACE = Path('/home/aparsons/projects/eigsep/terrain')
TX_PRODUCT = Path('/mnt/data02/eigsep/marjum-2026-07/curation/transmitter_position.json')
ANT_REPORT = WORKSPACE / 'cv_antenna_repick_v1' / 'report.json'

tx = json.loads(TX_PRODUCT.read_text())
ant = json.loads(ANT_REPORT.read_text())
print('Tx product stamp:', tx['stamp'])
print('Antenna source:', ant['source'])""")

md(r"""## 1. Inputs & provenance

Both products are read directly from disk with no recomputation. Hashes and
generator commits are recorded in the products themselves; we display them
here rather than re-deriving them, to keep this notebook auditable against
the exact bytes it read.""")

code(r"""print('=== Transmitter position product provenance ===')
prov = tx['provenance']
print('generator:', prov['generator'], '@', prov['generator_commit'])
print('generated_utc:', prov['generated_utc'])
print(f"{'input':55s} sha256[:16]")
for i in prov['inputs']:
    print(f"{i['path']:55s} {i['sha256'][:16]}")

print()
print('=== Antenna repick provenance ===')
print('source:', ant['source'])
print('input_sha256[:16]:', ant['input_sha256'][:16] if isinstance(ant['input_sha256'], str) else ant['input_sha256'])
print('sigma_px:', ant['sigma_px'], ' camera_parameters_fixed:', ant['camera_parameters_fixed'],
      ' distortion_used:', ant['distortion_used'])""")

md(r"""## 2. Transmitter position — best estimate and uncertainty bracket

**Best estimate (ENU, m):** `{best}`
**Uncertainty:** a **{bound} m bounding radius** (not a posterior sigma) — see
`marjum_tx_uncertainty.py` / `TRANSMITTER_FIT.md` for the derivation rule.
""".format(best=tx['best_estimate_enu_m'], bound=tx['uncertainty']['bound_m']))

code(r"""u = tx['uncertainty']
print('bound_m:', u['bound_m'], '(', u['kind'], ')')
print('methodological_envelope_m:', u['basis']['methodological_envelope_m'])
print('independent_check_max_ray_miss_m:', u['basis']['independent_check_max_ray_miss_m'])
print('rule:', u['basis']['rule'])
print()
print(u['interpretation'])""")

md(r"""### 2a. Baseline diagnostics — per-generation transmitter position and independent-camera ray miss

The six "conditioned" cameras (2159/2171/2172/2198/2199/2203) are jointly fit
against the transmitter ray in every generation below, so their near-zero
misses are **circular** and are shown only for completeness — they are not
evidence of transmitter accuracy. Only 2210 and 2211 are independent checks.
`cv_transmitter_joint_v2` is excluded from the envelope as an outlier (see
column).""")

code(r"""gens = tx['generations']
rows = []
for name, g in gens.items():
    p = g['transmitter_enu_m']
    ind = g['independent_check_ray_miss_m']
    rows.append((name, p, ind.get('2210'), ind.get('2211'),
                 g['acceptance_accepted'], g['outlier_excluded_from_envelope']))

hdr = f"{'generation':32s} {'E':>9s} {'N':>9s} {'U':>9s} {'2210 miss(m)':>13s} {'2211 miss(m)':>13s} {'accepted':>9s} {'outlier':>8s}"
print(hdr)
print('-' * len(hdr))
for name, p, m2210, m2211, acc, out in rows:
    print(f"{name:32s} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f} "
          f"{('%.3f' % m2210) if m2210 is not None else '—':>13s} "
          f"{('%.3f' % m2211) if m2211 is not None else '—':>13s} "
          f"{str(acc):>9s} {str(out):>8s}")""")

code(r"""fig, ax = plt.subplots(1, 2, figsize=(11, 4))

names = [n for n in gens]
outlier = [gens[n]['outlier_excluded_from_envelope'] for n in names]
E = [gens[n]['transmitter_enu_m'][0] for n in names]
N = [gens[n]['transmitter_enu_m'][1] for n in names]
U = [gens[n]['transmitter_enu_m'][2] for n in names]
colors = ['tab:red' if o else 'tab:blue' for o in outlier]

ax[0].scatter(E, N, c=colors)
for n, e, nn in zip(names, E, N):
    ax[0].annotate(n.replace('cv_transmitter_', ''), (e, nn), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
be = tx['best_estimate_enu_m']
ax[0].scatter([be[0]], [be[1]], marker='*', s=200, c='k', label='bracket centroid (best est.)')
ax[0].set_xlabel('E (m)'); ax[0].set_ylabel('N (m)'); ax[0].set_title('Transmitter horizontal scatter across generations')
ax[0].legend(fontsize=7)

m2210 = [gens[n]['independent_check_ray_miss_m'].get('2210') for n in names]
m2211 = [gens[n]['independent_check_ray_miss_m'].get('2211') for n in names]
x = np.arange(len(names))
w = 0.35
ax[1].bar(x - w/2, m2210, width=w, label='2210 ray miss (m)')
ax[1].bar(x + w/2, m2211, width=w, label='2211 ray miss (m)')
ax[1].axhline(u['bound_m'], color='k', ls='--', lw=1, label=f"bound = {u['bound_m']} m")
ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace('cv_transmitter_', '') for n in names],
                                            rotation=60, ha='right', fontsize=7)
ax[1].set_ylabel('ray miss (m)'); ax[1].set_title('Independent-camera ray miss (not circular)')
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig('_b2_tx_diag.png', dpi=140)
plt.show()""")

md(r"""### 2b. Independent-pair solution (2210 + 2211 alone)

Triangulating **only** from the two cameras that are never conditioned on the
transmitter ray gives an independent cross-check on the fit-family envelope
above.""")

code(r"""ip = tx['independent_pair_solution']
print('reference generation:', ip['reference_generation'])
print('independent-pair solution (ENU):', ip['solution_enu_m'])
print('mutual ray gap (2210 vs 2211):', ip['mutual_ray_gap_m'], 'm')
print('offset from reference (dE, dN, dU, horiz, total):', ip['offset_from_reference_m'])
print()
print(ip['interpretation'])""")

md(r"""### 2c. Caveats — read before using this position downstream

1. **v4 is a candidate, not an accepted product.** `cv_transmitter_joint_v4/acceptance.json`
   records `accepted: false`, failing exactly one of 39 checks
   (`2211_transmitter_reprojection`). No transmitter generation in this family
   is currently accepted.
2. **Six of eight cameras are circular evidence.** Only 2210/2211 constrain
   transmitter accuracy independently; the conditioned six will always show
   near-zero ray miss because they are fit to the transmitter ray.
3. **The dominant uncertainty term is a real, unresolved ~1 m vertical group
   disagreement**, not a bad pick. Aaron independently verified 2211's pixel
   pick is correct (2026-09-13), retracting the earlier "different mast
   point" explanation. The two unconditioned cameras agree with each other to
   0.72 m and place the transmitter ~1.04 m *higher* than the joint-fit
   family, with only ~0.46 m horizontal offset — the signature of a biased
   solution (likely cameras 2199/2203 at ~2.6 m range absorbing angular
   weight with free poses), not scattered label error.
4. **DEM height quantization.** `marjum_dem_sw.npz` is int32; any elevation
   derived against it inherits a 1 m floor.
5. **No posterior, no coverage guarantee.** The 1.5 m figure is a deterministic
   bracket over fit generations (`ceil(max(envelope, independent_check) * 10) / 10`),
   not a Bayesian credible interval.""")

md(r"""## 3. Antenna position — best estimate and diagnostics

**Best estimate (ENU, m):** `{ant}` — camera-fixed robust triangulation,
`cv_antenna_repick_v1/fit_antenna.npz`, shifted `{shift}` m from the prior
(20-view) estimate.
""".format(ant=[round(x, 3) for x in ant['antenna']], shift=[round(x, 3) for x in ant['shift_m']]))

code(r"""print('initial_antenna:', ant['initial_antenna'])
print('antenna (fitted):', ant['antenna'])
print('shift_m:', ant['shift_m'])
print()
print('residual_median_px:', round(ant['residual_median_px'], 2),
      ' residual_rms_px:', round(ant['residual_rms_px'], 2),
      ' residual_max_px:', round(ant['residual_max_px'], 2))
print('leave-one-out median_px:', round(ant['loo_median_px'], 2),
      ' loo_rms_px:', round(ant['loo_rms_px'], 2),
      ' loo_max_px:', round(ant['loo_max_px'], 2))
print('optimizer:', ant['optimizer'])""")

md(r"""### 3a. Baseline diagnostics — per-image reprojection residuals

`before` = residual against the prior 20-view antenna estimate;
`after` = residual against the refit robust triangulation (23 views, camera
poses held fixed). `ray_miss_m` is the corresponding 3-D miss distance.""")

code(r"""before = {b['key']: b for b in ant['before']}
after = {a['key']: a for a in ant['after']}
keys = ant['keys']

hdr = f"{'image':8s} {'before px':>10s} {'after px':>9s} {'after ray_miss (m)':>19s}"
print(hdr); print('-' * len(hdr))
for k in keys:
    b = before[k]['error_px']; a = after[k]['error_px']; rm = after[k]['ray_miss_m']
    flag = '  <-- >1 m ray miss' if rm > 1.0 else ''
    print(f"{k:8s} {b:10.2f} {a:9.2f} {rm:19.3f}{flag}")""")

code(r"""fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(keys))
after_px = [after[k]['error_px'] for k in keys]
before_px = [before[k]['error_px'] for k in keys]
ax.bar(x - 0.2, before_px, width=0.4, label='before (prior 20-view antenna)')
ax.bar(x + 0.2, after_px, width=0.4, label='after (23-view robust refit)')
ax.set_xticks(x); ax.set_xticklabels(keys, rotation=60, ha='right', fontsize=8)
ax.set_ylabel('reprojection residual (px)')
ax.set_title('Antenna label reprojection residual by image')
ax.legend()
fig.tight_layout()
fig.savefig('_b2_ant_diag.png', dpi=140)
plt.show()""")

md(r"""### 3b. Caveats — read before using this position downstream

1. **Camera poses are held fixed** for this triangulation; it is not a joint
   refit, so it does not absorb camera pose error into the antenna estimate
   (by design), but it also cannot correct for camera pose error either.
2. **Five images have residuals of 40–58 px** (2209, 2210, 2211, 2217, 2235),
   corresponding to ray misses of 1.0–1.6 m. Per `ANTENNA_REPICK.md`, a local
   parallax-aware scan on one of these (2209) confirmed sub-meter camera
   adjustments preserve terrain residuals but do not remove the antenna
   residual — forcing it to zero raises held-out terrain residuals to ~13 px.
   **These labels must not be treated as exact Gaussian observations**; a
   robust mixture/error-floor or an added calibration degree of freedom is
   required before any downstream MCMC use.
3. LOO median (20.28 px) exceeds the plain fit median (15.47 px), i.e. the
   solution is somewhat sensitive to any single view — expected given the
   residual spread above, not evidence of a fitting bug.""")

md(r"""## 4. Summary

| Target | Best estimate (E, N, U) m | Uncertainty | Status |
|---|---|---|---|
| Transmitter | {tx_e} | **±{tx_u} m hard bracket** (not a sigma) | No accepted generation; use bracketed product |
| Antenna | {ant_e} | Not yet bracketed — residual too large/non-Gaussian on 5/23 views | Provisional; not MCMC-ready |

""".format(tx_e=tx['best_estimate_enu_m'], tx_u=tx['uncertainty']['bound_m'],
           ant_e=[round(x, 3) for x in ant['antenna']]))

md(r"""## 5. Decision requested

1. **Transmitter position:** may downstream work (e.g. delay/geometry
   calculations) cite `[1652.198, 2025.104, 1684.384] m ± 1.5 m` (hard
   bracket) as the current defensible value, given that no fit generation is
   accepted and the ~1 m vertical group disagreement remains unresolved?
2. **Transmitter follow-up:** is it worth spending further optimizer time to
   resolve the vertical group disagreement (free the established-camera
   poses in the joint fit, or downweight/exclude the ultra-close 2199/2203
   cameras), or should the ±1.5 m bracket stand as the delivered product for
   now and this be logged as a known open item?
3. **Antenna position:** the current best estimate
   `[1655.868, 2030.923, 1777.397] m` has no formal uncertainty bracket
   analogous to the transmitter's, and 5/23 labels have residuals too large
   to treat as Gaussian. Should B2's antenna deliverable be (a) a similar
   bracketing exercise over antenna-fit generations, (b) a robust-error-floor
   refit that down-weights the 5 problem views, or (c) is the current
   point estimate sufficient for the next downstream consumer's precision
   needs?

No further transmitter/antenna refit work has been started pending this
decision — per fleet-freeze protocol, this agent is not self-assigning
follow-on optimizer work.""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'arp', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}
path = _WORKSPACE / 'Marjum 2026-07 B2 Position Review.ipynb'
nbf.write(nb, path)
print('wrote', path)
