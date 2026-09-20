#!/usr/bin/env python3
"""Emit `comb_transition_waterfalls.ipynb`.

Frozen builder for the notebook, kept beside it so the notebook can be
regenerated rather than hand-edited (the repo's nbstripout clean filter
strips outputs, so the committed .ipynb is unexecuted and its rendered
.html/.pdf are the evidence).

Usage:  python3 build_comb_transition_waterfalls.py
"""

import nbformat as nbf

NB = "comb_transition_waterfalls.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# Digital self-comb transition boundaries — `np.diff` waterfalls

**Author:** rfi-analyst · **Campaign:** marjum-2026-07

## What this is

Six files that bracket the three *genuine* on/off transitions of the
**digital self-comb** (1.953125 MHz = exactly 8.000 channels,
channel-locked, residue 0 mod 8) in the early part of its
07-17/18 era. Each pair is an `ON -> off` file followed by the
`off -> ON` file that recovers:

| # | ON → off | off → ON |
|---|---|---|
| 1 | `corr_20260717_154410Z.h5` (15:44:10) | `corr_20260717_154619Z.h5` (15:46:19) |
| 2 | `corr_20260717_155704Z.h5` (15:57:04) | `corr_20260717_155912Z.h5` (15:59:12) |
| 3 | `corr_20260717_161623Z.h5` (16:16:23) | `corr_20260717_162916Z.h5` (16:29:16) |

These are the three transitions attributed to the comb itself. The other
17 box-air boundaries in the era are daemon transitions, a daemon
outage, or rfswitch/VNA states where the antenna was not connected —
i.e. the instrument stopped looking, not the comb stopping. Those are
deliberately excluded.

**Purpose:** visually confirm and characterise what actually changes at
these boundaries. This notebook *shows* the data; it does not settle the
comb's physical origin.

## Provenance

* Boundary determination: `marjum-2026-07/flags/diag/comb_inventory.jsonl`
  (best-phase tooth contrast at 8.000 ch, `lock8`), operating point
  `lock8 >= 30` with phase 0.
* Data access: `eigsep_data.index.MetadataIndex` +
  `eigsep_data.bundle.load_bundle` — the canonical loaders. The cross
  correlation is available through `antenna="cross"`, which resolves the
  `"04"` key per file and fixes conjugation orientation, so there is no
  reason to drop to raw `h5py` here.
* Units: raw correlator counts throughout. Frequencies in MHz, times UTC.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timezone

from eigsep_data.index import MetadataIndex
from eigsep_data.bundle import load_bundle
from eigsep_data.flagging.detectors import overflow_mask

DATA_DIR = "/mnt/data02/eigsep/marjum-2026-07/data"

# The six boundary files, in time order. Each (ON->off, off->ON) pair.
FILES = [
    "corr_20260717_154410Z.h5",   # ON  -> off   (transition 1)
    "corr_20260717_154619Z.h5",   # off -> ON
    "corr_20260717_155704Z.h5",   # ON  -> off   (transition 2)
    "corr_20260717_155912Z.h5",   # off -> ON
    "corr_20260717_161623Z.h5",   # ON  -> off   (transition 3)
    "corr_20260717_162916Z.h5",   # off -> ON
]
ROLE = dict(zip(FILES, ["ON->off", "off->ON"] * 3))

# lock8 as measured by the campaign comb inventory, box-air (input 4).
LOCK8_BOX_AIR = {
    "corr_20260717_154410Z.h5": 0.0,
    "corr_20260717_154619Z.h5": 164.6,
    "corr_20260717_155704Z.h5": 19.9,
    "corr_20260717_155912Z.h5": 136.2,
    "corr_20260717_161623Z.h5": 0.0,
    "corr_20260717_162916Z.h5": 64.6,
}

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
print("numpy", np.__version__)
""")

md(r"""
## Load

`patterns=` restricts the index to just these six files, and
`cache=False` keeps the notebook from writing an `.eigsep_index.h5`
sidecar into the shared campaign data directory. Both are deliberate:
this is a six-file study and it should leave no trace in shared state.
""")

code(r"""
index = MetadataIndex(DATA_DIR, patterns=tuple(FILES), cache=False, streams="all")
sel = index.select(files=FILES)

print(f"indexed {len(index.table)} integrations from {len(sel.files)} files")
print(f"skipped: {index.skipped or 'none'}")

bundles = {
    "box-gnd (auto, input 0)": load_bundle(sel, antenna="box-gnd"),
    "box-air (auto, input 4)": load_bundle(sel, antenna="box-air"),
    "cross (box-gnd x box-air)": load_bundle(sel, antenna="cross"),
}
for name, b in bundles.items():
    print(f"\n--- {name} ---")
    print(b.summary())
    print(f"dtype {b.data.dtype}")
""")

md(r"""
## Timebase

Every one of these six files has `sync_consistent = True`, so the
index's `time_best` is trustworthy here and is used directly for the UTC
labels below.

That is worth stating explicitly because it is *not* generally true of
this campaign: `header/times` is corrupt in 642 of 5,120 files
(May-2026 dates, negative spans), and the standing rule is to fall back
to the filename close time. These six need no such fallback — checked,
not assumed.
""")

code(r"""
meta = sel.meta
tsummary = (meta.groupby("file")
                .agg(n_int=("row", "size"),
                     sync_ok=("sync_consistent", "all"),
                     t_start=("time_best", "min"),
                     t_end=("time_best", "max")))
tsummary["utc_start"] = [datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")
                         for t in tsummary.t_start]
tsummary["utc_end"] = [datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")
                       for t in tsummary.t_end]
tsummary["role"] = [ROLE[f] for f in tsummary.index]
tsummary["lock8_box_air"] = [LOCK8_BOX_AIR[f] for f in tsummary.index]
display(tsummary[["role", "n_int", "sync_ok", "utc_start", "utc_end", "lock8_box_air"]])

assert bool(tsummary.sync_ok.all()), "a file has inconsistent sync; do not trust time_best"
print("all six files: sync_consistent = True")
""")

md(r"""
## int32 accumulator wrap — repaired on the autos only

The correlator's auto accumulators are int32 and **wrap**. Autos are
non-negative by construction, so `raw < 0` detects a wrap with no
threshold (`eigsep_data.flagging.detectors.overflow_mask`). In this
six-file window box-gnd has ~0.22% wrapped samples and box-air ~0.001%.

This matters more here than almost anywhere else: a wrap is a
$\pm 2^{32}$ step, and **this notebook differences along time**, so a
single unrepaired wrap produces a $4\times10^9$-count spike that sets
the colour scale and hides everything real.

**The cross correlation is deliberately left alone.** A cross
visibility is genuinely signed in both real and imaginary parts, so
`raw < 0` carries no information about wrapping there — applying the
auto repair to it would corrupt real data.
""")

code(r"""
def repair_auto(a):
    # +2**32 wherever the int32 auto accumulator wrapped.
    a = np.asarray(a, dtype=np.float64)
    wrapped = overflow_mask(a)
    return np.where(wrapped, a + 2.0 ** 32, a), int(wrapped.sum())


auto_gnd, n_g = repair_auto(bundles["box-gnd (auto, input 0)"].data)
auto_air, n_a = repair_auto(bundles["box-air (auto, input 4)"].data)
cross = bundles["cross (box-gnd x box-air)"].data          # complex, untouched

print(f"box-gnd wraps repaired: {n_g}")
print(f"box-air wraps repaired: {n_a}")
print(f"cross left as-is (complex, signed): dtype {cross.dtype}")

freqs = bundles["box-air (auto, input 4)"].freqs_mhz
t_best = bundles["box-air (auto, input 4)"].t
print(f"{freqs.size} channels, {freqs[0]:.3f}-{freqs[-1]:.3f} MHz")
""")

md(r"""
## Differencing

`np.diff` is taken **within each file**, never across a file join. The
six files are not contiguous in time (there are gaps of minutes between
them), so differencing a naively concatenated stack would manufacture a
bright artificial row at every boundary that has no physical meaning.

Each file's 240 integrations therefore yield 239 difference rows, and a
row of `NaN` is inserted between files as a visual separator.

For the cross, the displayed quantity is $\Delta|V|$ — the signed change
in visibility amplitude — so that all three panels show the same kind of
quantity and can share a colour interpretation. (`|ΔV|`, which also
responds to pure phase rotation, is a different question and is not what
is plotted here.)
""")

code(r"""
row_of = {f: np.flatnonzero((meta.file == f).values) for f in FILES}


def stacked_diff(arr):
    # Per-file np.diff along time, stacked with NaN separators. Returns
    # the stacked array plus the y-centre and y-extent of each file's
    # block, for labelling.
    blocks, centres, edges = [], [], []
    y = 0
    for i, f in enumerate(FILES):
        d = np.diff(arr[row_of[f]], axis=0)
        blocks.append(d)
        centres.append(y + d.shape[0] / 2.0)
        edges.append((y, y + d.shape[0]))
        y += d.shape[0]
        if i != len(FILES) - 1:
            blocks.append(np.full((1, arr.shape[1]), np.nan))
            y += 1
    return np.concatenate(blocks, axis=0), centres, edges


D_gnd, centres, edges = stacked_diff(auto_gnd)
D_air, _, _ = stacked_diff(auto_air)
D_crs, _, _ = stacked_diff(np.abs(cross))

print(f"stacked diff shape: {D_gnd.shape}  "
      f"({len(FILES)} files x 239 diff rows + {len(FILES)-1} separators)")

# Which stacked rows are antenna-to-antenna differences? A difference row
# is only a clean on-sky difference if BOTH of its endpoint integrations
# were on the antenna; if either end is a load or VNA state the row is a
# switch step, not a sky step. Built on the same stacking geometry so it
# lines up with the waterfalls row for row.
state = meta["rfswitch"].astype(str).values
ant_rows = []
for i, f in enumerate(FILES):
    s = state[row_of[f]]
    ant_rows.append((s[:-1] == "RFANT") & (s[1:] == "RFANT"))
    if i != len(FILES) - 1:
        ant_rows.append(np.zeros(1, dtype=bool))
ANT = np.concatenate(ant_rows)
print(f"antenna-to-antenna difference rows: {ANT.sum()} of {ANT.size}")
""")

md(r"""
## The figure

Three panels, `sharex=True, sharey=True`: box-gnd auto, box-air auto,
and the cross amplitude. Frequency runs along x; the concatenated
per-file difference rows run down y, with each file's block labelled by
name, role and UTC start.

The colour scale is symmetric about zero and set per panel — the three
panels differ in absolute scale by orders of magnitude, so a shared
colour scale would render two of them blank.

It is set from the **antenna-to-antenna difference rows only**. This is
not cosmetic. `corr_20260717_162916Z.h5` runs a VNA sweep, and its
switch steps reach $\sim10^8$ counts — two orders of magnitude above
anything on sky. Scaling to all rows lets that one file's calibration
saturate the box-air panel and render the other five files blank. Rows
outside the antenna state are still drawn, and simply clip.
""")

code(r"""
def robust_lim(D, pct=99.0, rows=None):
    # Scale from `rows` only (default: antenna-to-antenna rows), so
    # calibration switch steps cannot set the colour scale.
    sub = D if rows is None else D[rows]
    v = sub[np.isfinite(sub)]
    if v.size == 0:
        v = D[np.isfinite(D)]
    return float(np.percentile(np.abs(v), pct)) or 1.0


def waterfall(axes, panels, title_suffix=""):
    for ax, (label, D) in zip(axes, panels):
        lim = robust_lim(D, rows=ANT)
        im = ax.imshow(D, aspect="auto", origin="upper", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                       extent=[freqs[0], freqs[-1], D.shape[0], 0],
                       interpolation="nearest")
        ax.set_title(f"{label}{title_suffix}", fontsize=9)
        ax.set_xlabel("frequency [MHz]")
        for (y0, y1) in edges[1:]:
            ax.axhline(y0 - 0.5, color="k", lw=0.8, alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return axes


ylabels = []
for f in FILES:
    utc = datetime.fromtimestamp(tsummary.loc[f, "t_start"],
                                 timezone.utc).strftime("%H:%M:%S")
    ylabels.append(f"{f[5:-3]}\n{ROLE[f]}  {utc}Z\nlock8={LOCK8_BOX_AIR[f]:g}")

fig, axes = plt.subplots(1, 3, figsize=(15, 8.5), sharex=True, sharey=True)
waterfall(axes,
          [("box-gnd auto  $\\Delta$counts", D_gnd),
           ("box-air auto  $\\Delta$counts", D_air),
           ("cross  $\\Delta|V|$", D_crs)])
axes[0].set_yticks(centres)
axes[0].set_yticklabels(ylabels, fontsize=7)
axes[0].set_ylabel("file block  (per-file np.diff along time, UTC)")
fig.suptitle("Digital self-comb transition boundaries — time-difference waterfalls\n"
             "2026-07-17, three ON/off transition pairs (raw counts)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
""")

md(r"""
### Which rows are on sky

The same y geometry, showing the receiver state behind every difference
row. Green is an antenna-to-antenna difference (a real on-sky step);
grey is any row with a load, noise-source or VNA state at either end —
those rows are switch steps and carry no sky information.
""")

code(r"""
fig, ax = plt.subplots(figsize=(3.2, 8.5))
ax.imshow(ANT.reshape(-1, 1).astype(float), aspect="auto", origin="upper",
          cmap="Greens", vmin=0, vmax=1.6,
          extent=[0, 1, ANT.size, 0], interpolation="nearest")
for (y0, _) in edges[1:]:
    ax.axhline(y0 - 0.5, color="k", lw=0.8, alpha=0.6)
ax.set_xticks([])
ax.set_yticks(centres)
ax.set_yticklabels(ylabels, fontsize=7)
ax.set_title("antenna-to-antenna\ndifference rows", fontsize=9)
frac = [f"{f[5:-3]}: {ANT[y0:y1].mean()*100:5.1f}% on sky"
        for f, (y0, y1) in zip(FILES, edges)]
fig.tight_layout()
plt.show()
print("\n".join(frac))
""")

md(r"""
## Same difference, log space

The linear panels above are dominated by the brightest channels, which
is what a difference of raw counts does. The comb is a *narrowband,
low-amplitude* feature, so the same difference taken in
$\log_{10}$ space — i.e. a fractional change per integration — is what
actually makes a comb switching on or off legible.

This is the same operation on the same data, not a different selection.
The autos use $\log_{10}\max(\mathrm{counts}, 1)$; the cross uses
$\log_{10}\max(|V|, 1)$.
""")

code(r"""
L_gnd, _, _ = stacked_diff(np.log10(np.maximum(auto_gnd, 1.0)))
L_air, _, _ = stacked_diff(np.log10(np.maximum(auto_air, 1.0)))
L_crs, _, _ = stacked_diff(np.log10(np.maximum(np.abs(cross), 1.0)))

fig, axes = plt.subplots(1, 3, figsize=(15, 8.5), sharex=True, sharey=True)
waterfall(axes,
          [("box-gnd auto  $\\Delta\\log_{10}$", L_gnd),
           ("box-air auto  $\\Delta\\log_{10}$", L_air),
           ("cross  $\\Delta\\log_{10}|V|$", L_crs)])
axes[0].set_yticks(centres)
axes[0].set_yticklabels(ylabels, fontsize=7)
axes[0].set_ylabel("file block  (per-file np.diff along time, UTC)")
fig.suptitle("Same boundaries, difference taken in log space\n"
             "(fractional change per integration — narrowband features legible)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
""")

md(r"""
## Read this before interpreting the panels: the rfswitch confound

The receiver switches between the antenna (`RFANT`), an ambient load
(`RFAMB`), a noise source (`RFNON`) and several VNA states on a
calibration cadence *within* these files. Differencing along time
responds to every one of those switches, and a switch is a far larger
step than the comb.

In this particular six-file selection the cadence is **not** balanced
across the two roles:
""")

code(r"""
sw = (meta.assign(state=meta["rfswitch"].astype(str))
          .pivot_table(index="file", columns="state", values="row",
                       aggfunc="size", fill_value=0))
sw.insert(0, "role", [ROLE[f] for f in sw.index])
sw.insert(1, "lock8", [LOCK8_BOX_AIR[f] for f in sw.index])
display(sw)

pure = {f: bool((meta.loc[meta.file == f, "rfswitch"].astype(str) == "RFANT").all())
        for f in FILES}
print("\npure-RFANT (no cal switching in the file):")
for f in FILES:
    print(f"  {f}  role={ROLE[f]:8s}  pure_RFANT={pure[f]}")
""")

md(r"""
**All three `ON -> off` files are 100% `RFANT`. All three `off -> ON`
files contain calibration or VNA states.** In this sample the split is
perfect, 3/3 and 3/3.

That is a confound, and it has to be stated plainly: within *these six
files alone*, "comb detected" and "calibration cadence running" are not
separable, so nothing in the figures above can by itself distinguish
the comb switching on from the calibration hardware switching on.

It is, however, a property of this small selection and **not** of the
era. Across the full 07-17 15:37 → 07-18 03:00 era on box-air, at the
same `lock8 >= 30` operating point, the association runs the other way:

| | pure `RFANT` | cal/VNA active |
|---|---|---|
| comb ON | **176** | 29 |
| comb off | 49 | 56 |

So era-wide the comb is predominantly detected in *clean antenna* files,
and the six boundary files here are an unrepresentative stretch. The
honest reading is that these six make a good visual case study of what a
boundary looks like, and a **bad** basis for attribution.

One further caveat specific to `corr_20260717_162916Z.h5`: only 29 of
its 240 integrations are `RFANT`, so its `lock8 = 64.6` detection rests
on 29 antenna integrations, not 240.
""")

md(r"""
## Decision requested

1. Does the characterisation above match what you expect a genuine comb
   on/off boundary to look like, or do the panels instead read as
   calibration switching to you?
2. If the rfswitch confound is the dominant concern, the natural next
   step is to re-pick boundary files **matched on cal state** (both
   sides of a transition pure-`RFANT`) and repeat this figure. Era-wide
   there are 49 pure-`RFANT` comb-off files to draw from. That is a new
   selection and a new milestone, so I have not done it here.
3. `lock8 >= 30` is the operating point used to call these transitions;
   it is defensible (a stable plateau, and 216/220 agreement with
   beam-analyst's independent detector) but it is a choice, and
   transition *counts* move with it.

**Not claimed here:** anything about the comb's physical origin. This
notebook shows morphology at six boundaries; it runs no pointing test,
no null control, and no far-field discriminator.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
with open(NB, "w") as fh:
    nbf.write(nb, fh)
print(f"wrote {NB} ({len(cells)} cells)")
