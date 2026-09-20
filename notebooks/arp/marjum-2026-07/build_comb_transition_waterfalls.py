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

Eight files bracketing the three *genuine* on/off transitions of the
**digital self-comb** (1.953125 MHz = exactly 8.000 channels,
channel-locked, residue 0 mod 8) in the early part of its 07-17/18 era.

| # | ON → off | off → ON | extra context |
|---|---|---|---|
| 1 | `corr_20260717_154410Z.h5` (15:44:10) | `corr_20260717_154619Z.h5` (15:46:19) | — |
| 2 | `corr_20260717_155704Z.h5` (15:57:04) | `corr_20260717_155912Z.h5` (15:59:12) | — |
| 3 | `corr_20260717_161623Z.h5` (16:16:23) | `corr_20260717_162916Z.h5` (16:29:16) | `…162707Z` before, `…163125Z` after |

These are the three transitions attributed to the comb itself. The other
17 box-air boundaries in the era are daemon transitions, a daemon
outage, or rfswitch/VNA states where the antenna was not connected —
i.e. the instrument stopped looking, not the comb stopping. Those are
deliberately excluded.

**Transition 3 is bracketed with two extra files** because a
per-integration check (see "Where transition 3 actually happens")
showed the file-level label is misleading there: `…162916Z` is an
*isolated single file* — off on both inputs immediately before and
immediately after it — and the step inside it is at the very end, not
the start. `…162707Z` and `…163125Z` are its contiguous neighbours
(129 s file cadence, verified, no gap).

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
from pathlib import Path

from eigsep_data.index import MetadataIndex
from eigsep_data.bundle import load_bundle
from eigsep_data.flagging.detectors import overflow_mask

DATA_DIR = "/mnt/data02/eigsep/marjum-2026-07/data"

# Boundary files, in time order. Transitions 1 and 2 are (ON->off,
# off->ON) pairs; transition 3 is bracketed with a before and an after
# file (see the per-integration section for why).
FILES = [
    "corr_20260717_154410Z.h5",   # ON  -> off   (transition 1)
    "corr_20260717_154619Z.h5",   # off -> ON
    "corr_20260717_155704Z.h5",   # ON  -> off   (transition 2)
    "corr_20260717_155912Z.h5",   # off -> ON
    "corr_20260717_161623Z.h5",   # ON  -> off   (transition 3)
    "corr_20260717_162707Z.h5",   #   before  (contiguous predecessor)
    "corr_20260717_162916Z.h5",   # off -> ON  (isolated single file)
    "corr_20260717_163125Z.h5",   #   after   (contiguous successor)
]
ROLE = {
    "corr_20260717_154410Z.h5": "ON->off",
    "corr_20260717_154619Z.h5": "off->ON",
    "corr_20260717_155704Z.h5": "ON->off",
    "corr_20260717_155912Z.h5": "off->ON",
    "corr_20260717_161623Z.h5": "ON->off",
    "corr_20260717_162707Z.h5": "before",
    "corr_20260717_162916Z.h5": "off->ON",
    "corr_20260717_163125Z.h5": "after",
}

# lock8 as measured by the campaign comb inventory, box-air (input 4).
LOCK8_BOX_AIR = {
    "corr_20260717_154410Z.h5": 0.0,
    "corr_20260717_154619Z.h5": 164.6,
    "corr_20260717_155704Z.h5": 19.9,
    "corr_20260717_155912Z.h5": 136.2,
    "corr_20260717_161623Z.h5": 0.0,
    "corr_20260717_162707Z.h5": 0.0,
    "corr_20260717_162916Z.h5": 64.6,
    "corr_20260717_163125Z.h5": 0.0,
}
# Same, box-gnd (input 0) -- it disagrees with box-air at transition 3.
LOCK8_BOX_GND = {
    "corr_20260717_154410Z.h5": 0.0,
    "corr_20260717_154619Z.h5": 399.5,
    "corr_20260717_155704Z.h5": 4.4,
    "corr_20260717_155912Z.h5": 362.2,
    "corr_20260717_161623Z.h5": 0.0,
    "corr_20260717_162707Z.h5": 0.0,
    "corr_20260717_162916Z.h5": 212.4,
    "corr_20260717_163125Z.h5": 0.0,
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
tsummary["lock8_box_gnd"] = [LOCK8_BOX_GND[f] for f in tsummary.index]
display(tsummary[["role", "n_int", "sync_ok", "utc_start", "utc_end",
                  "lock8_box_air", "lock8_box_gnd"]])

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
    ylabels.append(f"{f[5:-3]}\n{ROLE[f]}  {utc}Z\n"
                   f"lock8 air={LOCK8_BOX_AIR[f]:g} gnd={LOCK8_BOX_GND[f]:g}")

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
## Where transition 3 actually happens

The file-level label says the comb returns at `corr_20260717_162916Z.h5`
(16:29:16). A file is 240 integrations over ~129 s, so that label
localises the step no better than to the file. The question worth asking
directly is *where inside the file* it happens — and the answer decides
whether the notebook is even showing the transition.

Below, `lock8` is recomputed per 8-integration block using
`comb_inventory.py`'s own `tooth_contrast_locked` — imported from the
script, not reimplemented, so the statistic cannot silently drift from
the one that produced the published boundary list. Two traces per input:

* **all** — every integration in the block, whatever the receiver state.
* **antenna-only** — restricted to `RFANT` integrations, which is what
  `comb_inventory.process()` itself does.
""")

code(r"""
import importlib.util
from scipy.ndimage import median_filter

_ci_path = (Path.cwd().resolve().parents[2]
            / "scripts/marjum-2026-07/flagging/comb_inventory.py")
_spec = importlib.util.spec_from_file_location("comb_inventory", _ci_path)
CI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CI)
DET = CI.D
print(f"tooth statistic imported from {_ci_path}")


def lock8_trace(fname, key, block=8):
    # Per-block lock8 for one file/input, both all-integration and
    # antenna-only, using comb_inventory's own statistic.
    import h5py
    with h5py.File(f"{DATA_DIR}/{fname}", "r") as h:
        freqs_f = h["header/freqs"][:]
        raw = h["data/" + key][:]
        rfsw = (h["metadata/rfswitch"][()]
                if ("metadata" in h and "rfswitch" in h["metadata"]) else None)
    ant = DET.antenna_mask(rfsw, raw.shape[0])
    val = raw.astype(np.float64)
    ovf = DET.overflow_mask(raw)
    if ovf.any():
        val = np.where(ovf, val + 2.0 ** 32, val)
    logp = np.log10(np.maximum(val, 1.0))
    chans = np.arange(logp.shape[1])
    in_band = (freqs_f >= CI.BAND[0]) & (freqs_f <= CI.BAND[1])

    rows = []
    for s in range(0, raw.shape[0], block):
        sl = slice(s, s + block)
        med = np.median(logp[sl], axis=0)
        c_all, _ = CI.tooth_contrast_locked(
            med - median_filter(med, size=17, mode="nearest"),
            chans, in_band, 8)
        m = ant[sl]
        if m.sum() >= 2:
            med2 = np.median(logp[sl][m], axis=0)
            c_ant, _ = CI.tooth_contrast_locked(
                med2 - median_filter(med2, size=17, mode="nearest"),
                chans, in_band, 8)
        else:
            c_ant = np.nan
        rows.append((s, int(m.sum()), c_all, c_ant))
    return pd.DataFrame(rows, columns=["int0", "n_ant", "lock8_all",
                                       "lock8_ant_only"])


TRIO = ["corr_20260717_162707Z.h5", "corr_20260717_162916Z.h5",
        "corr_20260717_163125Z.h5"]
traces = {(f, k): lock8_trace(f, k) for f in TRIO for k in ("4", "0")}
print("traces computed")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, f in zip(axes, TRIO):
    for k, colour, lab in (("4", "tab:blue", "box-air (input 4)"),
                           ("0", "tab:red", "box-gnd (input 0)")):
        tr = traces[(f, k)]
        ax.plot(tr.int0, tr.lock8_all, color=colour, lw=1.2, label=f"{lab} all")
        ax.plot(tr.int0, tr.lock8_ant_only, color=colour, lw=0, marker="o",
                ms=4, label=f"{lab} antenna-only")
    tr4 = traces[(f, "4")]
    ax.fill_between(tr4.int0, 0, 1, where=(tr4.n_ant.values > 0),
                    transform=ax.get_xaxis_transform(), color="green",
                    alpha=0.10, step="post")
    ax.axhline(30, color="k", ls=":", lw=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("integration index within file")
    ax.set_title(f"{f[5:-3]}  ({ROLE[f]})", fontsize=9)
axes[0].set_ylabel("lock8 (tooth contrast at 8.000 ch)")
axes[0].legend(fontsize=6.5, loc="upper left")
fig.suptitle("Transition 3, per-integration: where the comb actually appears\n"
             "dotted line = lock8 = 30 operating point; green shading = "
             "antenna (RFANT) integrations", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.86])
plt.show()

for f in TRIO:
    t4, t0 = traces[(f, "4")], traces[(f, "0")]
    print(f"{f[5:-3]}  RFANT ints {int(t4.n_ant.sum())}/240   "
          f"box-air lock8 max {np.nanmax(t4.lock8_all):7.1f}   "
          f"box-gnd lock8 max {np.nanmax(t0.lock8_all):7.1f}")
""")

md(r"""
### What the trace shows — the file-level label is wrong three ways

**1. On box-air the step is at the very *end* of `…162916Z`, not the
front.** `lock8` sits at ~0.1 for integrations 0–207 and only rises at
**integration 208 of 240 (87% into the file, 16:29:00)**, reaching
49–72 for the final 32. Those final integrations are *exactly* the
file's only `RFANT` integrations. So what box-air records is not the
comb switching on — it is the **antenna being reconnected** at the end
of a VNA sweep, revealing a comb box-air could not see while the
antenna was disconnected. The file-level `lock8 = 64.6` is computed
entirely from those 29 integrations.

**2. On box-gnd the comb turns on in the *previous* file.** `…162707Z`
box-gnd is flat at 0.0 for integrations 0–191, then jumps to 233/240/225
at integrations 192–215 — onset **16:26:43**, some 2.5 minutes before
the file-level boundary. Box-gnd is conducted and does not need the
antenna connected, so it can date the onset and box-air cannot.

**3. It turns off in the *following* file, and the file-level statistic
misses that entirely.** `…163125Z` is published as `lock8 = 0.0` on both
inputs. Per block it is nothing of the kind: the comb is **present for
integrations 0–71 on both inputs** (box-gnd 130–239, box-air 43–57),
then stops hard at **integration 72, 16:29:56**, and stays off for the
remaining 168. The published 0.0 is a **median artefact** — `lock8` is
computed from a median over all 240 integrations, so a feature with
~30% duty cycle within a file washes out completely and the file reads
as clean.

Putting the three together, transition 3 is not a single-file event at
16:29:16 at all. The actual episode is

> **16:26:43 → 16:29:56 UTC, about 3 min 13 s, spanning three files.**

The boundary list compresses that to one file and dates its start 2.5
minutes late. This is a **systematic** property, not a one-off: any
comb episode shorter than roughly half a file is liable to be missed
outright, and any episode whose start falls in a cal-heavy stretch will
be dated to whenever the antenna next comes back. Short episodes
elsewhere in the era should be assumed to be under-counted until
checked per-integration.

Adding the two neighbouring files is what makes all of this visible;
with the original six it was invisible by construction.
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
**Every `ON -> off` file is 100% `RFANT`; every `off -> ON` file
contains calibration or VNA states.** In the original six-file
selection the split was perfect, 3/3 and 3/3.

That is a confound, and it has to be stated plainly: within those six
files alone, "comb detected" and "calibration cadence running" are not
separable, so nothing in the waterfalls above can by itself distinguish
the comb switching on from the calibration hardware switching on.

**The two files added for transition 3 partly break that degeneracy**,
which is the main reason they earn their place beyond bracketing:

* `…163125Z` is **240/240 `RFANT`, 100% on sky** and shows the comb
  present on the antenna for its first 72 integrations and absent for
  the remaining 168 — an on *and* an off, both with the antenna
  connected throughout and no switch anywhere near either. That is the
  control the original six lacked entirely, and it rules out the
  reading that the comb is merely "whatever the VNA sweep puts into the
  band".
* `…162707Z` is 70.3% on sky and carries the box-gnd onset at
  integration ~192, i.e. the comb starts while the antenna is still
  connected, not at a switch edge.

So for transition 3 the confound is now **broken**, not merely
testable: the episode both starts and ends inside antenna-connected
data. It remains total for transitions 1 and 2.

It is in any case a property of this small selection and **not** of the
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
2. **The boundary list needs re-deriving per-integration, and this is
   now the biggest open item.** Transition 3 should read
   **16:26:43 → 16:29:56** across three files, not a single-file event
   at 16:29:16. Two distinct failure modes are demonstrated above: the
   file-level `lock8` median **misses** episodes with less than roughly
   half-file duty cycle (`…163125Z` publishes 0.0 while carrying the
   comb for 72 integrations), and where the antenna is disconnected it
   dates the *antenna returning* rather than the comb starting. Both
   bias the era's episode count and durations. I have not edited the
   boundary list. Re-deriving it per-integration on **box-gnd** — the
   input that sees through a cal cycle — is the fix, and it is a new
   milestone.
3. If the rfswitch confound is the dominant concern for transitions 1
   and 2, the natural next step is to re-pick boundary files **matched
   on cal state** (both sides pure-`RFANT`) and repeat this figure.
   Era-wide there are 49 pure-`RFANT` comb-off files to draw from. That
   is a new selection and a new milestone, so I have not done it here.
4. `lock8 >= 30` is the operating point used to call these transitions;
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
