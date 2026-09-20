#!/usr/bin/env python3
"""Emit `comb_transition_waterfalls.ipynb`.

Frozen builder, kept beside the notebook so it can be regenerated rather
than hand-edited (the repo's nbstripout clean filter strips outputs, so
the committed .ipynb is unexecuted and its rendered .html/.pdf are the
evidence).

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

**Author:** rfi-analyst · **Campaign:** marjum-2026-07 · **Rev 2**

## What this is

Time-difference waterfalls across eight boundaries of the **digital
self-comb** (1.953125 MHz = exactly 8.000 channels, channel-locked,
residue 0 mod 8) during its 07-17/18 era.

**Rev 2 rebuilds the file selection.** Rev 1 chose its files from
box-air file-level `lock8`, and that turned out to be the wrong ground
truth in two demonstrable ways (below). State and timing now come
exclusively from **box-gnd**, per-integration — Aaron's 2026-09-20
methodology decision. Box-gnd is conducted rather than antenna-switched,
so it sees through the calibration and VNA cadence.

**Box-gnd決定 determines *when*; it does not narrow what is plotted.**
Every figure still shows both autos and the cross, as before.

## Why the file-level list was replaced

`flags/diag/comb_inventory.jsonl` computes one `lock8` per (file, input)
from a median over all 240 integrations. Good for characterising a comb,
wrong for dating transitions:

* **It misses short episodes.** `corr_20260717_163125Z.h5` publishes
  `lock8 = 0.0` on both inputs while carrying the comb for its first 72
  integrations. A feature with under roughly half-file duty cycle washes
  out of a 240-sample median and the file reads clean.
* **Where the antenna is disconnected it dates the antenna returning,
  not the comb starting.** `corr_20260717_162916Z.h5`'s published
  detection rests entirely on its final 29 `RFANT` integrations.

Re-derived per-integration on box-gnd, the era contains **22 episodes /
43 transitions**, of which **11 episodes are shorter than one file** and
so were structurally invisible before. The aggregate duty cycle is
unchanged (87.1% vs the file-level 86%), which is the reassurance that
this resolves structure rather than inventing it.

## Provenance

* State and timing: `curation/self_comb_transitions_boxgnd.jsonl`, from
  `flags/diag/self_comb_per_integration_boxgnd.jsonl`
  (`flagging/scan_self_comb_per_integration.py` +
  `derive_self_comb_transitions.py`), box-gnd, 8-integration blocks
  (~4.3 s).
* The statistic is `comb_inventory.tooth_contrast_locked` at spacing 8,
  **imported from that module, not reimplemented**, so it cannot drift
  from the detector behind the published inventory.
* Data access: `eigsep_data.index.MetadataIndex` +
  `eigsep_data.bundle.load_bundle`. `antenna="cross"` resolves the `04`
  key per file and fixes conjugation, so there is no reason to drop to
  raw `h5py`.
* Units: raw correlator counts. Frequencies MHz, times UTC.
""")

code(r"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timezone
from pathlib import Path

from eigsep_data.index import MetadataIndex
from eigsep_data.bundle import load_bundle
from eigsep_data.flagging.detectors import overflow_mask

CAMPAIGN = Path("/mnt/data02/eigsep/marjum-2026-07")
DATA_DIR = CAMPAIGN / "data"
TRANSITIONS = CAMPAIGN / "curation/self_comb_transitions_boxgnd.jsonl"
TRACE = CAMPAIGN / "flags/diag/self_comb_per_integration_boxgnd.jsonl"

# Eight boundary files, chosen from the corrected box-gnd list. Every
# one is 100% RFANT -- see "The confound, and why it is now gone".
FILES = [
    "corr_20260717_153953Z.h5",
    "corr_20260717_154202Z.h5",
    "corr_20260717_154410Z.h5",
    "corr_20260717_155037Z.h5",
    "corr_20260717_161623Z.h5",
    "corr_20260717_163125Z.h5",
    "corr_20260717_164001Z.h5",
    "corr_20260717_170338Z.h5",
]

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
print("numpy", np.__version__)
""")

md(r"""
## The corrected transition list

Read from the product rather than restated here, so the notebook cannot
drift from it.
""")

code(r"""
recs = [json.loads(l) for l in TRANSITIONS.open()]
PROV = recs[0]["provenance"]
EPISODES = pd.DataFrame(recs[1:])

for k in ("input", "block_seconds", "state_rule", "n_episodes",
          "duty_cycle", "caveat"):
    print(f"{k:18s} {PROV[k]}")

show = EPISODES[["t_start_utc", "t_end_utc", "duration_s",
                 "shorter_than_file"]].copy()
show["t_start_utc"] = [t[11:19] for t in show.t_start_utc]
show["t_end_utc"] = [t[11:19] for t in show.t_end_utc]
display(show)
print(f"\n{int(EPISODES.shorter_than_file.sum())} of {len(EPISODES)} episodes "
      f"are shorter than one file, i.e. invisible to a file-level median.")
""")

md(r"""
### Which transitions land in the selected files

Each selected file is listed with the box-gnd transitions that fall
inside it. Note that almost every transition is *interior* to a file —
at 4.3 s resolution there is no reason for a physical switch to align
with a 129 s file boundary, and the fact that the old list appeared to
put them on file boundaries was an artefact of the file-level median.
""")

code(r"""
edges = []
for _, e in EPISODES.iterrows():
    edges.append((e.t_start_utc, "off->ON", e.file_start, e.block_start))
    edges.append((e.t_end_utc, "ON->off", e.file_end, e.block_end))
EDGES = pd.DataFrame(edges, columns=["t_utc", "kind", "file", "block"])

in_sel = EDGES[EDGES.file.isin(FILES)].sort_values("t_utc")
display(in_sel.assign(t_utc=[t[11:19] for t in in_sel.t_utc]))

LABEL = {}
for f in FILES:
    sub = in_sel[in_sel.file == f]
    LABEL[f] = " / ".join(f"{r.kind} {r.t_utc[11:19]}" for _, r in sub.iterrows()) \
        or "no interior transition"
    print(f"{f[5:-3]}  {LABEL[f]}")
""")

md(r"""
## Load

`patterns=` restricts the index to the selected files and `cache=False`
keeps the notebook from writing an `.eigsep_index.h5` sidecar into the
shared campaign data directory.
""")

code(r"""
index = MetadataIndex(str(DATA_DIR), patterns=tuple(FILES), cache=False,
                      streams="all")
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
""")

md(r"""
## Timebase

All selected files have `sync_consistent = True`, so the index's
`time_best` is trustworthy and is used directly. That is not generally
true of this campaign — `header/times` is corrupt in 642 of 5,120 files
and the standing fallback is the filename close time — so it is checked
here rather than assumed.
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
tsummary["transitions"] = [LABEL[f] for f in tsummary.index]
display(tsummary[["n_int", "sync_ok", "utc_start", "transitions"]])

assert bool(tsummary.sync_ok.all()), "inconsistent sync; do not trust time_best"
print("all selected files: sync_consistent = True")
""")

md(r"""
## int32 accumulator wrap — repaired on the autos only

Auto accumulators are int32 and wrap. Autos are non-negative by
construction, so `raw < 0` detects a wrap with no threshold
(`detectors.overflow_mask`).

This matters more here than almost anywhere else, because **this
notebook differences along time**: a single unrepaired wrap is a
$\pm 2^{32}$ step and would set the colour scale on its own.

**The cross is deliberately left alone.** A cross visibility is
genuinely signed in both real and imaginary parts, so `raw < 0` carries
no information about wrapping there, and applying the auto repair would
corrupt real data.
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
print(f"{freqs.size} channels, {freqs[0]:.3f}-{freqs[-1]:.3f} MHz")
""")

md(r"""
## Differencing

`np.diff` is taken **within each file**, never across a file join. The
selected files are not contiguous, so differencing a naively
concatenated stack would manufacture a bright artificial row at every
boundary. Each file's 240 integrations therefore give 239 difference
rows, with a `NaN` row inserted between files as a separator.

For the cross the displayed quantity is $\Delta|V|$ — the signed change
in visibility amplitude — so all three panels show the same kind of
quantity. ($|\Delta V|$, which also responds to pure phase rotation, is
a different question and is not plotted here.)
""")

code(r"""
row_of = {f: np.flatnonzero((meta.file == f).values) for f in FILES}


def stacked_diff(arr):
    # Per-file np.diff along time, stacked with NaN separators. Also
    # returns each file block's y-centre and y-extent, for labelling.
    blocks, centres, edges_ = [], [], []
    y = 0
    for i, f in enumerate(FILES):
        d = np.diff(arr[row_of[f]], axis=0)
        blocks.append(d)
        centres.append(y + d.shape[0] / 2.0)
        edges_.append((y, y + d.shape[0]))
        y += d.shape[0]
        if i != len(FILES) - 1:
            blocks.append(np.full((1, arr.shape[1]), np.nan))
            y += 1
    return np.concatenate(blocks, axis=0), centres, edges_


D_gnd, centres, edges_y = stacked_diff(auto_gnd)
D_air, _, _ = stacked_diff(auto_air)
D_crs, _, _ = stacked_diff(np.abs(cross))

# Antenna-to-antenna difference rows: both endpoints on the antenna.
state = meta["rfswitch"].astype(str).values
ant_rows = []
for i, f in enumerate(FILES):
    s = state[row_of[f]]
    ant_rows.append((s[:-1] == "RFANT") & (s[1:] == "RFANT"))
    if i != len(FILES) - 1:
        ant_rows.append(np.zeros(1, dtype=bool))
ANT = np.concatenate(ant_rows)

print(f"stacked diff shape: {D_gnd.shape}")
print(f"antenna-to-antenna difference rows: {ANT.sum()} of {ANT.size} "
      f"({100 * ANT.mean():.1f}%)")

# y position of each box-gnd transition, for marking on the waterfalls
TRANS_Y = []
for _, r in in_sel.iterrows():
    i = FILES.index(r.file)
    y0 = edges_y[i][0]
    TRANS_Y.append((y0 + r.block * 8, r.kind, r.t_utc[11:19]))
print(f"{len(TRANS_Y)} box-gnd transitions fall inside the selected files")
""")

md(r"""
## The figure

Three panels, `sharex=True, sharey=True`: box-gnd auto, box-air auto,
cross amplitude. Frequency on x; concatenated per-file difference rows
down y, each block labelled with its file and the box-gnd transitions
inside it. **Black arrows on the left mark where box-gnd says each
transition happens** — that is the ground truth this rev is built on.

The colour scale is symmetric about zero, per panel (the panels differ
in absolute scale by orders of magnitude), and set from a robust
percentile of antenna-to-antenna rows.
""")

code(r"""
def robust_lim(D, pct=99.0, rows=None):
    sub = D if rows is None else D[rows]
    v = sub[np.isfinite(sub)]
    if v.size == 0:
        v = D[np.isfinite(D)]
    return float(np.percentile(np.abs(v), pct)) or 1.0


ylabels = [f"{f[5:-3]}\n{LABEL[f]}" for f in FILES]


def waterfall(panels, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    for ax, (label, D) in zip(axes, panels):
        lim = robust_lim(D, rows=ANT)
        im = ax.imshow(D, aspect="auto", origin="upper", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                       extent=[freqs[0], freqs[-1], D.shape[0], 0],
                       interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("frequency [MHz]")
        for (y0, _) in edges_y[1:]:
            ax.axhline(y0 - 0.5, color="k", lw=0.8, alpha=0.6)
        for y, kind, _t in TRANS_Y:
            ax.axhline(y, color="k", lw=0.9, ls="--", alpha=0.75)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_yticks(centres)
    axes[0].set_yticklabels(ylabels, fontsize=6.5)
    axes[0].set_ylabel("file block  (per-file np.diff along time, UTC)")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


waterfall([("box-gnd auto  $\\Delta$counts", D_gnd),
           ("box-air auto  $\\Delta$counts", D_air),
           ("cross  $\\Delta|V|$", D_crs)],
          "Digital self-comb boundaries — time-difference waterfalls (raw counts)\n"
          "dashed lines = box-gnd transitions, per-integration ground truth")
""")

md(r"""
### Same difference, log space

The linear panels are dominated by the brightest channels, which is what
a difference of raw counts does. The comb is narrowband and
low-amplitude, so the same difference in $\log_{10}$ space — a
fractional change per integration — is what makes it legible. Same
operation, same data.
""")

code(r"""
L_gnd, _, _ = stacked_diff(np.log10(np.maximum(auto_gnd, 1.0)))
L_air, _, _ = stacked_diff(np.log10(np.maximum(auto_air, 1.0)))
L_crs, _, _ = stacked_diff(np.log10(np.maximum(np.abs(cross), 1.0)))

waterfall([("box-gnd auto  $\\Delta\\log_{10}$", L_gnd),
           ("box-air auto  $\\Delta\\log_{10}$", L_air),
           ("cross  $\\Delta\\log_{10}|V|$", L_crs)],
          "Same boundaries, difference taken in log space\n"
          "(fractional change per integration — narrowband features legible)")
""")

md(r"""
## The per-integration state trace

The box-gnd `lock8` trace behind every transition above, for each
selected file. This is the evidence for the boundary list, not an
illustration of it.

The state rule is hysteretic, and deliberately so. A plain threshold
chatters: the transition count runs 73 → 197 as the threshold moves
5 → 80, and 37 of 63 apparent off-runs are under 13 s — the statistic
wobbling across a threshold inside a continuously-on stretch. What makes
a clean rule possible is the statistic's floor: on comb-free data the
log-residual is exactly zero on more than half the in-band channels, the
MAD scale collapses, and `tooth_contrast_locked` returns exactly `0.0`.
Of 1,380 sub-threshold blocks in the era, **1,176 are exactly 0.0** and
only 204 lie in (0, 30). So "off" is a hard value, not a small one, and
the rule is: ON at `lock8 >= 30`, OFF at `lock8 == 0.0`, hold in
between. That gives 22 episodes, stable at 22–23 for any ON threshold
in 20–40.
""")

code(r"""
trace = pd.DataFrame(json.loads(l) for l in TRACE.open())
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
for ax, f in zip(axes.ravel(), FILES):
    t = trace[trace.file == f].sort_values("block")
    ax.plot(t.int0, t.lock8, color="tab:red", lw=1.3)
    ax.axhline(30, color="k", ls=":", lw=1)
    for _, r in in_sel[in_sel.file == f].iterrows():
        ax.axvline(r.block * 8, color="k", ls="--", lw=1,
                   alpha=0.8)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"{f[5:-3]}\n{LABEL[f]}", fontsize=7.5)
    ax.set_xlabel("integration")
axes[0, 0].set_ylabel("box-gnd lock8")
axes[1, 0].set_ylabel("box-gnd lock8")
fig.suptitle("box-gnd per-integration comb state (8-integration blocks, ~4.3 s)\n"
             "dotted = ON threshold 30; dashed = derived transition", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
""")

md(r"""
**Six of the eight are unambiguous** — `153953Z`, `154202Z`, `154410Z`,
`155037Z`, `161623Z` and `163125Z` switch between ~0 and 10²–10³ in a
single block, square-wave, nothing near the threshold.

**`164001Z` and `170338Z` are the marginal cases and are kept
deliberately.** Both spend time in the 20–60 range where the hysteresis
is doing real work rather than merely formalising an obvious call:
`164001Z` toggles four times in 129 s, and `170338Z`'s last third wanders
either side of the threshold. They are the honest worst case for this
detector, and a selection containing only the clean six would
misrepresent how well it does. If the derived transitions in those two
look wrong to you, that is the thing to say — it bears directly on
decision 3 below.
""")

md(r"""
## The confound, and why it is now gone

Rev 1 had a serious problem. Its six files were picked from box-air
file-level `lock8`, and in that selection all three `ON -> off` files
were 100% `RFANT` while all three `off -> ON` files carried
calibration/VNA states — a perfect 3/3 split. "Comb detected" and
"calibration cadence running" were not separable, so nothing in the
waterfalls could distinguish the comb switching on from the cal
hardware switching on.

That confound was **an artefact of the ground truth, not of the comb.**
A box-air file-level statistic can only register a change when the
antenna happens to be connected, so it preferentially dates transitions
to cal-cadence edges — it manufactures the correlation it then appears
to show.

With box-gnd per-integration as the state source it disappears:
""")

code(r"""
pure = {}
for f in meta.file.unique():
    s = meta.loc[meta.file == f, "rfswitch"].astype(str)
    pure[f] = float((s == "RFANT").mean())

sw = (meta.assign(state=meta["rfswitch"].astype(str))
          .pivot_table(index="file", columns="state", values="row",
                       aggfunc="size", fill_value=0))
sw.insert(0, "frac_RFANT", [pure[f] for f in sw.index])
display(sw)

print("on-sky fraction by selected file:")
for f in FILES:
    print(f"  {f[5:-3]}  {100 * pure[f]:5.1f}%")
print(f"\nall eight files pure RFANT: {all(pure[f] == 1.0 for f in FILES)}")
""")

md(r"""
**All eight selected files are 100% `RFANT`**, and across the whole era
**39 of the 43 box-gnd transitions have both sides in pure-`RFANT`
files**. There is no cal/VNA switching anywhere in this selection, so
the panels above cannot be showing calibration steps — every feature in
them is an antenna-connected, on-sky difference.

This is the substantive payoff of the methodology change, and it also
retires rev 1's open question. Rev 1 asked whether its panels showed the
comb or the cal cadence, and could not answer from its own data. Under
box-gnd ground truth the question does not arise: the transitions are in
clean antenna data to begin with.

The era-wide numbers quoted in rev 1 — comb ON 176 pure-`RFANT` vs 29
cal-active, comb off 49 vs 56 — were computed against the file-level
box-air state and should not be carried forward. They describe the old
ground truth's behaviour, not the comb's.
""")

md(r"""
## Caveats

* **Time resolution is one block, ~4.3 s.** Episodes shorter than about
  two integrations can still be missed. This is 30× finer than the
  file-level median it replaces, not a guarantee of completeness.
* **A returned `lock8 = 0.0` is ambiguous in principle** — it means "no
  measurable tooth structure *or* a degenerate MAD scale". Checked on
  750 sampled era blocks: 4.0% had a degenerate scale and every one had
  an unnormalised best-phase tooth excess of exactly 0.00000, so the
  degeneracy never hid a comb in this dataset. `tooth_excess_log10` is
  carried in the trace so this stays auditable.
* **Scope is the documented era**, 07-17 15:37:44 → 07-18 03:00:13.
  Because the file-level method under-counts short episodes, short
  episodes *outside* the era may also have been missed; the era bounds
  themselves inherit the old method. Not yet checked.
* **Nothing here is an attribution.** No pointing test, no null control,
  no far-field discriminator. This is morphology and timing only.
""")

md(r"""
## Decision requested

1. **Adopt `curation/self_comb_transitions_boxgnd.jsonl` as the
   authoritative self-comb transition list?** It supersedes file-level
   `lock8` *for timing only*; `comb_inventory.jsonl` remains correct for
   characterisation and I have not modified it.
2. **Extend the per-integration scan beyond the era?** Measured cost is
   ~29 s for 313 era files, so the full campaign is roughly 8 minutes.
   Given that the file-level method demonstrably misses sub-file
   episodes, this is the only way to know whether the era bounds are
   themselves right — and there are already two unexplained box-gnd
   detections on 07-12 in the file-level inventory.
3. **Do the 22 episodes need a minimum-duration filter for downstream
   use?** Eleven are shorter than one file and two are a single 4.3 s
   block. They are real by this detector, but a flag-mask consumer may
   want them merged or dropped rather than toggling state twice inside
   one integration window.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
with open(NB, "w") as fh:
    nbf.write(nb, fh)
print(f"wrote {NB} ({len(cells)} cells)")
