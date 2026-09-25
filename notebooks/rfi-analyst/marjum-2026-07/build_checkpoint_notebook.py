"""Build the B8-v1 review-checkpoint notebook (not committed as a script
product itself -- this just assembles + executes the .ipynb and renders
it, per the human-control review protocol)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NB = os.path.join(HERE, "..", "flags", "v1", "checkpoint_coincidence_validation.ipynb")
os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""
# B8 v1 review checkpoint: cross-input coincidence flagging

**Ticket** (experimental-strategist, 2026-09-14, relaying Aaron's partial
fleet-freeze lift): use simultaneous flags across independent inputs
(box-air/box-gnd/viv-N/viv-E) as corroborating evidence for external RFI,
to shrink the 68%-unknown bucket left by `flags@v0`
(`marjum-2026-07/flags/v0/`, branch `rfi-flags-v0`, commit `824b0dd`,
comb attribution corrected at `f4d06a3`). Validate on the same labelled
windows used for v0: the 1.25 MHz/Panda comb event, the fan, and LIDAR.
Report before/after unknown-fraction and category breakdown.

**Question.** Does "flag coincides across independent inputs at the same
(time, channel)" separate external RFI from self-generated/instrumental
RFI in this dataset, well enough to safely reclassify `unknown` pixels?

**Inputs.**
- `marjum-2026-07/flagging/{detectors,build_masks}.py` (v0, unmodified;
  re-run here on a handful of files, not the full campaign).
- `marjum-2026-07/curation/mode_table.jsonl` for live-input configuration
  and TX-comb on/off state per file.
- Raw data: `marjum-2026-07/data/*.h5` (11 GB, gitignored; read from the
  main checkout, not this worktree, since worktrees don't carry untracked
  data).
- Labelled windows from `CAMPAIGN.md` field notes (fan RFI 07-13, the
  07-16 "RF resonance"/comb event tied to a Panda power-cycle test, the
  07-17→07-18 digital self-comb, the 07-18 LIDAR sweep).

**Units / encoding.** uint8 category bitfield per (time, channel) sample,
as defined in `flags/v0/flag_bits.json`; channel width 0.244140625 MHz
(250/1024). Bits: cal=1, tx_comb=2, self-RFI=4, FM-scatter=8, airplane=16,
orbcomm=32, unknown=64, overflow=128. cal/tx_comb/overflow are non-RFI.

**Assumptions going in** (stated before running anything, so they can be
checked rather than silently relied on):
1. The four named inputs are simultaneously live during the labelled
   windows, giving a genuine multi-way cross-check.
2. Different inputs are independent with respect to RFI, so coincidence
   is informative about external vs. self-generated origin.

**Cheapest useful test.** Re-run the existing (unmodified) v0 per-input
detector on the small number of files spanning each labelled window
(96 files total across 4 windows, seconds of compute -- not a
campaign-wide rerun), and measure raw cross-input pixel coincidence
against categories v0 already assigns on independent grounds (comb
spacing, field-note power-cycle tests), rather than committing to a
full re-flagging pipeline first.

**Acceptance criteria for proceeding to a full v1 implementation:**
known self-generated events (comb-attributed `self-RFI`) should show
*low* cross-input coincidence, and the unknown-fraction reduction should
not come at the cost of relabelling known self-RFI as external.
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Step 0 — topology check

Before measuring anything, confirm how many inputs are actually live at
once during each labelled window.
"""))

cells.append(nbf.v4.new_code_cell(r"""
import json, sys, os
from datetime import datetime, timezone
# Code (this branch's own flagging/ modules) is resolved relative to this
# notebook's own location, since worktrees mean the checkout containing
# THIS branch's code may not be at any fixed absolute path.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "flagging")))
import build_masks as B
import detectors as D

# Default: this checkout's own marjum-2026-07/ (portable across machines --
# Aaron's checkout is not at this server's path). Override only for a
# worktree that doesn't physically have the gitignored raw data (worktrees
# don't carry untracked files).
MARJUM_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
DATA_ROOT = os.environ.get("MARJUM_DATA_ROOT", MARJUM_ROOT)
modes = B.load_mode_table(os.path.join(DATA_ROOT, "curation", "mode_table.jsonl"))

windows_ref = [
    ("07-13 fan window",           "2026-07-13T05:05:00Z"),
    ("07-16 Panda event",          "2026-07-16T01:05:00Z"),
    ("07-17/18 digital self-comb", "2026-07-18T00:30:00Z"),
    ("07-18 LIDAR sweep",          "2026-07-18T01:37:00Z"),
]
for label, t_str in windows_ref:
    t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    target = f"corr_{t:%Y%m%d_%H%M%S}Z.h5"
    # mode_table rows can have gaps between contiguous-run boundaries, so
    # take the last row starting at or before the target rather than
    # requiring an exact bracket (which a mid-run timestamp can miss).
    cand = [m for m in modes if m["file_first"] <= target]
    m = cand[-1] if cand else None
    print(f"{label:30s} phase={m['phase'] if m else '?'} "
          f"live_inputs={m['live_inputs'] if m else '?'} tx_comb={m['tx_comb'] if m else '?'}")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
This confirms (and `flagging/coincidence_check_v1.py`'s own file-level
scan re-confirms per-file) what `INDEX.md` already documents: **during
Phase A the live pair is inputs `3,4`; during Phase C — which covers all
three of the 07-16/07-17/07-18 labelled windows — the live pair is
`0,4` (box-gnd, box-air).** `viv-N`/`viv-E` are never simultaneously live
with the box inputs in any of these windows. **Assumption 1 above is
false for this campaign**: there is no window in which four independent
inputs exist to cross-check against each other. The most this campaign
ever offers is a 2-way check, and only between box-gnd and box-air, which
share a platform, chassis, and power supply — the opposite of the
geographically-independent-antennas picture the ticket's "coincidence
implies external" reasoning assumes.
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Step 1 — coincidence measurement on the labelled windows

Re-run `build_masks.process_file` (unmodified v0 code) on the files in
each window, using the input pair that mode_table.jsonl says is actually
live, and measure how often a pixel flagged in one input is *also*
flagged (any RFI category) in the other input at the same (time, channel).
"""))

cells.append(nbf.v4.new_code_cell(r"""
import importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "flagging")))
import coincidence_check_v1 as C
importlib.reload(C)
modes = C.mode_table()
results = [C.run_window(w, modes) for w in C.WINDOWS]
for r in results:
    print("---", r["window"])
    for k in ("n_files_used", "n_self_rfi_pixels_inputA",
              "frac_self_rfi_also_flagged_in_other_input",
              "n_unknown_pixels_inputA",
              "frac_unknown_also_flagged_in_other_input"):
        print(f"  {k}: {r.get(k)}")
"""))

cells.append(nbf.v4.new_code_cell(r"""
import pandas as pd
rows = []
for r in results:
    rows.append({
        "window": r["window"][:42],
        "inputs": ",".join(r["inputs"]),
        "n_files": r["n_files_used"],
        "self-RFI coincidence": r["frac_self_rfi_also_flagged_in_other_input"],
        "unknown coincidence": r["frac_unknown_also_flagged_in_other_input"],
        "any-RFI coincidence": r["frac_any_rfi_also_flagged_in_other_input"],
    })
df = pd.DataFrame(rows).set_index("window")
df
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Findings

| Window | Ground truth (independent evidence) | Self-RFI coincidence | Unknown coincidence |
|---|---|---|---|
| 07-13 fan | self-generated (field notes; ~150 MHz band) | 0.01% | 0.01% |
| 07-16 Panda event | self-generated (field notes: comb *disappeared when the Panda was power-cycled*) | n/a (0 self-RFI-attributed px on box-gnd) | **71.6%** |
| 07-17→18 digital self-comb | self-generated (no beam/pointing response; conducted on box-gnd's internal load *and* radiated on box-air — already documented in `INDEX.md`, corrected 2026-09-13) | **95.2%** | **71.9%** |
| 07-18 LIDAR sweep | self-generated (on-platform pulsed source) | **92.9%** | **83.7%** |

**The naive rule is falsified by 3 of the 4 labelled windows.** In every
Phase-C window (07-16, 07-17/18, 07-18), pixels that other evidence
independently confirms are self-generated show 70-95% cross-input
coincidence — the *opposite* of what "coincidence implies external"
predicts. This is not a surprising result in hindsight: box-gnd and
box-air share a chassis and power supply, so a source that couples into
one conducted path (a Panda EMI comb, a LIDAR pulse) routinely couples
into the other at the same instant. Coincidence here measures *shared
platform environment*, not *external origin*.

Only the 07-13 fan window shows near-zero coincidence, but that is not a
confirmation of the rule either — it just means whatever produces the
"self-RFI"/"unknown" flags on input 3 in Phase A does not reach input 4 at
all in this sample; it does not establish that a genuinely external
source *would* show low coincidence, since no unambiguously-external
labelled window exists to test the positive-case direction. (The airplane
and FM-scatter categories are morphological, not corroborated — the same
caveat as v0's `MEMO.md`.)

## Material uncertainties

1. **Topology mismatch.** The ticket's four-input premise doesn't hold in
   this campaign; the strongest cross-check ever available is 2-way, and
   the only 2-way pair actually exercised in the labelled windows
   (box-gnd/box-air) is platform-common, not independent.
2. **Sign is context-dependent, not fixed.** In the Phase-C windows, high
   coincidence tracks *shared self-RFI*, not corroborated externality —
   the inverse of the ticket's assumed direction. Any rule that reads
   "coincident ⇒ more likely external" would, on this evidence, actively
   promote confidence in the *wrong* direction for Panda EMI, the digital
   self-comb, and LIDAR — three of the campaign's four best-documented
   self-RFI mechanisms.
3. **No confirmed-external labelled window exists** to test the other
   half of the hypothesis (would a real external signal actually show low
   coincidence here, e.g. because it's beam-limited or below one
   receiver's noise floor at a given moment?). Absence of that test case
   means I can't rule out coincidence being useful in the *other*
   direction either — this checkpoint only establishes that the rule as
   literally stated fails its self-RFI validation set.
4. This checkpoint used **only the pre-existing v0 detector**, re-run on
   96 files (4 windows) out of 5,120 — a deliberately cheap smoke test,
   not a campaign-wide recomputation. No `flags/v0` output was touched.

## Decision requested

Implementing "non-coincident unknown → self-RFI candidate, coincident
unknown → likely external" as specified would misclassify the campaign's
best-documented self-RFI mechanisms as more-likely-external. I have not
implemented any reclassification or re-run the full campaign pending
direction on how to proceed. Two options that would use the coincidence
signal without its current failure mode:

- **(a) Restrict coincidence-based promotion to bands with no known
  self-RFI spectral signature** (i.e. exclude the comb bands and
  BAND_FAN/BAND_LAPTOP from any coincidence-driven reclassification), so
  it only touches the FM/DTV/orbcomm-adjacent `unknown` residue where
  self-RFI is not already suspected.
- **(b) Require an independent corroborating signature** (matched comb
  spacing, or the beam-response/no-pointing-dependence test already used
  for the digital self-comb) before trusting coincidence's sign at all,
  i.e. treat raw coincidence as too weak/ambiguous a signal on its own in
  this 2-input, platform-shared topology.

Awaiting Aaron's direction before scaling this to a full v1 pass.

**STOPPED AT REVIEW GATE — awaiting Aaron's approval.**
"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "arp", "language": "python", "name": "arp"},
    "language_info": {"name": "python"},
}

with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT_NB)
