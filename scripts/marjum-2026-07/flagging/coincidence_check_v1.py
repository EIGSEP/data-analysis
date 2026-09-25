"""B8 v1 cheapest-useful-test: does cross-input coincidence separate
self-RFI from external RFI, or does it get fooled by common-mode
self-RFI?

Ticket (experimental-strategist, 2026-09-14): use simultaneous flags
across independent inputs as corroborating evidence for external RFI;
non-coincident flags as candidates for the self-RFI/instrumental
bucket instead of `unknown`. Validate against the labelled windows in
CAMPAIGN.md used for v0 (1.25 MHz/Panda comb, fan, LIDAR).

**Topology check first.** The ticket's premise names four inputs
(box-air/box-gnd/viv-N/viv-E). None of the three labelled windows below
actually have four live inputs at once -- `mode_table.jsonl` shows
exactly TWO live inputs throughout: (3,4) for the 07-13 fan window,
(0,4) = (box-gnd, box-air) for 07-16 and 07-18. viv-N/viv-E are not
simultaneously live with the box inputs during any labelled window in
this campaign, so "cross-input coincidence" here means a 2-way check,
not a 4-way one, and its statistical power is correspondingly weaker
(a single corroborating witness, not three).

**Known confound, already on record (INDEX.md, corrected 2026-09-13):**
the 1.953125 MHz digital self-comb is present on BOTH live Phase-C
inputs at once -- conducted into box-gnd's internal load, radiated and
antenna-received on box-air -- and it is NOT external (no beam
response; PROGRAM.md-independent detector already sets self-RFI on
comb-spacing grounds). If raw pixel coincidence across the two live
inputs is used to promote confidence toward "external," this is the
test case that should catch it doing the wrong thing.

This script does NOT change flags/v0. It re-runs the existing v0
per-input detector (`build_masks.process_file`, unmodified) on a
handful of files inside each labelled window, then asks: of the
pixels v0 already calls `self-RFI` (comb-attributed, so we have
independent grounds to trust that label), what fraction are also
flagged (raw coincidence, not necessarily category-matched) in the
OTHER live input at the same (time, channel)? And separately, of the
pixels v0 calls `unknown`, what fraction would this rule reclassify?
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

import h5py
import numpy as np

from eigsep_data.flagging import build_masks as B
from eigsep_data.flagging import detectors as D
from eigsep_data.paths import get_campaign_root

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # sibling study modules in this directory


def _campaign_root():
    """Campaign root for this study script.

    ``MARJUM_DATA_ROOT`` still wins, for a worktree that lacks the
    gitignored raw data. Otherwise the package's own setting applies
    (``eigsep_data.set_campaign_root()`` or ``EIGSEP_CAMPAIGN_ROOT``).
    This script used to anchor on its own ``__file__``; it moved out of
    the campaign tree on 2026-09-19, so there is no such anchor.
    """
    env = os.environ.get("MARJUM_DATA_ROOT")
    if env:
        return env
    return str(get_campaign_root(required=True))


DATA_ROOT = _campaign_root()

WINDOWS = [
    {
        "name": "07-13 fan (~150 MHz, labelled self-RFI)",
        "day": "20260713",
        "t0": "2026-07-13T05:00:00Z",
        "t1": "2026-07-13T05:20:00Z",
        "inputs": ("3", "4"),
    },
    {
        "name": "07-16 Panda/1.25MHz-field-note event (labelled self-RFI, "
                "Panda power cycle)",
        "day": "20260716",
        "t0": "2026-07-16T01:00:00Z",
        "t1": "2026-07-16T01:30:00Z",
        "inputs": ("0", "4"),
    },
    {
        "name": "07-17/18 digital self-comb, 1.953125 MHz (labelled "
                "self-RFI, conducted+radiated on BOTH live inputs)",
        "day": "20260718",
        "t0": "2026-07-18T00:00:00Z",
        "t1": "2026-07-18T02:00:00Z",
        "inputs": ("0", "4"),
    },
    {
        "name": "07-18 LIDAR sweep, 15-min-interval RFI (labelled "
                "self-RFI, on-platform)",
        "day": "20260718",
        "t0": "2026-07-18T01:30:00Z",
        "t1": "2026-07-18T02:00:00Z",
        "inputs": ("0", "4"),
    },
]


def files_for_day(day):
    return sorted(glob.glob(os.path.join(DATA_ROOT, "data", f"corr_{day}_*.h5")))


def mode_table():
    return B.load_mode_table(os.path.join(DATA_ROOT, "curation", "mode_table.jsonl"))


def in_window(fname, t0, t1):
    t = D.file_close_time(fname)
    return t0 <= t <= t1


def run_window(w, modes):
    t0 = datetime.strptime(w["t0"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(w["t1"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    day_files = files_for_day(w["day"])
    files = [f for f in day_files if in_window(os.path.basename(f), t0, t1)]
    if not files:
        return {"window": w["name"], "error": "no files found in window"}

    per_input_cats = {k: [] for k in w["inputs"]}
    per_input_times = {k: [] for k in w["inputs"]}
    n_files_used = 0
    for path in files:
        fname = os.path.basename(path)
        m = B.mode_for(modes, fname)
        tx_on = bool(m and m.get("tx_comb") == "on")
        _fname, per_input, freqs, err = B.process_file((path, tx_on))
        if err:
            continue
        if not all(k in per_input for k in w["inputs"]):
            continue  # this file doesn't actually have both requested inputs live
        n_files_used += 1
        for k in w["inputs"]:
            cat = per_input[k]["cat"]
            per_input_cats[k].append(cat)

    if n_files_used == 0:
        return {"window": w["name"], "error": "no file had all requested inputs live"}

    # Concatenate along time within this window. Files are same length
    # (n_time, n_chan) per campaign convention; if not, truncate to min.
    stacked = {}
    for k in w["inputs"]:
        arrs = per_input_cats[k]
        n_min = min(a.shape[0] for a in arrs)
        stacked[k] = np.concatenate([a[:n_min] for a in arrs], axis=0)

    k0, k1 = w["inputs"]
    catA, catB = stacked[k0], stacked[k1]
    n = min(catA.shape[0], catB.shape[0])
    catA, catB = catA[:n], catB[:n]

    sky_a = (catA & (D.CAL | D.OVERFLOW)) == 0
    sky_b = (catB & (D.CAL | D.OVERFLOW)) == 0
    both_sky = sky_a & sky_b

    flagged_b_any_rfi = (catB & D.RFI_BITS) != 0

    def frac_coincident(mask_a):
        sel = mask_a & both_sky
        n_sel = int(sel.sum())
        if n_sel == 0:
            return None, 0
        coinc = int((sel & flagged_b_any_rfi).sum())
        return coinc / n_sel, n_sel

    self_a = (catA & D.SELF_RFI) != 0
    unk_a = (catA & D.UNKNOWN) != 0
    any_rfi_a = (catA & D.RFI_BITS) != 0

    f_self, n_self = frac_coincident(self_a)
    f_unk, n_unk = frac_coincident(unk_a)
    f_any, n_any = frac_coincident(any_rfi_a)

    def category_breakdown(cat, sky):
        out = {}
        for bit, name in D.CATEGORY_NAMES.items():
            out[name] = float(((cat & bit) != 0)[sky].mean()) if sky.sum() else 0.0
        return out

    return {
        "window": w["name"],
        "n_files_used": n_files_used,
        "inputs": w["inputs"],
        "n_time_samples": int(n),
        f"category_breakdown_input_{k0}": category_breakdown(catA, sky_a),
        f"category_breakdown_input_{k1}": category_breakdown(catB, sky_b),
        "n_self_rfi_pixels_inputA": n_self,
        "frac_self_rfi_also_flagged_in_other_input": f_self,
        "n_unknown_pixels_inputA": n_unk,
        "frac_unknown_also_flagged_in_other_input": f_unk,
        "n_any_rfi_pixels_inputA": n_any,
        "frac_any_rfi_also_flagged_in_other_input": f_any,
    }


def main():
    modes = mode_table()
    results = [run_window(w, modes) for w in WINDOWS]
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
