#!/usr/bin/env python3
"""Extract RF-calibration windows from the campaign scan.

The filtered correlator set retains the full `metadata/` group, including
`metadata/rfswitch` -- a per-spectrum record of which source the front-end
switch was looking at. The filter only dropped correlator *data* keys, so the
calibration cadence Christian ran is present in the filtered data and needs no
recourse to the raw T7 archive.

Switch states observed in this campaign
---------------------------------------
    RFANT     antenna (science data)
    RFAMB     ambient load            \\  noise-wave / Y-factor
    RFNON     noise source ON          |  calibration
    RFNOFF    noise source OFF        /
    RFSP1     spare / aux port
    VNAO      VNA -> OPEN standard    \\
    VNAS      VNA -> SHORT standard    |  1-port SOL calibration
    VNAL      VNA -> LOAD standard    /   + antenna reflection
    VNAANT    VNA -> antenna         <-   the S11 measurement
    VNAAMB / VNANON / VNANOFF / VNARF / VNASP1   VNA through other ports
    UNKNOWN   switch in transit between states

Output
------
curation/cal_windows.jsonl -- one row per contiguous run of files containing
any non-RFANT state, with a state histogram and a coarse cal-type label.

Usage
-----
    python curation/extract_cal_windows.py
    python curation/extract_cal_windows.py --summary
"""

from __future__ import annotations

import argparse
import os
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eigsep_data.paths import get_campaign_root


def _campaign_root():
    """Campaign root; ``MARJUM_DATA_ROOT`` wins, else the package setting.

    This script anchored on its own ``__file__`` until it moved out of
    the campaign tree on 2026-09-19.
    """
    env = os.environ.get("MARJUM_DATA_ROOT")
    if env:
        return Path(env)
    return get_campaign_root(required=True)


CAMPAIGN_ROOT = _campaign_root()
CSV_IN = CAMPAIGN_ROOT / "curation" / "file_state.csv"
JSONL_OUT = CAMPAIGN_ROOT / "curation" / "cal_windows.jsonl"

FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")

SOL = {"VNAO", "VNAS", "VNAL"}
NOISE = {"RFNON", "RFNOFF", "RFAMB"}
VNA_ANY = {"VNAO", "VNAS", "VNAL", "VNAANT", "VNAAMB", "VNANON", "VNANOFF", "VNARF", "VNASP1"}
NON_ANT = NOISE | VNA_ANY | {"RFSP1"}


def filename_time(fn: str) -> float:
    m = FNAME_RE.match(fn)
    if not m:
        raise ValueError(fn)
    return (
        datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Receiver regimes.
#
# rf-calibrator (B7) found a discrete receiver state change bracketing the
# logged 7U battery failure (~22:00 UTC 07-17) and box-air outage (~23:35):
# T_rx 212 -> 539 K, g_rx 1150 -> 672. The exact boundary is unresolved inside
# a 5.7 h observing gap. Calibration solutions MUST NOT be interpolated across
# it.
# --------------------------------------------------------------------------
RECEIVER_REGIMES = [
    ("2026-07-12T00:00:00Z", "2026-07-17T19:42:00Z", "rx-A"),
    # Boundary unresolved within this gap; do not interpolate across it.
    ("2026-07-17T19:42:00Z", "2026-07-18T01:24:00Z", "rx-transition"),
    ("2026-07-18T01:24:00Z", "2026-07-18T23:59:59Z", "rx-B"),
]


def receiver_regime(ts: float) -> str:
    for start, end, label in RECEIVER_REGIMES:
        if (datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
                <= ts <
                datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp()):
            return label
    return "unknown"


def cal_type(states: set[str]) -> str:
    """Coarse label for what this window is usable for.

    IMPORTANT (rf-calibrator, verified 2026-09-12): the VNA_* switch states are
    *timing markers only*. The actual S11 sweeps were written by vna_writer to
    separate ants11_*/recs11_*.h5 files that are NOT in this repo (only 2025
    S11 exists locally, under terrain/s11_data/). Cross-product coherence
    during VNA states is 0.001 -- there is no usable reflection measurement in
    the correlator stream. Gamma_ant is absent from local disk; the 2026 S11
    files are the top target if the T7 archive is ever mounted.

    So a window containing SOL states tells you WHEN a VNA sweep happened, not
    what it measured. Only the RFNON/RFAMB noise states carry calibratable
    in-band data.
    """
    has_sol = SOL <= states
    has_noise = {"RFNON", "RFAMB"} <= states
    if has_sol and has_noise:
        return "noise+vna_timing"
    if has_sol:
        return "vna_timing"
    if has_noise:
        return "noise"
    if states & VNA_ANY:
        return "vna_timing-partial"
    if states & NOISE:
        return "noise-partial"
    return "aux"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--csv-in", type=Path, default=CSV_IN)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    df["t"] = df["file"].map(filename_time)
    df = df.sort_values("t").reset_index(drop=True)

    def states_of(row) -> Counter:
        raw = row["rfswitch_states_json"]
        if not isinstance(raw, str) or not raw.strip():
            return Counter()
        try:
            return Counter(json.loads(raw))
        except json.JSONDecodeError:
            return Counter()

    df["_states"] = df.apply(states_of, axis=1)
    df["_has_cal"] = df["_states"].map(lambda c: bool(set(c) & NON_ANT))

    grp = (df["_has_cal"] != df["_has_cal"].shift()).cumsum()
    rows = []
    for _, g in df[df["_has_cal"]].groupby(grp):
        agg = Counter()
        for c in g["_states"]:
            agg.update(c)
        states = {k: int(v) for k, v in agg.items() if k in NON_ANT}
        rows.append({
            "t_start_utc": iso(g.iloc[0]["t"]),
            "t_end_utc": iso(g.iloc[-1]["t"]),
            "file_first": g.iloc[0]["file"],
            "file_last": g.iloc[-1]["file"],
            "n_files": int(len(g)),
            "phase": str(g.iloc[0]["filter_phase"]),
            "cal_type": cal_type(set(states)),
            "receiver_regime": receiver_regime(g.iloc[0]["t"]),
            "states": dict(sorted(states.items(), key=lambda kv: -kv[1])),
            "n_unknown": int(agg.get("UNKNOWN", 0)),
        })

    with JSONL_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_missing = int((~df["has_meta_rfswitch"].astype(bool)).sum())
    print(f"{len(rows)} calibration windows -> {JSONL_OUT.relative_to(CAMPAIGN_ROOT)}")
    print(f"  files lacking metadata/rfswitch entirely: {n_missing} "
          f"(all of phase A and early B; no cal possible there)")

    if args.summary:
        by = Counter(r["cal_type"] for r in rows)
        print("\n  windows by type: " + ", ".join(f"{k}={v}" for k, v in by.most_common()))
        tot = Counter()
        for r in rows:
            tot.update(r["states"])
        print("  total samples by state:")
        for k, v in tot.most_common():
            print(f"    {k:10s} {v:7d}")
        print("\n  first window per type:")
        seen = set()
        for r in rows:
            if r["cal_type"] not in seen:
                seen.add(r["cal_type"])
                print(f"    {r['cal_type']:14s} {r['t_start_utc']}  {r['file_first']}")


if __name__ == "__main__":
    main()
