#!/usr/bin/env python3
"""Detect int32 accumulator overflow (wrap) per file, per input, per channel.

The correlator autos are `int32` and are a positive-definite quantity, so a
**negative recorded value is unambiguously an accumulator wrap** -- the power
in that (time, channel) cell is not merely noisy, it is wrong by 2**32.

This is a per-CHANNEL defect, not a per-file one. A file with a handful of
bright wrapped channels is otherwise perfectly good data, so dropping the whole
file wastes it. The product here is therefore a per-channel record; a
file-level `accumulator-overflow` mask in select_files.py covers only the
severe cases, for callers who cannot apply channel masks.

Physically, wrapping got worse when `corr_acc_len` doubled at 15:54:59 UTC
07-15 (67108864 -> 134217728): twice the accumulation is twice the count, so
bright channels that previously fit in int32 stopped fitting.

Output
------
curation/overflow_channels.jsonl -- one row per affected (file, input):
    {file, t_utc, phase, corr_acc_len, input, n_cells, n_chans,
     frac_samples, channels: {chan: n_samples}}

Files with no overflow produce no rows.

Usage
-----
    python curation/scan_overflow.py
    python curation/scan_overflow.py --summary
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

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
DATA = CAMPAIGN_ROOT / "data"
JSONL_OUT = CAMPAIGN_ROOT / "curation" / "overflow_channels.jsonl"

FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")

PHASE_PIVOTS = [
    ("2026-07-15T00:32:17Z", "C"),
    ("2026-07-14T04:10:43Z", "B"),
]


def filename_time(fn: str) -> float:
    m = FNAME_RE.match(fn)
    if not m:
        raise ValueError(fn)
    return (datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            .replace(tzinfo=timezone.utc).timestamp())


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def phase_of(ts: float) -> str:
    for t, ph in PHASE_PIVOTS:
        if ts >= datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp():
            return ph
    return "A"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    # corr_acc_len and filter_phase are not readable from header/ in every file
    # (obs_config is empty in many), so join them from the scan, which resolves
    # them correctly. Fall back to None/derived if the scan is absent.
    acc_by_file, phase_by_file = {}, {}
    csv_path = CAMPAIGN_ROOT / "curation" / "file_state.csv"
    if csv_path.is_file():
        import csv as _csv
        with csv_path.open() as fh:
            for rec in _csv.DictReader(fh):
                try:
                    acc_by_file[rec["file"]] = int(float(rec["corr_acc_len"]))
                except (ValueError, KeyError, TypeError):
                    pass
                phase_by_file[rec["file"]] = rec.get("filter_phase") or None

    files = sorted(DATA.glob("corr_*.h5"))
    rows = []
    for i, path in enumerate(files):
        name = path.name
        try:
            ts = filename_time(name)
        except ValueError:
            continue
        try:
            with h5py.File(path, "r") as h:
                acc = acc_by_file.get(name)
                for key in sorted(h["data"]):
                    if len(key) != 1:          # autos only; crosses are signed
                        continue
                    arr = h["data"][key]
                    if arr.dtype != np.int32:
                        continue
                    a = arr[:]
                    neg = a < 0
                    if not neg.any():
                        continue
                    per_chan = neg.sum(axis=0)
                    chans = {int(c): int(per_chan[c])
                             for c in np.flatnonzero(per_chan)}
                    rows.append({
                        "file": name,
                        "t_utc": iso(ts),
                        "phase": phase_by_file.get(name) or phase_of(ts),
                        "corr_acc_len": acc,
                        "input": key,
                        "n_cells": int(neg.sum()),
                        "n_chans": len(chans),
                        "frac_samples": round(float(neg.mean()), 8),
                        "channels": chans,
                    })
        except Exception as e:                  # noqa: BLE001
            print(f"  ! {name}: {e}", file=sys.stderr)
        if args.summary and i % 1000 == 0:
            print(f"  ...{i}/{len(files)}", file=sys.stderr)

    with JSONL_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    affected = {r["file"] for r in rows}
    print(f"{len(rows)} (file,input) rows over {len(affected)} files "
          f"-> {JSONL_OUT.relative_to(CAMPAIGN_ROOT)}")

    if args.summary:
        by_acc = Counter()
        files_by_acc = {}
        for r in rows:
            by_acc[r["corr_acc_len"]] += 1
            files_by_acc.setdefault(r["corr_acc_len"], set()).add(r["file"])
        print("\n  by corr_acc_len:")
        for acc, n in sorted(by_acc.items(), key=lambda kv: (kv[0] or 0)):
            print(f"    {acc}: {n} rows, {len(files_by_acc[acc])} files")
        print("\n  by input:", dict(Counter(r["input"] for r in rows)))
        print("  by phase:", dict(Counter(r["phase"] for r in rows)))
        chan_hits = Counter()
        for r in rows:
            chan_hits.update(r["channels"].keys())
        print(f"\n  distinct channels ever affected: {len(chan_hits)}")
        print("  most-affected channels (chan: files):")
        for c, n in chan_hits.most_common(12):
            print(f"    {c:5s} {n}" if isinstance(c, str) else f"    {c:5d} {n}")


if __name__ == "__main__":
    main()
