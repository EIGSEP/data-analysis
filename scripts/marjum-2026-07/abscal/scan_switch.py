#!/usr/bin/env python3
"""Scan rfswitch state axis over a time range of Marjum 2026-07 corr files.

Targeted scan only -- data-archivist owns the authoritative campaign-wide
product (curation/cal_windows.jsonl). This exists so B7 can move before that
file lands, and to cross-check it when it does.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from eigsep_data.paths import campaign_data_dir

DATA = campaign_data_dir()
FNPAT = re.compile(r"corr_(\d{8})_(\d{6})Z\.h5")


def _jmeta(f, key):
    if key not in f["metadata"]:
        return None
    v = f["metadata"][key][()]
    return json.loads(v.decode() if isinstance(v, bytes) else v)


def scan_file(path):
    """Return per-file switch summary, or None if no rfswitch key."""
    with h5py.File(path, "r") as f:
        states = _jmeta(f, "rfswitch")
        if states is None:
            return None
        times = f["header/times"][:]
        phase = f.attrs.get("filter_phase")
        keys = list(f["data"].keys())
        therm = _jmeta(f, "rfswitch_therm")
    states = np.asarray(states, dtype=object)
    return {
        "file": Path(path).name,
        "phase": phase,
        "t0": float(times[0]),
        "t1": float(times[-1]),
        "n": len(states),
        "keys": keys,
        "hist": dict(Counter(states.tolist())),
        "states": states.tolist(),
        "times": times.tolist(),
        "n_therm": 0 if therm is None else len(therm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="corr_2026071[78]*.h5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    files = sorted(DATA.glob(args.glob))
    rows, no_switch = [], []
    for p in files:
        r = scan_file(p)
        if r is None:
            no_switch.append(p.name)
        else:
            rows.append(r)

    total = Counter()
    for r in rows:
        total.update(r["hist"])
    print(f"files scanned      : {len(files)}")
    print(f"  with rfswitch    : {len(rows)}")
    print(f"  without rfswitch : {len(no_switch)}")
    print(f"state histogram    : {dict(total)}")

    nonant = [r for r in rows if set(r["hist"]) - {"RFANT"}]
    print(f"files w/ non-RFANT : {len(nonant)}")
    for r in nonant[:400] if not args.summary_only else []:
        h = {k: v for k, v in r["hist"].items() if k != "RFANT"}
        print(f"  {r['file']}  phase={r['phase']}  {h}")

    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
