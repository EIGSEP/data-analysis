#!/usr/bin/env python3
"""Group the scanned switch states into contiguous cal windows."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["t0"])
    samples = []
    for r in rows:
        for s, t in zip(r["states"], r["times"]):
            samples.append((t, s if s is not None else "NONE", r["file"], r["phase"]))
    samples.sort(key=lambda x: x[0])
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--gap", type=float, default=120.0,
                    help="seconds of RFANT that closes a cal window")
    args = ap.parse_args()

    s = load(args.scan)
    windows, cur = [], None
    for t, st, fn, ph in s:
        cal = st not in ("RFANT",)
        if cal:
            if cur is None:
                cur = {"t0": t, "t1": t, "hist": Counter(), "files": set(), "phase": ph}
            elif t - cur["t1"] > args.gap:
                windows.append(cur)
                cur = {"t0": t, "t1": t, "hist": Counter(), "files": set(), "phase": ph}
            cur["t1"] = t
            cur["hist"][st] += 1
            cur["files"].add(fn)
    if cur:
        windows.append(cur)

    print(f"{len(windows)} cal windows\n")
    for i, w in enumerate(windows):
        h = dict(w["hist"])
        kind = []
        if {"VNAO", "VNAS", "VNAL"} <= set(h):
            kind.append("SOL")
        if "VNAANT" in h:
            kind.append("VNAANT")
        if {"RFNON", "RFAMB"} <= set(h):
            kind.append("LOADPAIR")
        print(f"[{i:3d}] {iso(w['t0'])} -> {iso(w['t1'])} "
              f"({w['t1']-w['t0']:6.0f}s) phase={w['phase']} "
              f"{'+'.join(kind) or '?':16s} nfile={len(w['files'])} {h}")


if __name__ == "__main__":
    main()
