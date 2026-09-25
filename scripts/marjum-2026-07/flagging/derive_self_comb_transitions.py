#!/usr/bin/env python3
"""Digital-self-comb on/off episodes from the box-gnd per-integration trace.

Consumes `scan_self_comb_per_integration.py`'s block trace and emits the
authoritative transition list.

State rule, and why it is hysteretic
------------------------------------
A plain threshold on `lock8` chatters badly here: the transition count
runs 73 -> 197 as the threshold moves 5 -> 80, and 37 of the 63 apparent
off-runs are 1-3 blocks (under 13 s). Those are the statistic wobbling
across the threshold inside a continuously-on stretch, not the comb
switching.

What makes a clean rule available is the statistic's own floor. On
comb-free data the log-residual is exactly zero on more than half the
in-band channels, the MAD scale collapses, and `tooth_contrast_locked`
returns exactly 0.0. Measured over the era: of 1,380 sub-threshold
blocks, **1,176 are exactly 0.0** and only 204 are in (0, 30). The off
state is therefore a hard, unambiguous value rather than a small one.

So:

    ON   when lock8 >= ON_THRESHOLD and phase == 0
    OFF  when lock8 == 0.0     (degenerate/no structure)
    otherwise hold the previous state

This yields 22 episodes, stable at 22-23 for any ON threshold in
20-40 and degrading gracefully (20 at 60, 16 at 200) -- against 73-197
transitions for a plain threshold over the same range. The aggregate
duty cycle, 87.1%, agrees with the file-level product's 86%, so the
method resolves structure without shifting the total.

Scope: box-gnd (input 0) only, per Aaron's 2026-09-20 decision. It is
conducted rather than antenna-switched, so it sees through the cal/VNA
cadence. This determines *state and timing* only.

Output: one JSON line of provenance, then one per episode:
    {episode, t_start_utc, t_end_utc, duration_s,
     file_start, block_start, file_end, block_end,
     shorter_than_file}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

BLOCK_S = 4.3          # 8 integrations at ~0.537 s
FILE_S = 128.85
ON_THRESHOLD = 30.0


def parse_t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def episodes(rows, on_threshold=ON_THRESHOLD):
    """Hysteretic on/off state -> list of (first_block, last_block)."""
    out, cur, state = [], None, False
    for r in rows:
        if r["lock8"] >= on_threshold and r["phase"] == 0:
            state = True
        elif r["lock8"] == 0.0:
            state = False
        if state and cur is None:
            cur = [r, r]
        elif state:
            cur[1] = r
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--on-threshold", type=float, default=ON_THRESHOLD)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.trace)]
    rows.sort(key=lambda r: (r["file"], r["block"]))
    eps = episodes(rows, args.on_threshold)

    span = (parse_t(rows[-1]["t_utc"]) - parse_t(rows[0]["t_utc"])).total_seconds()
    total_on = sum(
        (parse_t(b["t_utc"]) - parse_t(a["t_utc"])).total_seconds() + BLOCK_S
        for a, b in eps)

    with Path(args.out).open("w") as fh:
        fh.write(json.dumps({"provenance": {
            "product": "self_comb_transitions_boxgnd",
            "campaign": "marjum-2026-07",
            "version": "v1",
            "generator": "flagging/derive_self_comb_transitions.py",
            "trace": Path(args.trace).name,
            "trace_generator": "flagging/scan_self_comb_per_integration.py",
            "statistic": "comb_inventory.tooth_contrast_locked, spacing 8 ch",
            "input": "0 (box-gnd) -- authoritative for state and timing",
            "block_integrations": 8,
            "block_seconds": BLOCK_S,
            "state_rule": (f"ON when lock8>={args.on_threshold} and phase==0; "
                           "OFF when lock8==0.0; else hold"),
            "n_episodes": len(eps),
            "n_transitions": max(0, 2 * len(eps) - 1),
            "duty_cycle": round(total_on / span, 4),
            "supersedes": ("file-level lock8 in flags/diag/comb_inventory.jsonl "
                           "for transition timing only; that product remains "
                           "correct for comb characterisation"),
            "caveat": ("time resolution is one block (~4.3 s); episodes shorter "
                       "than ~2 integrations can still be missed"),
        }}) + "\n")
        for i, (a, b) in enumerate(eps):
            dur = (parse_t(b["t_utc"]) - parse_t(a["t_utc"])).total_seconds() + BLOCK_S
            fh.write(json.dumps({
                "episode": i,
                "t_start_utc": a["t_utc"],
                "t_end_utc": b["t_utc"],
                "duration_s": round(dur, 1),
                "file_start": a["file"], "block_start": a["block"],
                "file_end": b["file"], "block_end": b["block"],
                "shorter_than_file": bool(dur < FILE_S),
            }) + "\n")
    n_short = sum(1 for a, b in eps
                  if (parse_t(b["t_utc"]) - parse_t(a["t_utc"])).total_seconds()
                  + BLOCK_S < FILE_S)
    print(f"{len(eps)} episodes, {n_short} shorter than one file, "
          f"duty cycle {total_on / span:.1%} -> {args.out}")


if __name__ == "__main__":
    main()
