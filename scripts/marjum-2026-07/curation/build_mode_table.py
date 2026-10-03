#!/usr/bin/env python3
"""Build the campaign observing-mode table for marjum-2026-07.

One row per contiguous window in which every classified mode axis is constant:

    height era | wiring phase | rotation state | orientation bin |
    box-air 1 MHz EMI | transmitter | corr_acc_len | RF switch state | validity

The two comb axes (memo 001, "Combs, and which one is the transmitter"):

``boxair_emi``   box-air's 1.000 MHz self-EMI comb (07-16 01:16:57-16:49:03,
                 box-air only, gone when the Panda was power-cycled). Detected
                 per file from ``file_state.csv``'s ``a*_comb4mhz_score``.
                 Before memo 001 this column was called ``tx_comb`` and was
                 read as the transmitter; the values are unchanged.
``transmitter``  the beam-mapping transmitter's 8-channel comb (1.953125 MHz,
                 channels == 0 mod 8), "on" / "off" / "partial" per file from
                 box-gnd's per-integration episodes
                 (``curation/transmitter_transitions_boxgnd.jsonl`` over the
                 block trace ``flags/diag/transmitter_per_integration_boxgnd.jsonl``).
                 "partial" means the file holds both on and off blocks. Files
                 closing before the trace starts (07-17 15:37:44) are "off":
                 memo 001's campaign-wide 8-channel contrast first sees the
                 comb at 15:36:22. Files after the trace ends (07-18 03:00:13)
                 are "unknown".

Inputs
------
curation/file_state.csv   per-file scan (see scan_file_state.py)
events.jsonl              field-note timeline (height eras come from here)
curation/transmitter_transitions_boxgnd.jsonl, flags/diag/transmitter_per_integration_boxgnd.jsonl

Outputs
-------
curation/mode_table.jsonl     one JSON object per contiguous mode window
curation/mode_table.provenance.json   generator, commit, inputs, renames
curation/file_modes.csv       per-file mode assignment (regenerable, gitignored)

Why filename time and not header/times
--------------------------------------
`file_state.csv` carries `t_close_unix` read from `header/times`, which is
unreliable: 615 of 5,120 files report May-2026 or 07-09 timestamps because the
F-engine had not re-synced. Filenames are the file CLOSE time and are correct.
This script derives all time from the filename and records the header-vs-filename
discrepancy as a diagnostic (`hdr_time_bad`) rather than trusting it.

Usage
-----
    python curation/build_mode_table.py
    python curation/build_mode_table.py --az-tol 2.0 --summary
"""

from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
EVENTS = CAMPAIGN_ROOT / "events.jsonl"
JSONL_OUT = CAMPAIGN_ROOT / "curation" / "mode_table.jsonl"
CSV_OUT = CAMPAIGN_ROOT / "curation" / "file_modes.csv"
PROV_OUT = CAMPAIGN_ROOT / "curation" / "mode_table.provenance.json"
TX_TRANSITIONS = CAMPAIGN_ROOT / "curation" / "transmitter_transitions_boxgnd.jsonl"
TX_TRACE = CAMPAIGN_ROOT / "flags" / "diag" / "transmitter_per_integration_boxgnd.jsonl"

FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")

# --------------------------------------------------------------------------
# Height eras. Source: field notes via events.jsonl (category "mechanical").
# Field notes give day- or hour-level granularity for the lifts; these are the
# best available boundaries. LIDAR cannot supply them -- only 244 of 5,120
# files carry a reading and the rangefinder saturates at 250 m.
#
# Each entry: (start_utc_inclusive, label, source_tag)
# --------------------------------------------------------------------------
HEIGHT_ERAS = [
    ("2026-07-12T00:00:00Z", "~2m",    "fieldnotes:deflection-0.97m-loaded-160lb"),
    ("2026-07-15T00:00:00Z", "~30m",   "fieldnotes:lift-to-30m-completed-7/15"),
    ("2026-07-17T02:30:00Z", "~87.5m", "fieldnotes:raised-87.5m-by-2030-MDT-7/16"),
    ("2026-07-17T18:50:00Z", "~91m",   "fieldnotes:final-height-91m-1250-MDT"),
]

# Box-air 1 MHz EMI detection. a*_comb4mhz_score (a 16.4-channel harmonic sum
# that the 1.000 MHz comb lights up) is strongly bimodal: ~0.06 when the comb
# is absent, ~28-35 when present. Any threshold in between works.
BOXAIR_EMI_THRESHOLD = 1.0

# motor_*_std is exactly 0.0 when the axis is parked for the whole file and
# nonzero (>=30 at the 5th percentile) while slewing. Exact-zero is the test.
PARKED_STD = 0.0

# Default azimuth clustering tolerance, in potmon_az units. The pot reading is
# continuous and noisy, so "distinct orientation" requires a tolerance.
DEFAULT_AZ_TOL = 2.0


def _jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [r for r in rows if "provenance" not in r]


def transmitter_states() -> dict:
    """Per-file transmitter state from box-gnd's per-integration episodes.

    A block is "on" if its (file, block) lies inside an episode's
    [(file_start, block_start), (file_end, block_end)] span, in the same
    (file, block) order the episodes were derived in. A file is "on" if every
    block is on, "off" if none is, "partial" otherwise. Files before the
    trace are "off" (memo 001: first on 07-17 15:36:22); files after it, or
    missing from it, are "unknown" -- see ``state_for`` below.
    """
    eps = [((e["file_start"], e["block_start"]), (e["file_end"], e["block_end"]))
           for e in _jsonl(TX_TRANSITIONS)]
    blocks: dict[str, list[bool]] = {}
    for r in sorted(_jsonl(TX_TRACE), key=lambda r: (r["file"], r["block"])):
        k = (r["file"], r["block"])
        blocks.setdefault(r["file"], []).append(any(a <= k <= b for a, b in eps))
    first, last = min(blocks), max(blocks)

    class _States(dict):
        def __missing__(self, fn):
            return "off" if fn < first else "unknown"

    out = _States()
    for fn, on in blocks.items():
        out[fn] = "on" if all(on) else ("off" if not any(on) else "partial")
    return out


def short_sha(repo: Path) -> str:
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                            "--", "."], capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def filename_time(fn: str) -> float:
    """UTC close time from the filename. Authoritative; see module docstring."""
    m = FNAME_RE.match(fn)
    if not m:
        raise ValueError(f"unparseable filename: {fn}")
    return (
        datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def height_era(ts: float) -> tuple[str, str]:
    label, src = "unknown", "none"
    for start, lab, s in HEIGHT_ERAS:
        if ts >= pd.Timestamp(start).timestamp():
            label, src = lab, s
    return label, src


def live_inputs(phase: str) -> list[str]:
    """Correlator inputs carrying real sky signal, per data/README.md."""
    return {"A": ["3", "4"], "B": ["3", "4"], "C": ["0", "4"]}.get(phase, [])


def classify(df: pd.DataFrame, az_tol: float) -> pd.DataFrame:
    df = df.copy()
    df["t"] = df["file"].map(filename_time)
    # Sort by (time, name), not time alone. Duplicate-suffix files such as
    # corr_20260714_041044Z-1.h5 / -2.h5 / .h5 share a timestamp, and a
    # time-only sort leaves their relative order undefined. Consumers expand
    # window [file_first, file_last] spans against a lexicographic file
    # listing, so the two orderings must agree or files fall through the gaps.
    df = df.sort_values(["t", "file"]).reset_index(drop=True)

    # Diagnostic: header/times disagrees with the filename by > 1 hour.
    hdr = pd.to_numeric(df["t_close_unix"], errors="coerce")
    df["hdr_time_bad"] = (hdr - df["t"]).abs() > 3600

    eras = df["t"].map(height_era)
    df["height_era"] = [e[0] for e in eras]
    df["height_src"] = [e[1] for e in eras]

    # Rotation: parked iff both axes have exactly zero spread within the file.
    az_moving = df["motor_az_std"].fillna(0) > PARKED_STD
    el_moving = df["motor_el_std"].fillna(0) > PARKED_STD
    df["rot_state"] = np.select(
        [az_moving & el_moving, az_moving, el_moving],
        ["az+el-moving", "az-moving", "el-moving"],
        default="parked",
    )
    # Axis liveness: telemetry present at all for that axis in this file.
    # NOTE: these mean "MOTOR telemetry present for this axis", not "the axis
    # works" and NOT "the IMU is usable". pointing-analyst nearly misread them
    # as IMU liveness. The IMU picture is different and worse: imu_az is dead
    # from 07-16 onward, intermittent before, and where alive its yaw drifts
    # +188 deg/hr -- it is not an absolute azimuth reference anywhere in the
    # campaign. Use pointing_table for anything pointing-related.
    df["az_alive"] = df["motor_az_med"].notna()
    df["el_alive"] = df["motor_el_med"].notna()

    # Box-air 1 MHz EMI: on if any live input for the file's phase shows it
    # in the 4 MHz harmonic-sum score.
    comb = pd.Series(False, index=df.index)
    for ph in ("A", "B", "C"):
        sel = df["filter_phase"] == ph
        if not sel.any():
            continue
        cols = [f"a{i}_comb4mhz_score" for i in live_inputs(ph)]
        cols = [c for c in cols if c in df.columns]
        if cols:
            comb |= sel & (df[cols].max(axis=1) > BOXAIR_EMI_THRESHOLD)
    df["boxair_emi"] = np.where(comb, "on", "off")
    df["transmitter"] = df["file"].map(transmitter_states())

    # Orientation bin: cluster parked azimuths to a tolerance. Slewing files
    # get no orientation.
    df["orient_bin"] = pd.NA
    parked = (df["rot_state"] == "parked") & df["potmon_az_med"].notna()
    for (era, phase), grp in df[parked].groupby(["height_era", "filter_phase"]):
        vals = grp["potmon_az_med"].to_numpy()
        order = np.argsort(vals)
        binid = np.empty(len(vals), dtype=int)
        b, anchor = 0, vals[order[0]]
        for rank, i in enumerate(order):
            if vals[i] - anchor > az_tol:
                b += 1
                anchor = vals[i]
            binid[i] = b
        df.loc[grp.index, "orient_bin"] = [f"{era}/{phase}/az{n:03d}" for n in binid]

    df["rfsw"] = df.get("rfswitch_dominant", pd.Series(pd.NA, index=df.index))
    return df


MODE_KEYS = [
    "height_era",
    "filter_phase",
    "rot_state",
    "orient_bin",
    "boxair_emi",
    "transmitter",
    "corr_acc_len",
    "rfsw",
]


def windows(df: pd.DataFrame) -> list[dict]:
    key = df[MODE_KEYS].astype(str).agg("|".join, axis=1)
    grp = (key != key.shift()).cumsum()
    out = []
    for _, g in df.groupby(grp):
        first, last = g.iloc[0], g.iloc[-1]
        # integration_time_s is PER SPECTRUM (0.268 s or 0.537 s); each file
        # holds n_int spectra. Per-file total = integration_time_s * n_int.
        integ = (
            pd.to_numeric(g["integration_time_s"], errors="coerce")
            * pd.to_numeric(g["n_int"], errors="coerce")
        ).sum()
        out.append(
            {
                "t_start_utc": iso(first["t"]),
                "t_end_utc": iso(last["t"]),
                "file_first": first["file"],
                "file_last": last["file"],
                "n_files": int(len(g)),
                "integration_time_s": float(integ) if np.isfinite(integ) else None,
                "height_era": first["height_era"],
                "height_src": first["height_src"],
                "phase": first["filter_phase"],
                "live_inputs": live_inputs(str(first["filter_phase"])),
                "rot_state": first["rot_state"],
                "az_alive": bool(first["az_alive"]),
                "el_alive": bool(first["el_alive"]),
                "orient_bin": None if pd.isna(first["orient_bin"]) else str(first["orient_bin"]),
                "potmon_az_med": None if pd.isna(first["potmon_az_med"]) else float(first["potmon_az_med"]),
                "boxair_emi": first["boxair_emi"],
                "transmitter": first["transmitter"],
                "corr_acc_len": None if pd.isna(first["corr_acc_len"]) else int(first["corr_acc_len"]),
                "rfswitch_dominant": None if pd.isna(first["rfsw"]) else str(first["rfsw"]),
                "n_hdr_time_bad": int(g["hdr_time_bad"].sum()),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--az-tol", type=float, default=DEFAULT_AZ_TOL,
                    help=f"azimuth clustering tolerance in potmon units (default {DEFAULT_AZ_TOL})")
    ap.add_argument("--summary", action="store_true", help="print the orientation-census summary")
    ap.add_argument("--csv-in", type=Path, default=CSV_IN)
    args = ap.parse_args()

    df = classify(pd.read_csv(args.csv_in), args.az_tol)
    wins = windows(df)

    with JSONL_OUT.open("w") as f:
        for w in wins:
            f.write(json.dumps(w) + "\n")
    PROV_OUT.write_text(json.dumps({
        "product": "mode_table",
        "campaign": "marjum-2026-07",
        "generator": "data-analysis/scripts/marjum-2026-07/curation/build_mode_table.py",
        "generator_commit": short_sha(Path(__file__).resolve().parent),
        "inputs": {
            "file_state": args.csv_in.name,
            "transmitter_transitions": TX_TRANSITIONS.name,
            "transmitter_trace": TX_TRACE.name,
        },
        "n_files": int(len(df)),
        "n_windows": len(wins),
        "renamed_from": {
            "columns": {"tx_comb": "boxair_emi"},
            "reason": "the column tracks box-air's 1.000 MHz self-EMI, not the "
                      "transmitter (memo 001); values unchanged",
        },
        "added": {"transmitter": "on/off/partial/unknown per file from box-gnd's "
                                 "per-integration transmitter episodes"},
        "az_tol": args.az_tol,
    }, indent=2) + "\n")
    keep = ["file", "t", "height_era", "filter_phase", "rot_state", "orient_bin",
            "boxair_emi", "transmitter", "corr_acc_len", "rfsw", "hdr_time_bad",
            "potmon_az_med"]
    df[keep].to_csv(CSV_OUT, index=False)

    print(f"{len(df)} files -> {len(wins)} mode windows")
    print(f"  wrote {JSONL_OUT.relative_to(CAMPAIGN_ROOT)}")
    print(f"  wrote {CSV_OUT.relative_to(CAMPAIGN_ROOT)} (gitignored)")
    print(f"  files with unusable header/times: {int(df['hdr_time_bad'].sum())}")

    if args.summary:
        print("\n=== orientation census: parked, box-air EMI off, transmitter off, "
              "per height era + phase ===")
        sci = df[(df.rot_state == "parked") & (df.boxair_emi == "off")
                 & (df.transmitter == "off") & df.orient_bin.notna()].copy()
        sci["integ_s"] = (
            pd.to_numeric(sci["integration_time_s"], errors="coerce")
            * pd.to_numeric(sci["n_int"], errors="coerce")
        )
        for (era, phase), g in sci.groupby(["height_era", "filter_phase"]):
            per = g.groupby("orient_bin").agg(files=("file", "size"), integ_s=("integ_s", "sum"))
            print(f"\n  {era:8s} phase {phase}: {len(per):4d} distinct orientations, "
                  f"{len(g)} files, {per.integ_s.sum()/3600:.2f} h total")
            rich = per[per.integ_s >= 600]
            print(f"      orientations with >=10 min: {len(rich)}   "
                  f"median {per.integ_s.median()/60:.1f} min   "
                  f"max {per.integ_s.max()/3600:.2f} h")


if __name__ == "__main__":
    main()
