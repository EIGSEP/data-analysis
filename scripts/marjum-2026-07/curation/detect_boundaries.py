#!/usr/bin/env python3
"""Detect state-change boundaries from file_state.csv.

Two boundary layers:

  A. Deterministic-metadata boundaries. Any change in one of a fixed set of
     "fingerprint" columns between chronologically-adjacent files marks a
     boundary. These are exact (single-file precision).

  B. Numeric-signal boundaries. Median-filtered per-input power, live-channel
     count, argmax channel, and comb-detection scores. A jump larger than
     THRESH * local-MAD from a rolling baseline marks a boundary.

Output: boundaries.jsonl and STATE_CHANGES.md summary.
"""
from __future__ import annotations
import argparse
import csv
import os
import json
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
CSV_PATH = CAMPAIGN_ROOT / "curation" / "file_state.csv"
JSONL_OUT = CAMPAIGN_ROOT / "boundaries.jsonl"
MD_OUT = CAMPAIGN_ROOT / "STATE_CHANGES.md"

# Filenames are `corr_YYYYMMDD_HHMMSSZ[-N].h5` where the timestamp is file
# close time UTC. This is reliable; the in-file header/times array can be
# bogus (F-engine not yet re-synced), producing May 2026 timestamps.
FILENAME_RE = None  # unused; simple slicing below

# Columns whose *value* change between adjacent files marks a boundary.
FINGERPRINT_COLS = [
    "filter_phase",
    "data_keys",
    "mux_0to1",
    "mux_4to5",
    "adc_gain",
    "adc_mux_sel",
    "fft_shift",
    "corr_scalar",
    "corr_acc_len",
    "use_noise",
    "use_ref",
    "run_tag",
    "fpg_file",
    "snap_id",
    "ants",
    "ant_rx_ids",
    "ant_snap_inputs",
    "wiring_hash",
    "obs_config_hash",
    "input_to_ant_hash",
    "pol_delay_hash",
    "has_meta_rfswitch",
    "has_meta_motor",
    "has_meta_lidar",
    "has_meta_imu_el",
    "has_meta_potmon",
    "has_meta_system_current",
    "has_meta_tempctrl_lna",
    "cal_cadence_active",   # derived; see main()
]

# Columns to *bucket* first, then trigger on bucket change. Reduces noise from
# jittery near-boundary values.
#
# rfswitch_dominant is deliberately EXCLUDED: it cycles on a scheduled cal
# cadence (RFANT/RFAMB/RFNON/RFSP1/VNASP1/...) and its per-file transitions
# aren't state changes, they're the schedule ticking. Instead we derive a
# boolean "cal cadence active" signal below.
BUCKETED_COLS: dict[str, callable] = {
    # cadence: quantise to nearest 32 s
    "file_span_s": lambda x: int(round(x / 32.0)) if pd.notna(x) else None,
}

# When rfswitch_dominant is anything other than RFANT, the scheduled
# calibration cadence is firing. Its overall on/off state IS interesting.
CAL_STATES = {"RFAMB", "RFNON", "RFSP1", "RFSP1_SHORT", "RFSP1_OPEN", "VNASP1", "VNARF"}

# Numeric signal columns to run change-point on (per-input where applicable).
# We use log10(power) so LNA gain steps and battery-drop dropouts both look
# small in Z-units. `live_chans` and `argmax_ch` are checked separately with
# a stricter gate below because they flip on filter-phase-inappropriate
# inputs (e.g. a2 in phase B/C where input 2 carries noise).
NUMERIC_COLS_PER_INPUT = [
    "power", "argmax_ch",
    "comb4mhz_score", "comb1p25mhz_score",
]

# The per-phase "live" inputs. Only run Layer-B change-detection on these.
PHASE_INPUTS = {
    "A": ("0", "2", "3", "4"),
    "B": ("3", "4", "5"),
    "C": ("0", "4"),
}


def utc_iso(t: float) -> str:
    if not np.isfinite(t) or t <= 0:
        return ""
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def filename_close_unix(fname: str) -> float:
    """Parse close time from filename `corr_YYYYMMDD_HHMMSSZ[-N].h5`.

    This is the trustworthy timestamp; in-file header/times can be bogus when
    the F-engine hadn't been re-synced (values from May 2026 seen in the raw
    scan). Duplicate `-N` suffix bumps by N milliseconds so files sort stably.
    """
    stem = fname.split("/")[-1].removeprefix("corr_").removesuffix(".h5")
    if "-" in stem:
        stem, dup = stem.split("-", 1)
        try:
            dup_off = int(dup) * 0.001
        except ValueError:
            dup_off = 0.0
    else:
        dup_off = 0.0
    ts = datetime.strptime(stem, "%Y%m%d_%H%M%SZ").replace(tzinfo=timezone.utc)
    return ts.timestamp() + dup_off


def rolling_mad_jumps(series: pd.Series, window: int = 21, thresh: float = 5.0) -> list[int]:
    """Return integer indices where |value - rolling median| > thresh * rolling MAD.

    Only counts positive-going or negative-going *transitions*: a run of high values
    triggers only at the first index. NaNs are treated as break-point transparent.
    """
    s = series.astype("float64")
    if s.notna().sum() < window * 2:
        return []
    med = s.rolling(window, center=True, min_periods=window // 2).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=window // 2).median() + 1e-9
    z = (s - med).abs() / mad
    outlier = z > thresh
    idx = []
    prev = False
    for i, v in enumerate(outlier.fillna(False).values):
        if v and not prev:
            idx.append(i)
        prev = v
    return idx


def detect(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Use filename close time as the reliable clock; in-file times can be
    # bogus in files where F-engine sync hasn't been redone.
    df["file_close_unix"] = df["file"].apply(filename_close_unix)
    df = df.sort_values("file_close_unix").reset_index(drop=True)
    df["t_iso"] = df["file_close_unix"].apply(utc_iso)

    # Inter-file gap using filename-derived close-times. When it's roughly
    # equal to the previous file's own span, files are contiguous. A gap
    # significantly larger than the previous span is a real drop-out.
    df["prev_close"] = df["file_close_unix"].shift(1)
    df["prev_span"] = df["file_span_s"].shift(1).fillna(0)
    df["gap_s"] = df["file_close_unix"] - df["prev_close"] - df["prev_span"]

    # Derived: cal cadence active in this file? (rfswitch spent >5% of the
    # file in a non-RFANT state that is a scheduled cal state).
    def _cal_active(row):
        raw = row.get("rfswitch_states_json")
        if not isinstance(raw, str) or not raw:
            return False
        try:
            counts = json.loads(raw)
        except Exception:
            return False
        total = sum(counts.values()) or 1
        cal = sum(v for k, v in counts.items() if k in CAL_STATES)
        return (cal / total) > 0.05
    df["cal_cadence_active"] = df.apply(_cal_active, axis=1)

    events: list[dict] = []

    # --- Layer A: deterministic fingerprint deltas ---
    for col in FINGERPRINT_COLS:
        if col not in df.columns:
            continue
        prev = df[col].shift(1)
        cur = df[col]
        # NaN != NaN by default; treat NaN as its own bucket
        changed = ~((prev == cur) | (prev.isna() & cur.isna()))
        for i in np.where(changed.values)[0]:
            if i == 0:
                continue  # first file has no predecessor
            events.append({
                "t_utc": df["t_iso"].iat[i],
                "t_unix": float(df["file_close_unix"].iat[i]),
                "layer": "A_fingerprint",
                "signal": col,
                "before": _jsonable(df[col].iat[i - 1]),
                "after": _jsonable(df[col].iat[i]),
                "file": df["file"].iat[i],
                "prev_file": df["file"].iat[i - 1],
                "notes": None,
            })

    # bucketed columns
    for col, bucketiser in BUCKETED_COLS.items():
        if col not in df.columns:
            continue
        b = df[col].map(bucketiser)
        prev = b.shift(1)
        changed = ~((prev == b) | (prev.isna() & b.isna()))
        for i in np.where(changed.values)[0]:
            if i == 0:
                continue
            events.append({
                "t_utc": df["t_iso"].iat[i],
                "t_unix": float(df["file_close_unix"].iat[i]),
                "layer": "A_bucketed",
                "signal": col,
                "before": _jsonable(df[col].iat[i - 1]),
                "after": _jsonable(df[col].iat[i]),
                "file": df["file"].iat[i],
                "prev_file": df["file"].iat[i - 1],
                "notes": f"bucket change {bucketiser(df[col].iat[i-1])!r} -> {bucketiser(df[col].iat[i])!r}",
            })

    # gap events (gap of *idle time* between files, using filename close times)
    for i in np.where(df["gap_s"] > 300)[0]:
        events.append({
            "t_utc": df["t_iso"].iat[i],
            "t_unix": float(df["file_close_unix"].iat[i]),
            "layer": "A_gap",
            "signal": "gap_s",
            "before": None,
            "after": float(df["gap_s"].iat[i]),
            "file": df["file"].iat[i],
            "prev_file": df["file"].iat[i - 1],
            "notes": f"inter-file gap {df['gap_s'].iat[i]/60:.1f} min",
        })

    # sync_time changes: filter noise (many files inherit last upload_unix)
    prev_sync = df["sync_time"].shift(1)
    for i in np.where((df["sync_time"] != prev_sync) & prev_sync.notna())[0]:
        if i == 0: continue
        # only flag when it moves >1 s
        if abs(df["sync_time"].iat[i] - prev_sync.iat[i]) < 1.0:
            continue
        events.append({
            "t_utc": df["t_iso"].iat[i],
            "t_unix": float(df["file_close_unix"].iat[i]),
            "layer": "A_sync",
            "signal": "sync_time",
            "before": float(prev_sync.iat[i]),
            "after": float(df["sync_time"].iat[i]),
            "file": df["file"].iat[i],
            "prev_file": df["file"].iat[i - 1],
            "notes": f"F-engine re-synced (upload_unix {utc_iso(df['header_upload_unix'].iat[i])})",
        })

    # --- Layer B: numeric-signal deltas, per phase-appropriate input ---
    # We compute change-points within a rolling window on log10(power) so a
    # 10x drop shows as ~1 unit; we insist on a strict Z-threshold to keep
    # signal-only boundaries meaningful (many small ones already show as
    # config changes at Layer A).
    for phase, inputs in PHASE_INPUTS.items():
        phase_mask = df["filter_phase"] == phase
        if not phase_mask.any():
            continue
        sub = df[phase_mask].copy()
        sub_idx = sub.index.to_numpy()  # positions in the master df
        for inp in inputs:
            for name in NUMERIC_COLS_PER_INPUT:
                col = f"a{inp}_{name}"
                if col not in sub.columns:
                    continue
                s = sub[col].astype("float64")
                if s.notna().sum() < 50:
                    continue
                # power in log10 to compress large ranges
                if name == "power":
                    s = np.log10(s.where(s > 0)).astype("float64")
                for i_local in rolling_mad_jumps(s, window=21, thresh=8.0):
                    i = int(sub_idx[i_local])
                    events.append({
                        "t_utc": df["t_iso"].iat[i],
                        "t_unix": float(df["file_close_unix"].iat[i]),
                        "layer": "B_signal",
                        "signal": col,
                        "before": _jsonable(df[col].iat[i - 1]) if i > 0 else None,
                        "after": _jsonable(df[col].iat[i]),
                        "file": df["file"].iat[i],
                        "prev_file": df["file"].iat[i - 1] if i > 0 else None,
                        "notes": None,
                    })

    # sort by unix time then a stable priority (A before B)
    events.sort(key=lambda e: (e["t_unix"], 0 if e["layer"].startswith("A") else 1, e["signal"]))
    return df, events


def _jsonable(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def collapse_by_time(events: list[dict], within_s: float = 90.0) -> list[dict]:
    """Group events whose t_unix falls within `within_s` into a single 'boundary'."""
    out = []
    cur = None
    for e in events:
        if cur is None or e["t_unix"] - cur["t_unix"] > within_s or e["file"] != cur["file"]:
            if cur:
                out.append(cur)
            cur = {
                "t_utc": e["t_utc"],
                "t_unix": e["t_unix"],
                "file": e["file"],
                "prev_file": e["prev_file"],
                "signals": [],
            }
        cur["signals"].append({
            "layer": e["layer"],
            "signal": e["signal"],
            "before": e["before"],
            "after": e["after"],
            "notes": e["notes"],
        })
    if cur:
        out.append(cur)
    return out


def summarise_boundary(b: dict) -> str:
    """One-line human summary."""
    a_sigs = [s["signal"] for s in b["signals"] if s["layer"].startswith("A")]
    b_sigs = [s["signal"] for s in b["signals"] if s["layer"] == "B_signal"]
    parts = []
    if a_sigs:
        parts.append("config: " + ", ".join(sorted(set(a_sigs))))
    if b_sigs:
        parts.append("signal: " + ", ".join(sorted(set(b_sigs))))
    return f"{b['t_utc']}  {b['file']}  ({len(b['signals'])} signals) -> {'; '.join(parts)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--jsonl", type=Path, default=JSONL_OUT)
    ap.add_argument("--md", type=Path, default=MD_OUT)
    ap.add_argument("--within-s", type=float, default=90.0)
    args = ap.parse_args()

    df, events = detect(args.csv)
    print(f"raw events: {len(events)}")
    boundaries = collapse_by_time(events, within_s=args.within_s)
    print(f"collapsed to {len(boundaries)} boundaries (within {args.within_s:.0f} s)")

    with args.jsonl.open("w") as f:
        for b in boundaries:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    # Categorise each boundary by its dominant signal group
    def _category(b: dict) -> str:
        sigs = {s["signal"] for s in b["signals"]}
        if "filter_phase" in sigs or "wiring_hash" in sigs or "snap_id" in sigs or "ants" in sigs or "ant_snap_inputs" in sigs:
            return "wiring"
        if "cal_cadence_active" in sigs:
            return "cal-cadence"
        if any(s.startswith("has_meta_") for s in sigs):
            return "daemon"
        if "obs_config_hash" in sigs or "run_tag" in sigs or "fpg_file" in sigs:
            return "software-config"
        if "adc_gain" in sigs or "adc_mux_sel" in sigs or "fft_shift" in sigs or "corr_scalar" in sigs or "corr_acc_len" in sigs:
            return "correlator-config"
        if "mux_0to1" in sigs or "mux_4to5" in sigs:
            return "adc-mux"
        if "sync_time" in sigs or "gap_s" in sigs or "file_span_s" in sigs:
            return "run-boundary"
        if any("power" in s or "argmax_ch" in s or "comb" in s for s in sigs):
            return "signal"
        return "other"

    for b in boundaries:
        b["category"] = _category(b)

    # write the enriched jsonl (with category)
    with args.jsonl.open("w") as f:
        for b in boundaries:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    # Markdown summary: category counts + full table
    cat_counts: dict[str, int] = {}
    for b in boundaries:
        cat_counts[b["category"]] = cat_counts.get(b["category"], 0) + 1

    lines = [
        "# State-change boundaries",
        "",
        f"Derived from `file_state.csv` ({len(df)} files) by `detect_boundaries.py`. Boundaries collapse per-file within a {args.within_s:.0f} s window.",
        "",
        f"Total boundaries: **{len(boundaries)}**",
        "",
        "Boundary = the first file whose fingerprint differs from the previous file. Layers: `A_fingerprint` (exact metadata change), `A_bucketed` (quantised value change), `A_gap` (long inter-file gap), `A_sync` (F-engine re-sync), `B_signal` (>8-MAD jump in per-input power / argmax / comb-detection score, phase-appropriate inputs only). Category assigned by dominant signal.",
        "",
        "## By category",
        "",
        "| category | # |",
        "|---|---|",
    ]
    for c in sorted(cat_counts, key=lambda k: -cat_counts[k]):
        lines.append(f"| {c} | {cat_counts[c]} |")

    # Top-level rare-category events (wiring, daemon, correlator-config) are
    # the ones worth eyeballing in this doc; the routine chatter (signal, cal
    # cycle, run-boundary) lives in the jsonl for filtering.
    important = [b for b in boundaries if b["category"] in {"wiring", "daemon", "correlator-config", "adc-mux", "software-config"}]
    lines += [
        "",
        f"## Notable boundaries ({len(important)}) - wiring / daemon / correlator-config / adc-mux / software-config",
        "",
        "| # | UTC | file | category | signals |",
        "|---|---|---|---|---|",
    ]
    for i, b in enumerate(important, 1):
        sigs = sorted({s["signal"] for s in b["signals"]})
        lines.append(
            f"| {i} | `{b['t_utc']}` | `{b['file']}` | {b['category']} | {', '.join(sigs)} |"
        )

    lines += ["", "## All boundaries", "", "| # | UTC | file | category | # signals | signals |", "|---|---|---|---|---|---|"]
    for i, b in enumerate(boundaries, 1):
        sigs = sorted({s["signal"] for s in b["signals"]})
        lines.append(
            f"| {i} | `{b['t_utc']}` | `{b['file']}` | {b['category']} | {len(b['signals'])} | {', '.join(sigs)} |"
        )

    args.md.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.jsonl} and {args.md}")


if __name__ == "__main__":
    main()
