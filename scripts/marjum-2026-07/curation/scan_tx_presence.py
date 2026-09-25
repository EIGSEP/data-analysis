#!/usr/bin/env python3
"""Per-file TX-comb presence: was the ground transmitter on, integration by
integration, for the whole campaign.

Answers a direct question that was previously answered only piecemeal:
- `mode_table.jsonl`'s `tx_comb` column does NOT track the transmitter -- it
  tracks an unrelated, self-generated 1.953125 MHz digital comb (see
  `curation/tx_comb_eras.jsonl` and the retraction in INDEX.md). Confusing the
  two was the original error; this script does not repeat it.
- `curation/tx_comb_eras.jsonl` characterizes comb TYPE per era from 6-file
  medians -- it establishes what the TX comb looks like, but is not a
  per-file table and cannot say which specific integrations had it on.
- `eigsep_data/notebooks/arp/marjum-2026-07/comb_presence_beam_scan.json`
  is per-file but scoped to the 227-file beam-scan window only.

This is the missing piece: per file, per input, was the TX comb (real
external emitter, ~1.000 MHz = 4.096 channels, NOT locked to the ADC clock)
present, across all 5,120 files.

Method
------
`curation/scan_tx_comb.py` established the discriminator: the TX comb is a
harmonic family of peaks whose Rayleigh phase-concentration statistic S is
high at period P = 1.000/0.244140625 = 4.0961 channels, AND whose tone
centres sit on integer MHz (unlike the self-comb or unstructured RFI, which
do not). Both are independent, O(n_tones) computations -- no need for the
full continuous periodogram scan that makes the per-era tool too slow to run
per file.

Calibrated against six known cases (07-13/14/15 "off" era samples, the 07-16
"on" era sample, and two self-comb-dominated samples):

    case                S_tx    rms_offset_from_integer_MHz
    TX on   (07-16)     0.873   0.091
    TX off  (worst neg) 0.168   0.286   <- max S_tx seen with no TX
    TX off  (best neg)  0.022   0.318

Threshold: S_tx > 0.5 AND rms_offset < 0.15. Both margins are >2x the
worst-case separation observed in calibration, on independent statistics.

Autos are int32 and positive-definite; a negative value is an accumulator
wrap (see curation/overflow_channels.jsonl) and is repaired (+2**32) before
the median so a handful of bright wrapped samples cannot distort it.

Scans key='4' (box-air), which per data/README.md is live for the entire
campaign (present in phases A, B, and C) -- the one input that lets a single
scan cover 07-12 through 07-18 without a phase-dependent key.

Output: curation/tx_presence.jsonl, one row per file:
    {file, t_utc, phase, n_tones, S_tx, rms_offset_mhz, tx_on, note}

Usage
-----
    python curation/scan_tx_presence.py
    python curation/scan_tx_presence.py --summary
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
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
JSONL_OUT = CAMPAIGN_ROOT / "curation" / "tx_presence.jsonl"

FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")
CHAN_MHZ = 250.0 / 1024.0
P_TX = 1.000 / CHAN_MHZ          # 4.0961 channels
S_TX_THRESHOLD = 0.5
RMS_THRESHOLD_MHZ = 0.15
MIN_TONES = 8

PHASE_PIVOTS = [("2026-07-15T00:32:17Z", "C"), ("2026-07-14T04:10:43Z", "B")]


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


def find_tones(spec, lo, hi, n_med=21, snr=8.0):
    """Local maxima standing `snr` robust-sigma above a running median.

    Identical logic to scan_tx_comb.py's find_tones (kept independent here to
    avoid an eigsep_observing import, which is unnecessary overhead for a
    5,120-file scan and slows it roughly 5x).
    """
    idx = np.arange(lo, hi)
    band = spec[lo:hi]
    pad = n_med // 2
    padded = np.pad(band, pad, mode="edge")
    cont = np.array([np.median(padded[i:i + n_med]) for i in range(band.size)])
    excess = band - cont
    scale = 1.4826 * np.median(np.abs(excess - np.median(excess)))
    hits = (excess[1:-1] > excess[:-2]) & (excess[1:-1] >= excess[2:]) & \
           (excess[1:-1] > snr * max(scale, 1e-30))
    return idx[1:-1][hits]


def s_at(peaks: np.ndarray, period: float) -> float:
    """Rayleigh phase-concentration statistic at a fixed period."""
    return float(np.abs(np.exp(2j * np.pi * peaks / period).sum()) / peaks.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--hi", type=int, default=1000)
    args = ap.parse_args()

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
                if "4" not in h["data"]:
                    rows.append({"file": name, "t_utc": iso(ts),
                                "phase": phase_of(ts), "tx_on": None,
                                "note": "input 4 absent"})
                    continue
                a = np.asarray(h["data"]["4"][:], dtype=np.float64)
                a[a < 0] += 2 ** 32
                spec = np.median(a, axis=0)
                freqs = np.asarray(h["header"]["freqs"][:], dtype=float)
        except Exception as e:                  # noqa: BLE001
            rows.append({"file": name, "t_utc": iso(ts), "phase": phase_of(ts),
                        "tx_on": None, "note": f"read error: {e}"})
            print(f"  ! {name}: {e}", file=sys.stderr)
            continue

        peaks = find_tones(spec, args.lo, args.hi)
        if peaks.size < MIN_TONES:
            rows.append({"file": name, "t_utc": iso(ts), "phase": phase_of(ts),
                        "n_tones": int(peaks.size), "tx_on": False,
                        "note": "too few tones to test"})
            continue

        stx = s_at(peaks, P_TX)
        off_mhz = (freqs[peaks] % 1.0 + 0.5) % 1.0 - 0.5
        rms = float(np.std(off_mhz))
        tx_on = bool(stx > S_TX_THRESHOLD and rms < RMS_THRESHOLD_MHZ)
        rows.append({
            "file": name, "t_utc": iso(ts), "phase": phase_of(ts),
            "n_tones": int(peaks.size), "S_tx": round(stx, 4),
            "rms_offset_mhz": round(rms, 4), "tx_on": tx_on,
        })

        if args.summary and i % 500 == 0:
            print(f"  ...{i}/{len(files)}", file=sys.stderr)

    with JSONL_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_on = sum(1 for r in rows if r.get("tx_on") is True)
    n_off = sum(1 for r in rows if r.get("tx_on") is False)
    n_na = len(rows) - n_on - n_off
    print(f"{len(rows)} files -> {JSONL_OUT.relative_to(CAMPAIGN_ROOT)}")
    print(f"  tx_on=True: {n_on}   tx_on=False: {n_off}   unresolved: {n_na}")

    if args.summary:
        # Contiguous on/off windows, for a human-readable timeline.
        def key(r):
            return (r.get("tx_on"), r["phase"])
        groups = []
        cur = None
        for r in rows:
            k = key(r)
            if cur is None or k != cur[0]:
                if cur is not None:
                    groups.append(cur)
                cur = (k, [r])
            else:
                cur[1].append(r)
        if cur is not None:
            groups.append(cur)
        print(f"\n  {len(groups)} contiguous tx_on windows:")
        for (on, phase), grp in groups:
            if on is not True:
                continue
            print(f"    {grp[0]['t_utc']} -> {grp[-1]['t_utc']}  "
                  f"phase {phase}  {len(grp)} files")


if __name__ == "__main__":
    main()
