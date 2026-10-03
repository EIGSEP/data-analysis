#!/usr/bin/env python3
"""Per-file presence of box-air's 1.000 MHz self-EMI comb, for the whole campaign.

What this measures (memo 001, section "Combs")
----------------------------------------------
Box-air's own electronics radiated a 1.000 MHz comb (teeth at integer MHz,
~10 dB, box-air only) from 07-16 01:16:57 to 16:49:03 UTC; it vanished when
the Panda computer was power-cycled. This scan detects it per file on input
4 (box-air). It is NOT the beam-mapping transmitter: the transmitter is the
8-channel (1.953125 MHz) comb on channels == 0 mod 8 from 07-17 15:36:22,
whose per-file state is `curation/transmitter_presence.jsonl` (beam-scan
window) and `curation/transmitter_transitions_boxgnd.jsonl` (per-integration
episodes). Before memo 001 this product was `curation/tx_presence.jsonl`
with field `tx_on`, written when the 1 MHz comb was believed to be the
transmitter; the values are unchanged, only the names.

- `curation/comb_eras.jsonl` characterizes comb type per era from 6-file
  medians; it is not a per-file table.
- `mode_table.jsonl`'s `boxair_emi` column is a separate, cruder detector of
  the same comb (from `file_state.csv`'s `comb4mhz_score`).

Method
------
`curation/scan_comb_eras.py` established the discriminator: the 1 MHz comb is
a harmonic family of peaks whose Rayleigh phase-concentration statistic S is
high at period P = 1.000/0.244140625 = 4.0961 channels, AND whose tone
centres sit on integer MHz (unlike the 8-channel transmitter comb or
unstructured RFI). Both are independent, O(n_tones) computations.

Calibrated against six known cases (07-13/14/15 "off" era samples, the 07-16
"on" era sample, and two transmitter-dominated samples):

    case                     S_1mhz  rms_offset_from_integer_MHz
    1 MHz EMI on  (07-16)    0.873   0.091
    off (worst neg)          0.168   0.286   <- max S_1mhz seen without it
    off (best neg)           0.022   0.318

Threshold: S_1mhz > 0.5 AND rms_offset < 0.15. Both margins are >2x the
worst-case separation observed in calibration, on independent statistics.

Autos are int32 and positive-definite; a negative value is an accumulator
wrap (see curation/overflow_channels.jsonl) and is repaired (+2**32) before
the median so a handful of bright wrapped samples cannot distort it.

Scans key='4' (box-air), which per data/README.md is live for the entire
campaign (present in phases A, B, and C).

Output: curation/boxair_emi_presence.jsonl: one provenance line, then one row
per file:
    {file, t_utc, phase, n_tones, S_1mhz, rms_offset_mhz, boxair_emi, note}

Usage
-----
    python curation/scan_boxair_emi_presence.py
    python curation/scan_boxair_emi_presence.py --summary --out <path>
"""

from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
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
JSONL_OUT = CAMPAIGN_ROOT / "curation" / "boxair_emi_presence.jsonl"
HERE = Path(__file__).resolve().parent

FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")
CHAN_MHZ = 250.0 / 1024.0
P_1MHZ = 1.000 / CHAN_MHZ        # 4.0961 channels
S_1MHZ_THRESHOLD = 0.5
RMS_THRESHOLD_MHZ = 0.15
MIN_TONES = 8

PHASE_PIVOTS = [("2026-07-15T00:32:17Z", "C"), ("2026-07-14T04:10:43Z", "B")]


def short_sha(repo):
    """Commit of the generator's repo, with -dirty if it has local changes."""
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                            "--", "."], capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


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

    Identical logic to scan_comb_eras.py's find_tones (kept independent here to
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
    ap.add_argument("--out", type=Path, default=JSONL_OUT,
                    help="output JSONL (default: the campaign's curation/)")
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
                                "phase": phase_of(ts), "boxair_emi": None,
                                "note": "input 4 absent"})
                    continue
                a = np.asarray(h["data"]["4"][:], dtype=np.float64)
                a[a < 0] += 2 ** 32
                spec = np.median(a, axis=0)
                freqs = np.asarray(h["header"]["freqs"][:], dtype=float)
        except Exception as e:                  # noqa: BLE001
            rows.append({"file": name, "t_utc": iso(ts), "phase": phase_of(ts),
                        "boxair_emi": None, "note": f"read error: {e}"})
            print(f"  ! {name}: {e}", file=sys.stderr)
            continue

        peaks = find_tones(spec, args.lo, args.hi)
        if peaks.size < MIN_TONES:
            rows.append({"file": name, "t_utc": iso(ts), "phase": phase_of(ts),
                        "n_tones": int(peaks.size), "boxair_emi": False,
                        "note": "too few tones to test"})
            continue

        s1 = s_at(peaks, P_1MHZ)
        off_mhz = (freqs[peaks] % 1.0 + 0.5) % 1.0 - 0.5
        rms = float(np.std(off_mhz))
        emi = bool(s1 > S_1MHZ_THRESHOLD and rms < RMS_THRESHOLD_MHZ)
        rows.append({
            "file": name, "t_utc": iso(ts), "phase": phase_of(ts),
            "n_tones": int(peaks.size), "S_1mhz": round(s1, 4),
            "rms_offset_mhz": round(rms, 4), "boxair_emi": emi,
        })

        if args.summary and i % 500 == 0:
            print(f"  ...{i}/{len(files)}", file=sys.stderr)

    prov = {
        "product": "boxair_emi_presence",
        "campaign": "marjum-2026-07",
        "generator": "data-analysis/scripts/marjum-2026-07/curation/"
                     "scan_boxair_emi_presence.py",
        "generator_commit": short_sha(HERE),
        "renamed_from": {"file": "curation/tx_presence.jsonl",
                         "fields": {"tx_on": "boxair_emi", "S_tx": "S_1mhz"},
                         "reason": "the 1.000 MHz comb is box-air self-EMI, "
                                   "not the transmitter (memo 001)"},
        "measures": "box-air (input 4) 1.000 MHz self-EMI comb, per file",
        "threshold": {"S_1mhz_gt": S_1MHZ_THRESHOLD,
                      "rms_offset_mhz_lt": RMS_THRESHOLD_MHZ,
                      "min_tones": MIN_TONES},
    }
    with args.out.open("w") as f:
        f.write(json.dumps({"provenance": prov}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_on = sum(1 for r in rows if r.get("boxair_emi") is True)
    n_off = sum(1 for r in rows if r.get("boxair_emi") is False)
    n_na = len(rows) - n_on - n_off
    print(f"{len(rows)} files -> {args.out}")
    print(f"  boxair_emi=True: {n_on}   boxair_emi=False: {n_off}   unresolved: {n_na}")

    if args.summary:
        # Contiguous on/off windows, for a human-readable timeline.
        def key(r):
            return (r.get("boxair_emi"), r["phase"])
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
        print(f"\n  {len(groups)} contiguous boxair_emi windows:")
        for (on, phase), grp in groups:
            if on is not True:
                continue
            print(f"    {grp[0]['t_utc']} -> {grp[-1]['t_utc']}  "
                  f"phase {phase}  {len(grp)} files")


if __name__ == "__main__":
    main()
