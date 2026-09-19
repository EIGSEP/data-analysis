"""Decide *which* comb beam-analyst's per-file ON/OFF map is tracking.

Why this exists
---------------
`beam-analyst` measured a per-file comb-presence map across the 227 files of
the 07-17 18:51 -> 07-18 03:22 beam-scan window and reported 65 files "comb
off", including a contiguous ~2 h block. The detector was deliberately
*spacing-agnostic* (count channels in 480-800 whose second difference exceeds
20 MADs), which makes it blind to comb identity: it fires on any periodic
tone structure.

INDEX.md's settled position is that there are two combs in this campaign and
the beam-scan window contains only the *digital self-comb* (8.000 ch, locked,
residue 0, ours), the TX beam-mapping comb (1.000 MHz, 4.096 ch, walks) having
been on only 07-16 01:18-16:51. If that is right, a spacing-agnostic detector
run over the beam-scan window is a **self-comb** presence map wearing a TX
label, and the archive must not ingest it as `tx_comb`.

This script settles it by running the archive's own estimator
(`scan_tx_comb.py`: continuous periodogram + channel-lock test) on the exact
ON and OFF file groups beam-analyst labelled, per file rather than per era.

Reported per file:
  period_ch / S      continuous-periodogram fundamental and its Rayleigh score
  lock_frac/phase_ch channel-lock fraction and modal residue (0 => includes DC)
  mod8_contrast      median on-comb excess at ch % 8 == 0 (self-comb strength)
  mod4096_S          Rayleigh score at the *walking* TX period 4.096 ch
  identified_as      name from scan_tx_comb.REFERENCE, or "unidentified"

Units: channels (250/1024 = 0.244140625 MHz each). int32 wraps repaired before
the median, as everywhere else in this archive.

Output: curation/beamscan_comb_identity.jsonl via --json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scan_tx_comb import (  # noqa: E402
    CHAN_MHZ,
    comb_contrast,
    find_tones,
    fundamental,
    identify,
    median_spectrum,
    periodogram,
    top_peaks,
)


def rayleigh_at(peak_ch, period):
    """Phase concentration at one fixed real-valued period."""
    if peak_ch.size == 0:
        return 0.0
    return float(np.abs(np.exp(2j * np.pi * peak_ch / period).sum()) / peak_ch.size)


def analyse_file(path, key, lo, hi, snr):
    spec, _ = median_spectrum([path], key=key)
    rec = {"file": Path(path).name, "key": key}
    if spec is None:
        rec["verdict"] = f"key {key} absent"
        return rec
    rec["mod8_contrast"] = comb_contrast(spec, 8, 0)
    peaks = find_tones(spec, lo, hi, snr=snr)
    rec["n_tones"] = int(peaks.size)
    # TX comb is 1.000 MHz = 4.096 ch and does NOT lock to the grid, so it is
    # scored at its known period rather than required to win the periodogram.
    rec["mod4096_S"] = round(rayleigh_at(peaks, 1.000 / CHAN_MHZ), 3)
    rec["mod8_S"] = round(rayleigh_at(peaks, 8.0), 3)
    if peaks.size < 8:
        rec["verdict"] = "too few tones for a period estimate"
        return rec
    P, S = periodogram(peaks)
    picks = top_peaks(P, S)
    best_p, best_s, harmonic_ok, _ = fundamental(picks)
    step = int(round(best_p))
    resid = peaks % step if step > 1 else np.zeros_like(peaks)
    counts = np.bincount(resid, minlength=max(step, 1))
    rec.update(
        period_ch=round(best_p, 4),
        period_mhz=round(best_p * CHAN_MHZ, 6),
        S=round(best_s, 3),
        lock_frac=round(float(counts.max() / peaks.size), 3),
        phase_ch=int(counts.argmax()),
        harmonic_consistent=harmonic_ok,
    )
    rec["channel_locked"] = bool(
        abs(best_p - step) < 0.02 and rec["lock_frac"] > 0.6
    )
    rec["identified_as"] = (
        "no coherent comb (peaks not harmonic)"
        if not harmonic_ok
        else identify(best_p, rec["lock_frac"], rec["phase_ch"])
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presence-json", required=True,
                    help="beam-analyst's comb_presence_beam_scan.json")
    ap.add_argument("--data-dir", default=str(HERE.parent / "data"))
    ap.add_argument("--key", default="4", help="correlator input (box-air)")
    ap.add_argument("--band", type=int, nargs=2, default=(480, 800))
    ap.add_argument("--snr", type=float, default=8.0)
    ap.add_argument("--n-per-group", type=int, default=6,
                    help="files sampled from each contiguous ON/OFF run")
    ap.add_argument("--json", help="write JSONL here")
    args = ap.parse_args()

    recs = json.load(open(args.presence_json))["files"]
    data_dir = Path(args.data_dir)

    # Contiguous runs of equal comb_on, so each block is sampled on its own.
    runs, start = [], 0
    for i in range(1, len(recs) + 1):
        if i == len(recs) or recs[i]["comb_on"] != recs[start]["comb_on"]:
            runs.append((recs[start]["comb_on"], start, i))
            start = i

    out = []
    for state, i0, i1 in runs:
        idx = np.unique(np.linspace(i0, i1 - 1, min(args.n_per_group, i1 - i0))
                        .round().astype(int))
        for i in idx:
            path = data_dir / recs[i]["file"]
            if not path.exists():
                continue
            rec = analyse_file(str(path), args.key, *args.band, args.snr)
            rec.update(scan_index=int(i), beam_analyst_comb_on=bool(state),
                       run=[i0, i1], t_close_utc=recs[i]["t_close_utc"],
                       comb_snr=recs[i]["comb_snr"])
            out.append(rec)
            print(json.dumps(rec), flush=True)

    if args.json:
        with open(args.json, "w") as f:
            for rec in out:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
