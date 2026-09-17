"""Per-file TX-state detector for the 07-17/18 beam-scan window.

No product records transmitter state, so it has to be recovered from the comb
channels themselves.  Two matched detectors run side by side on each file:

  TX comb   -- spacing searched over 0.990-1.010 MHz with free phase.  The
               transmitter comb is *not* locked to the channel grid: on 07-16,
               where it is independently known to be on, it sits at 1.0000 MHz
               with an offset of 4.03 channels from DC.
  self comb -- spacing fixed at 250/128 MHz = 8.000 channels exactly, phase
               locked to DC, which is what an ADC-clock subharmonic looks like.

Both are reported with N_eff, the effective number of channels carrying the
tone excess.  Coherence alone is not evidence: when the excess sits in two or
three bright RFI lines, S is near 1 for almost any trial spacing.  A detection
requires high S *and* N_eff above threshold.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_observing import io

from comb_periodogram import effective_tones, excess_spectrum, periodogram

SELF_COMB_MHZ = 250.0 / 128          # 8.000 channels exactly
N_EFF_MIN = 10.0
S_MIN = 0.60


def analyse(spec, freqs, lo, hi, tx_grid):
    e = excess_spectrum(spec, lo, hi)
    f = freqs[lo:hi]
    n_eff = effective_tones(e)
    s_tx, ph_tx = periodogram(e, f, tx_grid)
    best = int(np.argmax(s_tx))
    s_self, ph_self = periodogram(e, f, np.array([SELF_COMB_MHZ]))
    df = freqs[1] - freqs[0]
    off_self = ((-ph_self[0] / (2 * np.pi)) * SELF_COMB_MHZ) % SELF_COMB_MHZ
    off_tx = ((-ph_tx[best] / (2 * np.pi)) * tx_grid[best]) % tx_grid[best]
    return {
        "n_eff": n_eff,
        "tx_spacing_mhz": float(tx_grid[best]),
        "tx_S": float(s_tx[best]),
        "tx_offset_ch": float(off_tx / df),
        "self_S": float(s_self[0]),
        "self_offset_ch": float(off_self / df),
    }


def classify(r):
    if r["n_eff"] < N_EFF_MIN:
        return "uncertain"
    return "on" if r["tx_S"] >= S_MIN else "off"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--key", default="4")
    ap.add_argument("--lo", type=int, default=240)
    ap.add_argument("--hi", type=int, default=960)
    ap.add_argument("--since", default="corr_20260717_185000Z")
    ap.add_argument("--until", default="corr_20260718_999999Z")
    ap.add_argument("--output-json", default="tx_state_beam_scan.json")
    args = ap.parse_args()

    tx_grid = np.arange(0.990, 1.010, 1e-5)
    files = [f for f in sorted(glob.glob(str(Path(args.data) / "*.h5")))
             if args.since <= Path(f).name <= args.until]
    print(f"{len(files)} files, input {args.key}, channels "
          f"{args.lo}-{args.hi}")

    rows = []
    for filename in files:
        dat, header, _ = io.read_hdf5(filename)
        a = np.asarray(dat[args.key]).astype(np.float64)
        a[a < 0] += 2 ** 32
        spec = np.median(a, axis=0)
        freqs = np.asarray(header["freqs"], float)
        r = analyse(spec, freqs, args.lo, args.hi, tx_grid)
        r["file"] = Path(filename).name
        r["tx_state"] = classify(r)
        rows.append(r)

    states = [r["tx_state"] for r in rows]
    print(f"\nTX state: on={states.count('on')}  off={states.count('off')}  "
          f"uncertain={states.count('uncertain')}  of {len(rows)}")

    conf = [r for r in rows if r["n_eff"] >= N_EFF_MIN]
    print(f"\nfiles with enough tone structure to judge: {len(conf)}")
    if conf:
        print(f"  best TX-band coherence : median {np.median([r['tx_S'] for r in conf]):.3f}, "
              f"max {max(r['tx_S'] for r in conf):.3f}")
        print(f"  self-comb coherence    : median {np.median([r['self_S'] for r in conf]):.3f}, "
              f"max {max(r['self_S'] for r in conf):.3f}")
        print(f"  self-comb offset from DC: median "
              f"{np.median([r['self_offset_ch'] for r in conf]):.3f} ch "
              f"(0 = locked to the channel grid)")
        hits = [r for r in conf if r["tx_S"] >= S_MIN]
        print(f"  files with a TX-band detection: {len(hits)}")
        for r in hits[:10]:
            print(f"    {r['file']}  D={r['tx_spacing_mhz']:.5f} MHz  "
                  f"S={r['tx_S']:.3f}  offset {r['tx_offset_ch']:.2f} ch  "
                  f"N_eff={r['n_eff']:.1f}")

    with open(args.output_json, "w") as stream:
        json.dump({"n_eff_min": N_EFF_MIN, "s_min": S_MIN,
                   "self_comb_mhz": SELF_COMB_MHZ,
                   "input": args.key, "files": rows}, stream, indent=2)
    print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
