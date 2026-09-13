"""Deep stack at the transmitter's *own* line channels across the beam scan.

Two objections to the "no TX in the beam scan" result deserve a quantitative
answer rather than an argument:

  narrowband -- the field notes describe a narrowband TX at ~78/80 MHz before
                the scan, which every comb detector would correctly miss;
  power      -- the scan ran with 13 dB attenuation, so a detector calibrated
                on the unattenuated 07-16 episode might simply sit above it.

Both are answered by ignoring comb structure entirely: take the channels where
the transmitter demonstrably emits during the confirmed TX-on era, then stack
every scan integration at those same channels and ask what upper limit the
stack supports.  13 dB is a factor of 20 in power, so the stack has to be
deep enough to see 1/20 of the TX-on line amplitude.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_observing import io

TX_PRESENCE = ("/home/aparsons/projects/eigsep/marjum-2026-07/"
               "curation/tx_presence.jsonl")
BAND = (280, 370)


def second_diff(a):
    out = np.zeros_like(a)
    out[..., 1:-1] = a[..., 1:-1] - 0.5 * (a[..., :-2] + a[..., 2:])
    return out


def stack(files, key="4"):
    """Mean second-difference spectrum and its standard error."""
    total = None
    total_sq = None
    n = 0
    for filename in files:
        dat, header, _ = io.read_hdf5(filename)
        a = np.asarray(dat[key]).astype(np.float64)
        a[a < 0] += 2 ** 32
        d = second_diff(a)
        total = d.sum(axis=0) if total is None else total + d.sum(axis=0)
        total_sq = (d ** 2).sum(axis=0) if total_sq is None else total_sq + (d ** 2).sum(axis=0)
        n += d.shape[0]
    mean = total / n
    var = np.maximum(total_sq / n - mean ** 2, 0.0)
    return mean, np.sqrt(var / n), n, np.asarray(header["freqs"], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tx-presence", default=TX_PRESENCE)
    ap.add_argument("--attenuation-db", type=float, default=13.0)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))
    rows = [json.loads(line) for line in open(args.tx_presence)]
    on_names = {Path(r["file"]).name for r in rows if r["tx_on"]}
    tx_on = [f for f in files if Path(f).name in on_names][:40]
    scan = [f for f in files if Path(f).name >= "corr_20260717_185000Z"]

    print(f"TX-on stack : {len(tx_on)} files")
    m_on, e_on, n_on, freqs = stack(tx_on)
    print(f"              {n_on} integrations")
    print(f"scan stack  : {len(scan)} files")
    m_sc, e_sc, n_sc, _ = stack(scan)
    print(f"              {n_sc} integrations")

    lo, hi = BAND
    # the transmitter's own lines: significant in the TX-on stack
    snr_on = m_on[lo:hi] / np.maximum(e_on[lo:hi], 1e-30)
    lines = np.arange(lo, hi)[snr_on > 20]
    # drop channels adjacent to a stronger line (second-difference sidelobes)
    keep = [c for c in lines
            if m_on[c] >= m_on[c - 1] and m_on[c] >= m_on[c + 1]]
    print(f"\nTX lines identified in the TX-on stack: {keep}")
    print(f"   = {np.round(freqs[keep], 3)} MHz")

    atten = 10 ** (args.attenuation_db / 10)
    print(f"\n{'ch':>5} {'MHz':>8} | {'TX-on amp':>12} {'sigma':>10} {'SNR':>8} "
          f"| {'scan amp':>12} {'sigma':>10} {'SNR':>8} | {'pred @-13dB':>12} "
          f"{'pred SNR':>9}")
    preds, obs = [], []
    for c in keep:
        pred = m_on[c] / atten
        preds.append(pred / max(e_sc[c], 1e-30))
        obs.append(m_sc[c] / max(e_sc[c], 1e-30))
        print(f"{c:5d} {freqs[c]:8.3f} | {m_on[c]:12.4g} {e_on[c]:10.3g} "
              f"{m_on[c] / max(e_on[c], 1e-30):8.1f} | {m_sc[c]:12.4g} "
              f"{e_sc[c]:10.3g} {m_sc[c] / max(e_sc[c], 1e-30):8.1f} | "
              f"{pred:12.4g} {pred / max(e_sc[c], 1e-30):9.1f}")

    preds, obs = np.array(preds), np.array(obs)
    print(f"\nIf the TX had been on at -{args.attenuation_db:.0f} dB during the "
          f"scan, these lines would appear at SNR "
          f"{np.median(preds):.0f} (median), {preds.min():.0f} (weakest).")
    print(f"Observed in the scan stack: SNR {np.median(obs):.1f} (median), "
          f"{obs.max():.1f} (largest).")
    # joint upper limit from the combined line set
    w = 1.0 / np.maximum(e_sc[keep], 1e-30) ** 2
    comb_amp = np.sum(w * m_sc[keep]) / np.sum(w)
    comb_sig = 1.0 / np.sqrt(np.sum(w))
    on_amp = np.sum(w * m_on[keep]) / np.sum(w)
    print(f"\nInverse-variance combined over all {len(keep)} TX lines:")
    print(f"   scan  : {comb_amp:.4g} +/- {comb_sig:.4g}  "
          f"({comb_amp / comb_sig:+.1f} sigma)")
    print(f"   2-sigma upper limit on TX amplitude in the scan: "
          f"{2 * comb_sig:.4g}")
    print(f"   TX-on amplitude at the same channels: {on_amp:.4g}")
    print(f"   => TX during the scan is below {2 * comb_sig / on_amp:.3g} "
          f"of its 07-16 level "
          f"({10 * np.log10(max(2 * comb_sig / on_amp, 1e-30)):.1f} dB), "
          f"vs -{args.attenuation_db:.0f} dB claimed.")


if __name__ == "__main__":
    main()
