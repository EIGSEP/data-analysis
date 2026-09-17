"""How much does int32 accumulator wrap actually corrupt the v007 beam fit?

``load_v007_data`` forms ``measured_tx[c] = auto[c] - 0.5*(auto[c-1]+auto[c+1])``,
so a wrapped sample at channel c contaminates the *three* channels c-1, c, c+1.
This script replays that construction on the exact file slice the fit uses and
counts corrupted (time, channel) entries at the channels we actually fit,
then measures the comb spacing in the fit band to test the 1.0010 MHz claim.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from eigsep_observing import io

CONSENSUS = [504, 520, 528, 536, 544, 552, 560, 568, 576, 584]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", type=int, default=-185)
    ap.add_argument("--stop", type=int, default=-150)
    ap.add_argument("--fit-lo", type=int, default=480)
    ap.add_argument("--fit-hi", type=int, default=800)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))[args.start:args.stop]
    raws, freqs = [], None
    for filename in files:
        dat, header, _ = io.read_hdf5(filename)
        raws.append(np.asarray(dat["4"]))
        if freqs is None:
            freqs = np.asarray(header["freqs"], float)
    raw = np.concatenate(raws, axis=0)
    df = freqs[1] - freqs[0]
    print(f"{raw.shape[0]} spectra x {raw.shape[1]} channels, dtype {raw.dtype}")

    wrapped = raw < 0
    # repaired: single int32 wrap -> add 2**32 and read as uint32
    repaired = raw.astype(np.float64)
    repaired[wrapped] += 2 ** 32

    def second_diff(a):
        out = np.zeros_like(a)
        out[:, 1:-1] = a[:, 1:-1] - 0.5 * (a[:, :-2] + a[:, 2:])
        return out

    tx_bad = second_diff(raw.astype(np.float64))
    tx_ok = second_diff(repaired)

    # a channel's measured_tx is contaminated if c-1, c, or c+1 wrapped
    contam = np.zeros_like(wrapped)
    contam[:, 1:-1] = wrapped[:, 1:-1] | wrapped[:, :-2] | wrapped[:, 2:]

    sel = np.arange(504, 785, 8)
    print("\n=== contamination at the fitted comb channels ===")
    print(f"{sel.size} fitted channels, {raw.shape[0]} times = "
          f"{sel.size * raw.shape[0]} samples")
    n_c = int(contam[:, sel].sum())
    print(f"contaminated samples: {n_c} "
          f"({100 * n_c / (sel.size * raw.shape[0]):.4f}%)")
    hit = sel[contam[:, sel].any(axis=0)]
    print(f"affected fitted channels: {hit}  ({freqs[hit].round(2)} MHz)")
    print(f"  of which in the default consensus set: "
          f"{sorted(set(hit.tolist()) & set(CONSENSUS))}")
    print(f"directly wrapped fitted channels (c itself): "
          f"{sel[wrapped[:, sel].any(axis=0)]}")

    print("\n=== size of the error these samples inject ===")
    for c in hit:
        m = contam[:, c]
        good = ~m
        typ = np.median(np.abs(tx_ok[good, c]))
        err = tx_bad[m, c] - tx_ok[m, c]
        print(f"  ch{c} ({freqs[c]:7.3f} MHz): {int(m.sum())} bad samples; "
              f"median |signal| = {typ:.4g}; "
              f"injected error median {np.median(np.abs(err)):.4g} "
              f"= {np.median(np.abs(err)) / max(typ, 1e-30):.1f}x signal; "
              f"sign of error: {np.sign(np.median(err)):+.0f}")

    print("\n=== comb spacing in the fit band "
          f"(ch {args.fit_lo}-{args.fit_hi}) ===")
    spec = np.median(repaired, axis=0)
    resid = np.zeros_like(spec)
    resid[1:-1] = spec[1:-1] - 0.5 * (spec[:-2] + spec[2:])
    band = slice(args.fit_lo, args.fit_hi)
    r = resid[band]
    scale = np.median(np.abs(r - np.median(r)))
    idx = np.arange(args.fit_lo, args.fit_hi)
    peaks = idx[1:-1][(r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:])
                      & (r[1:-1] > 20 * scale)]
    print(f"{peaks.size} peaks > 20 MAD in band")
    if peaks.size > 1:
        f0 = freqs[peaks]
        dfq = np.diff(f0)
        print(f"consecutive spacings (MHz): {np.unique(dfq.round(4))}")
        k = np.rint((f0 - f0[0]) / np.median(dfq))
        slope, icept = np.polyfit(k, f0, 1)
        rms = np.std(f0 - (slope * k + icept))
        print(f"comb spacing = {slope:.6f} MHz "
              f"({slope / df:.4f} channels), fit rms {rms * 1e3:.2f} kHz")
        for hyp, name in [(1.0010, "rfi-analyst 1.0010 MHz"),
                          (2.0020, "2 x 1.0010 MHz"),
                          (8 * df, "8 channels")]:
            kk = np.rint((f0 - f0[0]) / hyp)
            pred = f0[0] + kk * hyp
            print(f"  vs {name:24s}: max |peak - predicted| = "
                  f"{np.abs(f0 - pred).max() * 1e3:8.1f} kHz "
                  f"({np.abs(f0 - pred).max() / df:.2f} channels)")

    print("\n=== is there a weaker comb between the strong tones? ===")
    between = np.setdiff1d(idx, np.concatenate([peaks + o for o in (-1, 0, 1)]))
    print(f"median |resid| on strong tones: {np.median(resid[peaks]):.4g}")
    print(f"median |resid| off tones:       {np.median(np.abs(resid[between])):.4g}")
    sub = between[np.abs(resid[between]) > 5 * scale]
    print(f"off-tone channels > 5 MAD: {sub.size} -> {sub[:30]}")
    if sub.size > 1:
        print(f"  their freqs (MHz): {freqs[sub[:30]].round(3)}")


if __name__ == "__main__":
    main()
