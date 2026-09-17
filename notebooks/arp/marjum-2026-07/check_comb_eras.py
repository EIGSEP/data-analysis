"""Measure the TX comb spacing per era, on the raw spectrum.

The second-difference baseline subtraction used by ``load_v007_data`` nulls a
comb whose tones sit in adjacent channels, so comb identification is done here
on the raw median spectrum instead, via the autocorrelation of the
tone-detection mask and a least-squares fit to the detected tone frequencies.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from eigsep_observing import io


def median_spectrum(files, key="4"):
    specs, freqs = [], None
    for filename in files:
        dat, header, _ = io.read_hdf5(filename)
        auto = np.asarray(dat[key]).astype(np.float64)
        auto[auto < 0] += 2 ** 32          # undo single int32 wraps
        specs.append(np.median(auto, axis=0))
        if freqs is None:
            freqs = np.asarray(header["freqs"], float)
    return np.median(np.stack(specs), axis=0), freqs


def find_tones(spec, lo, hi, n_med=21, snr=8.0):
    """Peaks standing above a running-median continuum."""
    idx = np.arange(lo, hi)
    band = spec[lo:hi]
    pad = n_med // 2
    padded = np.pad(band, pad, mode="edge")
    cont = np.array([np.median(padded[i:i + n_med]) for i in range(band.size)])
    excess = band - cont
    scale = 1.4826 * np.median(np.abs(excess - np.median(excess)))
    hits = (excess[1:-1] > excess[:-2]) & (excess[1:-1] >= excess[2:]) & \
           (excess[1:-1] > snr * max(scale, 1e-30))
    return idx[1:-1][hits], excess, scale


def comb_fit(freqs_of_peaks, guess):
    k = np.rint((freqs_of_peaks - freqs_of_peaks[0]) / guess)
    keep = np.concatenate([[True], np.diff(k) > 0])   # drop duplicate ordinals
    k, f = k[keep], freqs_of_peaks[keep]
    slope, icept = np.polyfit(k, f, 1)
    rms = float(np.std(f - (slope * k + icept)))
    return slope, rms, k.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--hi", type=int, default=1000)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))
    eras = {
        "07-12 start": files[0:6],
        "07-13": files[1000:1006],
        "07-14": files[2000:2006],
        "07-15": files[3000:3006],
        "07-16": files[4000:4006],
        "07-17 beam-scan (fit input)": files[-185:-179],
        "07-18 end": files[-6:],
    }
    for label, group in eras.items():
        spec, freqs = median_spectrum(group)
        df = freqs[1] - freqs[0]
        peaks, excess, scale = find_tones(spec, args.lo, args.hi)
        name = Path(group[0]).name
        if peaks.size < 3:
            print(f"{label:30s} {name}  <3 tones found")
            continue
        f0 = freqs[peaks]
        dch = np.diff(peaks)
        # most common spacing is the robust comb step
        step_ch = int(np.bincount(dch).argmax())
        slope, rms, n = comb_fit(f0, max(step_ch, 1) * df)
        hyps = {"1.0010 MHz": 1.0010, "1.9533 MHz": 1.9533}
        verdict = {}
        for hname, h in hyps.items():
            kk = np.rint((f0 - f0[0]) / h)
            verdict[hname] = np.abs(f0 - (f0[0] + kk * h)).max() / df
        print(f"{label:30s} {name}")
        print(f"    {peaks.size:3d} tones in ch {args.lo}-{args.hi}; "
              f"modal spacing {step_ch} ch; "
              f"LSQ comb = {slope:.6f} MHz ({slope / df:.4f} ch), "
              f"rms {rms * 1e3:6.1f} kHz, n={n}")
        print(f"    max residual vs " + ";  ".join(
            f"{k}: {v:.2f} ch" for k, v in verdict.items()))


if __name__ == "__main__":
    main()
