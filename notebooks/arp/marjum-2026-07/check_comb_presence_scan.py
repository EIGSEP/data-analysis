"""Per-file TX comb presence across the 07-17/18 beam-scan window.

The comb spacing is 8 channels in this window, but the comb is not on for the
whole of it.  Any fit extended beyond the 35-file v007 slice needs to know
which files actually carry the comb.  Metric: median second-difference
amplitude on the 8-channel comb grid in the fit band, versus the same
statistic on the off-grid channels.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_observing import io

FIT_LO, FIT_HI = 480, 800


def comb_metrics(spec):
    sd = np.zeros_like(spec)
    sd[1:-1] = spec[1:-1] - 0.5 * (spec[:-2] + spec[2:])
    grid = np.arange(FIT_LO, FIT_HI, 8)
    off = np.setdiff1d(np.arange(FIT_LO, FIT_HI),
                       np.concatenate([grid + o for o in (-1, 0, 1)]))
    on_amp = float(np.median(sd[grid]))
    off_scale = float(1.4826 * np.median(np.abs(sd[off] - np.median(sd[off]))))
    return on_amp, off_scale, on_amp / max(off_scale, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--since", default="corr_20260717_185000Z")
    ap.add_argument("--snr", type=float, default=20.0,
                    help="comb-on threshold on median grid amplitude / off-grid MAD")
    ap.add_argument("--output-json", default="comb_presence_beam_scan.json")
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(str(Path(args.data) / "*.h5")))
             if Path(f).name >= args.since]
    print(f"{len(files)} files from {Path(files[0]).name}")

    rows = []
    for filename in files:
        dat, header, _ = io.read_hdf5(filename)
        auto = np.asarray(dat["4"]).astype(np.float64)
        n_wrapped = int((auto < 0).sum())
        auto[auto < 0] += 2 ** 32
        spec = np.median(auto, axis=0)
        on_amp, off_scale, snr = comb_metrics(spec)
        rows.append({
            "file": Path(filename).name,
            "t_close_utc": Path(filename).name[5:20],
            "comb_snr": snr,
            "comb_on": bool(snr >= args.snr),
            "n_wrapped": n_wrapped,
            "n_spectra": int(auto.shape[0]),
        })

    on = np.array([r["comb_on"] for r in rows])
    snr = np.array([r["comb_snr"] for r in rows])
    print(f"\ncomb ON in {on.sum()} / {on.size} files "
          f"({100 * on.mean():.1f}%)")
    print(f"SNR when on : median {np.median(snr[on]):.1f}" if on.any() else "")
    print(f"SNR when off: median {np.median(snr[~on]):.2f}" if (~on).any() else "")

    # contiguous runs
    print("\ncontiguous stretches:")
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    bounds = np.concatenate([[0], edges, [on.size]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        state = "ON " if on[a] else "OFF"
        print(f"  {state} files[{a:3d}:{b:3d}]  "
              f"{rows[a]['file']} .. {rows[b - 1]['file']}  "
              f"({b - a} files, median SNR {np.median(snr[a:b]):8.1f})")

    wrapped_files = sum(1 for r in rows if r["n_wrapped"] > 0)
    print(f"\nint32 wrap: {wrapped_files} / {len(rows)} files affected "
          f"({100 * wrapped_files / len(rows):.1f}%), "
          f"{sum(r['n_wrapped'] for r in rows)} samples total")
    print(f"  comb-ON files wrapping : "
          f"{100 * np.mean([r['n_wrapped'] > 0 for r, o in zip(rows, on) if o]):.1f}%")
    print(f"  comb-OFF files wrapping: "
          f"{100 * np.mean([r['n_wrapped'] > 0 for r, o in zip(rows, on) if not o]):.1f}%")

    with open(args.output_json, "w") as stream:
        json.dump({"snr_threshold": args.snr, "files": rows}, stream, indent=2)
    print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
