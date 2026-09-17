"""Do alternate TX comb tones differ in amplitude, on *our* fit window?

data-archivist measured a 4.63x strong/weak split between alternate tones of
the 8-channel comb (a "strong sub-comb" 16 channels apart) in one file, one
correlator input, one 30 MHz band, and asked for it to be re-measured on the
actual v007 fit window before anyone relies on it.

This measures tone amplitude by channel parity (channel % 16) across the
comb-ON files of the fit slice, per sub-band, on the input the fit uses.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from eigsep_observing import io

from eigsep_data.beam_mapping.diagnostics import comb_present


def tone_amplitudes(spec, lo, hi, step=8):
    """Second-difference amplitude at each comb tone, and its channel."""
    second = np.zeros_like(spec)
    second[1:-1] = spec[1:-1] - 0.5 * (spec[:-2] + spec[2:])
    # lock the comb phase to the brightest tone in the band
    grid0 = np.arange(lo, hi, step)
    best = max(range(step),
               key=lambda o: np.median(second[np.clip(grid0 + o, 0, 1023)]))
    grid = np.clip(grid0 + best, 0, 1023)
    return grid, second[grid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--key", default="4", help="correlator input the fit uses")
    ap.add_argument("--all-inputs", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))[-185:-150]

    bands = [("56-85 MHz (archivist's band)", 232, 350),
             ("85-117 MHz", 350, 480),
             ("117-156 MHz (fit band lo)", 480, 640),
             ("156-195 MHz (fit band hi)", 640, 800),
             ("195-240 MHz", 800, 984)]

    keys = [args.key]
    if args.all_inputs:
        dat, _, _ = io.read_hdf5(files[0])
        keys = sorted(k for k in dat if str(k).isdigit())

    for key in keys:
        specs = []
        for filename in files:
            dat, header, _ = io.read_hdf5(filename)
            auto = np.asarray(dat[key]).astype(np.float64)
            auto[auto < 0] += 2 ** 32          # repair wrap before measuring
            spec = np.median(auto, axis=0)
            if comb_present(spec):
                specs.append(spec)
        if not specs:
            print(f"\ninput {key}: no comb-on files")
            continue
        spec = np.median(np.stack(specs), axis=0)
        print(f"\n=== input {key}: {len(specs)} comb-on files of {len(files)} ===")
        print(f"{'band':<28} {'n':>4} {'ch%16==a':>11} {'ch%16==b':>11} "
              f"{'ratio':>7} {'overlap':>8}")
        for label, lo, hi in bands:
            grid, amp = tone_amplitudes(spec, lo, hi)
            keep = amp > 0
            grid, amp = grid[keep], amp[keep]
            if grid.size < 6:
                print(f"{label:<28} {grid.size:>4}  too few tones")
                continue
            a = amp[(grid % 16) == (grid[0] % 16)]
            b = amp[(grid % 16) != (grid[0] % 16)]
            if a.size < 3 or b.size < 3:
                print(f"{label:<28} {grid.size:>4}  parity split too small")
                continue
            ma, mb = np.median(a), np.median(b)
            strong, weak = (ma, mb) if ma >= mb else (mb, ma)
            # do the two populations actually separate, or just scatter?
            overlap = float(np.mean(np.min(b) <= a)) if ma >= mb else \
                float(np.mean(np.min(a) <= b))
            lo_s, hi_s = np.percentile(a if ma >= mb else b, [10, 90])
            lo_w, hi_w = np.percentile(b if ma >= mb else a, [10, 90])
            sep = "clean" if hi_w < lo_s else "OVERLAPS"
            print(f"{label:<28} {grid.size:>4} {ma:11.3g} {mb:11.3g} "
                  f"{strong / max(weak, 1e-30):7.2f} {sep:>8}")
            print(f"{'':<28} strong 10-90%: [{lo_s:.3g}, {hi_s:.3g}]   "
                  f"weak 10-90%: [{lo_w:.3g}, {hi_w:.3g}]")

    print("\nNote: each channel carries its own free gain in score_channel and "
          "in the joint fit, and measured_sigma is radiometric (proportional "
          "to tone power), so a pure amplitude offset is absorbed. A parity "
          "split only bites where TX gain vs frequency is forced to be smooth.")


if __name__ == "__main__":
    main()
