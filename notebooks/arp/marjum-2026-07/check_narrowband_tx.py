"""Was the transmitter narrowband (~78/80 MHz) during the beam scan?

Field notes (fn:p181) record the TX final configuration immediately before the
scan as "Rx chain 1 @ ~80 MHz; Rx chain 6 @ 78 MHz".  If the transmitter put
out a small tone set rather than a Dirac comb, every comb-based detector is
correctly null while the TX is fully on, and the beam map would be keyed to
those tones rather than to comb structure.

Note ch320 = 78.125 MHz and ch328 = 80.078 MHz are both multiples of 8, so
they sit on the every-8th selection grid and could be carrying the fit without
anyone noticing.

Three questions, in order of how decisive they are:
  1. which channels actually carry the fit's leverage (pointing-modulated
     power), in raw channel terms;
  2. is there narrowband structure near 78/80 MHz in the scan window;
  3. does it modulate with pointing the way a transmitter must.
"""

import argparse

import numpy as np

from eigsep_data.beam_mapping.diagnostics import load_v007_data

CONSENSUS = [504, 520, 528, 536, 544, 552, 560, 568, 576, 584]
CELL_OFFSET = 1000


def cells(az, el, cell_deg):
    return ((np.round(az / cell_deg).astype(int) + CELL_OFFSET) * 10000
            + np.round(el / cell_deg).astype(int) + CELL_OFFSET)


def modulation_f(y, idx, min_per_cell=8):
    uniq, inv = np.unique(idx, return_inverse=True)
    counts = np.bincount(inv, minlength=uniq.size)
    keep = counts >= min_per_cell
    if keep.sum() < 5:
        return np.nan
    sums = np.bincount(inv, weights=y, minlength=uniq.size)
    means = np.zeros(uniq.size)
    means[counts > 0] = sums[counts > 0] / counts[counts > 0]
    within = np.bincount(inv, weights=(y - means[inv]) ** 2,
                         minlength=uniq.size)
    var_w = within[keep].sum() / max(int(counts[keep].sum() - keep.sum()), 1)
    grand = np.average(means[keep], weights=counts[keep])
    var_b = (np.sum(counts[keep] * (means[keep] - grand) ** 2)
             / max(int(keep.sum()) - 1, 1))
    return var_b / max(var_w, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cell-deg", type=float, default=15.0)
    args = ap.parse_args()

    data = load_v007_data(args.data, require_comb=False)
    real = data["times"] > 0
    tx = data["measured_tx"][real].astype(float)
    freqs = data["freqs"]
    idx = cells(data["az_deg"][real], data["el_deg"][real], args.cell_deg)

    print("=== Check 1: where is the fit's leverage, in raw channels? ===")
    amp = np.median(np.abs(tx), axis=0)
    F = np.array([modulation_f(tx[:, c], idx) for c in range(1, 1023)])
    chan = np.arange(1, 1023)
    # leverage ~ how much pointing-dependent power a channel contributes
    lev = amp[1:1023] * np.nan_to_num(F)
    order = np.argsort(lev)[::-1][:15]
    print(f"{'ch':>5} {'MHz':>8} {'%8':>3} {'median|tx|':>12} {'F(pointing)':>12} "
          f"{'leverage':>12}")
    for i in order:
        c = chan[i]
        print(f"{c:5d} {freqs[c]:8.3f} {c % 8:3d} {amp[c]:12.4g} "
              f"{F[i]:12.2f} {lev[i]:12.4g}")
    tot = np.nansum(lev)
    on8 = np.nansum(lev[(chan % 8) == 0])
    print(f"\nshare of total pointing-modulated leverage on channels ≡0 mod 8: "
          f"{100 * on8 / tot:.2f}%")

    print("\n=== Check 2: narrowband structure near 78 / 80 MHz ===")
    lo, hi = 280, 360
    band = np.median(np.abs(tx[:, lo:hi]), axis=0)
    scale = 1.4826 * np.median(np.abs(band - np.median(band)))
    print(f"channels {lo}-{hi} ({freqs[lo]:.1f}-{freqs[hi]:.1f} MHz), "
          f"MAD {scale:.4g}")
    print(f"{'ch':>5} {'MHz':>8} {'%8':>3} {'median|tx|':>12} {'MADs':>8} "
          f"{'F(pointing)':>12}")
    for c in range(lo, hi):
        if band[c - lo] > 8 * scale:
            print(f"{c:5d} {freqs[c]:8.3f} {c % 8:3d} {band[c - lo]:12.4g} "
                  f"{band[c - lo] / scale:8.1f} {F[c - 1]:12.2f}")

    print("\n=== the fit's own consensus channels ===")
    print(f"{'ch':>5} {'MHz':>8} {'median|tx|':>12} {'F(pointing)':>12}")
    for c in CONSENSUS:
        print(f"{c:5d} {freqs[c]:8.3f} {amp[c]:12.4g} {F[c - 1]:12.2f}")
    print(f"\nThe consensus set spans {freqs[CONSENSUS[0]]:.1f}-"
          f"{freqs[CONSENSUS[-1]]:.1f} MHz -- nowhere near 78/80 MHz.")


if __name__ == "__main__":
    main()
