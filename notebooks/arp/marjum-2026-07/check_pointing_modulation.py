"""Is the 8-channel comb in the v007 fit window a sky source or self-RFI?

data-archivist reports the 1.953125 MHz / 8-channel comb present during
07-17/18 is a digital self-comb from our own electronics, seen on internal
loads as well as the antenna, and that the TX comb (1.000 MHz, walking) was
off for the whole beam-scan window.  If that is right, the v007 beam fit has
been fitting self-RFI and is worthless.

Self-RFI does not care where the dish points.  A transmitter on a ridge, seen
through the antenna beam, must modulate strongly with pointing AND must put
every channel's maximum in the *same* direction.  Two tests:

  1. between-cell vs within-cell variance over an (az, el) grid -- an F-like
     statistic that is ~1 for signal with no pointing dependence;
  2. agreement of the peak direction across independent channels.

Control channels: off-comb channels, and the known strong RFI carriers.
"""

import argparse

import numpy as np

from eigsep_data.beam_mapping.diagnostics import load_v007_data


CELL_OFFSET = 1000       # keeps negative az/el out of the packed index


def cell_index(az, el, cell_deg):
    return ((np.round(az / cell_deg).astype(int) + CELL_OFFSET) * 10000
            + np.round(el / cell_deg).astype(int) + CELL_OFFSET)


def cell_to_azel(cell, cell_deg):
    return ((cell // 10000 - CELL_OFFSET) * cell_deg,
            (cell % 10000 - CELL_OFFSET) * cell_deg)


def modulation_stats(y, cells, min_per_cell=8):
    """Between-cell / within-cell variance, and the peak cell."""
    uniq, inv = np.unique(cells, return_inverse=True)
    counts = np.bincount(inv, minlength=uniq.size)
    keep = counts >= min_per_cell
    if keep.sum() < 5:
        return np.nan, None, 0
    sums = np.bincount(inv, weights=y, minlength=uniq.size)
    means = np.full(uniq.size, np.nan)
    means[counts > 0] = sums[counts > 0] / counts[counts > 0]
    within = np.bincount(inv, weights=(y - means[inv]) ** 2,
                         minlength=uniq.size)
    dof_w = max(int(counts[keep].sum() - keep.sum()), 1)
    var_within = within[keep].sum() / dof_w
    grand = np.average(means[keep], weights=counts[keep])
    var_between = (np.sum(counts[keep] * (means[keep] - grand) ** 2)
                   / max(int(keep.sum()) - 1, 1))
    peak = uniq[keep][np.nanargmax(means[keep])]
    return var_between / max(var_within, 1e-30), peak, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cell-deg", type=float, default=5.0)
    args = ap.parse_args()

    data = load_v007_data(args.data, require_comb=False)
    real = data["times"] > 0
    az = data["az_deg"][real]
    el = data["el_deg"][real]
    tx = data["measured_tx"][real]
    freqs = data["freqs"]
    cells = cell_index(az, el, args.cell_deg)
    print(f"{real.sum()} spectra, az {az.min():.1f}..{az.max():.1f} deg, "
          f"el {el.min():.1f}..{el.max():.1f} deg, "
          f"{np.unique(cells).size} cells of {args.cell_deg} deg")

    groups = {
        "comb, strong phase (ch%16==8)": [c for c in range(488, 800, 16)],
        "comb, weak phase (ch%16==0)": [c for c in range(480, 800, 16)],
        "off-comb control (ch%8==4)": [c for c in range(484, 800, 8)],
        "RFI carriers control": [126, 135, 386, 547, 638, 908],
    }

    peaks = {}
    for label, channels in groups.items():
        rows = []
        for ch in channels:
            f, peak, ncell = modulation_stats(tx[:, ch].astype(float), cells)
            rows.append((ch, f, peak))
        fs = np.array([r[1] for r in rows], float)
        good = np.isfinite(fs)
        print(f"\n--- {label} ({good.sum()} channels) ---")
        print(f"  between/within variance ratio: "
              f"median {np.nanmedian(fs):.2f}, "
              f"10-90% [{np.nanpercentile(fs, 10):.2f}, "
              f"{np.nanpercentile(fs, 90):.2f}]")
        pk = [r[2] for r in rows if r[2] is not None]
        if pk:
            vals, counts = np.unique(pk, return_counts=True)
            top = vals[np.argmax(counts)]
            frac = counts.max() / len(pk)
            taz, tel = cell_to_azel(top, args.cell_deg)
            pazel = np.array([cell_to_azel(c, args.cell_deg) for c in pk])
            med = np.median(pazel, axis=0)
            spread = np.median(np.hypot(*(pazel - med).T))
            print(f"  most common peak cell: az {taz:.0f} el {tel:.0f} deg, "
                  f"shared by {counts.max()}/{len(pk)} channels ({100 * frac:.0f}%)")
            print(f"  median peak direction: az {med[0]:.0f} el {med[1]:.0f} deg; "
                  f"median angular scatter about it {spread:.0f} deg")
            peaks[label] = (taz, tel, frac)
        for ch, f, peak in rows[:6]:
            if peak is None:
                continue
            paz, pel = cell_to_azel(peak, args.cell_deg)
            print(f"    ch{ch} ({freqs[ch]:7.2f} MHz): F={f:9.2f}  "
                  f"peak az {paz:6.0f} el {pel:6.0f}")

    print("\n=== verdict inputs ===")
    print("  A sky/TX source: F >> 1 on comb channels, and independent "
          "channels agree on one peak direction.")
    print("  Self-RFI:        F ~ 1 on comb channels, peak directions "
          "scattered like the controls.")


if __name__ == "__main__":
    main()
