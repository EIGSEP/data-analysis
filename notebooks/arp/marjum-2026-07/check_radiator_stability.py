"""Is the self-comb radiator a fixed ground source we could beam-map against?

A source that modulates with pointing is not co-rotating with the platform: it
is fixed in the topocentric frame.  If so, the v007 machinery measured a real
antenna response to a real external source, and the fitted "TX heading" is an
estimate of *that radiator's* direction.  Whether it is usable as a beam probe
turns on one testable question: does the fitted direction stay put?

Fits the heading in independent time blocks across the beam scan and reports
elevation and azimuth separately, because they are not equally trustworthy.
Elevation is absolutely referenced; azimuth is not -- imu_az is dead and the
motor azimuth suffers a documented slip episode at 20:41:24 that loses 27.4
deg in 12 minutes.  So **elevation stability is the real test**; azimuth is
expected to walk and is reported only for completeness.
"""

import argparse

import numpy as np

from eigsep_data.beam_mapping import HFSSBeamSet
from eigsep_data.beam_mapping.diagnostics import (
    fit_v007_beam_joint,
    load_v007_data,
)

CONSENSUS = [504, 520, 528, 536, 544, 552, 560, 568, 576, 584]


def heading_angles(h):
    """Elevation above horizon and azimuth of a unit heading vector."""
    h = np.asarray(h, float)
    el = np.degrees(np.arcsin(np.clip(-h[2], -1, 1)))
    az = np.degrees(np.arctan2(h[1], h[0]))
    return el, az


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--beam", default="../../../hfss_beam_maps/bowtie_beam.npz")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--first", type=int, default=-227)
    ap.add_argument("--last", type=int, default=-5)
    args = ap.parse_args()

    beam = HFSSBeamSet.from_npz(args.beam)
    span = args.last - args.first
    width = span // args.blocks
    print(f"fitting {args.blocks} blocks of {width} files across the beam scan")
    print(f"{'block':<7} {'files':<14} {'n':>7} {'el (deg)':>10} "
          f"{'az (deg)':>10} {'alpha':>8} {'nrms':>8}")

    els, azs, rows = [], [], []
    for b in range(args.blocks):
        lo = args.first + b * width
        hi = lo + width
        try:
            data = load_v007_data(args.data, start=lo, stop=hi)
            n = int((data["times"] > 0).sum())
            if n < 500:
                print(f"{b:<7} [{lo}:{hi}]     {n:>7}  too few valid spectra")
                continue
            fit = fit_v007_beam_joint(data, beam, beam_channels=CONSENSUS)
        except Exception as exc:
            print(f"{b:<7} [{lo}:{hi}]  failed: {exc}")
            continue
        el, az = heading_angles(fit.heading)
        els.append(el)
        azs.append(az)
        rows.append((b, lo, hi, n, el, az, fit.alpha_deg))
        print(f"{b:<7} [{lo}:{hi}]  {n:>7} {el:10.3f} {az:10.3f} "
              f"{fit.alpha_deg:8.3f} {getattr(fit, 'normalized_rms', float('nan')):8.4f}")

    if len(els) < 2:
        print("\nnot enough blocks fitted to judge stability")
        return
    els = np.array(els)
    azs = np.array(azs)
    print(f"\nelevation: mean {els.mean():.3f} deg, "
          f"std {els.std(ddof=1):.3f} deg, "
          f"range {els.max() - els.min():.3f} deg")
    print(f"azimuth  : mean {azs.mean():.3f} deg, "
          f"std {azs.std(ddof=1):.3f} deg, "
          f"range {azs.max() - azs.min():.3f} deg")
    print("\nA fixed ground radiator should hold elevation to well inside the")
    print("4.435 deg scan cell across the whole scan. Drift much larger than")
    print("that means the fit is not tracking one stable direction, and the")
    print("apparatus cannot be repurposed as a beam probe without first")
    print("explaining what moved.")


if __name__ == "__main__":
    main()
