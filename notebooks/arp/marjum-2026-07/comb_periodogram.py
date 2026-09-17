"""Continuous comb periodogram: what comb spacings are present, really?

Estimators that find the modal *integer* channel gap between peaks can only
ever return integer multiples of the channel width, which manufactures
"channel-locked" combs.  This avoids that entirely: for a trial spacing D it
evaluates the Fourier component of the tone-excess spectrum at period D in
*frequency* space,

    S(D) = | sum_c e_c exp(2 pi i f_c / D) | / sum_c e_c

with no reference to the channel grid.  S is a coherence in [0, 1]; the phase
of the sum gives the comb's offset, so a comb sitting exactly on channel
multiples (residue 0, i.e. through DC) is distinguishable from one that walks.

Validation: run it on a 07-16 file, where the transmitter comb is independently
known to be ~1.000 MHz, before trusting it anywhere else.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from eigsep_observing import io


def excess_spectrum(spec, lo, hi):
    """Tone excess above a local continuum, clipped at zero."""
    second = np.zeros_like(spec)
    second[1:-1] = spec[1:-1] - 0.5 * (spec[:-2] + spec[2:])
    e = second[lo:hi].copy()
    e[e < 0] = 0.0
    return e


def periodogram(e, freqs, spacings):
    """Coherence S(D) and comb phase for each trial spacing."""
    total = e.sum()
    if total <= 0:
        return np.zeros_like(spacings), np.zeros_like(spacings)
    phase = 2 * np.pi * freqs[None, :] / spacings[:, None]
    z = (e[None, :] * np.exp(1j * phase)).sum(axis=1) / total
    return np.abs(z), np.angle(z)


def effective_tones(e):
    """(sum e)^2 / sum e^2 -- how many channels actually carry the excess.

    Coherence alone is meaningless when the excess sits in a handful of bright
    lines: |sum e exp(i phi)| / sum e is then near 1 for almost any trial
    spacing.  A comb claim needs N_eff large as well as S high.
    """
    e = np.asarray(e, float)
    denom = np.sum(e ** 2)
    return float(np.sum(e) ** 2 / denom) if denom > 0 else 0.0


def report(label, spec, freqs, lo, hi, spacings, top=4):
    e = excess_spectrum(spec, lo, hi)
    f = freqs[lo:hi]
    n_eff = effective_tones(e)
    S, ph = periodogram(e, f, spacings)
    # local maxima of the coherence
    peaks = np.flatnonzero((S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:])) + 1
    peaks = peaks[np.argsort(S[peaks])[::-1]][:top]
    df = freqs[1] - freqs[0]
    print(f"\n--- {label} ---")
    print(f"   N_eff = {n_eff:.1f} channels carry the excess"
          f"{'   <-- too concentrated, comb claims unreliable' if n_eff < 10 else ''}")
    if not peaks.size:
        print("   no coherent comb")
        return
    for p in sorted(peaks, key=lambda q: -S[q]):
        D = spacings[p]
        # comb offset: where the tones sit relative to DC, in channels
        offset_mhz = (-ph[p] / (2 * np.pi)) * D % D
        print(f"   D = {D:8.5f} MHz ({D / df:7.4f} ch)  S = {S[p]:.3f}   "
              f"offset {offset_mhz:7.4f} MHz "
              f"({offset_mhz / df:6.3f} ch from DC)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--key", default="4")
    ap.add_argument("--lo", type=int, default=240)
    ap.add_argument("--hi", type=int, default=960)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))
    df = 250.0 / 1024
    # fine, non-quantised grid of trial spacings
    spacings = np.arange(0.40, 5.0, 2e-5)

    def med_spec(group, key=None):
        acc = []
        for filename in group:
            dat, header, _ = io.read_hdf5(filename)
            a = np.asarray(dat[key or args.key]).astype(np.float64)
            a[a < 0] += 2 ** 32
            acc.append(np.median(a, axis=0))
        return np.median(np.stack(acc), axis=0), np.asarray(header["freqs"], float)

    def pick(name):
        return [f for f in files if Path(f).name == name]

    cases = [
        ("07-16 TX era (validation)",
         [f for f in files if "corr_20260716_1043" <= Path(f).name[5:18] <= "corr_20260716_1055"][:4]
         or files[4000:4004]),
        ("07-17 v007 fit window (8-ch comb ON)", files[-185:-181]),
        ("07-17 21:55-23:42 (8-ch comb OFF block)",
         [f for f in files if "20260717_2200" <= Path(f).name[5:18] <= "20260717_2320"][:4]),
        ("07-17 pre-self-comb 15:00 (before 15:37 onset)",
         [f for f in files if "20260717_1500" <= Path(f).name[5:18] <= "20260717_1530"][:4]),
    ]
    for label, group in cases:
        if not group:
            print(f"\n--- {label} ---\n   no files matched")
            continue
        spec, freqs = med_spec(group)
        report(f"{label}  [{Path(group[0]).name}, input {args.key}]",
               spec, freqs, args.lo, args.hi, spacings)

    print(f"\nchannel width {df:.6f} MHz; 8 ch = {8 * df:.6f} MHz; "
          f"4.096 ch = {4.096 * df:.6f} MHz")
    print("A comb at offset ~0 ch from DC is locked to the channel grid "
          "(clock-derived); one at an arbitrary offset is not.")


if __name__ == "__main__":
    main()
