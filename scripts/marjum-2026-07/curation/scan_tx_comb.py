"""Identify combs per era and decide whether each is channel-locked.

Why this exists
---------------
Two wrong comb spacings have circulated for this campaign, and both came from
the same class of estimator: find the modal *integer* gap between detected
peaks, seed a least-squares fit with it, and report the slope. That estimator
is circular — the seed forces tone ordinals onto an integer-channel lattice, so
it can only ever return an integer multiple of the channel width. Run it on a
band with no comb at all and it still returns a confident "2-channel comb".

This tool uses a **continuous periodogram** over the detected tone positions
instead::

    S(P) = | sum_j exp(2*pi*i * ch_j / P) | / n_tones

P is scanned on a fine real-valued grid, so nothing anchors the answer to an
integer. S is the Rayleigh statistic for phase concentration: ~1 for a perfect
comb of period P channels, ~1/sqrt(n) for unstructured peaks. A band with no
comb produces no peak above ~0.4, which is the outcome the seeded estimator
cannot represent.

The decisive question is not the spacing but **channel lock**:

- ``lock_frac`` — fraction of tones sharing one residue modulo the rounded
  period. ~1.0 means every tone sits at a fixed channel index.
- ``phase_ch`` — that residue. A comb generated inside the signal chain is a
  harmonic family of some clock, so it includes DC and its residue is **0**.
  An external transmitter has no reason to align with either the channel grid
  or DC.

A comb that is an exact integer number of channels *and* has residue 0 is
phase-locked to the ADC sample clock and is therefore **ours**, not the sky's.
``flagging/detectors.py`` encodes the same rule; this script is the per-era
measurement behind it.

Units: MHz throughout; ``freqs`` from ``header/freqs``. Channel width is
250/1024 = 0.244140625 MHz. int32 wraps are repaired (``auto[auto<0] += 2**32``)
before the median so bright wrapped RFI does not distort the continuum.

Output: ``curation/tx_comb_eras.jsonl`` via ``--json``.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_observing import io

CHAN_MHZ = 250.0 / 1024.0

# Reference spacings, for naming a detected period. "locked" means an exact
# integer number of channels, i.e. phase-locked to the ADC clock.
REFERENCE = [
    ("tx 1.000 MHz (4.096 ch, walks)", 1.000 / CHAN_MHZ, False),
    ("laptop 2.000 MHz (8.192 ch, walks)", 2.000 / CHAN_MHZ, False),
    ("lna 1.250 MHz (5.120 ch, walks)", 1.250 / CHAN_MHZ, False),
    ("digital self 1.953125 MHz (8 ch, locked)", 8.0, True),
]

# Eras sampled as indices into sorted(glob("data/*.h5")). Six files is plenty:
# a comb is a configuration property, not a noisy measurement. Note that the
# first and last few files of the campaign are calibration (RF switch on
# internal loads), so they are labelled as such rather than as sky.
ERAS = [
    ("07-12 start (cal)", 0, 6),
    ("07-13", 1000, 1006),
    ("07-14", 2000, 2006),
    ("07-15", 3000, 3006),
    ("07-16 (TX on)", 4000, 4006),
    ("07-17 beam scan", -185, -179),
    ("07-18 end (cal)", -6, None),
]


def median_spectrum(files, key="4"):
    specs, freqs = [], None
    for filename in files:
        dat, header, _ = io.read_hdf5(filename)
        if key not in dat:
            continue
        auto = np.asarray(dat[key]).astype(np.float64)
        auto[auto < 0] += 2 ** 32          # undo single int32 wraps
        specs.append(np.median(auto, axis=0))
        if freqs is None:
            freqs = np.asarray(header["freqs"], float)
    if not specs:
        return None, None
    return np.median(np.stack(specs), axis=0), freqs


def find_tones(spec, lo, hi, n_med=21, snr=8.0):
    """Local maxima standing ``snr`` robust-sigma above a running median."""
    idx = np.arange(lo, hi)
    band = spec[lo:hi]
    pad = n_med // 2
    padded = np.pad(band, pad, mode="edge")
    cont = np.array([np.median(padded[i:i + n_med]) for i in range(band.size)])
    excess = band - cont
    scale = 1.4826 * np.median(np.abs(excess - np.median(excess)))
    hits = (excess[1:-1] > excess[:-2]) & (excess[1:-1] >= excess[2:]) & \
           (excess[1:-1] > snr * max(scale, 1e-30))
    return idx[1:-1][hits]


def periodogram(peak_ch, p_min=1.8, p_max=12.0, n=120001):
    """Rayleigh phase-concentration statistic against real-valued period."""
    P = np.linspace(p_min, p_max, n)
    S = np.abs(np.exp(2j * np.pi * peak_ch[:, None] / P[None, :]).sum(0))
    return P, S / peak_ch.size


def top_peaks(P, S, k=6, sep=0.2):
    picks = []
    for i in np.argsort(S)[::-1]:
        if all(abs(P[i] - p[0]) > sep for p in picks):
            picks.append((float(P[i]), float(S[i])))
        if len(picks) == k:
            break
    return picks


def fundamental(picks, frac=0.85):
    """Pick the fundamental, not a subharmonic alias.

    A comb of period P also concentrates phase at P/2, P/3, ..., and those
    aliases can score marginally higher than P when tones are missing. So:
    take every peak within ``frac`` of the best score, call the **largest**
    period the fundamental, and report whether the rest are integer
    submultiples of it. If they are not, the peaks are unstructured and there
    is no comb, whatever the top score says.
    """
    s_max = max(s for _, s in picks)
    strong = [(p, s) for p, s in picks if s >= frac * s_max]
    p0, s0 = max(strong, key=lambda ps: ps[0])
    ratios = [p0 / p for p, _ in strong]
    consistent = all(abs(r - round(r)) < 0.03 for r in ratios)
    return p0, s0, consistent, len(strong)


def identify(period_ch, lock_frac, phase_ch):
    """Name the comb, or say it is unidentified."""
    for name, ref, locked in REFERENCE:
        if abs(period_ch - ref) < 0.05:
            if locked and not (lock_frac > 0.6 and phase_ch == 0):
                continue
            return name
    return "unidentified"


def comb_contrast(spec, step, phase, lo=448, hi=985):
    """Median on-comb excess over off-comb continuum, as a ratio."""
    ch = np.arange(lo, hi)
    on = ch[ch % step == phase]
    off = ch[ch % step != phase]
    base = np.median(spec[off])
    if base <= 0 or on.size == 0:
        return None
    return float(np.median(spec[on]) / base - 1.0)


def analyse(label, files, lo, hi, snr, key):
    spec, freqs = median_spectrum(files, key=key)
    rec = {"era": label, "key": key, "first_file": Path(files[0]).name,
           "last_file": Path(files[-1]).name, "n_files": len(files),
           "band_ch": [lo, hi]}
    if spec is None:
        rec["verdict"] = f"key {key} absent"
        return rec
    peaks = find_tones(spec, lo, hi, snr=snr)
    rec["n_tones"] = int(peaks.size)
    if peaks.size < 8:
        rec["verdict"] = "too few tones for a period estimate"
        return rec
    P, S = periodogram(peaks)
    picks = top_peaks(P, S)
    rec["periodogram_top"] = [{"period_ch": round(p, 4),
                               "period_mhz": round(p * CHAN_MHZ, 6),
                               "S": round(s, 3)} for p, s in picks]
    best_p, best_s, harmonic_ok, n_strong = fundamental(picks)
    rec["harmonic_consistent"] = harmonic_ok
    rec["n_strong_peaks"] = n_strong
    step = int(round(best_p))
    resid = peaks % step if step > 1 else np.zeros_like(peaks)
    counts = np.bincount(resid, minlength=max(step, 1))
    rec["period_ch"] = round(best_p, 4)
    rec["period_mhz"] = round(best_p * CHAN_MHZ, 6)
    rec["S"] = round(best_s, 3)
    rec["lock_frac"] = round(float(counts.max() / peaks.size), 3)
    rec["phase_ch"] = int(counts.argmax())
    rec["channel_locked"] = bool(abs(best_p - step) < 0.02
                                 and rec["lock_frac"] > 0.6)
    if not harmonic_ok:
        rec["identified_as"] = "no coherent comb (peaks not harmonic)"
    else:
        rec["identified_as"] = identify(best_p, rec["lock_frac"],
                                        rec["phase_ch"])
    rec["mod8_contrast"] = comb_contrast(spec, 8, 0)
    if rec["mod8_contrast"] is not None:
        rec["mod8_contrast"] = round(rec["mod8_contrast"], 4)
    # Are tone centres on integer MHz? That is the TX comb's signature.
    off_mhz = (freqs[peaks] % 1.0 + 0.5) % 1.0 - 0.5
    rec["rms_offset_from_integer_mhz"] = round(float(np.std(off_mhz)), 4)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--key", default="4", help="correlator auto key")
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--hi", type=int, default=1000)
    ap.add_argument("--snr", type=float, default=8.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))
    if not files:
        raise SystemExit(f"no *.h5 under {args.data}")

    for label, i0, i1 in ERAS:
        group = files[i0:i1] if i1 is not None else files[i0:]
        rec = analyse(label, group, args.lo, args.hi, args.snr, args.key)
        if args.json:
            print(json.dumps(rec))
            continue
        print(f"{rec['era']:20s} {rec['first_file']}  "
              f"n_tones={rec.get('n_tones', 0)}")
        if "period_ch" in rec:
            print(f"    period {rec['period_ch']:.4f} ch = "
                  f"{rec['period_mhz']:.6f} MHz  (S={rec['S']:.2f})  "
                  f"lock {rec['lock_frac']:.2f} at residue {rec['phase_ch']}"
                  f"  -> {'CHANNEL-LOCKED' if rec['channel_locked'] else 'walks'}"
                  f"{'' if rec['harmonic_consistent'] else '  [peaks not harmonic]'}")
            print(f"    identified as: {rec['identified_as']};  "
                  f"mod-8 contrast {rec['mod8_contrast']};  "
                  f"rms offset from integer MHz "
                  f"{rec['rms_offset_from_integer_mhz']:.3f} MHz")
        else:
            print(f"    {rec['verdict']}")


if __name__ == "__main__":
    main()
