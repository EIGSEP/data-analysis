"""Definitive comb inventory for marjum-2026-07.

Why a new detector rather than reusing the periodogram
------------------------------------------------------

Three estimators have disagreed about this campaign's combs, and each
disagreement was an artefact of the estimator, not of the data:

* **FFT argmax on a windowed difference spectrum** (mine, first pass)
  quantises the period to a bin. Over a 553-channel span, bins near
  4 channels are ~0.1 channel apart, which is the entire difference
  between the 4.000 and 4.096 hypotheses.
* **Modal tone spacing** (data-archivist) is 4 channels under *both*
  hypotheses: a 4.096-channel comb produces ~90% 4s and ~10% 5s, so
  reporting the mode as "exactly 4 channels" cannot distinguish them.
  The discriminator is the *asymmetry* of the 3s and 5s.
* **Autocorrelation of a time-averaged spectrum** (natural-experimenter)
  is blind to walking combs by construction: a comb whose teeth drift
  across the grid smears out under time-averaging, so "no comb" from
  that method is not evidence of absence.

This module uses **tooth contrast with an explicit phase scan**, which
puts locked and walking hypotheses on equal footing:

    resid  = logP - median_filter(logP, 2s+1)
    teeth  = channels predicted by (spacing, phase)
    contrast = [median(resid[teeth]) - median(resid[~teeth])] / MAD(resid)

maximised over phase. For a **locked** hypothesis teeth are at fixed
channel indices; for a **walking** hypothesis they are computed in
frequency. Nothing about the statistic prefers one kind.

The identification principle, corrected
---------------------------------------

An earlier version of this work asserted "integer-channel spacing =>
self-generated". **That is wrong for the transmitter.** PROGRAM.md §5
and EIGSEP paper §4.6 describe the beam-mapping transmitter as a
*clock-locked* Dirac comb -- it is deliberately locked to the same
reference as the receiver, so it produces a channel-locked comb by
design. Channel-locking therefore separates *clock-referenced* sources
(TX and our own digital electronics) from *free-running external* ones
(broadcast, laptops). It does not by itself identify self-RFI.

What distinguishes the TX from digital self-RFI is **beam response**:
the TX is a source in the far field, so the power the rotating antenna
receives from it is modulated by the antenna pattern as the platform
turns. Self-generated EMI is conducted or near-field and does not track
pointing. That test is implemented in `beam_modulation`.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.ndimage import median_filter

from eigsep_data.flagging import detectors as D

CW = D.CHAN_WIDTH_MHZ

# Hypotheses. "locked" spacings are exact channel counts; "walking"
# spacings are frequencies that are not integer channel multiples.
LOCKED_CHAN = [2, 4, 8, 16, 32, 64, 256]
WALKING_MHZ = [1.000, 1.250, 2.000, 3.900]

BAND = (50.0, 215.0)


def tooth_contrast_locked(resid, chans, in_band, spacing):
    """Best-phase tooth contrast for a comb at an exact channel spacing."""
    best = (-np.inf, -1)
    scale = 1.4826 * np.median(np.abs(resid[in_band] - np.median(resid[in_band])))
    if scale < 1e-12:
        return 0.0, -1
    for phase in range(spacing):
        teeth = in_band & (chans % spacing == phase)
        other = in_band & ~teeth
        if teeth.sum() < 8 or other.sum() < 8:
            continue
        c = (np.median(resid[teeth]) - np.median(resid[other])) / scale
        if c > best[0]:
            best = (float(c), phase)
    return best


def tooth_contrast_walking(resid, freqs, in_band, spacing_mhz, n_phase=16):
    """Best-phase tooth contrast for a comb at a non-integer spacing."""
    scale = 1.4826 * np.median(np.abs(resid[in_band] - np.median(resid[in_band])))
    if scale < 1e-12:
        return 0.0, -1.0
    best = (-np.inf, -1.0)
    for i in range(n_phase):
        ph = i * spacing_mhz / n_phase
        off = (freqs - ph) / spacing_mhz
        dist = np.abs(off - np.round(off)) * spacing_mhz
        teeth = in_band & (dist <= 0.5 * CW)
        other = in_band & ~teeth
        if teeth.sum() < 8 or other.sum() < 8:
            continue
        c = (np.median(resid[teeth]) - np.median(resid[other])) / scale
        if c > best[0]:
            best = (float(c), float(ph))
    return best


def analyse_spectrum(med, freqs):
    """All comb hypotheses against one median log spectrum."""
    chans = np.arange(med.size)
    in_band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    out = {}
    for s in LOCKED_CHAN:
        resid = med - median_filter(med, size=2 * s + 1, mode="nearest")
        c, ph = tooth_contrast_locked(resid, chans, in_band, s)
        out[f"lock{s}"] = round(c, 3)
        out[f"lock{s}_phase"] = ph
    for f0 in WALKING_MHZ:
        w = max(int(round(2 * f0 / CW)) | 1, 5)
        resid = med - median_filter(med, size=w, mode="nearest")
        c, ph = tooth_contrast_walking(resid, freqs, in_band, f0)
        out[f"walk{f0:g}"] = round(c, 3)
    return out


def process(path):
    fn = os.path.basename(path)
    recs = []
    try:
        with h5py.File(path, "r") as h:
            freqs = h["header/freqs"][:]
            keys = sorted(k for k in h["data"] if len(k) == 1)
            rfsw = h["metadata/rfswitch"][()] if (
                "metadata" in h and "rfswitch" in h["metadata"]) else None
            for k in keys:
                raw = h["data/" + k][:]
                ant = D.antenna_mask(rfsw, raw.shape[0])
                if ant.sum() < 4:
                    ant = np.ones(raw.shape[0], dtype=bool)
                # Repair int32 wrap before measuring: a wrap is a huge
                # downward spike that would corrupt the tooth statistic.
                ovf = D.overflow_mask(raw)
                val = raw.astype(np.float64)
                if ovf.any():
                    val = np.where(ovf, val + 2.0 ** 32, val)
                logp = np.log10(np.maximum(val, 1.0))
                med = np.median(logp[ant], axis=0)
                if not np.isfinite(med).all() or np.median(med) <= 0:
                    continue
                rec = {"file": fn, "input": k, "n_ovf": int(ovf.sum())}
                rec.update(analyse_spectrum(med, freqs))
                recs.append(rec)
    except Exception as exc:
        recs.append({"file": fn, "error": f"{type(exc).__name__}: {exc}"})
    return recs


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    files = sorted(glob.glob(os.path.join(data, "*.h5")))
    n = int(os.environ.get("NWORKERS", "12"))
    done = 0
    with ProcessPoolExecutor(max_workers=n) as ex:
        for recs in ex.map(process, files, chunksize=8):
            for r in recs:
                sys.stdout.write(json.dumps(r) + "\n")
            done += 1
            if done % 500 == 0:
                print(f"# {done}/{len(files)}", file=sys.stderr, flush=True)
    print(f"# done {done}", file=sys.stderr)


if __name__ == "__main__":
    main()
