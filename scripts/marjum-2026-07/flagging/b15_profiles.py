"""B15 step 1a/2: raw-amplitude time-domain profiles, duration and
amplitude distributions, and frequency-domain PCA, for airplane and
FM-scatter (micrometeor) flagged instants. Reads flags/v0 (regenerated,
unmodified) + raw data; writes nothing back to flags/v0 or flags/v1.
"""
import glob
import json
import os
import sys

import h5py
import numpy as np

from eigsep_data.flagging import build_masks as B
from eigsep_data.flagging import detectors as D
from eigsep_data.paths import get_campaign_root

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # sibling study modules in this directory
import b15_event_survey as EV  # noqa: E402

MARJUM_ROOT = (
    os.environ.get("MARJUM_DATA_ROOT") or str(get_campaign_root(required=True))
)
SCRATCH_DIR = os.path.join(MARJUM_ROOT, "flags", "b15", "_scratch")
# Default: this checkout's own marjum-2026-07/ (portable across machines).
# Override for a worktree that doesn't physically have the gitignored raw
# data (worktrees don't carry untracked files).
DATA_ROOT = MARJUM_ROOT
INPUT = "4"  # box-air / sky: physically the meaningful leg for external RFI
N_SEEDS = 400  # per class, capped for compute/report size -- see caveats
WIN_S = 20.0   # seconds of context on each side of a seed instant
RNG_SEED = 20260914

modes = B.load_mode_table(os.path.join(DATA_ROOT, "curation", "mode_table.jsonl"))


def collect_seeds():
    """All (file, input, time_index) instants where AIRPLANE / FM_DTV_MS is
    set on >=1 FM/DTV channel, from the regenerated per-day masks."""
    seeds = {"airplane": [], "ms": []}
    for dpath in EV.day_files():  # raises a clear error if REGEN is empty/missing
        with h5py.File(dpath, "r") as h:
            freqs = h["freqs_mhz"][:]
            fm_dtv = (
                ((freqs >= D.BAND_FM[0]) & (freqs <= D.BAND_FM[1]))
                | ((freqs >= D.BAND_DTV_LO[0]) & (freqs <= D.BAND_DTV_LO[1]))
                | ((freqs >= D.BAND_DTV_HI[0]) & (freqs <= D.BAND_DTV_HI[1]))
            )
            for fname in h["mask"].keys():
                grp = h["mask"][fname]
                if INPUT not in grp:
                    continue
                cat = grp[INPUT][:]
                air_t = np.flatnonzero(((cat[:, fm_dtv] & D.AIRPLANE) != 0).any(axis=1))
                ms_t = np.flatnonzero(((cat[:, fm_dtv] & D.FM_DTV_MS) != 0).any(axis=1))
                for i in air_t:
                    seeds["airplane"].append((os.path.basename(dpath), fname, int(i)))
                for i in ms_t:
                    seeds["ms"].append((os.path.basename(dpath), fname, int(i)))
    return seeds


def subsample(seeds, n, rng):
    if len(seeds) <= n:
        return seeds
    idx = rng.choice(len(seeds), size=n, replace=False)
    return [seeds[i] for i in sorted(idx)]


def load_fm_band_series(path):
    """Whole-file log|cross-power| time series (box-gnd x box-air, the
    `04` dataset), FM-band median, plus freqs, the overflow mask, and
    the per-channel log|cross-power| spectrum array.

    Switched from single-input (box-air) autocorrelation to cross-power
    per Aaron's correction (2026-09-15): cross-correlation between the
    two inputs resolves out receiver-local background that survives in
    either input's own autocorrelation. `logp` below is log10|cross
    power|, not log10(autocorrelation) -- same variable name kept so
    the rest of this module (baseline/MAD/PCA machinery) is unchanged.

    Overflow is still checked on BOTH inputs' own autocorrelations
    (`raw < 0`): an accumulator wrap in either input's accumulation
    corrupts any product built from that sample, including the cross
    term, even though the cross value itself has no sign-based wrap
    signature of its own (a legitimate cross real/imag part is signed).
    """
    with h5py.File(path, "r") as h:
        freqs = h["header/freqs"][:]
        raw0 = h["data/0"][:].astype(np.float64) if "0" in h["data"] else None
        raw4 = h["data/" + INPUT][:].astype(np.float64) if INPUT in h["data"] else None
        cross_key = "04" if "04" in h["data"] else ("40" if "40" in h["data"] else None)
        if cross_key is None:
            return None, None, freqs, None
        c = h["data/" + cross_key][:].astype(np.float64)
    cross_amp = np.abs(c[..., 0] + 1j * c[..., 1])
    logp = np.log10(np.maximum(cross_amp, 1.0))
    ovf = np.zeros_like(logp, dtype=bool)
    if raw0 is not None:
        ovf |= raw0 < 0
    if raw4 is not None:
        ovf |= raw4 < 0
    fm = (freqs >= D.BAND_FM[0]) & (freqs <= D.BAND_FM[1])
    fm_series = np.median(logp[:, fm], axis=1)
    ovf_time = ovf[:, fm].any(axis=1)  # any FM-band overflow (either input) at this sample
    return logp, fm_series, freqs, ovf_time


def profile_events(seeds_by_file, class_name):
    """For each seed, pull a +/-WIN_S window of the FM-band series, plus
    the full-band spectrum at the peak sample, from the seed's file
    (loading the neighbouring file too when the window runs off an edge)."""
    profiles = []       # list of (t_rel, normalized excursion) arrays
    durations_s = []
    peak_excursions = []
    residual_spectra = []
    freqs_ref = None
    n_rejected_overflow = 0
    n_rejected_degenerate_mad = 0
    # MAD floor below which "sigma" is not a meaningful scale -- typically
    # hit when a window's raw counts are so small that log10 quantizes to
    # a handful of repeated values. Below this, normalized excursions are
    # a division-by-~0 artifact, not a real amplitude measurement.
    MAD_FLOOR = 1e-3

    from scipy.ndimage import median_filter

    for (day, fname, seed_indices) in seeds_by_file:
        path = os.path.join(DATA_ROOT, "data", fname)
        if not os.path.exists(path):
            continue
        m = B.mode_for(modes, fname)
        integ = m["integration_time_s"] if m else 200.0
        logp, fm_series, freqs, ovf_time = load_fm_band_series(path)
        if logp is None:
            continue  # no cross-correlation dataset in this file
        if freqs_ref is None:
            freqs_ref = freqs
        nt = fm_series.size
        t = D.sample_times(fname, nt, integ)
        dt = float(np.median(np.diff(t))) if nt > 1 else 0.5
        win = max(int(round(WIN_S / max(dt, 1e-6))), 3)

        # Smooth local baseline via median filter (wide enough to not
        # eat the event itself), for normalizing each excursion.
        baseline = median_filter(fm_series, size=min(2 * win + 1, nt), mode="nearest")
        resid = fm_series - baseline
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))

        for i0 in seed_indices:
            lo, hi = max(i0 - win, 0), min(i0 + win + 1, nt)
            if hi - lo < 5:
                continue
            if ovf_time[lo:hi].any():
                # Accumulator wrap in this window: instrumental, not a
                # real amplitude excursion. Exclude rather than let a
                # clamped-to-zero sample masquerade as a modulation.
                n_rejected_overflow += 1
                continue
            if mad < MAD_FLOOR:
                # Degenerate local scale (quantized/near-constant raw
                # counts): dividing by it manufactures arbitrarily large
                # "MAD units" out of noise. Exclude rather than report.
                n_rejected_degenerate_mad += 1
                continue

            seg = resid[lo:hi] / mad
            t_rel = (np.arange(lo, hi) - i0) * dt
            profiles.append((t_rel, seg))

            peak_local = i0 - lo  # index of the seed sample within seg
            peak = float(np.abs(seg[peak_local]))
            peak_excursions.append(peak)

            # FWHM in seconds: walk outward from the SEED sample (not the
            # window-wide argmax) while |seg| stays above half its value,
            # so an unrelated bump elsewhere in the +/-20 s window can't
            # inflate the measured duration of this event.
            half = 0.5 * peak
            j_lo = peak_local
            while j_lo > 0 and abs(seg[j_lo - 1]) >= half:
                j_lo -= 1
            j_hi = peak_local
            while j_hi < len(seg) - 1 and abs(seg[j_hi + 1]) >= half:
                j_hi += 1
            durations_s.append(float((j_hi - j_lo + 1) * dt))

            # Residual spectrum (whole band) at the seed sample, relative
            # to that channel's file-median -- for PCA.
            chan_med = np.median(logp, axis=0)
            residual_spectra.append(logp[i0] - chan_med)

    return {
        "profiles": profiles,
        "durations_s": durations_s,
        "peak_excursions": peak_excursions,
        "residual_spectra": np.array(residual_spectra) if residual_spectra else np.zeros((0, 1024)),
        "freqs": freqs_ref,
        "n_rejected_overflow": n_rejected_overflow,
        "n_rejected_degenerate_mad": n_rejected_degenerate_mad,
    }


def group_by_file(seeds):
    by_file = {}
    for day, fname, i in seeds:
        by_file.setdefault((day, fname), []).append(i)
    return [(d, f, sorted(idx)) for (d, f), idx in by_file.items()]


if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    seeds = collect_seeds()
    print("total seed instants:", {k: len(v) for k, v in seeds.items()})

    results = {}
    for cls in ("airplane", "ms"):
        sub = subsample(seeds[cls], N_SEEDS, rng)
        by_file = group_by_file(sub)
        print(cls, "n_seeds_used:", len(sub), "n_files:", len(by_file))
        res = profile_events(by_file, cls)
        results[cls] = res
        durs = np.array(res["durations_s"])
        amps = np.array(res["peak_excursions"])
        print(f"  rejected: overflow={res['n_rejected_overflow']} "
              f"degenerate_mad={res['n_rejected_degenerate_mad']} "
              f"of {len(sub)} seeds -> {len(durs)} usable")
        print(f"  duration_s: median={np.median(durs):.3f} mean={np.mean(durs):.3f} "
              f"p90={np.percentile(durs,90):.3f}")
        print(f"  peak excursion (MAD units): median={np.median(amps):.2f} "
              f"p90={np.percentile(amps,90):.2f} max={np.max(amps):.2f}")

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    npz_path = os.path.join(SCRATCH_DIR, "b15_profiles.npz")
    pkl_path = os.path.join(SCRATCH_DIR, "b15_profiles_raw.pkl")
    np.savez(npz_path,
             airplane_durations=np.array(results["airplane"]["durations_s"]),
             airplane_amps=np.array(results["airplane"]["peak_excursions"]),
             airplane_spectra=results["airplane"]["residual_spectra"],
             ms_durations=np.array(results["ms"]["durations_s"]),
             ms_amps=np.array(results["ms"]["peak_excursions"]),
             ms_spectra=results["ms"]["residual_spectra"],
             freqs=results["airplane"]["freqs"] if results["airplane"]["freqs"] is not None else results["ms"]["freqs"],
             )
    # profiles (variable-length t_rel arrays) saved separately as pickled list
    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump({"airplane": results["airplane"]["profiles"],
                     "ms": results["ms"]["profiles"]}, f)
    print(f"wrote {npz_path} and {pkl_path}")
