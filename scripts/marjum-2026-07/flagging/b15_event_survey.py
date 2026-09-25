"""B15 step 1-2: band survey + event segmentation for airplane/FM-scatter
categories, from the regenerated (unmodified) v0 masks. Does not touch
flags/v0 or flags/v1; reads from a scratch regeneration only.
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


def _campaign_root():
    """Campaign root for this study script.

    ``MARJUM_DATA_ROOT`` still wins, for a worktree that lacks the
    gitignored raw data. Otherwise the package's own setting applies
    (``eigsep_data.set_campaign_root()`` or ``EIGSEP_CAMPAIGN_ROOT``).
    This script used to anchor on its own ``__file__``; it moved out of
    the campaign tree on 2026-09-19, so there is no such anchor.
    """
    env = os.environ.get("MARJUM_DATA_ROOT")
    if env:
        return env
    return str(get_campaign_root(required=True))


DATA_ROOT = _campaign_root()
# Scratch reproduction of v0's masks -- gitignored, regenerable.
# Anchored on DATA_ROOT, not on __file__: it used to resolve under
# whatever checkout this module lived in, so MARJUM_DATA_ROOT pointed
# the raw data at the real tree while the masks were still looked for
# in an empty worktree -- the module worked only where it happened to
# sit. Same bug class as read_flags.ROOT before its fix; see
# PATH_PORTABILITY_PROPOSAL.md.
REGEN = os.path.join(DATA_ROOT, "flags", "b15", "_scratch", "flags_v0_regen")
INPUTS = ("0", "4")  # box-gnd, box-air -- live for the vast majority of the
                      # campaign (Phase C, 07-15 00:32 -> 07-18 03:24) and
                      # the pair used throughout this survey.

modes = B.load_mode_table(os.path.join(DATA_ROOT, "curation", "mode_table.jsonl"))


def day_files():
    files = sorted(glob.glob(os.path.join(REGEN, "flags_*.h5")))
    if not files:
        raise FileNotFoundError(
            f"no regenerated flags/v0 masks found under {REGEN} -- this is "
            "a gitignored scratch build product, not checked into git, and "
            "must be regenerated once per checkout. Run:\n"
            f"  eigsep-build-masks --data {os.path.join(DATA_ROOT, 'data')} "
            f"--out {REGEN}\n"
            "then re-run this."
        )
    return files


def self_comb_channel_mask():
    """Campaign-wide, fixed per-channel exclusion mask: True where that
    channel is EVER flagged SELF_RFI (comb teeth: digital self-comb,
    Panda EMI, fan/laptop bands) on either box-gnd or box-air, anywhere
    in the regenerated v0 masks.

    Added per Aaron's correction (2026-09-14): transmitter/self-comb
    channels confound PCA's basis pursuit and must be excluded from the
    channel set going in, not left in and hoped to wash out. Built from
    v0's own SELF_RFI bit (don't re-derive comb identification) rather
    than a hand-picked channel list, so it tracks whatever v0 actually
    flagged, including the walking Panda comb which has no fixed
    channel set.

    **Correction (2026-09-15, Aaron):** `detectors.categorise()` sets
    SELF_RFI in `BAND_LAPTOP`/`BAND_FAN` (145-160 MHz) whenever *any*
    persistent-track anomaly lands in that band -- not only when a
    comb is actually identified there (the claimed laptop/fan combs are
    "zero detections campaign-wide" per `COMB_INVENTORY.md`). Folding
    that band membership into this function's "ever flagged" campaign-
    wide, all-time exclusion turned an occasional, dynamic flag into a
    permanent static one for ~150 MHz, which Aaron does not want: that
    band should go through the same per-instance dynamic flagging
    (v0's own per-pixel bitfield, applied as usual) as every other
    band, not a blanket exclusion regardless of what's happening there
    at a given time. Carved out below -- the comb-teeth-based exclusion
    (digital self-comb, Panda EMI) is unaffected.
    """
    freqs = None
    ever_self = None
    for dpath in day_files():
        with h5py.File(dpath, "r") as h:
            if freqs is None:
                freqs = h["freqs_mhz"][:]
                ever_self = np.zeros(freqs.size, dtype=bool)
            for fname in h["mask"].keys():
                grp = h["mask"][fname]
                for k in INPUTS:
                    if k not in grp:
                        continue
                    cat = grp[k][:]
                    ever_self |= ((cat & D.SELF_RFI) != 0).any(axis=0)
    not_dynamic_band = (freqs >= D.BAND_LAPTOP[0]) & (freqs <= D.BAND_LAPTOP[1])
    ever_self = ever_self & ~not_dynamic_band
    return freqs, ever_self


def band_survey():
    """Per-channel flagged fraction for airplane / FM-scatter, campaign-wide."""
    freqs = None
    n_sky = {k: None for k in INPUTS}
    n_air = {k: None for k in INPUTS}
    n_ms = {k: None for k in INPUTS}
    n_total_files = {k: 0 for k in INPUTS}

    for dpath in day_files():
        with h5py.File(dpath, "r") as h:
            if freqs is None:
                freqs = h["freqs_mhz"][:]
                for k in INPUTS:
                    n_sky[k] = np.zeros(freqs.size, dtype=np.int64)
                    n_air[k] = np.zeros(freqs.size, dtype=np.int64)
                    n_ms[k] = np.zeros(freqs.size, dtype=np.int64)
            for fname in h["mask"].keys():
                grp = h["mask"][fname]
                for k in INPUTS:
                    if k not in grp:
                        continue
                    cat = grp[k][:]
                    sky = (cat & (D.CAL | D.OVERFLOW)) == 0
                    n_sky[k] += sky.sum(axis=0)
                    n_air[k] += ((cat & D.AIRPLANE) != 0).sum(axis=0)
                    n_ms[k] += ((cat & D.FM_DTV_MS) != 0).sum(axis=0)
                    n_total_files[k] += 1
    return freqs, n_sky, n_air, n_ms, n_total_files


def event_segments(min_gap_s=30.0):
    """Contiguous-run events for AIRPLANE and FM_DTV_MS, per input, using
    real UTC time (not sample index) so cross-file runs are stitched
    correctly and gaps longer than min_gap_s end an event."""
    events = {k: {"airplane": [], "ms": []} for k in INPUTS}

    for dpath in day_files():
        with h5py.File(dpath, "r") as h:
            freqs = h["freqs_mhz"][:]
            fm_dtv = (
                ((freqs >= D.BAND_FM[0]) & (freqs <= D.BAND_FM[1]))
                | ((freqs >= D.BAND_DTV_LO[0]) & (freqs <= D.BAND_DTV_LO[1]))
                | ((freqs >= D.BAND_DTV_HI[0]) & (freqs <= D.BAND_DTV_HI[1]))
            )
            fnames = sorted(h["mask"].keys())  # chronological: filenames sort by close time
            for k in INPUTS:
                t_all, air_all, ms_all = [], [], []
                for fname in fnames:
                    grp = h["mask"][fname]
                    if k not in grp:
                        continue
                    cat = grp[k][:]
                    m = B.mode_for(modes, fname)
                    integ = m["integration_time_s"] if m else 200.0
                    t = D.sample_times(fname, cat.shape[0], integ)
                    air_time = ((cat[:, fm_dtv] & D.AIRPLANE) != 0).any(axis=1)
                    ms_time = ((cat[:, fm_dtv] & D.FM_DTV_MS) != 0).any(axis=1)
                    t_all.append(t)
                    air_all.append(air_time)
                    ms_all.append(ms_time)
                if not t_all:
                    continue
                t_all = np.concatenate(t_all)
                order = np.argsort(t_all)
                t_all = t_all[order]
                air_all = np.concatenate(air_all)[order]
                ms_all = np.concatenate(ms_all)[order]

                for label, mask in (("airplane", air_all), ("ms", ms_all)):
                    idx = np.flatnonzero(mask)
                    if idx.size == 0:
                        continue
                    # break into runs where either the index isn't
                    # contiguous OR the real time gap exceeds min_gap_s
                    breaks = np.flatnonzero(
                        (np.diff(idx) > 1) |
                        (np.diff(t_all[idx]) > min_gap_s)
                    )
                    starts = np.concatenate(([0], breaks + 1))
                    ends = np.concatenate((breaks, [idx.size - 1]))
                    for s, e in zip(starts, ends):
                        i0, i1 = idx[s], idx[e]
                        events[k][label].append({
                            "day": os.path.basename(dpath),
                            "t_start": float(t_all[i0]),
                            "t_end": float(t_all[i1]),
                            "duration_s": float(t_all[i1] - t_all[i0]),
                            "n_samples": int(i1 - i0 + 1),
                        })
    return events


if __name__ == "__main__":
    freqs, n_sky, n_air, n_ms, n_files = band_survey()
    print("files with input 0/4:", n_files)
    for k in INPUTS:
        frac_air = np.divide(n_air[k], np.maximum(n_sky[k], 1))
        frac_ms = np.divide(n_ms[k], np.maximum(n_sky[k], 1))
        top_air = np.argsort(frac_air)[::-1][:5]
        top_ms = np.argsort(frac_ms)[::-1][:5]
        print(f"--- input {k} ---")
        print("top airplane channels (MHz, frac):",
              list(zip(np.round(freqs[top_air], 3), np.round(frac_air[top_air], 5))))
        print("top ms channels (MHz, frac):",
              list(zip(np.round(freqs[top_ms], 3), np.round(frac_ms[top_ms], 5))))
        print("overall airplane frac:", round(float(n_air[k].sum() / n_sky[k].sum()), 5))
        print("overall ms frac:", round(float(n_ms[k].sum() / n_sky[k].sum()), 5))

    events = event_segments()
    scratch_dir = os.path.join(DATA_ROOT, "flags", "b15", "_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    with open(os.path.join(scratch_dir, "b15_events.json"), "w") as f:
        json.dump(events, f)
    for k in INPUTS:
        for label in ("airplane", "ms"):
            durs = [e["duration_s"] for e in events[k][label]]
            print(f"input {k} {label}: n_events={len(durs)}",
                  f"median_dur_s={np.median(durs) if durs else None}",
                  f"mean_dur_s={np.mean(durs) if durs else None}")
