"""Reference detector for unphysical elevation-solution slews.

This is the implementation behind `disc_mask.npy` and section 8 of
`beam_metric_outliers_checkpoint.ipynb`. It was originally run inline; it is
checked in here so the production `pointing_table` flag (EL_SOLUTION_GLITCH)
has an exact reference to reproduce.

Criterion
---------
For each pair of temporally adjacent samples (i, i+1) that are

  * in the same file,
  * separated by 0.1 s < dt < 2 s (the campaign's two cadences are 0.2684 s
    and 0.5369 s; this bracket accepts both and rejects file-edge and
    bad-clock pairs),
  * both with finite elevation,

compute |el[i+1] - el[i]| / dt. If it exceeds SLEW_MAX_DEG_S, flag **both**
i and i+1 -- the transition identifies a bad pair, and which member is the
bad one is not determined by the rate alone.

Why 20 deg/s: the commanded scan slews at ~5 deg/s (campaign p99 of the
in-file rate is 5.0 deg/s), so 20 is 4x the working rate. The count is
insensitive in this region -- campaign-wide, >20 deg/s gives 808 transitions,
>30 gives 752 and >50 gives 713 -- so the threshold sits in a sparse valley
between real motion and the glitch population. Dropping to 10 deg/s nearly
triples it (2240) by reaching into real motion.

Do NOT substitute a "differs from both neighbours" test. It misses glitch
runs of 2-3 consecutive bad samples, which occur (see
corr_20260717_201739Z.h5 samples 25-27), and only partially heals the
affected elevation bands.

Scope warning
-------------
The *mechanism* -- a single-sample escape from the el ~ +/-180 park emitting a
spurious |el| ~ 59-60 or ~0 -- is established only for the post-drive-failure
era (EL_POST_FAILURE set). Campaign-wide the pre-failure firings are a
different population: they show almost none of the |el| ~ 59 signature (1
sample vs 192) and 404 of 439 are already `quality == "suspect"`. So the bit
should be documented by its *criterion* ("elevation solution moved faster than
the drive can"), not by that mechanism.
"""
from __future__ import annotations

import numpy as np

SLEW_MAX_DEG_S = 20.0
DT_MIN_S = 0.1
DT_MAX_S = 2.0


def el_slew_glitch_mask(el_deg, t_s, file_key, slew_max_deg_s=SLEW_MAX_DEG_S):
    """Flag samples adjacent to an unphysical elevation slew.

    Parameters
    ----------
    el_deg : (n,) float
        Elevation in degrees, in acquisition order.
    t_s : (n,) float
        Timestamps in seconds, same order. Non-positive values are treated as
        invalid (the loader's sentinel for comb-off / missing headers).
    file_key : (n,) array
        Per-sample file identity; transitions across a change are not tested.
    slew_max_deg_s : float
        Rate above which a transition is considered unphysical.

    Returns
    -------
    (n,) bool
        True for both members of every offending transition.
    """
    el = np.asarray(el_deg, dtype=float)
    t = np.asarray(t_s, dtype=float)
    key = np.asarray(file_key)
    n = el.size
    if not (t.size == n and key.size == n):
        raise ValueError("el_deg, t_s and file_key must be the same length")

    dt = np.diff(t)
    d_el = np.abs(np.diff(el))
    same_file = key[1:] == key[:-1]
    finite = np.isfinite(d_el) & np.isfinite(dt) & (t[:-1] > 0) & (t[1:] > 0)
    testable = same_file & finite & (dt > DT_MIN_S) & (dt < DT_MAX_S)

    bad = np.zeros(n - 1, dtype=bool)
    bad[testable] = (d_el[testable] / dt[testable]) > slew_max_deg_s

    mask = np.zeros(n, dtype=bool)
    mask[:-1] |= bad        # the earlier member of the pair
    mask[1:] |= bad         # the later member of the pair
    return mask


def _self_test():
    """Reproduce disc_mask.npy from the explorer cache + sidecar."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    cache = np.load(here / "beam_explorer_cache.npz")
    side = np.load(here / "beam_explorer_sidecar.npz", allow_pickle=True)
    mask = el_slew_glitch_mask(cache["el_deg"].astype(float),
                               side["times"], side["file_index"])
    print(f"flagged {mask.sum()} of {mask.size} samples "
          f"({100*mask.mean():.3f}%)")
    ref = here / "disc_mask.npy"
    if ref.exists():
        old = np.load(ref)
        agree = int((mask == old).sum())
        print(f"vs disc_mask.npy: {agree}/{mask.size} agree, "
              f"{int((mask & ~old).sum())} new, {int((old & ~mask).sum())} lost")
        assert np.array_equal(mask, old), "does not reproduce disc_mask.npy"
        print("EXACT MATCH")
    el = cache["el_deg"].astype(float)
    h, e = np.histogram(np.abs(el[mask]),
                        bins=[0, 5, 20, 50, 57.5, 61, 75, 150, 175, 181])
    print("by |el|: " + ", ".join(f"{a:g}-{b:g}: {c}"
                                  for a, b, c in zip(e[:-1], e[1:], h) if c))


if __name__ == "__main__":
    _self_test()
