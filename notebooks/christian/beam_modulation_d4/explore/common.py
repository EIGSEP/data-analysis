"""Shared loader for the deployment-4 rotation beam-modulation exploration.

Deployment 4 (2025-07-15..20, Utah) has no working motor or IMU telemetry, so the
rotation windows were found from the radio data itself.  Everything here works off a
sidecar extracted from the external disk by 01_extract.py.
"""
import numpy as np, h5py
from pathlib import Path

REPO = Path("/home/christian/Documents/research/eigsep/data-analysis")
DISK = Path("/media/christian/Samsung_T5/eigsep_data/deployment4/corr_data")
OUT = REPO / "notebooks/christian/beam_modulation_d4/explore"
SIDECAR = REPO / "notebooks/christian/beam_modulation_d4/rotation_20250720_0101.h5"

# the chosen episode: local 07-19 18:15, UTC 2025-07-20 01:11-01:28, bracketed by
# stationary data on both sides
FILES = [f"corr_20250719_18{s}.h5" for s in
         ("0532", "0950", "1408", "1826", "2244", "2702", "3120", "3539")]

ROT = "2"            # the receiver that modulates
CTRL = ("3", "4")    # the two that do not
KEYS = (ROT,) + CTRL

# deployment-4 comb: tones on ch % 16 == 0 (deployment 5 had them on residue 8)
COMB_RESIDUE = 0
COMB_SPILL = (15, 0, 1)
FM_BAND = (86.0, 110.0)


def load():
    with h5py.File(SIDECAR, "r") as f:
        d = {k: f[f"data/{k}"][:].astype(np.float64) for k in KEYS}
        t = f["times"][:]
        sw = f["sw_state"][:]
        fr = f["freqs"][:]
    for k in d:
        d[k][d[k] <= 0] = np.nan
    return dict(d=d, t=t, sw=sw, fr=fr, ch=np.arange(fr.size),
                tsec=t - t[0], sky=sw == 0)


def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(x)


def flanking(c, nchan, half=(3, 7)):
    """Non-comb channels either side of tone channel c, clear of its spillover."""
    lo, hi = half
    offs = [o for o in range(-hi + 1, hi) if lo <= abs(o) < hi]
    return [c + o for o in offs if 0 <= c + o < nchan and (c + o) % 16 not in COMB_SPILL]
