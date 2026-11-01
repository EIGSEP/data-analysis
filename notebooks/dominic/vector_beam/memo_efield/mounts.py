"""
Inputs and mounts for the E-field memo (EIGSEP.github.io, memos/marjum_efield_beam).

Inputs: the Phase 2 bracketed cross-correlations, pointing and HFSS beam cached by
``../bracketed_fill_check/dump_inputs.py`` (path in ``EFIELD_INPUTS``, default
``inputs.npz`` here). Extrapolated DPSS fills are NaN'd (the bracketed-fill rule)
and each tooth's valid samples are those finite in both arms.

Mounts (az offset, alpha0, el correction, axle tilt, all deg):

- ``R``: right-handed azimuth, geometry v0013 (memo 005), no tilt. The geometry
  stored in the inputs; the memo's first draft.
- ``L0``: left-handed azimuth (memo 003, pointing table v5), memo 002's v0005
  posterior-mean positions, az zero +15.8 deg (-164.2 deg here, the same by the
  dipole's 180 deg symmetry), alpha0 8 deg, el zero +0.05 deg, no tilt.
- ``L``: ``L0`` with the v0016 elevation-axle tilt of +4.7 deg (memo 003 sec. 6).

The inputs' azimuth is pointing table v3's (the readings' sense); the left-handed
mounts multiply it by ``beam_mapping.AZ_SENSE``, as pointing table v5 does.
"""

import os
import sys
from pathlib import Path

import numpy as np

from eigsep_data.beam_mapping import AZ_SENSE, TxGeometryModel
from eigsep_data.beam_mapping import vector_beam as vb
from eigsep_data.rfi import build_frequency_trust_mask, flag_unbracketed_fills

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import geometry_v0013  # noqa: E402

INPUTS = Path(os.environ.get("EFIELD_INPUTS", HERE / "inputs.npz"))
OUT = HERE / "output"
NSIDE = 32
MAP_D = (62, 74, 86)

# memo 002: geometry v0005 posterior means (UTM raster grid of DEM v0002, m)
ANTENNA_ENU = np.array([1655.13, 2030.96, 1777.85])
TRANSMITTER_ENU = np.array([1650.61, 2027.05, 1684.62])
MOUNTS = {"L0": (-164.2, 8.0, 0.05, 0.0), "L": (-164.2, 8.0, 0.05, 4.7)}


def tooth_freq(d):
    return 171.875 + (d - 60) * 1.953125


def load(path=INPUTS):
    """The cached inputs and one dict per tooth (d89 has no valid samples and is left out)."""
    z = dict(np.load(path))
    nc = z["Ex_all"].shape[1]
    trust = build_frequency_trust_mask(tooth_freq(np.arange(nc)), min_freq=87.0, fm_low=87.0,
                                       fm_high=108.0, max_freq=230.0)
    Exa, Eya = z["Ex_all"].copy(), z["Ey_all"].copy()
    Exa[flag_unbracketed_fills(z["Ex_meas"], trust)] = np.nan
    Eya[flag_unbracketed_fills(z["Ey_meas"], trust)] = np.nan
    ch = []
    for k, d in enumerate(z["d_idx"][:-1]):
        d = int(d)
        Ex, Ey = Exa[:, d], Eya[:, d]
        v = z["sel"] & ~z["bad_time"] & np.isfinite(Ex) & np.isfinite(Ey)
        ch.append(dict(d=d, k=k, arm=d % 2, freq=tooth_freq(d), Ex=np.where(v, Ex, 0),
                       Ey=np.where(v, Ey, 0), v=v))
    return z, ch


def left_model(off, alpha, tilt):
    return TxGeometryModel(psi_deg=geometry_v0013.PSI_DEG, antenna_enu=ANTENNA_ENU,
                           transmitter_enu=TRANSMITTER_ENU, alpha0_deg=alpha, az_offset_deg=off,
                           axle_tilt_deg=tilt)


def geometry(name, z):
    """(pxs, r_hat, E1v, E2v) of every sample under mount ``name``."""
    if name == "R":
        return z["pxs"], z["r_hat"], z["E1v"], z["E2v"]
    off, alpha, el0, tilt = MOUNTS[name]
    return vb.fields_from_model(left_model(off, alpha, tilt), [0, 0, el0, 0, 0, 0],
                                AZ_SENSE * np.asarray(z["az"]), z["el"], NSIDE)


def truth(z, k, pix):
    """HFSS beam (npix, 3) of tooth ``k`` at ``pix``."""
    return z["beam"][z["h_idx"][k]][:, pix].T


def power_scale(u, t):
    """One real least-squares factor on total power, |u|^2 -> |t|^2 (the maps' scale)."""
    P, H = np.sum(np.abs(u) ** 2, 1), np.sum(np.abs(t) ** 2, 1)
    return np.sum(P * H) / np.sum(P * P)
