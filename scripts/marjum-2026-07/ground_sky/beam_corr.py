"""Columns for a low-order correction to the assumed beam, for ``global_fit.py``.

The beam is taken as B0 (1 + sum_lm c_lm Y_lm), with real spherical harmonics
Y_lm in the antenna body frame (boresight +z), 1 <= l <= lmax. B0's full-sphere
integral does not depend on pointing, so the change in normalization is a
constant absorbed by the gain, and each c_lm enters linearly through two columns
per bin: the B0 Y_lm-weighted GSM above the terrain horizon (sky part) and the
B0 Y_lm-weighted solid angle below it (ground part, multiplied by T_gnd in the
fit, so its coefficient is c_lm T_gnd). Both are normalized like the design
matrix (by the full-sphere integral of B0), so they are in GSM kelvin.

The integration runs on a fixed body-frame HEALPix grid, rotated to ENU and
Galactic per bin. With lmax = 0 it reproduces C_GSM and C_gnd, which is the
check that this integration matches ``build_design_matrix``.
"""

from __future__ import annotations

import healpy
import numpy as np
from astropy.time import Time

from eigsep_base.const import MARJUM_PASS
from eigsep_base.rotations import mount_rotation
from eigsep_sim.design_matrix import HorizonProfile
from eigsep_sim.observer import EarthSurface

import sun as sunmod
from common import ERAS, HORIZON_PROFILES, mount_offsets
from raster_sky import real_ylm_maps


def beam_corr_columns(df, freqs, beam, gsm, lmax, campaign, nside_int=32, chunk=500):
    """(cS, cN, labels): (nf, 1 + n_lm, nb) sky and ground columns. Index 0 is
    the uncorrected beam (C_GSM, C_gnd); 1.. are the Y_lm corrections."""
    az_off, el_off, psi = mount_offsets(campaign)
    nside_sky = healpy.npix2nside(gsm.shape[1])
    eras = sorted(set(df.era))
    hz = [sunmod.true_horizon(HorizonProfile.from_npz(campaign / f"{HORIZON_PROFILES}.npz", ERAS[e]))
          for e in eras]
    hidx = np.array([eras.index(e) for e in df.era])
    t = df.t.to_numpy()
    rg = EarthSurface(*MARJUM_PASS).rot_gal2top_stack(Time(t, format="unix")).astype(float)
    rb = mount_rotation(df.az.to_numpy() + az_off, df.el.to_numpy() + el_off, psi)
    nint = healpy.nside2npix(nside_int)
    dirs_b = np.array(healpy.pix2vec(nside_int, np.arange(nint)))      # (3, nint) body frame
    B0 = beam(dirs_b)                                                   # (nf, nint)
    B0 = B0 / B0.sum(1, keepdims=True)
    ylm, labels = real_ylm_maps(nside_int, lmax) if lmax else (np.zeros((0, nint)), [])
    W = B0[:, None, :] * np.concatenate([np.ones((1, nint)), ylm])[None]  # (nf, 1+nlm, nint)
    nf, nb = len(freqs), len(df)
    cS = np.zeros((nf, W.shape[1], nb))
    cN = np.zeros((nf, W.shape[1], nb))
    for a in range(0, nb, chunk):
        for i in range(a, min(a + chunk, nb)):
            enu = rb[i] @ dirs_b
            vis = hz[hidx[i]].visible(enu)
            gal = rg[i].T @ enu
            T = np.where(vis[None], gsm[:, healpy.vec2pix(nside_sky, *gal)], 0.0)  # (nf, nint)
            cS[:, :, i] = np.einsum("fkn,fn->fk", W, T)
            cN[:, :, i] = W[:, :, ~vis].sum(2)
        print(f"  beam-correction columns {min(a + chunk, nb)}/{nb}", flush=True)
    return cS, cN, ["00"] + labels
