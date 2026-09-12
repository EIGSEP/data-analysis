"""Before/after beammap for the SH-augmented fit at ch616 (best case)
and ch680 (typical). Uses the same contour style and az/el conventions
as make_beammap_style.py, in three rows:
  row 1: baseline K=4 HFSS-PCA fit
  row 2: SH-augmented fit (lmax=8, ridge=0.02, linearized)
  row 3: difference of residuals (baseline - augmented)
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beam_pca import compute_beam_pca
from data_space_rfi import data_space_rfi_mask
from fit_v007_pca_beam import (
    _project_templates,
    fit_channel,
    pointing_valid_mask,
    tx_arm_for_channel,
)
from rotation_beam import TransmitterGeometry
from tx_beam_sim import HFSSBeamSet, simulate_hfss_coupling
from v007_beam_diagnostic import (
    channel_validity_masks,
    gross_power_time_flags,
    load_v007_data,
)

MODEL_JSON = "v007_pca_beam_model_v5.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
CHANNELS = [int(c) for c in (sys.argv[1:] or [616, 680])]
LMAX = 8
RIDGE = 0.02

az_edges = np.arange(-182.5, 2.5, 5.0)
el_edges = np.arange(-182.5, 187.5, 5.0)
azc = 0.5 * (az_edges[1:] + az_edges[:-1])
elc = 0.5 * (el_edges[1:] + el_edges[:-1])


def grid(az, el, vals):
    sums = np.zeros((azc.size, elc.size))
    cnts = np.zeros_like(sums)
    ia = np.digitize(az, az_edges) - 1
    ie = np.digitize(el, el_edges) - 1
    ok = (ia >= 0) & (ia < azc.size) & (ie >= 0) & (ie < elc.size)
    np.add.at(sums, (ia[ok], ie[ok]), vals[ok])
    np.add.at(cnts, (ia[ok], ie[ok]), 1)
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)


def sh_basis_beams(nside, lmax):
    npix = hp.nside2npix(nside)
    n_ylm = (lmax + 1) ** 2
    alm_size = hp.Alm.getsize(lmax)
    ylm_maps = np.empty((n_ylm, npix))
    ell_of_mode = np.empty(n_ylm, dtype=int)
    idx = 0
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            alm = np.zeros(alm_size, complex)
            if m == 0:
                alm[hp.Alm.getidx(lmax, ell, 0)] = 1.0
            elif m > 0:
                alm[hp.Alm.getidx(lmax, ell, m)] = 1.0 / np.sqrt(2)
            else:
                sgn = -1 if (-m) % 2 else 1
                alm[hp.Alm.getidx(lmax, ell, -m)] = 1j / np.sqrt(2) * sgn
            ylm_maps[idx] = hp.alm2map(alm, nside, lmax=lmax, verbose=False)
            ell_of_mode[idx] = ell
            idx += 1
    beam_cart = np.zeros((3 * n_ylm, 3, npix), complex)
    ell_full = np.repeat(ell_of_mode, 3)
    for i in range(n_ylm):
        for j in range(3):
            beam_cart[3 * i + j, j] = ylm_maps[i]
    beam = HFSSBeamSet(
        beam_cart=beam_cart,
        gain_th=np.zeros((3 * n_ylm, npix)),
        gain_ph=np.zeros((3 * n_ylm, npix)),
        freqs_mhz=np.arange(3 * n_ylm, dtype=float),
    )
    return beam, ell_full


with open(MODEL_JSON) as f:
    cfg = json.load(f)
beam = HFSSBeamSet.from_npz(BEAM_FILE)
geometry = TransmitterGeometry(
    cfg["geometry"]["heading"], cfg["geometry"]["alpha_deg"])
data = load_v007_data("data")
clean = data_space_rfi_mask(
    "data", beam.freqs_mhz.min(), beam.freqs_mhz.max(), min_votes=5,
) & pointing_valid_mask(data)
pca = compute_beam_pca(beam, n_components=4)
T_pca = _project_templates(pca.components, data, geometry)

sh_beam, ell_full = sh_basis_beams(beam.nside, LMAX)
T_sh = {}
for arm in (0, 1):
    coup, _ = simulate_hfss_coupling(
        sh_beam, data["az_deg"], data["el_deg"], geometry,
        np.full(data["az_deg"].size, arm, dtype=int))
    T_sh[arm] = coup

for ch in CHANNELS:
    arm = tx_arm_for_channel(ch)
    freq_mhz = float(data["freqs"][ch])
    fit = fit_channel(data, pca, T_pca, ch, clean, ridge_lambda=0.003)
    a = (np.array(fit["coefficients_real"])
         + 1j * np.array(fit["coefficients_imag"]))
    m_pca = np.abs(np.conj(a) @ T_pca[arm]) ** 2
    y = data["measured_tx"][:, ch].astype(float)
    az, el = data["az_deg"], data["el_deg"]
    base_valid = channel_validity_masks(data, [ch])[:, 0]
    gross, _, _ = gross_power_time_flags(data, [ch])
    used = base_valid & ~gross & clean
    keep_style = ~((np.round(az, 4) == 0.0) & (np.round(el, 4) == 0.0))

    # Linearized augmentation.
    E_pca = np.conj(a) @ T_pca[arm]
    cE = np.conj(E_pca)
    A_r = 2.0 * np.real(cE[None, :] * T_sh[arm])
    A_i = -2.0 * np.imag(cE[None, :] * T_sh[arm])
    A = np.concatenate([A_r, A_i], axis=0).T
    r = (y - m_pca)[used]
    w_r = 1.0 / (1.0 + ell_full)
    widths = np.concatenate([w_r, w_r])
    rms_r = float(np.sqrt(np.mean(r ** 2)))
    prior_pen = (RIDGE * rms_r) / widths
    lhs = A[used].T @ A[used] + np.diag(prior_pen ** 2)
    rhs = A[used].T @ r
    c = np.linalg.solve(lhs, rhs)
    m_aug = m_pca + A @ c

    rms_pca = float(np.sqrt(np.mean((y[used] - m_pca[used]) ** 2))
                    / np.sqrt(np.mean(y[used] ** 2)))
    rms_aug = float(np.sqrt(np.mean((y[used] - m_aug[used]) ** 2))
                    / np.sqrt(np.mean(y[used] ** 2)))

    dmap = grid(az[keep_style], el[keep_style], y[keep_style])
    m_pca_map = grid(az[keep_style], el[keep_style], m_pca[keep_style])
    m_aug_map = grid(az[keep_style], el[keep_style], m_aug[keep_style])
    r_pca_map = dmap - m_pca_map
    r_aug_map = dmap - m_aug_map

    vmax = np.nanpercentile(dmap, 99.5)
    rs = np.nanpercentile(np.abs(r_pca_map), 99)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    panels = [
        (axes[0, 0], dmap,       f"DATA ch{ch} @ {freq_mhz:.3f} MHz",
         dict(levels=np.linspace(0, vmax, 30), cmap="plasma", extend="both")),
        (axes[0, 1], m_pca_map,  f"MODEL: K=4 HFSS-PCA (rms={rms_pca:.3f})",
         dict(levels=np.linspace(0, vmax, 30), cmap="plasma", extend="both")),
        (axes[0, 2], r_pca_map,  "DATA - MODEL (K=4)",
         dict(levels=np.linspace(-rs, rs, 31), cmap="RdBu_r", extend="both")),
        (axes[1, 0], dmap,       "DATA (same)",
         dict(levels=np.linspace(0, vmax, 30), cmap="plasma", extend="both")),
        (axes[1, 1], m_aug_map,  f"MODEL: K=4 + SH_lmax={LMAX} (rms={rms_aug:.3f})",
         dict(levels=np.linspace(0, vmax, 30), cmap="plasma", extend="both")),
        (axes[1, 2], r_aug_map,  f"DATA - MODEL (SH-augmented)",
         dict(levels=np.linspace(-rs, rs, 31), cmap="RdBu_r", extend="both")),
    ]
    for ax, img, title, kw in panels:
        cs = ax.contourf(elc, azc, np.nan_to_num(img, nan=0.0), **kw)
        ax.set_xlabel("Elevation [deg]")
        ax.set_ylabel("Azimuth [deg]")
        ax.set_ylim(-180, 0)
        ax.set_title(title)
        fig.colorbar(cs, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = f"aug_beammap_ch{ch}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}   K=4 rms={rms_pca:.4f} -> augmented rms={rms_aug:.4f}"
          f"  ({100*(1-rms_aug/rms_pca):.1f}% reduction)")
