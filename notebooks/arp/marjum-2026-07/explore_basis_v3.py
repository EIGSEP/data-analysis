"""Round 2 (fast): SH augmentation linearized fit as a function of
lmax. Also sweep the ridge strength to find the operating point,
since the linearized fit is guaranteed to be a lower bound on the
joint nonlinear fit's improvement.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import healpy as hp

from eigsep_data.beam_mapping import compute_beam_pca
from eigsep_data.beam_mapping import data_space_rfi_mask
from fit_v007_pca_beam import (
    _project_templates,
    fit_channel,
    pointing_valid_mask,
    tx_arm_for_channel,
)
from eigsep_data.beam_mapping import TransmitterGeometry
from eigsep_data.beam_mapping import HFSSBeamSet, simulate_hfss_coupling
from eigsep_data.beam_mapping.diagnostics import (
    channel_validity_masks,
    gross_power_time_flags,
    load_v007_data,
)

MODEL_JSON = "v007_pca_beam_model_v5.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
TEST_CHANNELS = [536, 552, 616, 680, 760]


def normalized_rms(y, m, mask):
    good = mask & np.isfinite(y) & np.isfinite(m)
    if good.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean((y[good] - m[good]) ** 2)) /
                 max(np.sqrt(np.mean(y[good] ** 2)), 1e-30))


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


def _prepare():
    with open(MODEL_JSON) as f:
        cfg = json.load(f)
    beam = HFSSBeamSet.from_npz(BEAM_FILE)
    geometry = TransmitterGeometry(
        cfg["geometry"]["heading"], cfg["geometry"]["alpha_deg"])
    data = load_v007_data("data")
    clean = data_space_rfi_mask(
        "data", beam.freqs_mhz.min(), beam.freqs_mhz.max(), min_votes=5,
    ) & pointing_valid_mask(data)
    return beam, geometry, data, clean


def linearized_aug_fit(y, m_pca, a, T_pca, T_sh, ell_full, arm, used,
                       ridge_scale=0.05):
    """Solve for complex SH augmentation coefficients with a
    1/(l+1)^2 prior width scaling on top of ``ridge_scale``.
    """
    E_pca = np.conj(a) @ T_pca[arm]
    cE = np.conj(E_pca)
    A_r = 2.0 * np.real(cE[None, :] * T_sh[arm])
    A_i = -2.0 * np.imag(cE[None, :] * T_sh[arm])
    A = np.concatenate([A_r, A_i], axis=0).T
    r = (y - m_pca)[used]
    Au = A[used]
    # Prior width per mode: shrinks with ell (smoother mode -> more freedom).
    w_r = 1.0 / (1.0 + ell_full)
    widths = np.concatenate([w_r, w_r])
    rms_r = float(np.sqrt(np.mean(r ** 2)))
    prior_pen = (ridge_scale * rms_r) / widths
    lhs = Au.T @ Au + np.diag(prior_pen ** 2)
    rhs = Au.T @ r
    c = np.linalg.solve(lhs, rhs)
    m_aug = m_pca + A @ c
    return m_aug, c


def sweep_lmax_and_ridge():
    print("\n=== SH augmentation: lmax + ridge sweep (linearized, "
          "lower bound on joint fit) ===")
    beam, geometry, data, clean = _prepare()
    K = 4
    pca = compute_beam_pca(beam, n_components=K)
    T_pca = _project_templates(pca.components, data, geometry)

    # Precompute channel-baseline fits (K=4) once.
    baseline = {}
    for ch in TEST_CHANNELS:
        arm = tx_arm_for_channel(ch)
        fit = fit_channel(data, pca, T_pca, ch, clean, ridge_lambda=0.003)
        a = (np.array(fit["coefficients_real"])
             + 1j * np.array(fit["coefficients_imag"]))
        m_pca = np.abs(np.conj(a) @ T_pca[arm]) ** 2
        y = data["measured_tx"][:, ch].astype(float)
        base = channel_validity_masks(data, [ch])[:, 0]
        gross, _, _ = gross_power_time_flags(data, [ch])
        used = base & ~gross & clean
        rms_base = normalized_rms(y, m_pca, used)
        baseline[ch] = (arm, a, m_pca, y, used, rms_base)

    for lmax in [4, 6, 8, 10, 12]:
        sh_beam, ell_full = sh_basis_beams(beam.nside, lmax)
        n_modes = sh_beam.beam_cart.shape[0]
        T_sh = {}
        for arm in (0, 1):
            coup, _ = simulate_hfss_coupling(
                sh_beam, data["az_deg"], data["el_deg"], geometry,
                np.full(data["az_deg"].size, arm, dtype=int))
            T_sh[arm] = coup

        print(f"\n  lmax={lmax}, {n_modes} SH modes (3*(lmax+1)^2)")
        print("     ridge   " + "  ".join(f"ch{ch}" for ch in TEST_CHANNELS)
              + "     mean")
        for ridge_scale in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
            row = []
            for ch in TEST_CHANNELS:
                arm, a, m_pca, y, used, rms_base = baseline[ch]
                m_aug, _ = linearized_aug_fit(
                    y, m_pca, a, T_pca, T_sh, ell_full, arm, used,
                    ridge_scale=ridge_scale)
                rms_aug = normalized_rms(y, m_aug, used)
                row.append(rms_aug)
            mean = float(np.nanmean(row))
            cells = "  ".join(f"{v:.4f}" for v in row)
            print(f"     {ridge_scale:.2f}   {cells}    {mean:.4f}")

        # Report improvement at best ridge per channel.
        print("     best per-ch improvement so far vs K=4 baseline:")
        for ch in TEST_CHANNELS:
            arm, a, m_pca, y, used, rms_base = baseline[ch]
            best = rms_base
            for ridge_scale in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
                m_aug, _ = linearized_aug_fit(
                    y, m_pca, a, T_pca, T_sh, ell_full, arm, used,
                    ridge_scale=ridge_scale)
                best = min(best, normalized_rms(y, m_aug, used))
            print(f"       ch{ch}: {rms_base:.4f} -> {best:.4f} "
                  f"({100*(1-best/rms_base):.1f}%)")


if __name__ == "__main__":
    sweep_lmax_and_ridge()
