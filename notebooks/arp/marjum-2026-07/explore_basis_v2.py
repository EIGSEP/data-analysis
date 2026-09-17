"""Round 2: SH augmentation with a real C_ell prior, and a joint
nonlinear fit for one channel to check the linearized lower bound.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import healpy as hp
from scipy.optimize import least_squares

from eigsep_data.beam_mapping import compute_beam_pca
from eigsep_data.beam_mapping import data_space_rfi_mask
from fit_v007_pca_beam import (
    _project_templates,
    fit_channel,
    pointing_valid_mask,
    tx_arm_for_channel,
)
from eigsep_data.beam_mapping import TransmitterGeometry, vector_to_spherical
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
    """Real-valued Y_lm HEALPix maps as complex 'eigen-beams' for each
    Cartesian field component. Returns HFSSBeamSet with (lmax+1)^2 *
    3 modes.
    """
    npix = hp.nside2npix(nside)
    n_ylm = (lmax + 1) ** 2
    # Build each real Y_lm by inverse-SH from a single-nonzero alm.
    alm_size = hp.Alm.getsize(lmax)
    ylm_maps = np.empty((n_ylm, npix))
    idx = 0
    ell_of_mode = np.empty(n_ylm, dtype=int)
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            alm = np.zeros(alm_size, complex)
            if m == 0:
                alm[hp.Alm.getidx(lmax, ell, 0)] = 1.0
            elif m > 0:
                # sqrt(2) so ||Y_lm^real|| matches ||Y_lm^complex||
                alm[hp.Alm.getidx(lmax, ell, m)] = 1.0 / np.sqrt(2)
            else:
                alm[hp.Alm.getidx(lmax, ell, -m)] = 1j / np.sqrt(2) * (
                    -1 if (-m) % 2 else 1)
            ylm_maps[idx] = hp.alm2map(alm, nside, lmax=lmax, verbose=False)
            ell_of_mode[idx] = ell
            idx += 1
    # 3 field-component copies (mode along x, y, z).
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


def hfss_cl_prior(beam):
    """Angular power spectrum of the actual HFSS field, per component,
    averaged over frequency slices. Used to give the SH augmentation
    an HFSS-consistent smoothness prior.
    """
    lmax = 3 * beam.nside - 1
    # Only lmax up to 32 for the prior (matches SH augmentation)
    lmax = min(lmax, 32)
    ell = np.arange(lmax + 1)
    cl_sum = np.zeros(lmax + 1)
    n = 0
    for f in range(beam.beam_cart.shape[0]):
        for j in range(3):
            mp_re = np.real(beam.beam_cart[f, j])
            mp_im = np.imag(beam.beam_cart[f, j])
            cl_sum += hp.anafast(mp_re, lmax=lmax)
            cl_sum += hp.anafast(mp_im, lmax=lmax)
            n += 2
    return ell, cl_sum / n


def experiment_sh_augmentation(lmax=8):
    print(f"\n=== SH augmentation (lmax={lmax}, C_ell prior from HFSS) ===")
    beam, geometry, data, clean = _prepare()
    K = 4
    pca = compute_beam_pca(beam, n_components=K)
    T_pca = _project_templates(pca.components, data, geometry)

    # SH modes, evaluated at the beam's native nside for the field.
    sh_beam, ell_full = sh_basis_beams(beam.nside, lmax)
    print(f"  {sh_beam.beam_cart.shape[0]} SH-field modes "
          f"(3 * (lmax+1)^2)")

    ell_prior, cl_prior = hfss_cl_prior(beam)
    # Interpolate C_ell prior to our ell values; small floor for
    # numerical safety.
    cl_at_ell = np.interp(ell_full, ell_prior, cl_prior)
    cl_at_ell = np.maximum(cl_at_ell, 1e-6 * cl_prior.max())
    # Prior width for each SH DOF ~ sqrt(C_ell).
    sqrt_cl = np.sqrt(cl_at_ell)  # (nmodes,)

    T_sh = {}
    for arm in (0, 1):
        coup, _ = simulate_hfss_coupling(
            sh_beam, data["az_deg"], data["el_deg"], geometry,
            np.full(data["az_deg"].size, arm, dtype=int))
        T_sh[arm] = coup

    header = ("    ch    K=4 rms   +SH linear   +SH joint    "
              "improvement (linear/joint)")
    print(header)

    results = {}
    for ch in TEST_CHANNELS:
        arm = tx_arm_for_channel(ch)
        fit = fit_channel(data, pca, T_pca, ch, clean, ridge_lambda=0.003)
        if fit is None:
            continue
        a = (np.array(fit["coefficients_real"])
             + 1j * np.array(fit["coefficients_imag"]))
        m_pca = np.abs(np.conj(a) @ T_pca[arm]) ** 2
        y = data["measured_tx"][:, ch].astype(float)
        base = channel_validity_masks(data, [ch])[:, 0]
        gross, _, _ = gross_power_time_flags(data, [ch])
        used = base & ~gross & clean
        rms_base = normalized_rms(y, m_pca, used)

        # Linearized: solve for complex SH amplitudes with C_ell prior.
        E_pca = np.conj(a) @ T_pca[arm]
        cE = np.conj(E_pca)
        # T_sh[arm] shape: (nmodes, ntime). Design matrix rows for
        # real+imag c: A_r = 2 Re{cE*T}, A_i = -2 Im{cE*T}.
        A_r = 2.0 * np.real(cE[None, :] * T_sh[arm])
        A_i = -2.0 * np.imag(cE[None, :] * T_sh[arm])
        A = np.concatenate([A_r, A_i], axis=0).T
        r = (y - m_pca)[used]
        Au = A[used]
        # Prior width per complex mode: sqrt(cl_at_ell) * data_amp.
        # Use r's RMS as the amplitude scale.
        rms_r = float(np.sqrt(np.mean(r ** 2)))
        # Concatenate widths for real then imag parts.
        widths = np.concatenate([sqrt_cl, sqrt_cl])
        # Higher ridge = more regularization. Use inverse of widths.
        prior_pen = (0.05 * rms_r) / (widths / widths.max())
        lhs = Au.T @ Au + np.diag(prior_pen ** 2)
        rhs = Au.T @ r
        c_lin = np.linalg.solve(lhs, rhs)
        m_lin = m_pca + A @ c_lin
        rms_lin = normalized_rms(y, m_lin, used)

        # Joint nonlinear fit: optimize both K=4 PCA (starting from
        # baseline) AND SH amplitudes together. This tells us how
        # much the linearized fit under-estimates. Only run for ch536
        # and ch616 since it's expensive.
        rms_joint = np.nan
        if ch in (536, 616):
            k_pca = K
            n_sh_modes = A.shape[1] // 2  # complex SH DOFs
            # Params: 2K-1 real for PCA (fix overall phase gauge),
            # then 2*n_sh_modes reals for SH.
            def unpack(x):
                a_pca = np.zeros(k_pca, complex)
                a_pca[0] = x[0]
                a_pca[1:].real = x[1:k_pca * 2 - 1:2] if False else x[1:2 * k_pca - 1:2]
                a_pca[1:] = x[1:2 * k_pca - 1:2] + 1j * x[2:2 * k_pca - 1:2]
                c_sh = x[2 * k_pca - 1:]  # length 2*n_sh_modes
                return a_pca, c_sh

            # Simpler unpack.
            def unpack2(x):
                a_pca = np.empty(k_pca, complex)
                a_pca[0] = x[0]
                for kk in range(1, k_pca):
                    a_pca[kk] = x[2 * kk - 1] + 1j * x[2 * kk]
                c_sh = x[2 * k_pca - 1:]
                return a_pca, c_sh

            def pack(a_pca_init, c_sh_init):
                x = np.zeros(2 * k_pca - 1 + 2 * n_sh_modes)
                x[0] = a_pca_init[0].real
                for kk in range(1, k_pca):
                    x[2 * kk - 1] = a_pca_init[kk].real
                    x[2 * kk] = a_pca_init[kk].imag
                x[2 * k_pca - 1:] = c_sh_init
                return x

            # SH field per pointing given complex c = c_r + i c_i:
            #   E_sh = sum_j (c_r_j + i c_i_j) * T_sh[arm][j]
            # But we've defined A with the complex convention already.
            # Model total field: E_total = conj(a_pca) @ T_pca + E_sh.
            # We need to build E_sh from real+imag halves consistently.
            def residual(x):
                a_pca, c_sh = unpack2(x)
                c_r = c_sh[:n_sh_modes]
                c_i = c_sh[n_sh_modes:]
                c_complex = c_r + 1j * c_i
                E = np.conj(a_pca) @ T_pca[arm] + c_complex @ T_sh[arm]
                m = np.abs(E) ** 2
                data_r = (m[used] - y[used]) / rms_r
                # SH ridge on all SH DOFs, weighted by inv prior width.
                pen_scale = 0.05
                pen_r = pen_scale * c_r / (sqrt_cl / sqrt_cl.max())
                pen_i = pen_scale * c_i / (sqrt_cl / sqrt_cl.max())
                # Mild PCA ridge for numerical stability.
                pen_pca_r = 0.001 * np.array(
                    [a_pca[kk].real for kk in range(1, k_pca)])
                pen_pca_i = 0.001 * np.array(
                    [a_pca[kk].imag for kk in range(1, k_pca)])
                return np.concatenate(
                    [data_r, pen_r, pen_i, pen_pca_r, pen_pca_i])

            x0 = pack(a, np.zeros(2 * n_sh_modes))
            res = least_squares(residual, x0, method="lm", max_nfev=500,
                                xtol=1e-9, ftol=1e-9)
            a_pca_fit, c_sh_fit = unpack2(res.x)
            c_r = c_sh_fit[:n_sh_modes]
            c_i = c_sh_fit[n_sh_modes:]
            c_complex = c_r + 1j * c_i
            E = np.conj(a_pca_fit) @ T_pca[arm] + c_complex @ T_sh[arm]
            m_joint = np.abs(E) ** 2
            rms_joint = normalized_rms(y, m_joint, used)

        imp_lin = 100 * (1 - rms_lin / rms_base)
        imp_joint = (100 * (1 - rms_joint / rms_base)
                     if np.isfinite(rms_joint) else np.nan)
        print(f"    {ch:>4d}   {rms_base:.4f}    {rms_lin:.4f}      "
              f"{rms_joint if np.isnan(rms_joint) else f'{rms_joint:.4f}':<8}   "
              f"{imp_lin:>5.1f}% / "
              f"{'---' if np.isnan(imp_joint) else f'{imp_joint:>5.1f}%'}")
        results[ch] = (rms_base, rms_lin, rms_joint)
    return results


if __name__ == "__main__":
    experiment_sh_augmentation(lmax=8)
    experiment_sh_augmentation(lmax=12)
