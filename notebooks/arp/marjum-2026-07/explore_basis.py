"""Diagnose the basis-expansion frontier for the v007 beam fit.

Three back-to-back experiments, no code changes to fit_v007_pca_beam.py:

1. **K-sweep**: rerun the same coefficient fit with n_components =
   4, 6, 8, 12, 20, 32. If the residual/data ratio keeps dropping,
   the K=4 basis is under-truncated inside HFSS's own span. If it
   plateaus at a value well above the ~7.5% repeatability floor,
   we have hit HFSS's dimensionality limit and any further gain has
   to come from a basis with modes HFSS does not contain.

2. **Angular power spectrum of the fit residual** (K=4). Grid the
   power residual to a coarse HEALPix map in the transmitter-frame
   (theta, phi) coordinates that the fit already uses, run
   healpy.anafast, and see which ell scales dominate. That is the
   angular content the augmented basis has to cover.

3. **Spanning-basis prototype (VSH-lite via HEALPix bump modes)**:
   pick a small set of narrow HEALPix-space "bumps" spread over the
   sampled sky, treat each bump as an additional complex eigen-beam
   plugged into the same simulate_hfss_coupling machinery, and fit
   K=4 HFSS-PCA + N_bump additive modes on a handful of channels
   with a mild ridge on the bump amplitudes. This is a feasibility
   probe, not the final basis: it tells us whether *any* extra
   spanning modes actually pull the residual down, and by how much.
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
from rotation_beam import TransmitterGeometry, vector_to_spherical
from tx_beam_sim import HFSSBeamSet, simulate_hfss_coupling
from v007_beam_diagnostic import (
    channel_validity_masks,
    gross_power_time_flags,
    load_v007_data,
)

MODEL_JSON = "v007_pca_beam_model_v5.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
TEST_CHANNELS = [536, 552, 616, 680, 760]


def normalized_rms(y, m, mask):
    """|y - m| / |y| on the mask, both power-space."""
    good = mask & np.isfinite(y) & np.isfinite(m)
    if good.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean((y[good] - m[good]) ** 2)) /
                 max(np.sqrt(np.mean(y[good] ** 2)), 1e-30))


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


def experiment_1_k_sweep():
    print("\n=== Experiment 1: K-sweep on HFSS-PCA basis ===")
    beam, geometry, data, clean = _prepare()
    Ks = [4, 6, 8, 12, 20, 32]
    header = "  K  " + "  ".join(f"ch{ch:>4d}" for ch in TEST_CHANNELS) + "     mean   exp_var"
    print(header)
    rows_out = {}
    for K in Ks:
        if K > beam.beam_cart.shape[0]:
            continue
        pca = compute_beam_pca(beam, n_components=K)
        T = _project_templates(pca.components, data, geometry)
        cumulative = float(np.sum(pca.explained_variance_ratio))
        row = []
        for ch in TEST_CHANNELS:
            arm = tx_arm_for_channel(ch)
            fit = fit_channel(data, pca, T, ch, clean, ridge_lambda=0.003)
            if fit is None:
                row.append(np.nan)
                continue
            a = (np.array(fit["coefficients_real"])
                 + 1j * np.array(fit["coefficients_imag"]))
            m = np.abs(np.conj(a) @ T[arm]) ** 2
            y = data["measured_tx"][:, ch].astype(float)
            base = channel_validity_masks(data, [ch])[:, 0]
            gross, _, _ = gross_power_time_flags(data, [ch])
            mask = base & ~gross & clean
            row.append(normalized_rms(y, m, mask))
        mean = float(np.nanmean(row))
        rows_out[K] = (row, mean, cumulative)
        cells = "  ".join(f"{v:.4f}" for v in row)
        print(f"  {K:>2d}  {cells}    {mean:.4f}   {cumulative:.6f}")
    return rows_out


def experiment_2_residual_spectrum():
    print("\n=== Experiment 2: angular power spectrum of the K=4 residual ===")
    beam, geometry, data, clean = _prepare()
    K = 4
    pca = compute_beam_pca(beam, n_components=K)
    T = _project_templates(pca.components, data, geometry)
    nside = beam.nside
    # Recompute per-pointing beam-frame pixel index (same as inside
    # simulate_hfss_coupling; no need to reload the beam).
    heading = geometry.heading_top
    az = np.deg2rad(data["az_deg"])
    el = np.deg2rad(data["el_deg"])
    ca, sa = np.cos(az), -np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    Rs = np.empty((az.size, 3, 3), float)
    Rs[:, 0] = np.stack([ca, -sa, np.zeros_like(ca)], axis=1)
    Rs[:, 1] = np.stack([ce * sa, ce * ca, -se], axis=1)
    Rs[:, 2] = np.stack([se * sa, se * ca, ce], axis=1)
    rhat = np.einsum("nij,j->ni", Rs.transpose(0, 2, 1), heading)
    th, ph = vector_to_spherical(rhat)
    px_beamframe = hp.ang2pix(nside, th, ph)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ch in TEST_CHANNELS:
        arm = tx_arm_for_channel(ch)
        fit = fit_channel(data, pca, T, ch, clean, ridge_lambda=0.003)
        if fit is None:
            continue
        a = (np.array(fit["coefficients_real"])
             + 1j * np.array(fit["coefficients_imag"]))
        m = np.abs(np.conj(a) @ T[arm]) ** 2
        y = data["measured_tx"][:, ch].astype(float)
        base = channel_validity_masks(data, [ch])[:, 0]
        gross, _, _ = gross_power_time_flags(data, [ch])
        used = base & ~gross & clean
        res = y - m

        npix = hp.nside2npix(nside)
        sums = np.zeros(npix)
        cnts = np.zeros(npix)
        np.add.at(sums, px_beamframe[used], res[used])
        np.add.at(cnts, px_beamframe[used], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            resmap = np.where(cnts > 0, sums / cnts, 0.0)
        # Normalize by the same-map data RMS so channels are comparable.
        data_sums = np.zeros(npix)
        np.add.at(data_sums, px_beamframe[used], y[used])
        with np.errstate(invalid="ignore", divide="ignore"):
            datamap = np.where(cnts > 0, data_sums / cnts, 0.0)
        # anafast on sampled cells only; unsampled = 0 acts as a
        # spatial window that biases C_ell down at low ell but keeps
        # relative shape across ell intact for comparison.
        cl_res = hp.anafast(resmap, lmax=48, use_pixel_weights=False)
        cl_dat = hp.anafast(datamap, lmax=48, use_pixel_weights=False)
        norm = np.sqrt(np.sum((2 * np.arange(cl_dat.size) + 1) * cl_dat))
        ell = np.arange(cl_res.size)
        axes[0].semilogy(ell, cl_res / max(norm ** 2, 1e-30),
                         label=f"ch{ch} @ {data['freqs'][ch]:.1f} MHz")
        axes[1].plot(ell, np.cumsum((2 * ell + 1) * cl_res)
                     / max(np.sum((2 * ell + 1) * cl_res), 1e-30),
                     label=f"ch{ch}")
        print(f"  ch{ch}: residual C_ell peaks at ell="
              f"{int(np.argmax(cl_res[1:]) + 1)},  50%/90% cumulative "
              f"variance by ell="
              f"{int(np.searchsorted(np.cumsum((2*ell+1)*cl_res), 0.5*np.sum((2*ell+1)*cl_res)))}"
              f"/{int(np.searchsorted(np.cumsum((2*ell+1)*cl_res), 0.9*np.sum((2*ell+1)*cl_res)))}")
    axes[0].set(xlabel="ell", ylabel="C_ell (residual) / data power",
                title="Angular power spectrum of K=4 residual")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="ell", ylabel="cumulative fraction of residual power",
                title="Cumulative residual power vs ell", ylim=(0, 1))
    axes[1].axhline(0.9, color="k", linestyle=":", linewidth=0.5)
    axes[1].axhline(0.5, color="k", linestyle=":", linewidth=0.5)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("explore_residual_cl.png", dpi=120)
    plt.close(fig)
    print("saved explore_residual_cl.png")


def experiment_3_bump_augmentation():
    """Feasibility probe: HFSS-PCA (K=4) + N HEALPix bump modes as
    additional complex eigen-beams. The bumps span the sampled part
    of the beam-frame sphere, so together they can represent *any*
    smooth additive shape the data supports there. Two configs:

    - light: 12 bumps (a small nside=1 HEALPix grid) -> K_total=16
    - heavy: 48 bumps (nside=2)                       -> K_total=52
    """
    print("\n=== Experiment 3: HEALPix-bump augmentation ===")
    beam, geometry, data, clean = _prepare()
    K = 4
    pca = compute_beam_pca(beam, n_components=K)
    T_pca = _project_templates(pca.components, data, geometry)
    nside_beam = beam.nside

    def make_bump_basis(nside_bumps, bump_fwhm_deg=25.0):
        """Complex 'eigen-beams' at low-nside centers, radiating on x,y,z."""
        npix_bumps = hp.nside2npix(nside_bumps)
        npix_beam = hp.nside2npix(nside_beam)
        centers_th, centers_ph = hp.pix2ang(nside_bumps, np.arange(npix_bumps))
        # We build 3 modes per bump: unit E in x, y, z. That covers
        # every polarization channel the model can couple to.
        modes = []
        centers_used = []
        for k in range(npix_bumps):
            xhat = hp.ang2vec(centers_th[k], centers_ph[k])
            all_vec = np.asarray(hp.pix2vec(nside_beam,
                                            np.arange(npix_beam)))
            cos_ang = np.clip(all_vec.T @ xhat, -1.0, 1.0)
            sigma = np.deg2rad(bump_fwhm_deg) / 2.355
            bump = np.exp(-0.5 * (np.arccos(cos_ang) / sigma) ** 2)
            # Only add bumps whose center is inside the sampled region
            # of the beam-frame (see coverage check below).
            centers_used.append((centers_th[k], centers_ph[k], bump))
        return centers_used

    # Coverage map: which beam-frame pixels are ever sampled at all?
    heading = geometry.heading_top
    az = np.deg2rad(data["az_deg"])
    el = np.deg2rad(data["el_deg"])
    ca, sa = np.cos(az), -np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    Rs = np.empty((az.size, 3, 3), float)
    Rs[:, 0] = np.stack([ca, -sa, np.zeros_like(ca)], axis=1)
    Rs[:, 1] = np.stack([ce * sa, ce * ca, -se], axis=1)
    Rs[:, 2] = np.stack([se * sa, se * ca, ce], axis=1)
    rhat = np.einsum("nij,j->ni", Rs.transpose(0, 2, 1), heading)
    th_s, ph_s = vector_to_spherical(rhat)
    covered_pixels = set(hp.ang2pix(nside_beam, th_s, ph_s).tolist())

    for label, nside_bumps in [("light", 1), ("heavy", 2)]:
        centers = make_bump_basis(nside_bumps, bump_fwhm_deg=25.0)
        # Keep bump only if its center pixel (at beam-nside) is covered.
        kept = []
        for th_c, ph_c, bump in centers:
            px_c = hp.ang2pix(nside_beam, th_c, ph_c)
            if px_c in covered_pixels:
                kept.append((th_c, ph_c, bump))
        # 3 field-components per kept bump.
        n_bumps = len(kept)
        n_modes = 3 * n_bumps
        # Build synthetic HFSSBeamSet holding these modes as "frequencies".
        beam_cart = np.zeros((n_modes, 3, hp.nside2npix(nside_beam)),
                             dtype=complex)
        for i, (_, _, bump) in enumerate(kept):
            for j in range(3):
                beam_cart[3 * i + j, j] = bump
        bump_beam = HFSSBeamSet(
            beam_cart=beam_cart,
            gain_th=np.zeros((n_modes, hp.nside2npix(nside_beam))),
            gain_ph=np.zeros((n_modes, hp.nside2npix(nside_beam))),
            freqs_mhz=np.arange(n_modes, dtype=float),
        )
        # Coupling templates for each arm, same machinery.
        T_bump = {}
        for arm in (0, 1):
            coup, _ = simulate_hfss_coupling(
                bump_beam, data["az_deg"], data["el_deg"], geometry,
                np.full(data["az_deg"].size, arm, dtype=int))
            T_bump[arm] = coup

        print(f"\n  [{label}] {n_bumps} bumps kept -> {n_modes} extra "
              f"complex modes on top of K=4; K_total={4 + n_modes}")
        header = "    ch    K=4 rms   augmented rms   improvement"
        print(header)
        for ch in TEST_CHANNELS:
            arm = tx_arm_for_channel(ch)
            # baseline K=4
            fit_base = fit_channel(data, pca, T_pca, ch, clean,
                                   ridge_lambda=0.003)
            if fit_base is None:
                continue
            a = (np.array(fit_base["coefficients_real"])
                 + 1j * np.array(fit_base["coefficients_imag"]))
            m_pca = np.abs(np.conj(a) @ T_pca[arm]) ** 2
            y = data["measured_tx"][:, ch].astype(float)
            base = channel_validity_masks(data, [ch])[:, 0]
            gross, _, _ = gross_power_time_flags(data, [ch])
            used = base & ~gross & clean
            rms_base = normalized_rms(y, m_pca, used)

            # Linearized augmentation: hold the PCA coefficients fixed
            # at the baseline fit and solve a linear least squares for
            # complex bump amplitudes c such that
            #   y ~ |E_pca + sum_j c_j * T_bump_j|^2
            # Expand and drop c*c terms (they'll be tiny if the fit is
            # already close): power ~ |E_pca|^2 + 2 Re{conj(E_pca) *
            # sum_j c_j * T_bump_j}. That is linear in c. Ridge on c.
            E_pca = np.conj(a) @ T_pca[arm]        # complex (ntime,)
            # Design matrix rows: for each bump mode j, contribution to
            # power is 2 Re{conj(E_pca) * T_bump[arm][j]}. But c is
            # complex, so split into (c_r, c_i) real DOFs with columns
            # 2 Re{conj(E_pca)*T} and -2 Im{conj(E_pca)*T}.
            cE = np.conj(E_pca)
            A_r = 2.0 * np.real(cE[None, :] * T_bump[arm])   # (nmodes,ntime)
            A_i = -2.0 * np.imag(cE[None, :] * T_bump[arm])
            A = np.concatenate([A_r, A_i], axis=0).T          # (ntime, 2*nmodes)
            r = (y - m_pca)[used]
            Au = A[used]
            # Ridge on all bump DOFs equally.
            ridge = 0.05 * float(np.sqrt(np.mean(r ** 2)))
            lhs = Au.T @ Au + (ridge ** 2) * np.eye(A.shape[1])
            rhs = Au.T @ r
            c = np.linalg.solve(lhs, rhs)
            m_aug = m_pca + A @ c
            rms_aug = normalized_rms(y, m_aug, used)
            print(f"    {ch:>4d}   {rms_base:.4f}      {rms_aug:.4f}     "
                  f"{100*(1-rms_aug/rms_base):>5.1f}%")


if __name__ == "__main__":
    ex1 = experiment_1_k_sweep()
    experiment_2_residual_spectrum()
    experiment_3_bump_augmentation()
