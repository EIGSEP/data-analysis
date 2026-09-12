"""Fit a low-order, HFSS-informed, data-corrected empirical beam model.

Reduces the HFSS complex-vector beam to a handful of orthogonal spatial
eigen-beams via PCA across frequency (see ``eigsep_data.beam_mapping.basis``), then --
holding the transmitter geometry/polarization fixed at the value
already recovered by ``fit_v007_multichannel_consensus.py`` -- fits
every candidate TX comb channel's data as a linear combination of
those eigen-beams' *complex* polarization coupling, projected through
the real scan trajectory.

This is a genuine generalization of the existing single-frequency
fit, not just a reparameterization: instead of trusting HFSS's exact
predicted beam shape at each channel's own frequency and fitting only
one free real gain per channel, each channel gets ``n_components``
free complex coefficients. The dominant component absorbs the unknown
transmitter gain vs frequency; the higher components let the fit pull
the beam's frequency-dependent shape away from HFSS's own prediction
when the data actually supports it -- that is the "empirical, HFSS-
informed" part.

Power is the squared magnitude of a coherent sum of complex fields, so
the coupling is combined *before* squaring (see
``eigsep_data.beam_mapping.tx_model.simulate_hfss_coupling``): summing each component's own
*power* would silently discard the interference cross-terms between
components and give a physically wrong answer.

Two independent layers of outlier/trust handling, matching the two
kinds of contamination described:
  * Per-time, shared across every channel: ``eigsep_data.beam_mapping.rfi`` flags
    RFI-contaminated times using only the raw data -- smooth-in-time
    departures in off-comb channels and the 0x4 cross-correlation --
    never a fitted beam model. An earlier, model-residual-based
    approach (flagging a channel's own fit residual) was tried and
    found unreliable: a too-simple beam model produces large, coherent
    residuals that look like isolated RFI to a residual-based
    detector, and the resulting reject/refit loop iteratively carved
    out whichever ~quarter of the data didn't fit, making the reported
    fit quality nearly insensitive to how much real model freedom was
    allowed. This is the "outlier data points ... signs of RFI"
    protection now.
  * Across channels, vs frequency: the smooth model of each
    coefficient vs frequency is a DPSS (Slepian) fit via
    ``hera_filters.dspec`` -- the same band-limited, gap-tolerant basis
    HERA's own pipeline uses for inpainting RFI-flagged channels --
    weighted by each channel's own fit quality (1 / normalized_rms**2),
    with an outer sigma-clip that drops channels whose fit is
    inconsistent with their neighbors. FM-band channels are not given
    a hard frequency mask: they simply carry little weight and get
    clipped if they are actual outliers. The DPSS fit is evaluated on
    the full native-resolution channel grid, not just the candidate
    tones, which is what makes "interpolate to any frequency" work --
    untrusted/missing channels are filled in exactly like RFI gaps are
    inpainted in HERA's own spectra.

The saved model is a dense grid of the fitted complex coefficients vs
frequency plus the eigen-beam maps, which is enough to reconstruct the
empirical beam (as a HEALPix map) at any frequency, or -- combined
with the known geometry -- predict the coupled power at any pointing.
"""

import argparse
import json

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from hera_filters import dspec
from scipy.optimize import least_squares

from eigsep_data.beam_mapping import compute_beam_pca
from eigsep_data.beam_mapping import data_space_rfi_mask
from eigsep_data.beam_mapping import TransmitterGeometry, vector_to_spherical
from eigsep_data.beam_mapping import HFSSBeamSet, simulate_hfss_coupling
from eigsep_data.beam_mapping.diagnostics import (
    channel_validity_masks,
    gross_power_time_flags,
    load_v007_data,
    tx_arm_for_channel,
)


def candidate_channels(data, beam):
    """Every eight-bin TX comb channel spanning the HFSS frequency support.

    Matches the comb scan in ``screen_v007_tx_channels.py`` so the two
    scripts agree on which raw channels are physically TX tones.
    """
    df = float(data["freqs"][1] - data["freqs"][0])
    first = int(np.ceil(beam.freqs_mhz.min() / df / 8.0) * 8)
    last = int(np.floor(beam.freqs_mhz.max() / df / 8.0) * 8)
    return np.arange(first, last + 1, 8, dtype=int)


def _unpack_coefficients(x, k):
    """Inverse of :func:`_pack_coefficients`."""
    a = np.empty(k, dtype=complex)
    a[0] = x[0]
    a[1:] = x[1::2][: k - 1] + 1j * x[2::2][: k - 1]
    return a


def _pack_coefficients(a):
    """Flatten complex coefficients to reals, fixing the overall phase.

    An overall complex phase rotation of every coefficient leaves the
    coupled power ``|a @ T|**2`` unchanged, so one real degree of
    freedom is not observable. Fix it by requiring the dominant
    (index 0) component to be real; the caller should rotate its
    initial guess so that component's phase is already zero.
    """
    a = np.asarray(a, dtype=complex)
    k = a.size
    x = np.empty(2 * k - 1)
    x[0] = a[0].real
    x[1::2] = a[1:].real
    x[2::2] = a[1:].imag
    return x


def _basis_from_direction(u):
    """Unitary matrix whose first column is proportional to unit vector ``u``.

    A Householder reflector with ``H @ u = alpha * e0`` for some unit-
    modulus ``alpha``; since ``H`` is Hermitian and unitary (hence its
    own inverse), ``H @ e0 = u / alpha`` -- so ``H``'s first column is
    ``u`` up to a fixed phase. That phase doesn't matter here: it is
    absorbed into the same overall-phase gauge freedom already fixed
    by requiring the fit's leading (index 0) coefficient to be real.
    """
    k = u.size
    e0 = np.zeros(k, dtype=complex)
    e0[0] = 1.0
    alpha = -np.exp(1j * np.angle(u[0])) if np.abs(u[0]) > 1e-12 else -1.0
    v = u - alpha * e0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return np.eye(k, dtype=complex)
    v = v / norm_v
    return np.eye(k, dtype=complex) - 2 * np.outer(v, np.conj(v))


def _project_templates(components, data, geometry):
    """Complex coupling templates for each eigen-component, both TX arms."""
    ntime = data["az_deg"].size
    templates = {}
    for arm in (0, 1):
        coupling, _ = simulate_hfss_coupling(
            components, data["az_deg"], data["el_deg"], geometry,
            np.full(ntime, arm, dtype=int))
        templates[arm] = coupling  # (n_components, ntime), complex
    return templates


def pointing_valid_mask(data):
    """Drop samples where the mount was parked at the motor origin.

    ``load_v007_data`` takes motor ``az_pos``/``el_pos`` at face value,
    and a large block of this dataset (~15%, 1294 of 8400 samples) sits
    at exactly ``az_pos == el_pos == 0`` with ``az_target == el_target
    == 0`` -- the mount genuinely parked at its origin during pre-scan
    idle time, confirmed against the potentiometer (~0 deg). The
    measured TX power nonetheless swings over its full dynamic range
    (~0 to 1.3e7) across that stretch, which a stationary antenna
    cannot do via its own beam: the transmitter side was being set up
    / switched while the mount sat still. Fed into the fit as if they
    were real measurements at pointing (0, 0), these samples alone
    contributed ~77% of the total squared residual while being only
    ~20% of the data, and they all pile into a single HEALPix pixel
    (the 1295-sample pixel that also distorted the coverage map).

    Excluding them halves the per-channel residual RMS (0.42 -> 0.24
    mean over six test channels) and lifts data/model correlation from
    ~0.80 to ~0.95, with no change to the model or fitting machinery.
    A broader "exclude anything stationary" cut gains nothing further
    (0.2382 vs 0.2372), so this stays deliberately narrow and keeps
    legitimate brief pauses mid-scan.
    """
    az = np.asarray(data["az_deg"], float)
    el = np.asarray(data["el_deg"], float)
    parked = (np.round(az, 4) == 0.0) & (np.round(el, 4) == 0.0)
    return ~parked


def hfss_prior_vector(pca, frequency_mhz):
    """HFSS's own predicted coefficient vector at ``frequency_mhz``.

    This *is* "the HFSS model projected onto the PCA basis": ``pca``
    was built directly from the HFSS field, so its per-frequency
    loadings, interpolated to an arbitrary frequency, reconstruct
    exactly what HFSS alone (no data) would predict there.
    """
    real_part = np.array([
        np.interp(frequency_mhz, pca.freqs_mhz, pca.loadings[:, k].real)
        for k in range(pca.loadings.shape[1])])
    imag_part = np.array([
        np.interp(frequency_mhz, pca.freqs_mhz, pca.loadings[:, k].imag)
        for k in range(pca.loadings.shape[1])])
    return real_part + 1j * imag_part


def fit_channel(data, pca, templates_by_arm, channel, clean_mask,
                gross_reference_percentile=99.0, gross_outlier_factor=5.0,
                ridge_lambda=0.003):
    """Robustly fit one candidate channel's PCA coefficients.

    RFI rejection is handled entirely upstream via ``clean_mask`` (see
    ``eigsep_data.beam_mapping.rfi.data_space_rfi_mask``): a single, model-independent,
    data-space flag shared across every channel, rather than an
    iterative reject-and-refit loop keyed to this channel's own beam-
    model residual. The latter was tried first and found to be
    unreliable: with a low-order (K=4) beam model, real, coherent
    model-inadequacy produces large residuals that look "isolated" to
    a per-channel detector, and the reject/refit loop would iteratively
    carve out whichever ~quarter of the data didn't fit well, making
    the final reported fit quality nearly insensitive to how much
    genuine shape freedom was allowed -- confirmed directly: forcing
    the shape correction to exactly zero changed the reported RMS by
    only a few percent, when it should have reverted to the much worse
    single-scale-gain RMS from the pre-PCA model.

    The fitted coefficient vector is reparameterized in a basis whose
    first axis is exactly the HFSS-predicted direction at this
    channel's frequency (see :func:`_basis_from_direction`), rather
    than the arbitrary standard basis. This makes the one genuinely
    unresolvable degree of freedom -- the overall amplitude, which is
    degenerate with the unknown transmitting antenna's own gain/
    efficiency vs frequency -- an explicit fit parameter (``gain``)
    tied to (i.e. multiplying) the HFSS prior direction, instead of
    letting it silently ride along inside whichever standard-basis
    component happened to dominate. The remaining ``k - 1`` complex
    coefficients (``shape``) are then genuine, HFSS-orthogonal shape
    corrections -- deviations from HFSS's own predicted relative beam
    shape at this frequency -- not confounded with the unknown gain.

    Each shape component is ridge-regularized toward zero (i.e. toward
    the HFSS prior) with a prior width set by
    ``sqrt(pca.explained_variance_ratio)`` for that component: HFSS's
    own PCA spectrum *is* an angular power spectrum in this basis (each
    component's own angular power is identical by SVD orthonormality,
    but its explained-variance-ratio measures how much genuine
    frequency-dependent beam variation HFSS predicts along that
    direction -- confirmed separately via a direct spherical-harmonic
    decomposition: components ordered by explained variance are also
    ordered by increasing angular scale, l_99 running from 4 up to 8
    across the four retained components). A component HFSS says barely
    varies (e.g. component 3, 0.05% of variance) gets a tight prior and
    needs strong data support to move; a component HFSS says varies a
    lot (component 1, 9.5%) is allowed more freedom. This is standard
    ridge regression, not a spatial mask: it acts through the actual
    per-channel design matrix, so a combination of shape components
    that has almost no leverage on the real (sampled) data -- e.g. one
    that only manifests in the unsampled cap around boresight -- has a
    small singular value there and gets shrunk hard toward zero
    regardless of the prior width, while well-constrained combinations
    pass through close to their unregularized data-fit value.
    """
    arm = tx_arm_for_channel(channel)
    templates = templates_by_arm[arm]
    k = templates.shape[0]
    frequency_mhz = float(data["freqs"][channel])
    y = data["measured_tx"][:, channel].astype(float)
    sigma = data["measured_sigma"][:, channel].astype(float)
    base = channel_validity_masks(data, [channel])[:, 0]
    gross, _, _ = gross_power_time_flags(
        data, [channel], gross_reference_percentile, gross_outlier_factor)
    valid = base & ~gross
    if valid.sum() < 2 * k + 10:
        return None

    a_hfss = hfss_prior_vector(pca, frequency_mhz)
    gain_hfss = float(np.linalg.norm(a_hfss))
    u = a_hfss / max(gain_hfss, 1e-30)
    basis = _basis_from_direction(u)
    templates_rot = basis.conj().T @ templates
    if ridge_lambda > 0:
        prior_width = ridge_lambda * gain_hfss * np.sqrt(
            np.maximum(pca.explained_variance_ratio[1:], 1e-12))
    else:
        prior_width = np.full(k - 1, np.inf)

    used = valid & clean_mask
    if used.sum() < 2 * k + 10:
        return None

    prior_model = np.abs(np.conj(a_hfss) @ templates) ** 2
    denom = max(float(np.median(prior_model[used])), 1e-30)
    gain_guess = np.sqrt(max(float(np.median(y[used])), 0.0) / denom)
    x0 = _pack_coefficients(
        np.r_[gain_guess * gain_hfss, np.zeros(k - 1, dtype=complex)])
    normalization_rms = max(float(np.sqrt(np.mean(y[used] ** 2))), 1e-30)

    def residual(x):
        params = _unpack_coefficients(x, k)
        field = np.conj(params) @ templates_rot
        model = np.abs(field) ** 2
        data_residual = (model[used] - y[used]) / normalization_rms
        shape_ridge = params[1:] / prior_width
        return np.concatenate(
            [data_residual, shape_ridge.real, shape_ridge.imag])

    result = least_squares(residual, x0, x_scale="jac")
    params = _unpack_coefficients(result.x, k)
    model = np.abs(np.conj(params) @ templates_rot) ** 2
    a = basis @ params
    residual_used = y[used] - model[used]
    initial_rms = float(np.sqrt(np.mean(y[used] ** 2)))
    normalized_rms = float(np.sqrt(np.mean(residual_used ** 2)) / max(initial_rms, 1e-30))
    standardized = residual_used / np.where(sigma[used] > 0, sigma[used], np.nan)
    dof = max(int(used.sum()) - (2 * k - 1), 1)
    reduced_chisq = float(np.nansum(standardized ** 2) / dof)
    return {
        "channel": int(channel),
        "frequency_mhz": frequency_mhz,
        "tx_arm": int(arm),
        "n_valid": int(valid.sum()),
        "n_used": int(used.sum()),
        "n_rfi_flagged": int(valid.sum() - used.sum()),
        "coefficients_real": a.real.tolist(),
        "coefficients_imag": a.imag.tolist(),
        "gain_fitted": float(params[0].real),
        "gain_hfss": gain_hfss,
        "shape_correction_real": params[1:].real.tolist(),
        "shape_correction_imag": params[1:].imag.tolist(),
        "normalized_rms": normalized_rms,
        "reduced_chisq": reduced_chisq,
        "initial_rms": initial_rms,
    }


def _dense_frequency_grid(data, beam):
    """Native-resolution channel grid spanning the HFSS frequency support.

    Every candidate TX comb channel lands exactly on this grid (they
    are a subset of the same raw channel indices), so it doubles as
    the "inpainting" grid DPSS fits onto: channels with data get
    nonzero weight, everything else is filled in by the fit.
    """
    df = float(data["freqs"][1] - data["freqs"][0])
    lo = int(np.ceil(beam.freqs_mhz.min() / df))
    hi = int(np.floor(beam.freqs_mhz.max() / df))
    return lo, np.asarray(data["freqs"][lo:hi + 1], float)


def _dpss_robust_fit(freq_mhz, mask_weight, values, filter_half_width_ns,
                     clip_sigma=4.0, max_iterations=6, eigenval_cutoff=1e-9):
    """Fit ``values`` (defined where ``mask_weight`` > 0) with a DPSS basis.

    Uses ``hera_filters.dspec`` directly rather than a hand-rolled
    polynomial: DPSS/Slepian sequences are the standard low-order,
    band-limited basis for exactly this "smooth function known at a
    sparse, partly-flagged set of frequencies" problem in the
    HERA/21cm pipeline, and unlike a polynomial they don't ring at the
    edges or need a hand-picked degree -- ``eigenval_cutoff`` picks how
    many terms the requested smoothness scale (``filter_half_width_ns``)
    actually supports over this frequency span. Evaluating the fit on
    ``freq_mhz`` directly (rather than only where data exists) is the
    same "inpainting" pattern HERA uses to fill RFI-flagged channels:
    zero-weight the untrusted/missing points and let the DPSS model
    fill them in.
    """
    freq_hz = freq_mhz * 1e6
    design, nterms = dspec.dpss_operator(
        freq_hz, [0.0], [filter_half_width_ns * 1e-9],
        eigenval_cutoff=[eigenval_cutoff])
    used = mask_weight > 0
    model = None
    for _ in range(max_iterations + 1):
        weights = np.where(used, mask_weight, 0.0)
        solution = dspec.fit_solution_matrix(weights, design)
        params = solution @ values
        model = design @ params
        residual = values - model
        good = residual[used]
        scale = 1.4826 * float(np.median(np.abs(good - np.median(good))))
        if scale < 1e-12:
            break
        outliers = used & (np.abs(residual) > clip_sigma * scale)
        if not np.any(outliers):
            break
        used = used & ~outliers
    # dpss_operator always returns a complex128 design matrix (shared with
    # its DFT counterpart); for real input the imaginary part is exactly
    # zero, so this is a dtype cast, not a loss of information.
    return model.real, used, int(nterms[0])


def build_model(data_path, beam_file, consensus_json, n_components=4,
                channel_clip_sigma=4.0, filter_half_width_ns=20.0,
                ridge_lambda=0.003, rfi_clip_sigma=5.0, rfi_min_votes=5):
    with open(consensus_json) as stream:
        consensus = json.load(stream)
    geometry = TransmitterGeometry(consensus["heading"], consensus["alpha_deg"])

    data = load_v007_data(data_path)
    beam = HFSSBeamSet.from_npz(beam_file)
    pca = compute_beam_pca(beam, n_components=n_components)
    templates_by_arm = _project_templates(pca.components, data, geometry)
    clean_mask = data_space_rfi_mask(
        data_path, beam.freqs_mhz.min(), beam.freqs_mhz.max(),
        clip_sigma=rfi_clip_sigma, min_votes=rfi_min_votes)
    clean_mask = clean_mask & pointing_valid_mask(data)

    channels = candidate_channels(data, beam)
    rows = [fit_channel(data, pca, templates_by_arm, ch, clean_mask,
                        ridge_lambda=ridge_lambda)
           for ch in channels]
    rows = [row for row in rows if row is not None]

    freqs = np.array([row["frequency_mhz"] for row in rows])
    arms = np.array([row["tx_arm"] for row in rows])
    weights = 1.0 / np.maximum(
        np.array([row["normalized_rms"] for row in rows]), 1e-3) ** 2
    coeffs_real = np.array([row["coefficients_real"] for row in rows])
    coeffs_imag = np.array([row["coefficients_imag"] for row in rows])
    gain_fitted = np.array([row["gain_fitted"] for row in rows])
    gain_hfss = np.array([row["gain_hfss"] for row in rows])
    shape_real = np.array([row["shape_correction_real"] for row in rows])
    shape_imag = np.array([row["shape_correction_imag"] for row in rows])

    grid_lo, grid_freqs = _dense_frequency_grid(data, beam)
    channel_index = np.array([row["channel"] for row in rows]) - grid_lo
    grid_weight = np.zeros(grid_freqs.size)
    grid_weight[channel_index] = weights

    # HFSS's own prediction needs no fitting -- it *is* the basis this
    # was all built from -- so it's evaluated directly on the dense grid.
    grid_gain_hfss = np.array(
        [np.linalg.norm(hfss_prior_vector(pca, f)) for f in grid_freqs])

    grid_gain_values = np.zeros(grid_freqs.size)
    grid_gain_values[channel_index] = gain_fitted
    grid_gain, used_gain, nterms = _dpss_robust_fit(
        grid_freqs, grid_weight, grid_gain_values,
        filter_half_width_ns, clip_sigma=channel_clip_sigma)
    trusted = used_gain[channel_index].copy()

    grid_shape = {"real": [], "imag": []}
    for j in range(n_components - 1):
        grid_values_re = np.zeros(grid_freqs.size)
        grid_values_re[channel_index] = shape_real[:, j]
        grid_values_im = np.zeros(grid_freqs.size)
        grid_values_im[channel_index] = shape_imag[:, j]
        model_re, used_re, _ = _dpss_robust_fit(
            grid_freqs, grid_weight, grid_values_re,
            filter_half_width_ns, clip_sigma=channel_clip_sigma)
        model_im, used_im, _ = _dpss_robust_fit(
            grid_freqs, grid_weight, grid_values_im,
            filter_half_width_ns, clip_sigma=channel_clip_sigma)
        grid_shape["real"].append(model_re)
        grid_shape["imag"].append(model_im)
        trusted &= used_re[channel_index] & used_im[channel_index]

    return {
        "geometry": consensus,
        "pca": pca,
        "data": data,
        "channels": rows,
        "freqs": freqs,
        "arms": arms,
        "weights": weights,
        "coeffs_real": coeffs_real,
        "coeffs_imag": coeffs_imag,
        "gain_fitted": gain_fitted,
        "gain_hfss": gain_hfss,
        "shape_real": shape_real,
        "shape_imag": shape_imag,
        "trusted": trusted,
        "grid_freqs": grid_freqs,
        "grid_gain": grid_gain,
        "grid_gain_hfss": grid_gain_hfss,
        "grid_shape": grid_shape,
        "n_components": n_components,
        "dpss_nterms": nterms,
    }


def make_diagnostics(model, output_prefix):
    pca = model["pca"]
    freqs = model["freqs"]
    trusted = model["trusted"]
    n_components = model["n_components"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(pca.singular_values, "o-")
    ax.set_xlabel("HFSS PCA component")
    ax.set_ylabel("Singular value")
    ax.set_title("HFSS beam-vs-frequency PCA spectrum\n"
                 f"cumulative variance at k={n_components}: "
                 f"{np.cumsum(pca.explained_variance_ratio)[-1]:.5f}")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_spectrum.png", dpi=160)
    plt.close(fig)

    grid_freqs = model["grid_freqs"]
    n_shape = n_components - 1
    fig, axes = plt.subplots(1 + n_shape, 1, figsize=(9, 2.4 * (1 + n_shape)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.scatter(freqs[trusted], model["gain_fitted"][trusted],
              s=14, c="C0", label="fitted gain, trusted")
    ax.scatter(freqs[~trusted], model["gain_fitted"][~trusted],
              s=14, c="C0", marker="x", label="fitted gain, down-weighted/rejected")
    ax.plot(grid_freqs, model["grid_gain"], c="C0", label="fitted gain, DPSS smooth")
    ax.plot(grid_freqs, model["grid_gain_hfss"], c="k", linestyle="--",
           label="HFSS prior (no data)")
    ax.axvspan(88.0, 108.0, color="gray", alpha=0.15, label="FM band")
    ax.set_ylabel("gain (degenerate w/\nTX antenna efficiency)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Degenerate amplitude: data-fit gain vs the HFSS prior it's tied to")

    for j in range(n_shape):
        ax = axes[1 + j]
        ax.scatter(freqs[trusted], model["shape_real"][trusted, j],
                  s=14, c="C0", label="trusted, Re")
        ax.scatter(freqs[~trusted], model["shape_real"][~trusted, j],
                  s=14, c="C0", marker="x", label="down-weighted/rejected, Re")
        ax.scatter(freqs[trusted], model["shape_imag"][trusted, j],
                  s=14, c="C1", label="trusted, Im")
        ax.scatter(freqs[~trusted], model["shape_imag"][~trusted, j],
                  s=14, c="C1", marker="x", label="down-weighted/rejected, Im")
        ax.plot(grid_freqs, model["grid_shape"]["real"][j], c="C0")
        ax.plot(grid_freqs, model["grid_shape"]["imag"][j], c="C1")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1,
                  label="HFSS prior (zero, by construction)")
        ax.axvspan(88.0, 108.0, color="gray", alpha=0.15)
        ax.set_ylabel(f"shape correction {j + 1}\n(HFSS-orthogonal)")
        if j == 0:
            ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Frequency [MHz]")
    fig.suptitle("Gain (tied to HFSS direction) + orthogonal shape corrections vs "
                f"frequency (DPSS, {model['dpss_nterms']} terms)")
    fig.savefig(f"{output_prefix}_coefficients.png", dpi=160)
    plt.close(fig)


def reconstruct_beam_cart(pca, coefficients):
    """Reconstruct a complex-vector beam map from PCA coefficients.

    ``coefficients`` is a length-``n_components`` complex vector (e.g.
    from :func:`hfss_prior_vector`, or a fitted/smoothed model
    coefficient set); the result has the same shape as one frequency
    slice of the original HFSS beam (``(3, npix)``) and can be plugged
    into the same downstream machinery (``simulate_hfss_coupling``,
    power maps, etc).
    """
    return np.tensordot(coefficients, pca.components.beam_cart, axes=(0, 0))


def beam_power_map(pca, coefficients):
    """Total (unpolarized) power pattern vs HEALPix pixel: sum_c |E_c|**2."""
    beam_cart = reconstruct_beam_cart(pca, coefficients)
    return np.sum(np.abs(beam_cart) ** 2, axis=0)


def fitted_coefficients_at(model, frequency_mhz):
    """Interpolate this model's fitted (gain, shape) to an arbitrary frequency.

    Returns the full complex coefficient vector in the *standard* PCA
    basis (i.e. already rotated out of the per-frequency HFSS-direction
    basis), directly comparable to :func:`hfss_prior_vector`.
    """
    grid_freqs = model["grid_freqs"]
    gain = np.interp(frequency_mhz, grid_freqs, model["grid_gain"])
    shape = np.array([
        np.interp(frequency_mhz, grid_freqs, model["grid_shape"]["real"][j])
        + 1j * np.interp(frequency_mhz, grid_freqs, model["grid_shape"]["imag"][j])
        for j in range(model["n_components"] - 1)
    ])
    a_hfss = hfss_prior_vector(model["pca"], frequency_mhz)
    u = a_hfss / max(np.linalg.norm(a_hfss), 1e-30)
    basis = _basis_from_direction(u)
    params = np.r_[gain, shape]
    return basis @ params


def scan_coverage_counts(data, geometry, nside):
    """Number of real scan time samples landing in each beam-frame pixel.

    The receiver-frame direction probed at each time depends only on
    the scan trajectory (az/el) and the fixed geometry, not on
    frequency or channel -- so this is the same "was this part of the
    beam ever actually measured" map for every comparison. Used to
    flag where a shape correction is a genuine, data-constrained
    result versus a low-order-basis extrapolation into territory the
    scan never visited (verified case in point: every one of the
    largest shape-vs-HFSS discrepancies sits within 4.4 degrees of
    boresight, and every one of those pixels has zero samples --
    the scan's closest approach to boresight is 6.75 degrees).
    """
    az = np.deg2rad(data["az_deg"])
    el = np.deg2rad(data["el_deg"])
    heading = geometry.heading_top
    ca, sa = np.cos(az), -np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    rs = np.empty((az.size, 3, 3))
    rs[:, 0] = np.stack([ca, -sa, np.zeros_like(ca)], axis=1)
    rs[:, 1] = np.stack([ce * sa, ce * ca, -se], axis=1)
    rs[:, 2] = np.stack([se * sa, se * ca, ce], axis=1)
    rhat = np.einsum("nij,j->ni", rs.transpose(0, 2, 1), heading)
    theta, phi = vector_to_spherical(rhat)
    px = hp.ang2pix(nside, theta, phi)
    return np.bincount(px, minlength=hp.nside2npix(nside))


def make_beam_comparison(model, frequencies_mhz, output_prefix):
    """Compare the data-fit empirical beam to HFSS projected onto the same basis.

    For each requested frequency: HFSS's own prediction (projected
    onto the PCA basis, i.e. ``hfss_prior_vector``) and the fitted,
    data-informed model (``fitted_coefficients_at``) are both turned
    into total-power HEALPix maps via :func:`beam_power_map`, and
    plotted side by side with their fractional difference. This is the
    "beam model vs 2D angle" data product, and the direct answer to
    whether the data actually pulled the beam shape away from HFSS at
    a given frequency, or just rides on the (expected, degenerate)
    gain difference.

    The shape-difference panel is masked wherever the real scan never
    sampled that beam-frame pixel (see :func:`scan_coverage_counts`):
    without this, the largest-looking "discrepancies" are actually an
    unconstrained low-order-basis extrapolation into a never-observed
    cap around boresight, not a measured beam-shape disagreement.
    """
    pca = model["pca"]
    nside = pca.components.nside
    geometry = TransmitterGeometry(model["geometry"]["heading"],
                                   model["geometry"]["alpha_deg"])
    coverage = scan_coverage_counts(model["data"], geometry, nside)
    uncovered = coverage == 0
    fig, axes = plt.subplots(len(frequencies_mhz), 4,
                             figsize=(17, 3.4 * len(frequencies_mhz)))
    axes = np.atleast_2d(axes)
    maps = []
    for row, freq in enumerate(frequencies_mhz):
        a_hfss = hfss_prior_vector(pca, freq)
        a_fit = fitted_coefficients_at(model, freq)
        map_hfss = beam_power_map(pca, a_hfss)
        map_fit = beam_power_map(pca, a_fit)
        # Normalize by total (sky-integrated) power, not peak. The
        # cross term between the HFSS-tied gain and the orthogonal
        # shape correction integrates to exactly zero over the sphere
        # (verified numerically: ~1e-9 relative), so total-power
        # normalization makes the shape difference exactly zero-mean
        # by construction. Peak normalization was tried first but a
        # single reference pixel lets the shape correction's own
        # (strictly non-negative, unavoidable) |Delta|**2 self-power
        # bias bleed into the normalization itself, producing a
        # spurious whole-sphere offset that isn't a real shape
        # disagreement -- confirmed by comparing the two normalizations
        # directly (peak-normalized mean difference: +0.07 at 70 MHz to
        # -0.02 at 200 MHz; total-power-normalized mean: 0 at every
        # frequency, to numerical precision).
        norm_hfss = max(float(np.mean(map_hfss)), 1e-30)
        norm_fit = max(float(np.mean(map_fit)), 1e-30)
        shape_diff = map_fit / norm_fit - map_hfss / norm_hfss
        peak_hfss = max(float(np.max(map_hfss)), 1e-30)
        peak_fit = max(float(np.max(map_fit)), 1e-30)
        maps.append({
            "frequency_mhz": float(freq),
            "hfss_power_map": map_hfss.tolist(),
            "fitted_power_map": map_fit.tolist(),
            "scan_sample_counts": coverage.tolist(),
        })
        shape_diff_masked = shape_diff.copy()
        shape_diff_masked[uncovered] = hp.UNSEEN
        plt.axes(axes[row, 0])
        hp.mollview(10 * np.log10(map_hfss / peak_hfss), hold=True, cbar=True,
                   title=f"{freq:.1f} MHz: HFSS prior (dB, peak-normalized)",
                   unit="dB", min=-30, max=0)
        plt.axes(axes[row, 1])
        hp.mollview(10 * np.log10(map_fit / peak_fit), hold=True, cbar=True,
                   title=f"{freq:.1f} MHz: data-fit model (dB, peak-normalized)",
                   unit="dB", min=-30, max=0)
        plt.axes(axes[row, 2])
        hp.mollview(shape_diff_masked, hold=True, cbar=True,
                   title="shape difference (grey = never sampled by\n"
                         "the real scan -- extrapolation, not a measurement)",
                   min=-0.3, max=0.3, cmap="RdBu_r")
        plt.axes(axes[row, 3])
        hp.mollview(np.log10(coverage + 1), hold=True, cbar=True,
                   title="log10(1 + scan sample count) per pixel\n"
                         "(same for every frequency)",
                   cmap="viridis")
    fig.savefig(f"{output_prefix}_beam_comparison.png", dpi=140)
    plt.close(fig)
    return maps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("beam_file")
    ap.add_argument("consensus_json")
    ap.add_argument("--n-components", type=int, default=4)
    ap.add_argument("--channel-clip-sigma", type=float, default=4.0)
    ap.add_argument("--filter-half-width-ns", type=float, default=20.0,
                    help="DPSS smoothness scale for coefficients vs frequency")
    ap.add_argument("--ridge-lambda", type=float, default=0.003,
                    help="shape-correction ridge strength, in units of "
                         "gain_hfss * sqrt(explained_variance_ratio). "
                         "Validated by direct comparison: 3.0 (the initial "
                         "dimensional-analysis guess) barely constrained "
                         "anything; 0.003 suppresses the boresight "
                         "extrapolation artifact from ~0.5-0.8 fractional "
                         "power error down to ~0.01-0.08, at a median RMS "
                         "cost of about 2%.")
    ap.add_argument("--rfi-clip-sigma", type=float, default=5.0,
                    help="per-monitor-channel MAD threshold for the "
                         "data-space RFI flag")
    ap.add_argument("--rfi-min-votes", type=int, default=5,
                    help="number of independent monitor channels/products "
                         "that must simultaneously flag a time for it to "
                         "be treated as RFI")
    ap.add_argument("--comparison-freqs-mhz", type=float, nargs="+",
                    default=[70.0, 130.0, 150.0, 200.0],
                    help="frequencies for the HFSS-vs-data-fit beam map comparison")
    ap.add_argument("--output-json", default="v007_pca_beam_model.json")
    ap.add_argument("--output-prefix", default="v007_pca_beam_model")
    args = ap.parse_args()

    model = build_model(
        args.data_path, args.beam_file, args.consensus_json,
        n_components=args.n_components,
        channel_clip_sigma=args.channel_clip_sigma,
        filter_half_width_ns=args.filter_half_width_ns,
        ridge_lambda=args.ridge_lambda,
        rfi_clip_sigma=args.rfi_clip_sigma,
        rfi_min_votes=args.rfi_min_votes)
    make_diagnostics(model, args.output_prefix)
    comparison_maps = make_beam_comparison(
        model, args.comparison_freqs_mhz, args.output_prefix)

    report = {
        "n_components": model["n_components"],
        "explained_variance_ratio": model["pca"].explained_variance_ratio.tolist(),
        "channels": model["channels"],
        "trusted": model["trusted"].tolist(),
        "dpss_nterms": model["dpss_nterms"],
        "dpss_filter_half_width_ns": args.filter_half_width_ns,
        "frequency_grid_mhz": model["grid_freqs"].tolist(),
        "gain_fitted_grid": model["grid_gain"].tolist(),
        "gain_hfss_grid": model["grid_gain_hfss"].tolist(),
        "shape_correction_real_grid": [
            model["grid_shape"]["real"][j].tolist()
            for j in range(model["n_components"] - 1)
        ],
        "shape_correction_imag_grid": [
            model["grid_shape"]["imag"][j].tolist()
            for j in range(model["n_components"] - 1)
        ],
        "eigen_beam_cart_real": model["pca"].components.beam_cart.real.tolist(),
        "eigen_beam_cart_imag": model["pca"].components.beam_cart.imag.tolist(),
        "beam_comparison_maps": comparison_maps,
        "geometry": model["geometry"],
    }
    rendered = json.dumps(report)
    with open(args.output_json, "w") as stream:
        stream.write(rendered + "\n")
    print(json.dumps({
        "n_channels_fit": len(model["channels"]),
        "n_trusted": int(model["trusted"].sum()),
        "explained_variance_ratio": model["pca"].explained_variance_ratio.tolist(),
        "median_normalized_rms": float(np.median(
            [row["normalized_rms"] for row in model["channels"]])),
        "gain_fitted_vs_hfss_ratio_at_comparison_freqs": [
            float(np.interp(f, model["grid_freqs"], model["grid_gain"])
                 / np.interp(f, model["grid_freqs"], model["grid_gain_hfss"]))
            for f in args.comparison_freqs_mhz
        ],
        "output_json": args.output_json,
        "output_plots": [f"{args.output_prefix}_spectrum.png",
                         f"{args.output_prefix}_coefficients.png",
                         f"{args.output_prefix}_beam_comparison.png"],
    }, indent=2))
