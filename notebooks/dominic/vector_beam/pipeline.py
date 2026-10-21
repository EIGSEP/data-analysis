"""
Marjum 2026-07 inputs for the vector-beam solve (eigsep_data.vector_beam).

The method lives in the package; this file holds the campaign choices:
the data window, the cached geometry, the sky/ground RFI recipe, and the
constants of the cross-correlation reduction. It reproduces the inputs of
the September 2026 study (notebooks/dominic/closure_fit_explore_v2.py on
the old layout), which read potentiometer azimuth through
``extract_beam_mapping_data`` and a geometry fitted by the since-removed
``fit_multi_freq_joint`` (``beam_sim.transmitter_fields``). Phase 2
(run_phase2.py) swaps in pointing table v3 and the v0013 geometry.

``load()`` returns everything a solve needs, keyed by name.
"""

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    median_filter,
    percentile_filter,
)

from eigsep_data.beam_sim import read_beam, transmitter_fields
from eigsep_data.data import (
    calibrate_weak_arm,
    extract_beam_mapping_data,
    extract_clean_pot_data_v2,
)
from eigsep_data.rfi import (
    build_frequency_trust_mask,
    fit_dpss_model_per_time,
    flag_rfi_time,
    freq_axis_flag_reduced,
)
from eigsep_data.beam_mapping import vector_beam as vb

#: Geometry fitted by fit_multi_freq_joint on 2026-09-17 against the
#: corrected frequency axis, plus the data window it was fitted on.
CACHE = Path(
    os.environ.get(
        "VECTOR_BEAM_CACHE",
        Path(__file__).resolve().parents[1] / "joint_fit_cache.npz",
    )
)

CHS = np.arange(8, 1024, 8)  # transmitter comb tones, arms alternating
SHIFT = 27  # leading tones dropped (below the band of interest)
TAU_NS = 91.65  # cable delay, from the phase slope of measured tones
TRUST = dict(min_freq=87.0, fm_low=87.0, fm_high=108.0, max_freq=230.0)
NMODES, NW = 23, 4.0


def _freq_axis_flag(raw, chs, freqs, window=9, sigma=6.0, chronic=0.10):
    """Per-arm frequency-axis flag on raw comb tones."""
    chs = np.asarray(chs)
    arm = np.arange(len(chs)) % 2
    mask = np.zeros(raw.shape, dtype=bool)
    for a in (0, 1):
        chs_arm = chs[arm == a]
        chs_arm = chs_arm[np.argsort(freqs[chs_arm])]
        sub = raw[:, chs_arm].astype(float)
        m, _, _ = flag_rfi_time(sub.T, window=window, sigma=sigma)
        m = m.T
        m[:, m.mean(axis=0) > chronic] = False
        mask[:, chs_arm] = m
    return mask


def sky_ground_flags(sky, ground, freqs, chs=CHS):
    """The September sky/ground RFI recipe: time- and frequency-axis
    flags on each, plus broadband dropouts on sky and broadband
    brightenings on ground."""
    sky_m, _, _ = flag_rfi_time(sky, window=9)
    gnd_m, _, _ = flag_rfi_time(ground, window=9, floor_fraction=1.0)
    sky_m |= _freq_axis_flag(sky, chs, freqs)
    gnd_m |= _freq_axis_flag(ground, chs, freqs)

    tone_chs = chs[freqs[chs] > 50.0]
    base = percentile_filter(
        sky[:, tone_chs].astype(float),
        percentile=80,
        size=(21, 1),
        mode="nearest",
    )
    tone_drop = np.mean(sky[:, tone_chs] / np.maximum(base, 1e-9) < 0.6, 1)
    is_tone = np.zeros(sky.shape[1], dtype=bool)
    is_tone[chs] = True
    nontone = np.flatnonzero(~is_tone & (freqs > 50.0) & (freqs < 246.0))
    base = percentile_filter(
        sky[:, nontone].astype(float),
        percentile=80,
        size=(21, 1),
        mode="nearest",
    )
    nontone_drop = np.mean(sky[:, nontone] / np.maximum(base, 1e-9) < 0.6, 1)
    event = binary_dilation(
        (tone_drop > 0.5) & (nontone_drop > 0.1), np.ones(3, dtype=bool)
    )
    sky_m[event, :] = True

    base = median_filter(
        ground[:, tone_chs].astype(float), size=(21, 1), mode="nearest"
    )
    bright = np.mean(ground[:, tone_chs] / np.maximum(base, 1e-9) > 1.3, 1)
    event = binary_dilation(bright > 0.2, np.ones(3, dtype=bool))
    gnd_m[event, :] = True
    return sky_m, gnd_m


def cross_tone_flags(cross, scale_factors, sky_or_ground, freqs_mhz):
    """Flags on the reduced, shifted cross tones: any of the three raw
    channels behind a tone flagged in sky or ground, plus time- and
    frequency-axis flags on the derotated |cross|, gap closing and a
    one-sample dilation in time."""
    reduced = cross[:, CHS] - 0.5 * (cross[:, CHS - 1] + cross[:, CHS + 1])
    reduced[:, 1::2] *= np.sqrt(scale_factors)
    reduced = reduced[:, SHIFT:]
    inherited = (
        sky_or_ground[:, CHS]
        | sky_or_ground[:, CHS - 1]
        | sky_or_ground[:, CHS + 1]
    )[:, SHIFT:]
    derot = np.exp(-2j * np.pi * freqs_mhz * 1e6 * TAU_NS * 1e-9)
    mag = np.abs(reduced * derot[None, :])
    own, _, _ = flag_rfi_time(mag, window=9)
    own |= freq_axis_flag_reduced(mag, window=9)
    mask = binary_closing(own | inherited, np.ones((9, 1), dtype=bool))
    return binary_dilation(mask, np.ones((3, 1), dtype=bool))


def load(require_bracketed_fill=True):
    """require_bracketed_fill=False reproduces the September inputs
    exactly; the default NaNs DPSS fills that are extrapolations (see
    vb.reduce_cross)."""
    cache = np.load(CACHE, allow_pickle=True)
    ext = extract_beam_mapping_data(
        Path(str(cache["data_dir"])),
        str(cache["start_time"]),
        str(cache["end_time"]),
        file_patterns=("*.h5", "*.hdf5"),
        sweep_slice=slice(int(cache["sweep_slice_start"]), None),
    )
    freqs = ext["freqs"]
    sky, ground, cross = ext["sky"], ext["ground"], ext["cross"]

    az_deg = extract_clean_pot_data_v2(
        np.copy(ext["pot_az_angle"]), ext["az_pos"] / (113 / 1.8)
    )
    el_deg = ext["imu_el_deg"].copy()

    rot = fit_dpss_model_per_time(
        raw=sky,
        chs=CHS,
        freqs=freqs,
        nw=4.0,
        nterms=32,
        include_poly=True,
        poly_order=2,
        guard_bins=1,
        fit_min_chan=min(CHS) - 8,
        fit_max_chan=sky.shape[1],
        min_freq=87.0,
        use_log=True,
        ridge=1e-5,
        verbose=False,
    )["reduced"]
    scale_factors, _ = calibrate_weak_arm(el_deg, az_deg, rot)

    sky_m, gnd_m = sky_ground_flags(sky, ground, freqs)
    freqs_mhz = freqs[::8][SHIFT + 1 :]
    tone_mask = cross_tone_flags(
        cross, scale_factors, sky_m | gnd_m, freqs_mhz
    )
    trust = build_frequency_trust_mask(freqs_mhz, **TRUST)

    Ex_hat, Ey_hat, bad_time, Ex_meas, Ey_meas = vb.reduce_cross(
        cross,
        CHS,
        tone_mask,
        scale_factors,
        freqs_mhz,
        SHIFT,
        TAU_NS,
        trust_mask=trust,
        nmodes=NMODES,
        NW=NW,
        require_bracketed_fill=require_bracketed_fill,
    )

    beam, _, _ = read_beam()
    nside = int(np.sqrt(beam.shape[-1] / 12))
    theta = float(cache["best_theta"])
    phi = float(cache["best_phi"])
    alpha = float(cache["best_alpha"])
    pxs, r_hat, E1v, E2v = transmitter_fields(
        theta, phi, alpha, np.deg2rad(az_deg), np.deg2rad(el_deg), nside
    )

    freqs_shifted = np.asarray(cache["freqs_shifted"])
    channels = []
    for d, h in zip(cache["freq_map_d_idx"], cache["freq_map_h_idx"]):
        d, h = int(d), int(h)
        Ex, Ey = Ex_hat[:, d], Ey_hat[:, d]
        channels.append(
            dict(
                d_idx=d,
                h_idx=h,
                freq_mhz=float(freqs_shifted[d]),
                valid=~bad_time & np.isfinite(Ex) & np.isfinite(Ey),
                Ex=Ex,
                Ey=Ey,
                Ex_measured=Ex_meas[:, d],
                Ey_measured=Ey_meas[:, d],
            )
        )
    return dict(
        geometry=(pxs, r_hat, E1v, E2v),
        channels=channels,
        beam=beam,
        n_integrations=len(ext["times"]),
        times=np.asarray(ext["times"], dtype=float),
        theta_phi_alpha=(theta, phi, alpha),
    )
