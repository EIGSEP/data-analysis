"""Combined 6-column comparison, 5 rows (TX frequencies) per figure.

Columns: HFSS beam | data | model beam | model data | difference beam
        | difference data

"beam" columns are full-sky maps from the smooth, DPSS-fit empirical
model vs the HFSS prior (what was ``all_freq_model_vs_hfss.png``).
"data" columns are the sparse, real-scan-sample-gridded measurement vs
the per-channel fitted model evaluated at those same samples (what was
``all_freq_data_vs_model.png``). Putting both side by side per
frequency answers, in one place: does the empirical beam differ from
HFSS, and separately, does the model actually explain what was
measured.
"""
import json
import math
import sys
import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from beam_pca import compute_beam_pca
from data_space_rfi import data_space_rfi_mask
from fast_mollview import fast_mollview
from fit_v007_pca_beam import (
    _project_templates,
    beam_power_map,
    fitted_coefficients_at,
    hfss_prior_vector,
    pointing_valid_mask,
)
from rotation_beam import TransmitterGeometry, vector_to_spherical
from tx_beam_sim import HFSSBeamSet
from v007_beam_diagnostic import channel_validity_masks, gross_power_time_flags, load_v007_data

MODEL_JSON = sys.argv[1] if len(sys.argv) > 1 else "v007_pca_beam_model_v5.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
DATA_PATH = "data"
ROWS_PER_FIGURE = 5
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "combined_v5"

t_start = time.time()


def log(msg):
    print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)


with open(MODEL_JSON) as f:
    d = json.load(f)

beam = HFSSBeamSet.from_npz(BEAM_FILE)
pca = compute_beam_pca(beam, n_components=d["n_components"])
nside = pca.components.nside
geometry = TransmitterGeometry(d["geometry"]["heading"], d["geometry"]["alpha_deg"])
data = load_v007_data(DATA_PATH)
log("loaded data/pca/geometry")

model = {
    "pca": pca,
    "grid_freqs": np.array(d["frequency_grid_mhz"]),
    "grid_gain": np.array(d["gain_fitted_grid"]),
    "grid_shape": {
        "real": [np.array(r) for r in d["shape_correction_real_grid"]],
        "imag": [np.array(r) for r in d["shape_correction_imag_grid"]],
    },
    "n_components": d["n_components"],
}

rows = sorted(d["channels"], key=lambda r: r["frequency_mhz"])
trusted_by_channel = {row["channel"]: t for row, t in zip(d["channels"], d["trusted"])}
n = len(rows)
log(f"{n} channels")

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
npix = hp.nside2npix(nside)

clean_mask = data_space_rfi_mask(
    DATA_PATH, beam.freqs_mhz.min(), beam.freqs_mhz.max(), min_votes=5)
clean_mask = clean_mask & pointing_valid_mask(data)
log("computed RFI + pointing-validity mask")
templates_by_arm = _project_templates(pca.components, data, geometry)


def grid_by_median(values, mask):
    out = np.full(npix, np.nan)
    px_masked = px[mask]
    vals_masked = values[mask]
    order = np.argsort(px_masked)
    px_sorted = px_masked[order]
    vals_sorted = vals_masked[order]
    boundaries = np.searchsorted(px_sorted, np.arange(npix + 1))
    for p in range(npix):
        lo, hi = boundaries[p], boundaries[p + 1]
        if hi > lo:
            out[p] = np.median(vals_sorted[lo:hi])
    return out


covered_static = ~np.isnan(grid_by_median(np.ones(px.size), clean_mask))

n_chunks = math.ceil(n / ROWS_PER_FIGURE)
for chunk in range(n_chunks):
    r0 = chunk * ROWS_PER_FIGURE
    r1 = min(r0 + ROWS_PER_FIGURE, n)
    chunk_rows = rows[r0:r1]
    fig, axes = plt.subplots(len(chunk_rows), 6, figsize=(19, 2.0 * len(chunk_rows)))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(chunk_rows):
        ch = row["channel"]
        arm = row["tx_arm"]
        freq = row["frequency_mhz"]
        trusted = trusted_by_channel[ch]
        tag = "" if trusted else " [UNTRUSTED]"

        # --- beam-level: HFSS prior vs smooth fitted model ---
        a_hfss = hfss_prior_vector(pca, freq)
        a_fit_smooth = fitted_coefficients_at(model, freq)
        map_hfss = beam_power_map(pca, a_hfss)
        map_fit_beam = beam_power_map(pca, a_fit_smooth)
        norm_hfss = max(float(np.mean(map_hfss)), 1e-30)
        norm_fit_beam = max(float(np.mean(map_fit_beam)), 1e-30)
        diff_beam = map_fit_beam / norm_fit_beam - map_hfss / norm_hfss
        diff_beam_masked = diff_beam.copy()
        diff_beam_masked[~covered_static] = np.nan
        scale_beam = max(np.percentile(np.abs(diff_beam[covered_static]), 99), 1e-4) \
            if covered_static.any() else 0.1

        # --- data-level: real measurements vs per-channel raw fitted model ---
        a_fit_raw = np.array(row["coefficients_real"]) + 1j * np.array(row["coefficients_imag"])
        templates = templates_by_arm[arm]
        model_power = np.abs(np.conj(a_fit_raw) @ templates) ** 2
        y = data["measured_tx"][:, ch].astype(float)
        base = channel_validity_masks(data, [ch])[:, 0]
        gross, _, _ = gross_power_time_flags(data, [ch])
        used = base & ~gross & clean_mask
        residual = y - model_power
        data_map = grid_by_median(y, used)
        model_data_map = grid_by_median(model_power, used)
        resid_data_map = grid_by_median(residual, used)
        finite_data = data_map[~np.isnan(data_map)]
        data_max = np.nanpercentile(finite_data, 99) if finite_data.size else 1.0
        finite_resid = resid_data_map[~np.isnan(resid_data_map)]
        scale_data = max(np.nanpercentile(np.abs(finite_resid), 95), 1e-6) \
            if finite_resid.size else 1.0

        fast_mollview(axes[i, 0], 10 * np.log10(map_hfss / map_hfss.max()), nside,
                     vmin=-30, vmax=0, title=f"{freq:.2f} MHz{tag}\nHFSS beam")
        fast_mollview(axes[i, 1], data_map, nside, vmin=0, vmax=data_max,
                     title=f"data (n={int(used.sum())})")
        fast_mollview(axes[i, 2], 10 * np.log10(map_fit_beam / map_fit_beam.max()), nside,
                     vmin=-30, vmax=0, title="model beam")
        fast_mollview(axes[i, 3], model_data_map, nside, vmin=0, vmax=data_max,
                     title="model data")
        fast_mollview(axes[i, 4], diff_beam_masked, nside, vmin=-scale_beam, vmax=scale_beam,
                     cmap="RdBu_r", title=f"beam - HFSS\n(+/-{scale_beam:.3f})")
        fast_mollview(axes[i, 5], resid_data_map, nside, vmin=-scale_data, vmax=scale_data,
                     cmap="RdBu_r", title=f"data - model\n(+/-{scale_data:.2e})")
    fig.tight_layout()
    outname = f"{PREFIX}_rows{r0:03d}-{r1-1:03d}.png"
    fig.savefig(outname, dpi=110)
    plt.close(fig)
    log(f"saved {outname} ({chunk+1}/{n_chunks})")

log("DONE")
