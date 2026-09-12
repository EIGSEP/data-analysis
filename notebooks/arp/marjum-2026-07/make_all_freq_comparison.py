"""One-off: giant per-channel comparison plots, one row per TX comb frequency.

Two separate figures, answering two different questions:
  * "model_vs_hfss": fitted (smooth, DPSS) empirical model vs the HFSS
    prior, both as full-sky maps -- how much did the data pull the
    beam shape away from HFSS.
  * "data_vs_model": the actual measured TX excess at each real scan
    sample minus the fitted model's prediction at that same sample,
    gridded onto the HEALPix pixel it was measured at -- how well the
    model explains what was actually observed. This is necessarily
    sparse (most pixels have 0-2 real samples); unsampled pixels are
    left grey.

Uses fast_mollview (direct projmap + imshow) instead of hp.mollview:
at ~600 total panels, mollview's per-call matplotlib/colorbar/graticule
setup overhead made this run for over an hour and it kept getting
killed by environment restarts before finishing. Also checkpoints each
figure to disk every 20 rows so a mid-run kill doesn't lose everything.
"""
import json
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
)
from rotation_beam import TransmitterGeometry, vector_to_spherical
from tx_beam_sim import HFSSBeamSet
from v007_beam_diagnostic import channel_validity_masks, gross_power_time_flags, load_v007_data

MODEL_JSON = "v007_pca_beam_model_v4.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
DATA_PATH = "data"

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

# ---- pixel each real scan sample lands on (frequency-independent) ----
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
log("computed RFI mask")
templates_by_arm = _project_templates(pca.components, data, geometry)


def grid_by_median(values, mask):
    """Median of `values[mask]` per pixel; NaN where no sample lands."""
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
row_h = 1.6
checkpoint_every = 20

# ================= Figure 1: model vs HFSS =================
fig1, axes1 = plt.subplots(n, 3, figsize=(11, row_h * n))
for i, row in enumerate(rows):
    freq = row["frequency_mhz"]
    trusted = trusted_by_channel[row["channel"]]
    a_hfss = hfss_prior_vector(pca, freq)
    a_fit = fitted_coefficients_at(model, freq)
    map_hfss = beam_power_map(pca, a_hfss)
    map_fit = beam_power_map(pca, a_fit)
    norm_hfss = max(float(np.mean(map_hfss)), 1e-30)
    norm_fit = max(float(np.mean(map_fit)), 1e-30)
    diff = map_fit / norm_fit - map_hfss / norm_hfss
    diff_masked = diff.copy()
    diff_masked[~covered_static] = np.nan
    scale = max(np.percentile(np.abs(diff[covered_static]), 99), 1e-4) \
        if covered_static.any() else 0.1
    tag = "" if trusted else " [UNTRUSTED]"
    fast_mollview(axes1[i, 0], 10 * np.log10(map_hfss / map_hfss.max()), nside,
                 vmin=-30, vmax=0, title=f"{freq:.2f} MHz{tag}: HFSS")
    fast_mollview(axes1[i, 1], 10 * np.log10(map_fit / map_fit.max()), nside,
                 vmin=-30, vmax=0, title="fit")
    fast_mollview(axes1[i, 2], diff_masked, nside, vmin=-scale, vmax=scale,
                 cmap="RdBu_r", title=f"fit - HFSS (+/-{scale:.3f})")
    if (i + 1) % checkpoint_every == 0 or i == n - 1:
        fig1.tight_layout()
        fig1.savefig("all_freq_model_vs_hfss.png", dpi=100)
        log(f"fig1 row {i+1}/{n} (checkpoint saved)")
plt.close(fig1)
log("saved all_freq_model_vs_hfss.png (final)")

# ================= Figure 2: data vs model (fit residual) =================
fig2, axes2 = plt.subplots(n, 3, figsize=(11, row_h * n))
for i, row in enumerate(rows):
    ch = row["channel"]
    arm = row["tx_arm"]
    freq = row["frequency_mhz"]
    a = np.array(row["coefficients_real"]) + 1j * np.array(row["coefficients_imag"])
    templates = templates_by_arm[arm]
    model_power = np.abs(np.conj(a) @ templates) ** 2
    y = data["measured_tx"][:, ch].astype(float)
    base = channel_validity_masks(data, [ch])[:, 0]
    gross, _, _ = gross_power_time_flags(data, [ch])
    used = base & ~gross & clean_mask
    residual = y - model_power
    data_map = grid_by_median(y, used)
    model_map = grid_by_median(model_power, used)
    resid_map = grid_by_median(residual, used)
    finite_data = data_map[~np.isnan(data_map)]
    data_max = np.nanpercentile(finite_data, 99) if finite_data.size else 1.0
    finite_resid = resid_map[~np.isnan(resid_map)]
    resid_scale = max(np.nanpercentile(np.abs(finite_resid), 95), 1e-6) \
        if finite_resid.size else 1.0
    fast_mollview(axes2[i, 0], data_map, nside, vmin=0, vmax=data_max,
                 title=f"{freq:.2f} MHz: data (n_used={int(used.sum())})")
    fast_mollview(axes2[i, 1], model_map, nside, vmin=0, vmax=data_max, title="model")
    fast_mollview(axes2[i, 2], resid_map, nside, vmin=-resid_scale, vmax=resid_scale,
                 cmap="RdBu_r", title=f"data - model (+/-{resid_scale:.2e})")
    if (i + 1) % checkpoint_every == 0 or i == n - 1:
        fig2.tight_layout()
        fig2.savefig("all_freq_data_vs_model.png", dpi=100)
        log(f"fig2 row {i+1}/{n} (checkpoint saved)")
plt.close(fig2)
log("saved all_freq_data_vs_model.png (final)")
log("DONE")
