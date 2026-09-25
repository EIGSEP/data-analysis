"""Data vs model in native (azimuth, elevation) scan coordinates.

The HEALPix beam-frame maps used elsewhere are a *lossy* view of this
dataset: two different (az, el) pointings can map to the same
beam-frame pixel while carrying a different polarization orientation,
so gridding to pixels smears together samples the forward model
legitimately treats as distinct. Plotting in the native scan
coordinates -- exactly how the raw comb channel looks straight out of
the file -- is the faithful comparison, and is where the model's
failure to track the data is actually legible.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from eigsep_data.beam_mapping import compute_beam_pca
from eigsep_data.beam_mapping import data_space_rfi_mask
from fit_v007_pca_beam import _project_templates, pointing_valid_mask
from eigsep_data.beam_mapping import TransmitterGeometry
from eigsep_data.beam_mapping import HFSSBeamSet
from eigsep_data.beam_mapping.diagnostics import (
    channel_validity_masks,
    gross_power_time_flags,
    load_v007_data,
)

MODEL_JSON = "v007_pca_beam_model_v5.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
CHANNELS = [int(c) for c in (sys.argv[1:] or [600])]

with open(MODEL_JSON) as f:
    d = json.load(f)
beam = HFSSBeamSet.from_npz(BEAM_FILE)
pca = compute_beam_pca(beam, n_components=d["n_components"])
geometry = TransmitterGeometry(d["geometry"]["heading"], d["geometry"]["alpha_deg"])
data = load_v007_data("data")
clean = data_space_rfi_mask(
    "data", beam.freqs_mhz.min(), beam.freqs_mhz.max(), min_votes=5
) & pointing_valid_mask(data)
templates = _project_templates(pca.components, data, geometry)
rows = {r["channel"]: r for r in d["channels"]}

az_edges = np.arange(-182.5, 85, 5.0)
el_edges = np.arange(-182.5, 185, 5.0)


def grid(az, el, vals):
    """Median of vals in each (az, el) cell; NaN where empty."""
    out = np.full((az_edges.size - 1, el_edges.size - 1), np.nan)
    ia = np.digitize(az, az_edges) - 1
    ie = np.digitize(el, el_edges) - 1
    ok = (ia >= 0) & (ia < out.shape[0]) & (ie >= 0) & (ie < out.shape[1])
    for a, e, v in zip(ia[ok], ie[ok], vals[ok]):
        if np.isnan(out[a, e]):
            out[a, e] = v
        else:
            out[a, e] = 0.5 * (out[a, e] + v)
    return out


for ch in CHANNELS:
    r = rows[ch]
    a = np.array(r["coefficients_real"]) + 1j * np.array(r["coefficients_imag"])
    model_power = np.abs(np.conj(a) @ templates[r["tx_arm"]]) ** 2
    y = data["measured_tx"][:, ch].astype(float)
    base = channel_validity_masks(data, [ch])[:, 0]
    gross, _, _ = gross_power_time_flags(data, [ch])
    used = base & ~gross & clean
    az, el = data["az_deg"][used], data["el_deg"][used]
    scale = np.sum(model_power[used] * y[used]) / np.sum(model_power[used] ** 2)
    m = scale * model_power[used]
    dmap = grid(az, el, y[used])
    mmap = grid(az, el, m)
    rmap = dmap - mmap
    vmax = np.nanpercentile(dmap, 99)
    rscale = np.nanpercentile(np.abs(rmap), 99)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    extent = [el_edges[0], el_edges[-1], az_edges[-1], az_edges[0]]
    for ax, img, title, kw in [
        (axes[0], dmap, f"DATA  ch{ch} @ {r['frequency_mhz']:.3f} MHz",
         dict(vmin=0, vmax=vmax, cmap="plasma")),
        (axes[1], mmap, "MODEL (HFSS-PCA fit)",
         dict(vmin=0, vmax=vmax, cmap="plasma")),
        (axes[2], rmap, "DATA - MODEL",
         dict(vmin=-rscale, vmax=rscale, cmap="RdBu_r")),
    ]:
        im = ax.imshow(img, origin="upper", extent=extent, aspect="auto", **kw)
        ax.set_xlabel("Elevation [deg]")
        ax.set_ylabel("Azimuth [deg]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = f"azel_compare_ch{ch}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    resid = np.sqrt(np.nanmean(rmap ** 2)) / np.sqrt(np.nanmean(dmap ** 2))
    print(f"saved {out}   gridded resid/data = {resid:.4f}", flush=True)
