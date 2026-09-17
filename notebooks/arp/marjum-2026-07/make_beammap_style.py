"""Beam map in the native scan coordinates, matching the raw-data view.

Deliberately mirrors how the comb channel looks straight out of the
file (adjacent-channel-differenced excess, contour-filled, elevation
on x, azimuth on y with 0 at top): that is the faithful view of this
dataset. The HEALPix beam-frame projection used elsewhere is lossy
here -- two different (az, el) pointings can share a beam-frame pixel
while carrying different polarization orientation -- which is why
structure that is obvious in these coordinates looks scrambled there.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from eigsep_data.beam_mapping import compute_beam_pca
from fit_v007_pca_beam import _project_templates
from eigsep_data.beam_mapping import TransmitterGeometry
from eigsep_data.beam_mapping import HFSSBeamSet
from eigsep_data.beam_mapping.diagnostics import load_v007_data

MODEL_JSON = "v007_pca_beam_model_v5.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
CHANNELS = [int(c) for c in (sys.argv[1:] or [600])]

with open(MODEL_JSON) as f:
    d = json.load(f)
beam = HFSSBeamSet.from_npz(BEAM_FILE)
pca = compute_beam_pca(beam, n_components=d["n_components"])
geometry = TransmitterGeometry(d["geometry"]["heading"], d["geometry"]["alpha_deg"])
data = load_v007_data("data")
templates = _project_templates(pca.components, data, geometry)
rows = {r["channel"]: r for r in d["channels"]}

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


for ch in CHANNELS:
    r = rows[ch]
    a = np.array(r["coefficients_real"]) + 1j * np.array(r["coefficients_imag"])
    model_power = np.abs(np.conj(a) @ templates[r["tx_arm"]]) ** 2
    y = data["measured_tx"][:, ch].astype(float)
    az, el = data["az_deg"], data["el_deg"]
    # exclude only the parked-at-origin block, which is not scan data at all
    keep = ~((np.round(az, 4) == 0.0) & (np.round(el, 4) == 0.0))
    scale = np.sum(model_power[keep] * y[keep]) / np.sum(model_power[keep] ** 2)
    dmap = grid(az[keep], el[keep], y[keep])
    mmap = grid(az[keep], el[keep], scale * model_power[keep])
    rmap = dmap - mmap
    vmax = np.nanpercentile(dmap, 99.5)
    rs = np.nanpercentile(np.abs(rmap), 99)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    for ax, img, title, kw in [
        (axes[0], dmap, f"DATA  ch{ch} @ {r['frequency_mhz']:.3f} MHz",
         dict(levels=np.linspace(0, vmax, 30), cmap="plasma", extend="both")),
        (axes[1], mmap, "MODEL (HFSS-PCA fit)",
         dict(levels=np.linspace(0, vmax, 30), cmap="plasma", extend="both")),
        (axes[2], rmap, "DATA - MODEL",
         dict(levels=np.linspace(-rs, rs, 31), cmap="RdBu_r", extend="both")),
    ]:
        cs = ax.contourf(elc, azc, np.nan_to_num(img, nan=0.0), **kw)
        ax.set_xlabel("Elevation [deg]")
        ax.set_ylabel("Azimuth [deg]")
        ax.set_ylim(-180, 0)
        ax.set_title(title)
        fig.colorbar(cs, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = f"beammap_style_ch{ch}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    good = ~np.isnan(rmap) & ~np.isnan(dmap)
    print(f"saved {out}  resid/data={np.sqrt(np.mean(rmap[good]**2))/np.sqrt(np.mean(dmap[good]**2)):.4f}",
          flush=True)
