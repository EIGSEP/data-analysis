"""Core data-loading and plotting for the RFI explorer notebook.

Pulls, for an arbitrary range of already-processed campaign files: raw
counts (data/*.h5), the B16 DPSS smooth-band model
(derived/smooth_model/v0/*.h5), and the flags/v2 mask --
and builds the same 3-panel view already established in
build_b16_notebook.py / the full-campaign checkpoint: data (log color) |
data - DPSS model (residual, symlog) | v2-mask x residual (kept-only).

Deliberately reads raw data fresh rather than reconstructing it from the
companion's stored model+residual: hera_filters zeroes the residual at
flagged pixels by default (write_companion()'s own docstring), so
model+residual would silently hide exactly the comb/RFI spikes the data
panel exists to show. The existing waterfall code in this repo has never
trusted that reconstruction either -- it always recomputes residual =
data - model itself, which this module does too, for the same reason.

Antenna resolution: "gnd" transparently prefers raw key "0", falling
back to "3" if "0" isn't live for that file (same physical box-gnd, a
different SNAP input during campaign Phase A(late)/B -- see
b16_dpss_model.py's INPUTS comment). "air" is always raw key "4". A
file with neither live for the requested antenna is skipped from the
range, not an error.
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detectors as D  # noqa: E402
import b15_event_survey as EV  # noqa: E402
import b16_dpss_model as M16  # noqa: E402
from eigsep_data.bundle import Campaign  # noqa: E402
from eigsep_data.products import get as _get_product  # noqa: E402

# The flags reader now lives in the package (2026-09-17); read_flags.py
# is a deprecating shim over this same call, so go straight to it and
# skip the warning. Arrays are bit-identical -- verified across 24
# files x v0/v2 x both call forms.
FLAGS_VERSION = "v2"
_CAMPAIGN = Campaign(M16.DATA_ROOT)
_FLAGS = _get_product("flags")

GND_KEYS = ("0", "3")   # try "0" first, fall back to "3" (same physical box)
AIR_KEY = "4"

_self_freqs, _self_mask = None, None
_band = None
_band_idx = None


def _ensure_band():
    global _self_freqs, _self_mask, _band, _band_idx
    if _band is None:
        _self_freqs, _self_mask = EV.self_comb_channel_mask()
        _band = (_self_freqs >= D.BAND_ANALYSIS[0]) & (_self_freqs <= D.BAND_ANALYSIS[1])
        _band_idx = np.nonzero(_band)[0]
    return _self_freqs, _band, _band_idx


def list_files():
    """Sorted list of all campaign data filenames (basenames)."""
    paths = sorted(glob.glob(os.path.join(M16.DATA_ROOT, "data", "*.h5")))
    return [os.path.basename(p) for p in paths]


def resolve_key(fname, antenna):
    """Return the raw input key ("0"/"3"/"4") actually live for this file
    and antenna ("gnd" or "air"), or None if neither is live."""
    path = os.path.join(M16.DATA_ROOT, "data", fname)
    try:
        with h5py.File(path, "r") as h:
            keys = set(k for k in h["data"] if len(k) <= 2)
    except Exception:
        return None
    if antenna == "gnd":
        for k in GND_KEYS:
            if k in keys:
                return k
        return None
    elif antenna == "air":
        return AIR_KEY if AIR_KEY in keys else None
    raise ValueError(f"antenna must be 'gnd' or 'air', got {antenna!r}")


def load_one_file(fname, antenna):
    """Load (freqs_band_mhz, raw_band, model_band, mask_bits_band, t)
    for one file/antenna, band-restricted to the analysis band. Returns
    None if this antenna has no live raw key or no B16 companion for
    this file (not an error -- caller skips it)."""
    key = resolve_key(fname, antenna)
    if key is None:
        return None

    self_freqs, band, band_idx = _ensure_band()
    path = os.path.join(M16.DATA_ROOT, "data", fname)
    with h5py.File(path, "r") as h:
        raw = h["data/" + key][:].astype(np.float64)
    raw_band = raw[:, band]

    companion_path = os.path.join(M16.DERIVED_DIR, fname)
    if not os.path.isfile(companion_path):
        return None
    with h5py.File(companion_path, "r") as h:
        if f"input_{key}" not in h:
            return None
        model_band = h[f"input_{key}"]["model"][:]

    if model_band.shape != raw_band.shape:
        raise ValueError(
            f"shape mismatch for {fname}/{key}: raw {raw_band.shape} vs "
            f"model {model_band.shape}")

    mask_bits, _freqs = _FLAGS.read_file(  # (n_time, 1024) uint16
        _CAMPAIGN, FLAGS_VERSION, fname, key)
    mask_bits_band = mask_bits[:, band]

    t0 = D.file_close_time(fname).timestamp()
    nt = raw_band.shape[0]
    t = t0 - 128.0 + np.arange(nt) * (128.0 / max(nt - 1, 1))

    return self_freqs[band], raw_band, model_band, mask_bits_band, t


def load_range(start_fname, end_fname, antenna, all_files=None):
    """Concatenate every file in [start_fname, end_fname] (inclusive,
    by sorted filename order) that has live data for `antenna`. Returns
    a dict with freqs_mhz, data, model, residual, mask_bits, t (all
    time-concatenated and time-sorted), plus which/how many files were
    used vs skipped, or raises ValueError if nothing usable is found."""
    if all_files is None:
        all_files = list_files()
    if start_fname not in all_files or end_fname not in all_files:
        raise ValueError("start/end filename not found in the real file list")
    i0, i1 = all_files.index(start_fname), all_files.index(end_fname)
    if i1 < i0:
        i0, i1 = i1, i0
    window = all_files[i0:i1 + 1]

    freqs_mhz = None
    data_list, model_list, mask_list, t_list = [], [], [], []
    used, skipped = [], []
    for fname in window:
        r = load_one_file(fname, antenna)
        if r is None:
            skipped.append(fname)
            continue
        freqs_mhz, raw_band, model_band, mask_bits_band, t = r
        data_list.append(raw_band)
        model_list.append(model_band)
        mask_list.append(mask_bits_band)
        t_list.append(t)
        used.append(fname)

    if not used:
        raise ValueError(
            f"no files in range [{start_fname}, {end_fname}] have live "
            f"'{antenna}' data with a B16 companion -- {len(skipped)} "
            f"files scanned, all skipped")

    data = np.concatenate(data_list, axis=0)
    model = np.concatenate(model_list, axis=0)
    mask_bits = np.concatenate(mask_list, axis=0)
    t = np.concatenate(t_list)
    order = np.argsort(t)
    residual = data - model  # recomputed here, NOT the companion's own
    # (zeroed-at-flags) residual -- see module docstring.
    return {
        "freqs_mhz": freqs_mhz,
        "data": data[order],
        "model": model[order],
        "residual": residual[order],
        "mask_bits": mask_bits[order],
        "t": t[order],
        "used_files": used,
        "skipped_files": skipped,
    }


def plot_three_panel(result, antenna, title_extra=""):
    """The established 3-panel view: data | data-model (residual) |
    v2-mask x residual (kept-only), same conventions as
    build_b16_notebook.py's waterfall cell. Returns the Figure."""
    freqs_mhz = result["freqs_mhz"]
    data, residual, mask_bits, t = (
        result["data"], result["residual"], result["mask_bits"], result["t"])
    flagged = mask_bits != 0  # v2: ANY bit set, including the new DPSS-outlier bit
    kept_residual = residual * (1 - flagged.astype(float))

    minutes = (t - t.min()) / 60
    extent = [freqs_mhz.min(), freqs_mhz.max(), minutes.max(), minutes.min()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    d = np.maximum(data, 1)
    im0 = axes[0].imshow(d, aspect="auto", extent=extent,
                          norm=mcolors.LogNorm(vmin=max(d.min(), 1), vmax=d.max()),
                          cmap="viridis")
    axes[0].set_title(f"{antenna}: data (counts){title_extra}", fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax_r = np.percentile(np.abs(residual), 99) or 1e-3
    im1 = axes[1].imshow(residual, aspect="auto", extent=extent,
                          norm=mcolors.SymLogNorm(linthresh=vmax_r / 50, vmin=-vmax_r, vmax=vmax_r),
                          cmap="RdBu_r")
    axes[1].set_title("data - DPSS model (counts)", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(kept_residual, aspect="auto", extent=extent,
                          norm=mcolors.SymLogNorm(linthresh=vmax_r / 50, vmin=-vmax_r, vmax=vmax_r),
                          cmap="RdBu_r")
    axes[2].set_title("flags/v2 mask x residual (kept-only)", fontsize=9)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Frequency (MHz)")
    axes[0].set_ylabel("minutes into range")
    plt.tight_layout()
    return fig
