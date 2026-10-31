"""Shared paths, provenance and per-file labels for the ground_sky scripts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def campaign_root():
    return Path(os.environ["EIGSEP_CAMPAIGN_ROOT"]).resolve()


def workspace_root():
    """The meta-repo holding the campaign and the package checkouts."""
    return campaign_root().parent


# Horizon profiles (curation/horizon_profiles_vNNNN.{json,npz}) for every
# script here; v0005 is traced from the geometry release v0005 antenna on DEM v0004
# (which extends v0003 6 km south), with its own profile for the 91 m re-raise.
HORIZON_PROFILES = "curation/horizon_profiles_v0005"
# The DEM the Sun's terrain trace reads (same grid as the profiles).
DEM_PATH = "derived/dem/v0004/marjum_dem.npz"

# The empirical beam product: its beam (dpss/empirical_beam.npz) and the mount
# geometry it was fitted in (dpss/diagnostics.npz). v0013 is v0012 refitted on
# geometry v0004 and is the reference beam.
BEAM_PRODUCT = "derived/beam/empirical_raster_v0013"


def beam_dir(campaign=None):
    return Path(campaign or campaign_root()) / BEAM_PRODUCT / "dpss"


def mount_offsets(campaign=None):
    """(az_add, el_add, psi) in degrees for ``mount_rotation(az + az_add, el + el_add, psi)``.

    The beam was fitted with the table az shifted by ``az_offset_deg`` plus the
    fitted ``az_zero_delta_deg``, the el by the fitted ``el_zero_delta_deg``,
    and the elevation axle at ``psi_deg``; using the beam in any other frame
    misplaces it. Before 2026-10-09 these scripts applied ``az_offset_deg`` and
    ``psi_deg`` only.
    """
    from eigsep_data.beam_mapping.tx_fit import PARAM_NAMES

    with np.load(beam_dir(campaign) / "diagnostics.npz") as d:
        p = dict(zip(PARAM_NAMES, np.asarray(d["params"], float)))
        return (float(d["az_offset_deg"]) + p["az_zero_delta_deg"],
                p["el_zero_delta_deg"], float(d["psi_deg"]))


# Horizon-profile keys by pointing-table era.
ERAS = {"~30m": "30m", "~87.5m": "87.5m", "~91m": "91m",
        "~91m-reraise": "91m-reraise"}   # its own profile from horizon_profiles_v0005

# Channel width of the correlator, MHz.
CHANNEL_MHZ = 250.0 / 1024

# Channel offsets from each beam slice that bin_raw.py averages: midway
# between transmitter teeth.
RAW_CHANNEL_OFFSETS = np.array([-5, -4, -3, 3, 4, 5])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_rev(path):
    """Commit, branch and dirtiness of the repository holding ``path``."""
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain", "--", ".")),
    }


def _read_jsonl(path):
    rows = [json.loads(line) for line in open(path)]
    return pd.DataFrame([r for r in rows if "file_first" in r])


def era_of_files(files, campaign=None):
    """Height era of each raw file, from ``curation/mode_table.jsonl``.

    The pointing table leaves ``height_era`` blank on 13.6 % of rows (its
    README); the mode table's file ranges cover them.
    """
    campaign = campaign or campaign_root()
    modes = _read_jsonl(campaign / "curation" / "mode_table.jsonl")
    out = {}
    for f in np.unique(files):
        hit = modes[(modes.file_first <= f) & (modes.file_last >= f)]
        out[f] = hit.height_era.iloc[0] if len(hit) else ""
    return np.array([out[f] for f in files], dtype=object)


def rows_at_height(files, era, campaign=None, require=True):
    """Mask of the rows whose height era (mode table, by source file) has the horizon
    profile ``era`` (e.g. "30m"), and the mode-table eras the window holds. Rows on the
    ground, being raised, of unknown height or at another height are dropped; with
    ``require``, a window with none at ``era`` is an error, so a fixed-height analysis
    never runs on rows that were not at that height."""
    eras = era_of_files(files, campaign)
    keep = np.array([ERAS.get(e) == era for e in eras], bool)
    names, n = np.unique(eras, return_counts=True)
    counts = {str(k) or "(none)": int(c) for k, c in zip(names, n)}
    if require and not keep.any():
        raise SystemExit(f"no rows at {era} in this window; its mode-table eras: {counts}")
    return keep, counts


def regime_of_files(files, campaign=None):
    """Receiver regime of each raw file: that of the latest cal window
    starting at or before it (``curation/cal_windows.jsonl``)."""
    campaign = campaign or campaign_root()
    windows = _read_jsonl(campaign / "curation" / "cal_windows.jsonl")
    windows = windows.sort_values("file_first")
    out = {}
    for f in np.unique(files):
        before = windows[windows.file_first <= f]
        out[f] = before.receiver_regime.iloc[-1] if len(before) else ""
    return np.array([out[f] for f in files], dtype=object)


# The correlator input tcal calibrates: box-air.
TCAL_KEY = "4"


def tcal_coefficients(t, version, campaign=None, chunk=1000):
    """``(scale, offset, freqs_mhz)`` of ``tcal@<version>`` at Unix times ``t``.

    The coefficients of T = scale * P + offset at plane P, on every correlator
    channel, from ``eigsep_data``'s tcal product: the same interpolation between
    calibration visits and S11 sweeps, and the same nearest load thermistor,
    that ``Bundle.calibrated`` applies to each row. NaN wherever tcal does not
    reach."""
    from eigsep_data.bundle import Campaign
    from eigsep_data.products.base import get

    camp = Campaign(campaign or campaign_root())
    tc = get("tcal")
    sols = tc._solutions(camp, version)
    if sols is None:
        raise FileNotFoundError(f"no tcal@{version} under {camp.root}")
    if TCAL_KEY not in set(np.asarray(sols["keys"]).astype(str)):
        raise ValueError(f"tcal@{version} does not calibrate input {TCAL_KEY}")
    t = np.asarray(t, float)
    nchan = sols["freqs"].size
    scale = np.full((t.size, nchan), np.nan)
    offset = np.full((t.size, nchan), np.nan)
    for i in range(0, t.size, chunk):
        out = tc.fetch(camp, version, None, None, TCAL_KEY, None, t[i:i + chunk])
        scale[i:i + chunk] = out["scale"]
        offset[i:i + chunk] = out["offset"]
    return scale, offset, np.asarray(sols["freqs"], float)


def calibrate(z, version, campaign=None):
    """Kelvin at plane P for a ``raw_binned`` product, from ``tcal@<version>``.

    ``z`` is the product's arrays (``t``, ``chans``, ``yc``, ``nc``, ``sem``).
    Each offset channel's bin mean is calibrated with that channel's
    coefficients at the bin's mean time, then the channels are averaged with
    their unmasked sample counts as weights, as the binner averages power.
    The coefficients move over hours (between calibration visits), so one
    time per bin stands for its rows.

    Returns a dict of (nbin, nfreq) arrays: ``T``; ``sigma``, the bin's
    standard error ``sem`` times ``scale`` (the offset adds no scatter);
    ``scale`` and ``offset``, the sample-weighted mean coefficients, so that
    ``T`` ~ ``scale * y + offset``. A bin is NaN at a frequency where tcal
    does not cover any of its unmasked channels."""
    s, o, _ = tcal_coefficients(z["t"], version, campaign)
    cols = np.asarray(z["chans"])[:, None] + RAW_CHANNEL_OFFSETS[None, :]
    sc, oc = s[:, cols], o[:, cols]                      # (nbin, nfreq, noff)
    w = np.asarray(z["nc"], float)
    yc = np.asarray(z["yc"], float)
    use = w > 0
    # A channel with data but no coefficient: the bin is not calibrated there.
    bad = (use & ~(np.isfinite(sc) & np.isfinite(oc))).any(-1)
    wsum = w.sum(-1)

    def wmean(a):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(use, a * w, 0.0).sum(-1) / wsum

    T, scale, offset = wmean(sc * yc + oc), wmean(sc), wmean(oc)
    for a in (T, scale, offset):
        a[bad | (wsum == 0)] = np.nan
    return {"T": T, "sigma": scale * np.asarray(z["sem"], float), "scale": scale, "offset": offset}
