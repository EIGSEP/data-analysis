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
# script here; v0003 is traced from the release v0004 antenna on DEM v0003.
HORIZON_PROFILES = "curation/horizon_profiles_v0003"
# The DEM the Sun's terrain trace reads (same grid as the profiles).
DEM_PATH = "derived/dem/v0003/marjum_dem.npz"

# Horizon-profile keys by pointing-table era.
ERAS = {"~30m": "30m", "~87.5m": "87.5m", "~91m": "91m"}

# Channel width of the correlator, MHz.
CHANNEL_MHZ = 250.0 / 1024


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
