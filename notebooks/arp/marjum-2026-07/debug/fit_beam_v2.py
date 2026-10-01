"""beam-fits-v2, D2 milestone: measured A(nu, theta, phi) under the
CORRECTED transmitter identity (1.953125 MHz, 8-channel-locked comb = TX,
per rfi-analyst's rotation-modulation test, PLAN.md CONTESTED section,
2026-09-14 -- NOT fully closed; see caveat in the output report).

Reuses the existing v007 PCA/least-squares machinery (eigsep_data.beam_mapping
+ fit_v007_pca_beam.py) almost unmodified -- its comb extraction was already
spacing-agnostic and never hard-coded to the old (wrong) comb identity, so
this is a re-windowing + re-pointing + re-labeling, not a rewrite. Two real
changes from v007:
  1. File window pinned to the documented 07-17 18:50 -> 07-18 03:22 UTC
     beam-scan era (computed boundary: files[-227:-1] of the sorted campaign
     glob, 226 files -- one short of the doc-cited 227; boundary-inclusivity
     rounding, not a different window).
  2. Pointing comes from pointing_table@v1 (joined per (file, sample_idx)),
     not v007's raw motor/pot arrays -- gives real sigmas and quality/flags
     instead of face-value motor counts.

Geometry (TX heading/alpha) is held fixed at v007's existing consensus fit,
per the existing pipeline's own design (build_model never re-fits geometry
jointly with the beam shape). This is a legitimate reuse, not corner-cutting:
the geometry model is a pure far-field heading regardless of what the
transmitter's physical identity was believed to be when it was fit, so
relabeling the source doesn't invalidate the direction that was recovered
from the same data.

2026-09-17 revision (Aaron's ruling on the D2 review gate). Three changes,
all corrections rather than new analysis:
  3. The sample mask now excludes samples where the receiver was NOT on the
     antenna, read per sample from ``metadata/rfswitch``. This was missing
     entirely -- 23 of the 226 files in the window are calibration files, and
     the off-antenna samples carried a median 71.3% of the residual power
     across the 101 channels.
  4. The mask also honours ``pointing_table@v1.2``'s EL_SOLUTION_GLITCH bit.
     Small in the aggregate but locally decisive: it is what destroys the
     |el| ~ 59 deg elevation bands.
  5. ``coverage_map`` is rewritten under geometer's Q8 closure -- the
     |el| ~ 180 "wrap cluster" is boresight-on-transmitter, not an angle-wrap
     artifact, and is now counted rather than footnoted out.

This file previously lived only in /tmp and was not under version control.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = str(Path(__file__).resolve().parent)
# Resolved from __file__, not hardcoded: the enclosing repo was renamed
# eigsep_data -> data-analysis on 2026-09-17 and every absolute path in
# this file broke. REPO is the repo root (two levels above notebooks/arp).
REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/data02/eigsep/marjum-2026-07/curation")

# eigsep_data split out of data-analysis into its own repo on 2026-09-17 and is
# editable-installed from /mnt/data02/eigsep/eigsep_data/src. The old fallback
# to data-analysis/src is gone with the package, so there is nothing to shim:
# a failed import here is a real environment problem, not a path problem.

from eigsep_data.beam_mapping import (
    compute_beam_pca, data_space_rfi_mask, TransmitterGeometry,
    HFSSBeamSet,
)
from eigsep_data.beam_mapping.diagnostics import load_v007_data
import fit_v007_pca_beam as v007

DATA_DIR = "/mnt/data02/eigsep/marjum-2026-07/data"
POINTING_PARQUET = "/mnt/data02/eigsep/marjum-2026-07/curation/pointing_table.parquet"
# The HFSS beam map lives with the eigsep_data PACKAGE since the 2026-09-17
# split, not in this (work-storage) repo. Resolve it from the installed package
# so it follows the package wherever that goes; data-analysis still carries a
# byte-identical leftover copy, which will drift the moment the model changes.
def _beam_file():
    import eigsep_data
    pkg = Path(eigsep_data.__file__).resolve().parents[2] / "hfss_beam_maps" / "bowtie_beam.npz"
    if pkg.exists():
        return str(pkg)
    legacy = Path(REPO) / "hfss_beam_maps" / "bowtie_beam.npz"
    if legacy.exists():
        return str(legacy)
    raise FileNotFoundError("bowtie_beam.npz not found beside the package or in this repo")


BEAM_FILE = _beam_file()
CONSENSUS_JSON = f"{HERE}/v007_multichannel_consensus.json"

FILES_SLICE = (-227, -1)  # 226 files; see module docstring
EL_POST_FAILURE = 256
EL_SOLUTION_GLITCH = 1024
OUT_PREFIX = f"{HERE}/beam_fits_v2"
ANTENNA_MASK_NPZ = f"{HERE}/antenna_mask.npz"


def receiver_on_antenna_mask(data):
    """Per-sample "the receiver was looking at the antenna" mask.

    Built by ``build_antenna_mask.py`` from the per-sample
    ``metadata/rfswitch`` stream via ``flagging/detectors.antenna_mask``.
    NOT the ``rfswitch_dominant`` column in ``curation/file_state.csv``:
    that is a file-level majority vote, and 14 of the 23 calibration files
    in this window are *majority* RFANT, so a dominant-state test keeps
    them. Per-sample is the correct granularity -- it catches the mixed
    files and keeps the genuine on-antenna samples inside a cal file.
    """
    cached = np.load(ANTENNA_MASK_NPZ, allow_pickle=True)
    keep = ~cached["exclude"]
    n = len(data["times"])
    if keep.size != n:
        raise ValueError(
            f"antenna_mask.npz has {keep.size} samples, data has {n}; "
            "rebuild it with build_antenna_mask.py for this window")
    return keep


def el_solution_glitch_mask(data):
    """True where pointing_table@v1.2 does NOT flag EL_SOLUTION_GLITCH."""
    return (data["flags_v1"] & EL_SOLUTION_GLITCH) == 0


def attach_pointing_v1(data, files):
    """Overwrite data['az_deg']/['el_deg'] with pointing_table@v1, joined by
    (file, sample_idx) -- the columns that product carries expressly for
    this. Adds data['quality_v1'] and data['flags_v1'] alongside."""
    n = len(data["times"])
    az = np.full(n, np.nan)
    el = np.full(n, np.nan)
    az_sigma = np.full(n, np.nan)
    el_sigma = np.full(n, np.nan)
    quality = np.array(["gap"] * n, dtype=object)
    flags = np.zeros(n, dtype=np.int64)

    t = pq.read_table(POINTING_PARQUET, columns=[
        "file", "sample_idx", "az_deg", "el_deg",
        "az_sigma_deg", "el_sigma_deg", "quality", "flags",
    ])
    df = t.to_pandas()
    df = df[df["file"].isin(set(Path(f).name for f in files))]
    df = df.set_index(["file", "sample_idx"])

    for i, filename in enumerate(files):
        fname = Path(filename).name
        sl_lo = 240 * i
        # nt for this file = wherever times[sl] stayed nonzero/valid;
        # load_v007_data zeroes the whole 240-block's times for comb-off
        # files, and leaves the true nt elsewhere -- use the block bound
        # directly, matched against how many rows pointing_table has for
        # this file (<=240 in every observed case).
        sub = df.loc[fname] if fname in df.index.get_level_values(0) else None
        if sub is None or len(sub) == 0:
            continue
        idxs = sub.index.to_numpy()
        rows = sl_lo + idxs
        valid = rows < n
        rows, idxs = rows[valid], idxs[valid]
        az[rows] = sub.loc[idxs, "az_deg"].to_numpy()
        el[rows] = sub.loc[idxs, "el_deg"].to_numpy()
        az_sigma[rows] = sub.loc[idxs, "az_sigma_deg"].to_numpy()
        el_sigma[rows] = sub.loc[idxs, "el_sigma_deg"].to_numpy()
        quality[rows] = sub.loc[idxs, "quality"].to_numpy()
        flags[rows] = sub.loc[idxs, "flags"].to_numpy().astype(np.int64)

    finite = np.isfinite(az) & np.isfinite(el)
    data["pointing_v1_finite"] = finite
    # simulate_hfss_coupling/healpy need in-range values for every sample,
    # including ones with no pointing_table@v1 row -- those are excluded
    # downstream via pointing_v1_valid_mask, so 0/0 here is a safe dummy,
    # never a value that ends up in the fit.
    data["az_deg"] = np.nan_to_num(az, nan=0.0)
    data["el_deg"] = np.nan_to_num(el, nan=0.0)
    data["az_sigma_deg"], data["el_sigma_deg"] = az_sigma, el_sigma
    data["quality_v1"], data["flags_v1"] = quality, flags
    return data


def pointing_v1_valid_mask(data):
    ok = data["quality_v1"] == "ok"
    return ok & data["pointing_v1_finite"]


def coverage_map(data, mask, cell_deg=5.0):
    az = data["az_deg"][mask]
    el = data["el_deg"][mask]
    flags = data["flags_v1"][mask]
    post_ef = (flags & EL_POST_FAILURE) != 0

    # Q8 CLOSED 2026-09-16 (geometer): `el` is the boresight ZENITH ANGLE --
    # el = 0 zenith, el = +90 horizon, el = +180 nadir. Proven independently
    # of Aaron by LIDAR return character vs elevation (493 real ground returns
    # at el ~ +90, median 92.31 m; zero returns at el ~ -90, i.e. sky; 2412
    # out-of-range sentinels at el ~ 0, shooting across the canyon). The
    # +90/-90 asymmetry is the sign-resolving observation.
    #
    # Consequence: the transmitter is essentially straight down (85.8 deg
    # below horizontal, i.e. near nadir), so |el| ~ 180 puts it ON BORESIGHT.
    # The "wrap cluster" previously excluded as a suspected angle-wrap
    # artifact is therefore the most on-source pointing in the dataset. It is
    # now counted in the coverage claim, and reported separately as the
    # on-source subset rather than as a footnoted exclusion.
    on_source = np.abs(np.abs(el) - 180.0) < 20.0
    off_source = ~on_source

    def stats(sub_az, sub_el):
        if sub_az.size == 0:
            return {"n_samples": 0, "n_cells": 0, "n_eff": 0.0,
                    "az_range_deg": None, "el_range_deg": None}
        az_cell = np.floor(sub_az / cell_deg)
        el_cell = np.floor(sub_el / cell_deg)
        cells = az_cell * 100000 + el_cell
        _, counts = np.unique(cells, return_counts=True)
        p = counts / counts.sum()
        n_eff = float(1.0 / np.sum(p ** 2))
        return {"n_samples": int(sub_az.size), "n_cells": int(counts.size),
                "n_eff": n_eff,
                "az_range_deg": [float(sub_az.min()), float(sub_az.max())],
                "el_range_deg": [float(sub_el.min()), float(sub_el.max())]}

    return {
        "cell_deg": cell_deg,
        "elevation_convention": (
            "el is the boresight ZENITH ANGLE: el=0 zenith, el=+90 horizon, "
            "el=+180 nadir. Q8 closed 2026-09-16 by geometer, proven by LIDAR "
            "return character vs elevation (ground returns at +90, none at "
            "-90). Supersedes the earlier 'wrap cluster may be an angle-wrap "
            "artifact' footnote: the transmitter is near nadir, so |el|~180 "
            "is boresight-on-transmitter."),
        "el_post_failure_fraction": float(post_ef.mean()) if post_ef.size else None,
        "total_coverage": stats(az, el),
        "on_source_near_nadir": {
            **stats(az[on_source], el[on_source]),
            "note": ("|el| within 20 deg of 180, i.e. boresight on the "
                     "transmitter. Previously excluded as a suspected "
                     "angle-wrap artifact; now counted."),
        },
        "off_source": stats(az[off_source], el[off_source]),
        "caveats": [
            "The el zero-point is constrained only to |offset| <~ 2 deg: "
            "geometer's MAD profile is flat within DEM quantisation over "
            "el_nadir ~ 87.5-91.5.",
            "Absolute AZIMUTH zero is still OPEN. The pot azimuth is "
            "body-frame, not north-referenced, so the azimuth axis of this "
            "coverage map has no absolute north anchor.",
        ],
    }


def main():
    files = sorted(glob.glob(str(Path(DATA_DIR) / "*.h5")))[slice(*FILES_SLICE)]
    print(f"[beam-fits-v2] window: {len(files)} files, "
          f"{Path(files[0]).name} .. {Path(files[-1]).name}", flush=True)

    with open(CONSENSUS_JSON) as stream:
        consensus = json.load(stream)
    geometry = TransmitterGeometry(consensus["heading"], consensus["alpha_deg"])

    data = load_v007_data(DATA_DIR, start=FILES_SLICE[0], stop=FILES_SLICE[1])
    data = attach_pointing_v1(data, data["files"])

    beam = HFSSBeamSet.from_npz(BEAM_FILE)
    pca = compute_beam_pca(beam, n_components=4)
    templates_by_arm = v007._project_templates(pca.components, data, geometry)

    clean_mask = data_space_rfi_mask(
        DATA_DIR, beam.freqs_mhz.min(), beam.freqs_mhz.max(),
        files_slice=FILES_SLICE, clip_sigma=5.0, min_votes=5)
    clean_mask = clean_mask & pointing_v1_valid_mask(data)

    # --- corrections applied 2026-09-17, per Aaron's ruling on the D2 review
    # gate (see beam_metric_outliers_checkpoint). Both were missing before:
    #  (1) the receiver spent part of this window on a VNA / noise source /
    #      load rather than the antenna, and nothing in the mask chain was a
    #      file- or sample-level campaign gate, so those samples were fit as
    #      if they were beam measurements. They carried a median 71.3% of the
    #      residual power across the 101 channels.
    #  (2) pointing_table@v1.2's EL_SOLUTION_GLITCH bit: samples where the
    #      IMU elevation solution jumped faster than the drive can move.
    on_antenna = receiver_on_antenna_mask(data)
    no_glitch = el_solution_glitch_mask(data)
    n_before = int(clean_mask.sum())
    n_cal = int((clean_mask & ~on_antenna).sum())
    n_glitch = int((clean_mask & on_antenna & ~no_glitch).sum())
    clean_mask = clean_mask & on_antenna & no_glitch
    print(f"[beam-fits-v2] mask: {n_before} -> {int(clean_mask.sum())} "
          f"(-{n_cal} off-antenna, -{n_glitch} EL_SOLUTION_GLITCH)",
          flush=True)

    cov = coverage_map(data, pointing_v1_valid_mask(data) & on_antenna
                       & no_glitch)
    print("[beam-fits-v2] coverage map:", json.dumps(cov, indent=2), flush=True)

    channels = v007.candidate_channels(data, beam)
    rows = [v007.fit_channel(data, pca, templates_by_arm, ch, clean_mask,
                             ridge_lambda=0.003)
           for ch in channels]
    rows = [row for row in rows if row is not None]
    print(f"[beam-fits-v2] fit {len(rows)}/{len(channels)} candidate channels",
          flush=True)

    freqs = np.array([row["frequency_mhz"] for row in rows])
    weights = 1.0 / np.maximum(
        np.array([row["normalized_rms"] for row in rows]), 1e-3) ** 2
    gain_fitted = np.array([row["gain_fitted"] for row in rows])
    gain_hfss = np.array([row["gain_hfss"] for row in rows])
    normalized_rms = np.array([row["normalized_rms"] for row in rows])

    report = {
        "milestone": "D2 beam-fits-v2",
        "tx_identity_caveat": (
            "This fit assumes the CORRECTED identity -- the 1.953125 MHz, "
            "8-channel-locked comb is the transmitter (rfi-analyst's "
            "rotation-modulation test, PLAN.md CONTESTED section). This is "
            "NOT fully closed as of 2026-09-14. If the identity resolves "
            "the other way, this is a near-field self-comb map of an "
            "instrument-internal radiator, not a far-field transmitter "
            "beam map -- same numbers, opposite physical meaning."
        ),
        "window": {
            "files_slice_from_full_campaign_glob": list(FILES_SLICE),
            "n_files": len(files),
            "first_file": Path(files[0]).name,
            "last_file": Path(files[-1]).name,
            "documented_era_cited": "07-17 18:50 -> 07-18 03:22 UTC, 227 files (marjum-2026-07/INDEX.md:962); "
                                     "this window computes to 226 files -- boundary-inclusivity rounding.",
        },
        "receiver_regime_caveat": (
            "rf-calibrator's B7 finding: a receiver regime change (T_rx "
            "212->539 K, g_rx 1150->672) sits between 07-17 19:42 and "
            "07-18 01:24 UTC, bracketing nearly this entire window (beam "
            "scan starts ~20:26 UTC). The channel-differenced measured_tx "
            "quantity is largely gain-ratio-invariant to first order, but "
            "this has not been separately verified here -- material "
            "uncertainty on absolute amplitude scale, not beam shape."
        ),
        "geometry_source": f"held fixed from {Path(CONSENSUS_JSON).name} (not re-fit jointly, matching v007's own design)",
        "coverage_map": cov,
        "n_candidate_channels": int(len(channels)),
        "n_channels_fit": int(len(rows)),
        "median_normalized_rms": float(np.median(normalized_rms)),
        "gain_fitted_vs_hfss_ratio_median": float(np.median(gain_fitted / np.maximum(gain_hfss, 1e-30))),
        "channels": rows,
    }
    out_json = f"{OUT_PREFIX}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[beam-fits-v2] wrote {out_json}", flush=True)
    print("[beam-fits-v2] DONE", flush=True)


if __name__ == "__main__":
    main()
