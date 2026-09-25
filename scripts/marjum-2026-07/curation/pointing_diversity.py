"""Dwell-weighted pointing diversity (N_eff) for any slice of the campaign.

Why this product exists
-----------------------
INDEX.md has carried two different answers to "how much orientation diversity
does era X have", and they disagree by ~5x:

  * a **file-level** count from ``mode_table.jsonl`` ("14 orientations at 30 m",
    "the TX-on window is ``rot_state = parked``"), and
  * a **dwell-weighted** participation ratio over the per-sample pointing table.

The file-level answer is the wrong instrument and INDEX.md now says so: one
slewing file sweeps many pointings that a per-file label cannot represent, and
a catalogued orientation held for 20 s counts the same as one held for 3 h.
But the dwell-weighted numbers were being recomputed ad hoc, by different
agents, in scratch notebooks, with different cell grids -- which is how you get
two archive claims that cannot be diffed.

This script is the one place the number is computed. It reads
``pointing_table.parquet`` (per correlator sample, the archive's own product)
and reports, for a named slice:

  ``n_samples``   finite-pointing samples (0.2684 s or 0.5369 s each)
  ``cells``       occupied (az, el) cells at the requested cell size
  ``n_eff``       participation ratio (sum w)^2 / sum w^2 over cell dwell
                  fractions -- the effective number of independent pointings
  ``top_frac``    dwell fraction in the single most-occupied cell
  ``top3_frac``   dwell fraction in the three most-occupied cells

N_eff is *geometric* diversity. It knows nothing about the beam: two distinct
orientations can still give near-degenerate beam weightings, so N_eff bounds
what the data can do rather than promising it.

Two pointing sources, and they disagree where it matters
--------------------------------------------------------
``--source motor`` uses ``motor_az_deg``/``motor_el_deg`` (commanded axis
position); ``--source fused`` uses ``az_deg``/``el_deg`` (the fused solution,
IMU-referenced in elevation). Report **both** for near-degenerate slices. On a
stare, the fused solution's sensor scatter straddles cell edges and inflates
N_eff without any real pointing change -- the TX-on era reads N_eff 1.14
(motor) but 1.96 (fused) at a 4.435 deg cell, purely from bin splitting. When
the two disagree by more than ~0.3, the slice is a stare and neither number
should be quoted to two decimals.

Cell size: pass several. A genuine multi-pointing slice loses N_eff as cells
grow; a stare returns the same N_eff at every cell size, because one dwell
dominates however you bin it. That invariance is the diagnostic, not the value.

Units: degrees. Times are UTC. Cells are fixed-width in (az, el) -- no
cos(el) area weighting, so cells near the zenith are smaller on sky than the
nominal size. This is deliberate: N_eff here measures diversity of *commanded
orientation*, which is what a rotation-separation budget spends.

Output: one JSON record per (slice, source, cell) via --json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Named slices the archive makes claims about. Each is (description, selector).
SLICES = {
    "tx_on": (
        "the 436 files with tx_on=true in tx_presence.jsonl "
        "(07-16 01:18 -> 07-17 14:11) -- the only transmitter-live data",
        lambda df: df.file.isin(_tx_on_files()),
    ),
    "beam_scan": (
        "the dedicated beam scan, 07-17 18:51 -> 07-18 03:22 (227 files)",
        lambda df: _between(df, "2026-07-17T18:51Z", "2026-07-18T03:22Z"),
    ),
    "era_2m": ("~2 m height era", lambda df: df.height_era.astype(str).str.startswith("2")),
    "era_30m": ("~30 m height era", lambda df: df.height_era.astype(str).str.startswith("30")),
    "campaign": ("every sample in the campaign", lambda df: pd.Series(True, index=df.index)),
}

SOURCES = {"motor": ("motor_az_deg", "motor_el_deg"), "fused": ("az_deg", "el_deg")}


def _tx_on_files():
    path = HERE / "tx_presence.jsonl"
    rows = [json.loads(line) for line in open(path)]
    return {r["file"] for r in rows if r.get("tx_on")}


def _between(df, t0, t1):
    t = pd.to_datetime(df.t_utc, unit="ns", utc=True)
    return (t >= pd.Timestamp(t0)) & (t <= pd.Timestamp(t1))


def diversity(az, el, cell):
    """Participation ratio over (az, el) cells of width `cell` degrees."""
    keep = np.isfinite(az) & np.isfinite(el)
    az, el = az[keep], el[keep]
    if az.size == 0:
        return {"n_samples": 0, "cells": 0, "n_eff": None,
                "top_frac": None, "top3_frac": None}
    key = np.stack((np.floor(az / cell).astype(np.int64),
                    np.floor(el / cell).astype(np.int64)))
    _, counts = np.unique(key, axis=1, return_counts=True)
    w = counts / counts.sum()
    order = np.sort(w)[::-1]
    return {
        "n_samples": int(az.size),
        "cells": int(w.size),
        "n_eff": round(float(1.0 / np.sum(w ** 2)), 3),
        "top_frac": round(float(order[0]), 5),
        "top3_frac": round(float(order[:3].sum()), 5),
    }


def short_sha(repo):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default=str(HERE / "pointing_table.parquet"))
    ap.add_argument("--slice", nargs="*", default=["tx_on", "beam_scan"],
                    choices=sorted(SLICES), help="named slices to report")
    ap.add_argument("--source", nargs="*", default=["motor", "fused"],
                    choices=sorted(SOURCES))
    ap.add_argument("--cell", type=float, nargs="*", default=[4.435, 5.0, 10.0],
                    help="cell sizes in degrees; 4.435 is the beam scan's "
                         "actual commanded step (nominally 5)")
    ap.add_argument("--quality", default=None,
                    help="restrict to rows with this pointing_table quality "
                         "(ok / suspect); default: no restriction, because a "
                         "stalled-but-known axis is sound pointing")
    ap.add_argument("--json", help="write JSONL here")
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    if args.quality:
        df = df[df.quality == args.quality]

    prov = {
        "product": "pointing_diversity",
        "campaign": "marjum-2026-07",
        "version": "v1",
        "generator": "curation/pointing_diversity.py",
        "generator_commit": short_sha(HERE.parent.parent),
        "input": "curation/pointing_table.parquet",
        "quality_filter": args.quality,
        "metric": "participation ratio (sum w)^2 / sum w^2 over per-sample "
                  "dwell in fixed-width (az, el) cells",
        "warning": "Geometric diversity only -- says nothing about beam "
                   "degeneracy, and nothing about whether the transmitter was "
                   "on. Cross-check tx_presence.jsonl before budgeting a "
                   "transmitter measurement against any of these numbers.",
    }

    out = [{"provenance": prov}]
    for name in args.slice:
        desc, sel = SLICES[name]
        sub = df[sel(df)]
        for source in args.source:
            az_col, el_col = SOURCES[source]
            for cell in args.cell:
                rec = {"slice": name, "description": desc, "source": source,
                       "cell_deg": cell,
                       **diversity(sub[az_col].values, sub[el_col].values, cell)}
                out.append(rec)
                print(json.dumps(rec), flush=True)

    if args.json:
        with open(args.json, "w") as f:
            for rec in out:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())
