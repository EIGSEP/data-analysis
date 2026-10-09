"""Build a bounded antenna-position product for marjum-2026-07 (B2.4 follow-on).

Mirrors `marjum_tx_uncertainty.py`'s methodology for the transmitter: no
single antenna fit carries a posterior, so the defensible bound is the
envelope over independently-defensible fit generations, checked against the
one genuinely independent-style evidence this fit has available: its own
leave-one-out (LOO) held-out reprojection misses (each LOO refit excludes one
view's constraint, then is scored against exactly that excluded view).

Runs no new optimizer fits — every number below is read from already-computed
`.npz`/`report.json` outputs.

Output is written under `terrain/` (this repo), not into the marjum-2026-07
curation directory, since that directory is data-archivist's territory; hand
this product to data-archivist for durable curation if it is accepted.

Usage: marjum_antenna_uncertainty.py [OUT.json]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_OUT = WORKSPACE / "antenna_position_bracket.json"

# Independently-defensible antenna position estimates: the incremental
# scale-up sequence (POSITION_REFINEMENT.md: starting joint fit -> position-only
# -> joint poses -> poses+focal -> orientation refinement) plus the current
# camera-fixed robust repick. Values are the frozen "antenna" field in each
# generation's own saved state.
GENERATIONS = {
    "cv_position_absolute/baseline": WORKSPACE / "cv_position_absolute" / "baseline.npz",
    "cv_position_absolute/orientations": WORKSPACE / "cv_position_absolute" / "orientations.npz",
    "cv_position_absolute/positions": WORKSPACE / "cv_position_absolute" / "positions.npz",
    "cv_position_absolute/poses": WORKSPACE / "cv_position_absolute" / "poses.npz",
    "cv_position_absolute/focal": WORKSPACE / "cv_position_absolute" / "focal.npz",
    "cv_antenna_repick_v1": WORKSPACE / "cv_antenna_repick_v1" / "fit_antenna.npz",
}
# Deliberately-retained failed/translated-start diagnostic (POSITION_REFINEMENT.md:
# "the shifted-start candidate is deliberately retained even if it has a worse
# objective... its horizon is not an equally probable alternative"). Excluded
# from the envelope as an outlier, exactly as cv_transmitter_joint_v2 is for
# the transmitter product.
OUTLIER_GENERATIONS = {
    "cv_position_absolute/alternative": WORKSPACE / "cv_position_absolute" / "alternative.npz",
}
CURRENT_BEST = "cv_antenna_repick_v1"
LOO_REPORT = WORKSPACE / "cv_antenna_repick_v1" / "report.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo):
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_antenna(path):
    with np.load(path) as f:
        return np.asarray(f["antenna"], float)


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    gens = {}
    inputs = []
    for name, path in {**GENERATIONS, **OUTLIER_GENERATIONS}.items():
        pos = load_antenna(path)
        gens[name] = {
            "antenna_enu_m": [round(x, 3) for x in pos.tolist()],
            "outlier_excluded_from_envelope": name in OUTLIER_GENERATIONS,
        }
        inputs.append({"path": f"terrain/{path.relative_to(WORKSPACE)}", "sha256": sha256(path)})

    retained = {g: v for g, v in gens.items() if not v["outlier_excluded_from_envelope"]}
    P = np.array([v["antenna_enu_m"] for v in retained.values()])
    centroid = P.mean(0)
    max_radius = float(np.linalg.norm(P - centroid, axis=1).max())

    loo = json.loads(LOO_REPORT.read_text())
    inputs.append({"path": "terrain/cv_antenna_repick_v1/report.json", "sha256": sha256(LOO_REPORT)})
    best = np.asarray(loo["antenna"], float)
    loo_entries = loo["leave_one_out"]
    loo_ray_miss = {e["key"]: round(float(e["ray_miss_m"]), 3) for e in loo_entries}
    loo_pos_dist = {
        e["key"]: round(float(np.linalg.norm(np.asarray(e["fit_position"], float) - best)), 3)
        for e in loo_entries
    }
    max_ray_miss = max(loo_ray_miss.values())
    worst_view = max(loo_ray_miss, key=loo_ray_miss.get)

    bound = float(np.ceil(max(max_radius, max_ray_miss) * 10) / 10)

    product = {
        "provenance": {
            "product": "antenna_position",
            "campaign": "marjum-2026-07",
            "version": "v1",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "terrain/marjum_antenna_uncertainty.py",
            "generator_commit": git_commit(WORKSPACE),
            "inputs": inputs,
            "params": {
                "envelope_generations": sorted(GENERATIONS),
                "outlier_generations": sorted(OUTLIER_GENERATIONS),
                "independent_check": "leave-one-out held-out reprojection ray miss, "
                                      "from the 23-view camera-fixed robust repick "
                                      "(cv_antenna_repick_v1/report.json:leave_one_out)",
            },
        },
        "stamp": f"marjum-2026-07/antenna_position@v1+{git_commit(WORKSPACE)}",
        "frame": "local working grid (ENU metres), anchored per marjum_bundle.working_grid",
        "best_estimate_enu_m": [round(float(x), 3) for x in best.tolist()],
        "uncertainty": {
            "bound_m": bound,
            "kind": "bounding radius, not a posterior sigma",
            "basis": {
                "methodological_envelope_m": round(max_radius, 3),
                "leave_one_out_max_ray_miss_m": round(max_ray_miss, 3),
                "leave_one_out_worst_view": worst_view,
                "rule": "bound = ceil(max(envelope, loo_max_ray_miss) * 10) / 10",
            },
            "interpretation": (
                "Treat as a hard bracket on the antenna position, built the same way as "
                "the transmitter's bracket. The generation envelope (incremental "
                "scale-up sequence + current repick) alone would give "
                f"{round(max_radius, 3)} m; the bound is instead set by the worst "
                f"leave-one-out held-out reprojection miss ({worst_view}, "
                f"{round(max_ray_miss, 3)} m), which is the closest thing this fit has "
                "to independent evidence — each LOO refit never sees the view it is then "
                "scored against."
            ),
        },
        "generations": gens,
        "leave_one_out": {
            "ray_miss_m": loo_ray_miss,
            "position_shift_from_best_m": loo_pos_dist,
        },
        "findings": {
            "large_residual_views": (
                "2209, 2210, 2211, 2217, 2235 all show 1.0-1.6 m LOO ray misses despite "
                "<0.16 m LOO position shift, i.e. removing any single one of them barely "
                "moves the antenna estimate, but the antenna estimate is a genuinely poor "
                "predictor of those views' labels. Per ANTENNA_REPICK.md, a local "
                "parallax-aware scan on 2209 confirmed sub-meter camera adjustments do not "
                "remove this residual, and forcing it to zero raises held-out terrain "
                "residuals to ~13 px -- so this is logged as an open item (candidate causes: "
                "camera pose error not absorbed by this camera-fixed triangulation, or a "
                "labelling/parallax effect specific to those 5 views), not pursued further "
                "under the no-new-optimizer-fits constraint on this product."
            ),
            "translated_start_outlier": (
                "cv_position_absolute/alternative is a deliberately-retained failed/"
                "translated-start diagnostic (14 m from the other generations); excluded "
                "from the envelope as it is not an equally probable alternative "
                "(POSITION_REFINEMENT.md)."
            ),
        },
        "limitations": [
            "This is a deterministic bracket over fit generations and LOO refits, not a "
            "Bayesian posterior. It has no coverage guarantee.",
            "The completed legacy MCMC pilot for this antenna is unconverged and supplies "
            "no usable posterior (per v0001_marjum_geometry shared.json); this bracket "
            "does not depend on it.",
            "LOO refits still share the same camera poses/labels/distortion model as the "
            "full fit; they are an internal robustness check, not a fully independent "
            "second instrument the way 2210/2211 are for the transmitter.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(product, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(f"  best estimate {product['best_estimate_enu_m']}  bound +/-{bound} m")
    print(f"  stamp {product['stamp']}")


if __name__ == "__main__":
    main()
