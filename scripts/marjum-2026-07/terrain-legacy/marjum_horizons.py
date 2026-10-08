"""Per-height terrain horizon profiles for marjum-2026-07 (B2.2).

Produces 0.1-degree-bearing horizon elevation profiles at the three height eras
that matter (~30 m, 87.5 m, 91 m), plus the sensitivity of those profiles to the
known antenna horizontal uncertainty. forward-modeler consumes these as the
Omega_below masks.

Vertical reference convention
----------------------------
The 91 m era is the only one with multi-view parallax, so it anchors the
absolute altitude: u = 1777.397 m from cv_antenna_repick_v1. The other two eras
are placed using the *differences* between field-note heights (91, 87.5, 30),
not their absolute values, because the field notes are tether-anchor-referenced
while photogrammetry is DEM-referenced and the two differ by ~2.5 m. Differences
cancel that reference offset if the notes were taken consistently; absolute
values do not. This is stated as an assumption, not a result.

Horizontal position is held at the 91 m solution for all three eras. That
assumes the highline anchor and trolley did not move between eras, which is
*load-dependent* - field notes record ~0.97 m of deflection at the 2 m
configuration - so it is carried as a stated systematic, not a footnote.

Usage: marjum_horizons.py [OUT_DIR]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eigsep_terrain.marjum_dem import MarjumDEM
from marjum_bundle import Terrain

WORKSPACE = Path(__file__).resolve().parent
# marjum-2026-07/ is a fixed sibling directory in the same checkout (not a
# separate/movable one -- confirmed with data-archivist, 2026-09-16).
CAMPAIGN_DIR = WORKSPACE.parent / "marjum-2026-07"
DEFAULT_OUT = CAMPAIGN_DIR / "curation"

DEM_CACHE = "marjum_dem_sw.npz"
ANTENNA_FIT = "cv_antenna_repick_v1/fit_antenna.npz"

# Anchor: the 91 m era, the only one with real multi-view parallax.
ANCHOR_ERA_M = 91.0
# Field-note heights per era (tether-anchor referenced).
ERAS = {"30m": 30.0, "87.5m": 87.5, "91m": 91.0}
# Known horizontal disagreement between bundle adjustment and MCMC (metres).
HORIZONTAL_UNCERTAINTY_M = 3.42
BEARING_STEP_DEG = 0.1
SKYLINE_SAMPLES = 768  # 2x the marjum_bundle default, for a finer radial sweep


def skyline_safe(terrain, cam, azimuth, count=SKYLINE_SAMPLES):
    """Skyline with a corrected tile-edge distance.

    Both marjum_bundle.Terrain.skyline and marjum_position.RefinedTerrain.skyline
    compute the edge distance as

        safe = where(abs(d) > 1e-12, d, 1e-12)
        return where(d >= 0, (hi - c)/safe, (lo - c)/safe)

    When a direction component is a tiny NEGATIVE number, `safe` is forced to
    +1e-12 while the branch still selects the (lo - c) formula, yielding a large
    negative distance. `end` then goes negative and is clamped to 2 m, so the ray
    is only sampled 2 m out and the returned horizon collapses to the ground
    directly beneath the observer. On an exact 0.1-degree grid this fires at
    bearing 270 (cos = -1.8e-16): the unpatched skyline returns -88.80 deg where
    the true value is +4.92 deg.

    CORRECTED 2026-09-13: this docstring previously quoted the reference as
    +3.58 deg. That figure came from a 1500 m brute-force march and was range
    limited - the horizon at bearing 270 is formed by terrain at 2934 m. A 0.5 m
    march to the tile edge (3030 m) gives +4.921 deg. State the march limit
    whenever quoting a brute-force reference.

    FIXED UPSTREAM 2026-09-13: software-engineer applied the same correction via
    np.copysign to marjum_bundle.Terrain.skyline and
    marjum_position.RefinedTerrain.skyline. This wrapper is now redundant against
    current source; it is retained so this product reproduces independently of
    those files' state. Verified: the patched root and this wrapper agree to
    exactly 0.0 at all four axis bearings.

    Continuous optimizer azimuths essentially never land within 1e-12 of an axis,
    so the fitting pipeline is not affected in practice - but a regular bearing
    grid, which is exactly what a horizon product needs, hits it every time.
    That was my hypothesis and it has since been tested by software-engineer
    across 5 converged fits: smallest |dx| or |dy| found was 0.21 m. Converged
    solutions only; historical optimizer iterates were not re-run.
    """
    az = np.atleast_1d(azimuth)
    ce, sn = np.cos(az), np.sin(az)

    def edge(c, lo, hi, d):
        sign = np.where(d >= 0, 1.0, -1.0)
        safe = np.where(np.abs(d) > 1e-12, d, sign * 1e-12)
        return np.where(d >= 0, (hi - c) / safe, (lo - c) / safe)

    end = np.minimum(edge(cam[0], terrain.e[0], terrain.e[-1], ce),
                     edge(cam[1], terrain.n[0], terrain.n[-1], sn))
    end = np.maximum(end - terrain.res, 2.0)
    fraction = np.linspace(0.0, 1.0, count)
    dist = np.exp(np.log(end[:, None]) * fraction)
    z = terrain.height(cam[0] + ce[:, None] * dist, cam[1] + sn[:, None] * dist)
    angle = np.arctan2(z - cam[2], dist)
    return np.max(np.where(np.isfinite(angle), angle, -np.pi / 2), axis=1)


def validate_against_brute_force(terrain, cam, n_probe=24, max_range=1500.0, seed=0):
    """Independent check: march each probe ray at 1 m steps and compare."""
    rng = np.random.default_rng(seed)
    probes = np.sort(np.concatenate([
        np.array([0.0, 90.0, 180.0, 270.0]),           # the degenerate cases
        rng.uniform(0, 360, n_probe - 4),
    ]))
    fast = np.degrees(skyline_safe(terrain, cam, np.radians(probes)))
    d = np.arange(1.0, max_range, 1.0)
    brute = []
    for b in np.radians(probes):
        z = terrain.height(cam[0] + np.cos(b) * d, cam[1] + np.sin(b) * d)
        brute.append(np.nanmax(np.degrees(np.arctan2(z - cam[2], d))))
    brute = np.array(brute)
    # Only compare where the brute-force march is not range-limited relative to
    # the full tile sweep; the fast version legitimately sees farther.
    diff = fast - brute
    return probes, fast, brute, diff


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


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Terrain applies working_grid in __init__; do NOT hand it a raw MarjumDEM.
    terrain = Terrain(MarjumDEM(cache_file=DEM_CACHE))

    with np.load(WORKSPACE / ANTENNA_FIT) as f:
        ant = np.asarray(f["antenna"], float)
    ground = float(terrain.height(ant[0], ant[1]))

    bearings_deg = np.arange(0.0, 360.0, BEARING_STEP_DEG)
    az = np.radians(bearings_deg)

    # Local ground relief around the antenna: this is why "height above ground"
    # is ill-defined at the metre level and must not be quoted bare.
    ee = ant[0] + np.linspace(-5, 5, 41)
    nn = ant[1] + np.linspace(-5, 5, 41)
    EE, NN = np.meshgrid(ee, nn)
    relief = terrain.height(EE, NN)
    relief_min, relief_max = float(np.nanmin(relief)), float(np.nanmax(relief))

    profiles = {}
    for era, note_h in ERAS.items():
        u = ant[2] - (ANCHOR_ERA_M - note_h)
        cam = np.array([ant[0], ant[1], u])
        elev = skyline_safe(terrain, cam, az)
        profiles[era] = {
            "u_m": u,
            "agl_m": u - ground,
            "fieldnote_height_m": note_h,
            "elev_rad": elev,
        }
        print(f"  {era:6s} u={u:9.3f}  AGL={u-ground:6.2f} m  "
              f"horizon elev deg: min {np.degrees(elev).min():+.3f} "
              f"max {np.degrees(elev).max():+.3f} "
              f"mean {np.degrees(elev).mean():+.3f}", flush=True)

    # Sensitivity of the 91 m profile to the antenna horizontal uncertainty:
    # displace by HORIZONTAL_UNCERTAINTY_M in 8 compass directions and record
    # the spread in horizon elevation per bearing.
    base = profiles["91m"]["elev_rad"]
    u91 = profiles["91m"]["u_m"]
    perturbed = []
    for th in np.arange(0, 2 * np.pi, np.pi / 4):
        c = np.array([ant[0] + HORIZONTAL_UNCERTAINTY_M * np.cos(th),
                      ant[1] + HORIZONTAL_UNCERTAINTY_M * np.sin(th), u91])
        perturbed.append(skyline_safe(terrain, c, az))
    perturbed = np.array(perturbed)
    spread = perturbed.max(0) - perturbed.min(0)
    print(f"\n  horizontal-uncertainty ({HORIZONTAL_UNCERTAINTY_M} m) induced horizon spread:")
    print(f"    median {np.degrees(np.median(spread)):.4f} deg   "
          f"p95 {np.degrees(np.percentile(spread,95)):.4f} deg   "
          f"max {np.degrees(spread.max()):.4f} deg")
    print(f"    (compare requested bearing resolution {BEARING_STEP_DEG} deg)")

    npz_path = out_dir / "horizon_profiles.npz"
    np.savez_compressed(
        npz_path,
        bearings_deg=bearings_deg,
        **{f"elev_rad_{e}": p["elev_rad"] for e, p in profiles.items()},
        **{f"u_m_{e}": p["u_m"] for e, p in profiles.items()},
        horizontal_perturbation_elev_rad=perturbed,
        horizontal_perturbation_spread_rad=spread,
        antenna_enu_m=ant,
        ground_under_antenna_m=ground,
    )

    commit = git_commit(WORKSPACE)
    meta = {
        "provenance": {
            "product": "horizon_profiles",
            "campaign": "marjum-2026-07",
            "version": "v1",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "terrain/marjum_horizons.py",
            "generator_commit": commit,
            "inputs": [
                {"path": f"terrain/{ANTENNA_FIT}",
                 "sha256": sha256(WORKSPACE / ANTENNA_FIT)},
                {"path": f"terrain/{DEM_CACHE}", "sha256": sha256(WORKSPACE / DEM_CACHE)},
            ],
            "params": {
                "bearing_step_deg": BEARING_STEP_DEG,
                "skyline_samples": SKYLINE_SAMPLES,
                "anchor_era_m": ANCHOR_ERA_M,
                "horizontal_uncertainty_m": HORIZONTAL_UNCERTAINTY_M,
            },
        },
        "stamp": f"marjum-2026-07/horizon_profiles@v1+{commit}",
        "data_file": "horizon_profiles.npz",
        "frame": "local working grid (ENU m); bearings are atan2(dN,dE) degrees; "
                 "elevation is angle above horizontal in radians",
        "antenna_enu_m": [round(float(x), 3) for x in ant],
        "ground_under_antenna_m": round(ground, 2),
        "local_ground_relief_m": {
            "window": "+/-5 m around antenna",
            "min": round(relief_min, 1),
            "max": round(relief_max, 1),
            "spread": round(relief_max - relief_min, 1),
        },
        "eras": {
            e: {
                "fieldnote_height_m": p["fieldnote_height_m"],
                "antenna_u_m": round(float(p["u_m"]), 3),
                "agl_m_vs_ground_under_antenna": round(float(p["agl_m"]), 2),
                "horizon_elev_deg": {
                    "min": round(float(np.degrees(p["elev_rad"]).min()), 4),
                    "max": round(float(np.degrees(p["elev_rad"]).max()), 4),
                    "mean": round(float(np.degrees(p["elev_rad"]).mean()), 4),
                },
            }
            for e, p in profiles.items()
        },
        "horizontal_uncertainty_sensitivity": {
            "displacement_m": HORIZONTAL_UNCERTAINTY_M,
            "directions": 8,
            "horizon_elev_spread_deg": {
                "median": round(float(np.degrees(np.median(spread))), 4),
                "p95": round(float(np.degrees(np.percentile(spread, 95))), 4),
                "max": round(float(np.degrees(spread.max())), 4),
            },
            "note": (
                "Spread in horizon elevation induced by displacing the antenna by the "
                "known bundle-adjustment-vs-MCMC horizontal disagreement. This is the "
                "dominant geometric uncertainty on these profiles."
            ),
        },
        "assumptions": [
            "ANCHORING: absolute altitude comes from the 91 m era (the only era with "
            "multi-view parallax). The 30 m and 87.5 m eras are placed using field-note "
            "height DIFFERENCES, not absolute values, because field notes are "
            "tether-anchor-referenced and photogrammetry is DEM-referenced; the two "
            "disagree by ~2.5 m in absolute terms. Valid only if the field notes were "
            "taken consistently across eras.",
            "HORIZONTAL: all three eras use the 91 m horizontal position, assuming the "
            "highline anchor and trolley did not move between eras. This is "
            "load-dependent - field notes record ~0.97 m deflection at the 2 m "
            "configuration - so treat it as a stated systematic. The 30 m era is the "
            "most exposed, being furthest in load from the anchor era.",
            "The 30 m and 87.5 m eras have NO antenna pixel picks in the photo record "
            "(all 31 picks lie in IMG_2209-2245, the 91 m set), so their positions are "
            "not independently measurable from images. These profiles are a modelled "
            "bracket, not a measurement, at those two heights.",
        ],
        "known_upstream_bug_worked_around": {
            "affects": [
                "marjum_bundle.Terrain.skyline",
                "marjum_position.RefinedTerrain.skyline",
            ],
            "defect": (
                "Tile-edge distance is computed as safe=where(abs(d)>1e-12, d, 1e-12) "
                "followed by where(d>=0, (hi-c)/safe, (lo-c)/safe). For a tiny NEGATIVE "
                "direction component, safe is forced positive while the branch still "
                "selects the (lo-c) formula, producing a large negative distance. `end` "
                "then goes negative and is clamped to 2 m, so the ray is sampled only 2 m "
                "out and the horizon collapses to the ground directly beneath."
            ),
            "observed": (
                "On an exact 0.1-degree grid this fires at bearing 270 (cos = -1.8e-16): "
                "the unpatched skyline returns -88.80 deg where a 0.5 m brute-force march "
                "to the tile edge (3030 m) gives +4.921 deg. 1 of 3600 grid bearings "
                "affected; it was the profile minimum, so min elevation was wrong by "
                "~93 deg."
            ),
            "status": (
                "FIXED UPSTREAM 2026-09-13 by software-engineer, via np.copysign in both "
                "marjum_bundle.Terrain.skyline and marjum_position.RefinedTerrain.skyline. "
                "The patched root and this product's skyline_safe wrapper were verified to "
                "agree to exactly 0.0 at all four axis bearings - two independently written "
                "fixes in agreement."
            ),
            "fitting_pipeline_impact": (
                "No evidence of impact. The hypothesis was that continuous optimizer "
                "azimuths never land within 1e-12 of an axis; software-engineer tested it "
                "across 5 converged fits (cv_transmitter_v3, joint_v1, v5, joint_v5, "
                "joint_v2) and found the smallest |dx| or |dy| to be 0.21 m. Caveat: "
                "converged solutions only, historical optimizer iterates were not re-run."
            ),
            "workaround": (
                "This product uses marjum_horizons.skyline_safe, which preserves the sign "
                "of the perturbed denominator. Now redundant against patched source, but "
                "retained so the product reproduces independently of those files' state. "
                "Validated against a 0.5 m brute-force march to the tile edge: agreement "
                "0.005 deg at bearing 270, 0.000 deg at 262.6."
            ),
        },
        "limitations": [
            "DEM cache marjum_dem_sw.npz is int32: elevation quantized to 1 m. At the "
            "typical horizon ranges here that is the dominant DEM-side error.",
            "Terrain.skyline is a finite-resolution radial sampler, not full ray "
            "tracing - its own docstring calls it 'only a surrogate for full ray "
            "tracing'. Fine ridges between radial samples can be missed.",
            "Horizon elevations are bounded by the DEM tile edge; terrain beyond the "
            "tile is not represented and distant low horizons may be truncated.",
            "'Height above ground' is ill-defined at the metre level here: local relief "
            "within +/-5 m of the antenna spans "
            f"{round(relief_max - relief_min, 1)} m. AGL figures are quoted against the "
            "single DEM sample directly beneath the antenna.",
        ],
    }
    json_path = out_dir / "horizon_profiles.json"
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {npz_path}")
    print(f"wrote {json_path}")
    print(f"  stamp {meta['stamp']}")


if __name__ == "__main__":
    main()
