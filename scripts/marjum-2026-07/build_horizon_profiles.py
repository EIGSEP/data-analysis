"""Build curation/horizon_profiles_vNNNN: the terrain skyline seen from the antenna.

For each bearing (atan2(dN, dE), 0.1 deg steps) the DEM is sampled by
bilinear interpolation every 0.25 m from the antenna out to the edge of the
DEM tile, the Earth's curvature is removed geometrically (R = 6371 km, no
refraction), and the horizon is the largest elevation angle along the ray.
Three eras share the 91 m-era horizontal position; the 30 m and 87.5 m eras
are offset in height by the field-note height differences. A sensitivity
block repeats the 91 m profile with the antenna displaced horizontally in
eight directions.

v0002 was made by an uncommitted copy of this procedure on the int32
``terrain/marjum_dem_sw.npz``; ``--quantize`` reproduces that quantisation
from the float32 cache for validation (``--validate-against``).

Usage (current product):
    python build_horizon_profiles.py v0003 --out-dir <campaign>/curation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eigsep_data.paths import get_campaign_root

BEARING_STEP_DEG = 0.1
MARCH_STEP_M = 0.25
EARTH_R_M = 6371e3
ANCHOR_ERA_M = 91.0
ERAS = {"30m": 30.0, "87.5m": 87.5, "91m": 91.0}
DIRECTIONS = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo):
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                    text=True, check=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain",
                                                         "--untracked-files=no"))


class Grid:
    """A DEM cache's elevations on its raster (working) grid, in metres."""

    def __init__(self, path, quantize="none"):
        with np.load(path) as z:
            data = np.asarray(z["dem"], float)
            self.res = float(z["res"])
            self.e0, self.n0 = int(z["e0_px"]), int(z["n0_px"])
        if quantize == "round":
            data = np.round(data)
        elif quantize == "trunc":
            data = np.trunc(data)
        self.data = data
        self.dtype = "float32" if quantize == "none" else f"int32 ({quantize}, emulated)"
        ny, nx = data.shape
        self.extent_e = ((0 - self.e0) * self.res, (nx - 1 - self.e0) * self.res)
        self.extent_n = ((0 - self.n0) * self.res, (ny - 1 - self.n0) * self.res)

    def interp(self, e, n):
        x = e / self.res + self.e0
        y = n / self.res + self.n0
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        d = self.data
        return ((1 - fx) * (1 - fy) * d[y0, x0] + fx * (1 - fy) * d[y0, x0 + 1]
                + (1 - fx) * fy * d[y0 + 1, x0] + fx * fy * d[y0 + 1, x0 + 1])

    def edge_range(self, e, n, ux, uy):
        """Distance along (ux, uy) from (e, n) to the last interpolable sample."""
        lim = []
        for u, p, (lo, hi) in ((ux, e, self.extent_e), (uy, n, self.extent_n)):
            hi = hi - self.res          # keep the +1 neighbour inside the array
            if u > 1e-12:
                lim.append((hi - p) / u)
            elif u < -1e-12:
                lim.append((lo - p) / u)
        return min(lim)


def profile(grid, e, n, heights, bearings):
    """Horizon elevation (rad) per bearing for each antenna height in ``heights``."""
    out = np.empty((len(heights), len(bearings)))
    hs = np.asarray(heights, float)[:, None]
    for j, b in enumerate(np.radians(bearings)):
        ux, uy = np.cos(b), np.sin(b)
        r = np.arange(MARCH_STEP_M, grid.edge_range(e, n, ux, uy), MARCH_STEP_M)
        z = grid.interp(e + r * ux, n + r * uy) - r ** 2 / (2 * EARTH_R_M)
        out[:, j] = np.max(np.arctan2(z[None, :] - hs, r[None, :]), axis=1)
    return out


def build(grid, antenna, bearings):
    e, n, u91 = antenna
    heights = {era: u91 - (ANCHOR_ERA_M - h) for era, h in ERAS.items()}
    prof = profile(grid, e, n, list(heights.values()), bearings)
    elev = dict(zip(heights, prof))
    return heights, elev


def summary(elev_rad):
    d = np.degrees(elev_rad)
    return {"min": round(float(d.min()), 4), "max": round(float(d.max()), 4),
            "mean": round(float(d.mean()), 4)}


def change(new, old):
    d = np.abs(np.degrees(new - old))
    return {"median": round(float(np.median(np.degrees(new - old))), 4),
            "p95_abs": round(float(np.percentile(d, 95)), 4),
            "max_abs": round(float(d.max()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="product version, e.g. v0003")
    ap.add_argument("--release", default="v0004", help="geometry release supplying the antenna")
    ap.add_argument("--dem", default=None, help="DEM cache (default: derived/dem/v0002/marjum_dem.npz)")
    ap.add_argument("--quantize", choices=("none", "round", "trunc"), default="none")
    ap.add_argument("--antenna", type=float, nargs=3, default=None,
                    help="override the antenna E N U (validation only)")
    ap.add_argument("--displacement-m", type=float, default=1.7,
                    help="illustrative horizontal displacement for the sensitivity block")
    ap.add_argument("--previous", default="v0002", help="previous version to report changes against")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <campaign>/curation")
    ap.add_argument("--validate-against", type=Path, default=None,
                    help="compare elevations with this npz and print max |delta|; writes nothing")
    args = ap.parse_args()

    campaign = Path(get_campaign_root(required=True))
    workspace = campaign.parent
    shared_path = campaign / "imgs" / "fits" / f"{args.release}_marjum_geometry" / "shared.json"
    dem_path = Path(args.dem) if args.dem else campaign / "derived" / "dem" / "v0002" / "marjum_dem.npz"
    shared = json.loads(shared_path.read_text())
    antenna = np.asarray(args.antenna if args.antenna else
                         shared["antenna_91m_era"]["position_enu_m"], float)
    grid = Grid(dem_path, args.quantize)
    bearings = np.round(np.arange(0, 360, BEARING_STEP_DEG), 6)
    heights, elev = build(grid, antenna, bearings)

    if args.validate_against:
        with np.load(args.validate_against) as old:
            for era in ERAS:
                d = np.degrees(np.abs(elev[era] - old[f"elev_rad_{era}"]))
                print(f"{era}: max |delta elev| {d.max():.5f} deg, p99 {np.percentile(d, 99):.5f},"
                      f" median {np.median(d):.6f}; bearings > 0.01 deg: {(d > 0.01).sum()}")
            print("ground under antenna", float(grid.interp(antenna[0], antenna[1])),
                  "recorded", float(old["ground_under_antenna_m"]))
        return

    th = np.radians(np.arange(DIRECTIONS) * 360.0 / DIRECTIONS)
    pert = np.stack([profile(grid, antenna[0] + args.displacement_m * np.cos(t),
                             antenna[1] + args.displacement_m * np.sin(t),
                             [antenna[2]], bearings)[0] for t in th])
    spread = pert.max(0) - pert.min(0)
    ground = float(grid.interp(antenna[0], antenna[1]))
    win = np.arange(-5, 5.001, grid.res)
    E, N = np.meshgrid(antenna[0] + win, antenna[1] + win)
    local = grid.interp(E.ravel(), N.ravel())

    out_dir = args.out_dir or campaign / "curation"
    prev = out_dir / f"horizon_profiles_{args.previous}.npz"
    if not prev.exists():
        prev = campaign / "curation" / f"horizon_profiles_{args.previous}.npz"
    old = np.load(prev) if prev.exists() else None

    commit, dirty = git_state(Path(__file__).resolve().parent)
    stamp = f"marjum-2026-07/horizon_profiles@{args.version}+{commit}" + ("-dirty" if dirty else "")
    data_file = f"horizon_profiles_{args.version}.npz"
    arrays = {"bearings_deg": bearings, "antenna_enu_m": antenna,
              "ground_under_antenna_m": np.float64(ground),
              "horizontal_perturbation_elev_rad": pert,
              "horizontal_perturbation_spread_rad": spread}
    for era in ERAS:
        arrays[f"elev_rad_{era}"] = elev[era]
        arrays[f"u_m_{era}"] = np.float64(heights[era])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / data_file, **arrays)

    rel = lambda p: str(Path(p).absolute().relative_to(workspace.absolute())) \
        if str(Path(p).absolute()).startswith(str(workspace.absolute())) else str(p)
    eras = {}
    for era, h in ERAS.items():
        eras[era] = {"fieldnote_height_m": h, "antenna_u_m": round(heights[era], 3),
                     "agl_m_vs_ground_under_antenna": round(heights[era] - ground, 2),
                     "horizon_elev_deg": summary(elev[era])}
        if old is not None:
            eras[era][f"change_from_{args.previous}_deg"] = change(elev[era], old[f"elev_rad_{era}"])
    frame = shared.get("frame", {})
    product = {
        "provenance": {
            "product": "horizon_profiles", "campaign": "marjum-2026-07", "version": args.version,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "data-analysis/scripts/marjum-2026-07/build_horizon_profiles.py",
            "generator_commit": commit, "generator_dirty": dirty,
            "inputs": [{"path": rel(p), "sha256": sha256(p)} for p in (shared_path, dem_path)]
                      + ([{"path": rel(prev), "sha256": sha256(prev)}] if old is not None else []),
            "params": {"bearing_step_deg": BEARING_STEP_DEG, "march_step_m": MARCH_STEP_M,
                       "march_limit": "DEM tile edge along each bearing",
                       "interpolation": "bilinear",
                       "earth_curvature": "geometric, R = 6371 km, no refraction",
                       "anchor_era_m": ANCHOR_ERA_M,
                       "illustrative_displacement_m": args.displacement_m},
        },
        "stamp": stamp, "data_file": data_file,
        "supersedes": f"horizon_profiles {args.previous}",
        "frame": ("UTM raster grid of derived/dem/v0002 (EPSG:6341 minus raster origin "
                  f"{frame.get('raster_origin_m')}); bearings are atan2(dN,dE) degrees; "
                  "elevation is angle above horizontal in radians"),
        "antenna_enu_m": [round(float(x), 3) for x in antenna],
        "antenna_source": (f"marjum-2026-07/imgs/fits/{args.release}_marjum_geometry/shared.json "
                           f"antenna_91m_era (release {args.release}, "
                           f"{shared['antenna_91m_era'].get('status')})"),
        "geometry_release": args.release,
        "ground_under_antenna_m": round(ground, 2),
        "dem": {"cache": rel(dem_path), "dtype": grid.dtype, "grid_m": grid.res,
                "extent_e_m": list(grid.extent_e), "extent_n_m": list(grid.extent_n)},
        "local_ground_relief_m": {"window": "+/-5 m around antenna",
                                  "min": round(float(local.min()), 2),
                                  "max": round(float(local.max()), 2),
                                  "spread": round(float(local.max() - local.min()), 2)},
        "eras": eras,
        "horizontal_displacement_sensitivity": {
            "displacement_m": args.displacement_m, "directions": DIRECTIONS,
            "horizon_elev_spread_deg": {
                "median": round(float(np.degrees(np.median(spread))), 4),
                "p95": round(float(np.degrees(np.percentile(spread, 95))), 4),
                "max": round(float(np.degrees(spread.max())), 4)},
            "note": ("Spread in the 91 m horizon from displacing the antenna by this distance in "
                     "8 directions. Illustrative only: the geometry release gives no uncertainty, "
                     "and the displacement is the historical v0001 bounding radius."),
        },
        "assumptions": [
            "The 91 m era anchors altitude; 30 m and 87.5 m are offset by field-note height "
            "differences; all eras use the 91 m horizontal position.",
            "A horizon profile represents the terrain as everything below the skyline at each "
            "bearing. Terrain visible above a nearer, lower ridge is inside that region, so the "
            "mask is exact for a height field.",
        ],
        "limitations": [
            "No vegetation, talus or suspension structure in the DEM.",
            f"The antenna position (release {args.release}) has no established uncertainty.",
            "The march stops at the DEM tile edge, which is nearer to the west and south than "
            "in v0002's larger mosaic (see the extent).",
        ],
    }
    (out_dir / f"horizon_profiles_{args.version}.json").write_text(json.dumps(product, indent=1) + "\n")
    print(f"wrote {out_dir / data_file} and its JSON ({stamp})")


if __name__ == "__main__":
    main()
