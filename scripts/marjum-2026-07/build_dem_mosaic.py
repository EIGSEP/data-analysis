"""Build a larger Marjum DEM mosaic on the DEM v0002/v0003 raster grid.

Generalises ``build_dem_sw_mosaic.py`` (DEM v0003) to any rectangle of
USGS 3DEP 0.5 m tiles ``12STJ<east><north>`` from the same lidar project
(UT_WestEast_B22). The coordinate frame is kept exactly: ``raster_origin``
stays at v0002's southwest pixel centre and the added terrain to the west and
south is addressed through ``e0_px`` and ``n0_px``. Every pixel of the base
mosaic must be identical in the result, and the GeoTIFF tags of every tile
are checked (size, 0.5 m pixels, georeferencing, CRS, pixel convention, no
nodata). The output has the base cache's keys and dtypes, so
``eigsep_terrain.dem.DEM(cache_file=...)`` and ``build_horizon_profiles.py``
read it unchanged; ``files`` holds the tile names, indexed east then north.

Tiles are read with Pillow rather than ``eigsep_terrain`` so the builder needs
only numpy and Pillow; the base-overlap check ties it to the
``eigsep_terrain``-built base pixel for pixel.

Usage (DEM v0004: v0003 plus 6 km to the south):
    python build_dem_mosaic.py v0004 --east 90 93 --north 38 48 --fetch \\
        --expect-dem-sha256 42949f2335ebc8d7d719b07b820cd7a6a33e01a0e8492f8a2619ce0123131a0f
writes <campaign>/derived/dem/v0004/{marjum_dem.npz,manifest.json}, staging the
GeoTIFFs in its inputs/ (downloaded with --fetch).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from eigsep_data.paths import get_campaign_root

FILE_BASE = "USGS_OPR_UT_WestEast_B22_12STJ"
URL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects/"
       "UT_WestEast_B22/UT_WestEast_7_B22/TIFF/")
TILE_PX = 2000           # 1 km tiles at 0.5 m
RES = 0.5
UTM_E0, UTM_N0 = 200000.0, 4300000.0   # SW corner of the 12STJ 100 km square, EPSG:6341


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo):
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                        text=True, check=True).stdout.strip()
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain",
                                                  "--untracked-files=no"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def tile_name(east, north):
    return f"{FILE_BASE}{east:02d}{north:02d}.tif"


def read_tile(path, east, north):
    """One tile as float32, south-up like the DEM cache; tags are checked."""
    with Image.open(path) as im:
        if im.size != (TILE_PX, TILE_PX):
            raise ValueError(f"{path}: unexpected tile size {im.size}")
        if tuple(im.tag_v2[33550]) != (RES, RES, 0.0):
            raise ValueError(f"{path}: unexpected pixel scale")
        origin = (0.0, 0.0, 0.0, UTM_E0 + east * 1000, UTM_N0 + (north + 1) * 1000, 0.0)
        if tuple(im.tag_v2[33922]) != origin:
            raise ValueError(f"{path}: unexpected georeferencing")
        keys = tuple(im.tag_v2[34735])
        entries = {keys[i]: keys[i + 1:i + 4] for i in range(4, len(keys), 4)}
        if entries.get(3072) != (0, 1, 6341) or entries.get(1025) != (0, 1, 1):
            raise ValueError(f"{path}: unexpected CRS or pixel convention")
        a = np.array(im, dtype=np.float32)[::-1]
        nodata = im.tag_v2.get(42113)
        if not np.isfinite(a).all() or (nodata is not None and np.any(a == float(nodata))):
            raise ValueError(f"{path}: nonfinite or nodata samples")
        return a


def fetch(names, tiles_dir):
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = tiles_dir / name
        if path.exists():
            continue
        print("download", name, flush=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(URL + name, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="product version, e.g. v0004")
    ap.add_argument("--base", default="v0003", help="base DEM version (kept pixel for pixel)")
    ap.add_argument("--east", type=int, nargs=2, required=True, metavar=("LO", "HI"),
                    help="inclusive 1 km tile index range, east")
    ap.add_argument("--north", type=int, nargs=2, required=True, metavar=("LO", "HI"),
                    help="inclusive 1 km tile index range, north")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <campaign>/derived/dem/<version>")
    ap.add_argument("--tiles-dir", type=Path, default=None, help="default: <out-dir>/inputs")
    ap.add_argument("--fetch", action="store_true", help="download missing tiles")
    ap.add_argument("--expect-dem-sha256", default=None,
                    help="stop unless the SHA-256 of the dem array's bytes equals this")
    args = ap.parse_args()
    campaign = Path(get_campaign_root(required=True))
    base_dir = campaign / "derived" / "dem" / args.base
    args.out_dir = args.out_dir or campaign / "derived" / "dem" / args.version
    args.tiles_dir = args.tiles_dir or args.out_dir / "inputs"

    base_npz = base_dir / "marjum_dem.npz"
    base_manifest = json.loads((base_dir / "manifest.json").read_text())
    if sha256(base_npz) != base_manifest["files"]["marjum_dem.npz"]["sha256"]:
        raise ValueError("base DEM differs from its manifest")
    with np.load(base_npz, allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}

    east = range(args.east[0], args.east[1] + 1)
    north = range(args.north[0], args.north[1] + 1)
    names = np.array([[tile_name(e, n) for n in north] for e in east])
    if args.fetch:
        fetch(names.flat, args.tiles_dir)
    for name, digest in base_manifest["input_sha256"].items():
        path = args.tiles_dir / name
        if name.endswith(".tif") and name in names and sha256(path) != digest:
            raise ValueError(f"{name} differs from the copy the base DEM used")

    # pixel (0, 0) centre of the new mosaic, relative to the base raster origin
    sw = np.array([UTM_E0 + east[0] * 1000 + RES / 2, UTM_N0 + north[0] * 1000 + RES / 2])
    shift = (np.asarray(base["raster_origin"], float) - sw) / RES
    if float(base["res"]) != RES or not np.array_equal(shift, np.round(shift)) \
            or np.any(shift < 0):
        raise ValueError("base frame is not on this tile grid, or the mosaic does not "
                         "contain the base origin")
    e0_px, n0_px = int(shift[0]), int(shift[1])

    out = args.out_dir / "marjum_dem.npz"
    if out.exists():
        raise FileExistsError(f"refusing to replace {out}")
    dem = np.empty((len(north) * TILE_PX, len(east) * TILE_PX), dtype=np.float32)
    for i, e in enumerate(east):
        for j, n in enumerate(north):
            dem[j * TILE_PX:(j + 1) * TILE_PX, i * TILE_PX:(i + 1) * TILE_PX] = \
                read_tile(args.tiles_dir / names[i, j], e, n)

    # the base mosaic sits at its own e0/n0 offset inside the new one
    r0 = n0_px - int(base["n0_px"])
    c0 = e0_px - int(base["e0_px"])
    by, bx = base["dem"].shape
    if r0 < 0 or c0 < 0 or r0 + by > dem.shape[0] or c0 + bx > dem.shape[1]:
        raise ValueError("the new mosaic does not contain the base mosaic")
    if not np.array_equal(dem[r0:r0 + by, c0:c0 + bx], base["dem"]):
        raise ValueError("base pixels changed")
    digest = hashlib.sha256(np.ascontiguousarray(dem).tobytes()).hexdigest()
    if args.expect_dem_sha256 and digest != args.expect_dem_sha256:
        raise SystemExit(f"dem array hash {digest} differs from the expected one")

    saved = dict(base)
    saved.update(dem=dem, files=names, e0_px=np.asarray(e0_px, dtype=base["e0_px"].dtype),
                 n0_px=np.asarray(n0_px, dtype=base["n0_px"].dtype))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, **saved)

    commit, dirty = git_state(Path(__file__).resolve().parent)
    extent_e = (-e0_px * RES, (dem.shape[1] - 1 - e0_px) * RES)
    extent_n = (-n0_px * RES, (dem.shape[0] - 1 - n0_px) * RES)
    manifest = {
        "product": "dem", "version": args.version, "published": date.today().isoformat(),
        "description": (f"Float32 Marjum DEM, {len(east)} east x {len(north)} north 3DEP "
                        f"0.5 m tiles 12STJ{east[0]}-{east[-1]} x {north[0]}-{north[-1]}, on "
                        "the DEM v0002 UTM raster grid (origin unchanged)."),
        "files": {out.name: {"sha256": sha256(out), "bytes": out.stat().st_size}},
        "dem_array_sha256": digest,
        "input_sha256": {name: sha256(args.tiles_dir / name) for name in names.flat},
        "input_url_base": URL,
        "input_order_east_north": names.tolist(),
        "base": {"version": base_manifest.get("version"), "sha256": sha256(base_npz),
                 "overlap_identical": True, "row_col_offset": [r0, c0]},
        "validation": {"shape_north_east": list(dem.shape), "pixel_size_m": RES,
                       "epsg": int(base["crs_epsg"]),
                       "raster_origin_utm_m": np.asarray(base["raster_origin"]).tolist(),
                       "e0_px": e0_px, "n0_px": n0_px,
                       "extent_e_m": extent_e, "extent_n_m": extent_n,
                       "survey_offset_m": np.asarray(base["survey_offset"]).tolist(),
                       "map_crd": "copied from the base cache, as v0003 copied v0002's"},
        "builder": {"script": "data-analysis/scripts/marjum-2026-07/build_dem_mosaic.py",
                    "data_analysis_commit": commit, "data_analysis_dirty": dirty},
        "bulk_sync": "eigsep-sync",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"built {out}: shape {dem.shape}, e0/n0 {e0_px}/{n0_px}, "
          f"extent E {extent_e} N {extent_n}, dem sha256 {digest}")


if __name__ == "__main__":
    main()
