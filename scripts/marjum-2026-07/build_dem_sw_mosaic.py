"""Build a float32 DEM mosaic on the v0002 grid: derived/dem/v0003 and v0004.

DEM v0002 covers the 3 x 4 USGS 3DEP tiles 91-93 east, 45-48 north. The
antenna's skyline at 87.5-91 m height reaches terrain beyond its west and
south edges, so horizon products need a larger mosaic. This builds one from
the staged GeoTIFFs of the same 0.5 m project and keeps the v0002 coordinate
frame exactly: ``raster_origin`` stays at v0002's southwest pixel centre and
the added terrain to the west and south is addressed through ``e0_px`` and
``n0_px``. It checks that the base version's pixels are unchanged.

- v0003 (the defaults): tiles 90-93 east x 44-48 north on base v0002, i.e.
  1 km more to the west and south (``e0_px = n0_px = 2000``); the retired
  ``terrain/marjum_dem_sw.npz`` held the same mosaic.
- v0004: tiles 90-93 east x 38-48 north on base v0003, i.e. v0003 plus 6 km
  to the south (``n0_px = 14000``). v0003's south edge cut the skyline;
  ``coverage_certificate.py`` shows terrain outside v0004 cannot change it.

Usage:
    python build_dem_sw_mosaic.py --out-dir <campaign>/derived/dem/v0003
    python build_dem_sw_mosaic.py --out-dir <campaign>/derived/dem/v0004 \\
        --version v0004 --base v0003 --north 38 48 --fetch \\
        --expect-dem-sha256 42949f2335ebc8d7d719b07b820cd7a6a33e01a0e8492f8a2619ce0123131a0f
(stage the GeoTIFFs in <out-dir>/inputs first, or download them with --fetch).
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

from eigsep_data.paths import get_campaign_root

FILE_BASE = "USGS_OPR_UT_WestEast_B22_12STJ"
URL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects/"
       "UT_WestEast_B22/UT_WestEast_7_B22/TIFF/")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo):
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                    text=True, check=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))


def fetch(paths):
    for path in paths:
        if path.is_file():
            continue
        print("download", path.name, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(URL + path.name, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--version", default="v0003", help="product version for the manifest")
    ap.add_argument("--base", default="v0002", help="DEM version whose pixels must be kept")
    ap.add_argument("--east", type=int, nargs=2, default=(90, 93), metavar=("LO", "HI"),
                    help="inclusive 1 km tile range, east (default 90 93)")
    ap.add_argument("--north", type=int, nargs=2, default=(44, 48), metavar=("LO", "HI"),
                    help="inclusive 1 km tile range, north (default 44 48)")
    ap.add_argument("--fetch", action="store_true", help="download missing GeoTIFFs into inputs/")
    ap.add_argument("--expect-dem-sha256", default=None,
                    help="stop unless the SHA-256 of the dem array's bytes equals this")
    args = ap.parse_args()
    import eigsep_terrain
    from eigsep_terrain.dem import DEM

    campaign = Path(get_campaign_root(required=True))
    base_dir = campaign / "derived" / "dem" / args.base
    base_manifest = json.loads((base_dir / "manifest.json").read_text())
    base_file = base_dir / "marjum_dem.npz"
    if sha256(base_file) != base_manifest["files"]["marjum_dem.npz"]["sha256"]:
        raise ValueError(f"DEM {args.base} differs from its manifest")

    out = args.out_dir
    output = out / "marjum_dem.npz"
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    inputs = out / "inputs"
    east = range(args.east[0], args.east[1] + 1)
    north = range(args.north[0], args.north[1] + 1)
    names = np.array([[f"{FILE_BASE}{e}{n}.tif" for n in north] for e in east])
    tifs = [inputs / name for name in names.flat]
    if args.fetch:
        fetch(tifs)
    # map_crd is copied from the base below, so the XML is optional (v0003 staged it)
    xml = inputs / f"{FILE_BASE}9145.xml"
    paths = ([xml] if xml.is_file() else []) + tifs
    missing = [str(p) for p in tifs if not p.is_file()]
    if missing:
        raise FileNotFoundError("stage these inputs first (or use --fetch):\n" + "\n".join(missing))
    for name, digest in base_manifest["input_sha256"].items():
        if (inputs / name).is_file() and sha256(inputs / name) != digest:
            raise ValueError(f"{name} differs from the copy DEM {args.base} used")
    input_hashes = {p.name: sha256(p) for p in paths}

    dem = DEM()
    dem.load_tif(np.vectorize(str)(np.array([[inputs / name for name in row] for row in names])),
                 survey_offset=(0, 0, 3))
    if xml.is_file():
        dem.load_xml(str(xml))
    with np.load(base_file) as base:
        origin = np.asarray(base["raster_origin"], float)
        heights = base["dem"]
        base_e0, base_n0 = int(base["e0_px"]), int(base["n0_px"])
        map_crd = {k: base[k] for k in ("eastbc", "westbc", "northbc", "southbc")}
    # pixels from the new mosaic's southwest centre to the frame origin, and to the base's corner
    shift = (origin - dem.raster_origin) / float(dem.res)
    if not np.array_equal(shift, np.round(shift)) or np.any(shift < 0):
        raise ValueError(f"mosaic does not extend DEM {args.base} by whole pixels to the west and south")
    e0_px, n0_px = (int(s) for s in shift)
    r0, c0 = n0_px - base_n0, e0_px - base_e0
    overlap = dem.data[r0:r0 + heights.shape[0], c0:c0 + heights.shape[1]]
    if r0 < 0 or c0 < 0 or overlap.shape != heights.shape or not np.array_equal(overlap, heights):
        raise ValueError(f"mosaic overlap differs from DEM {args.base}")
    digest = hashlib.sha256(np.ascontiguousarray(dem.data).tobytes()).hexdigest()
    if args.expect_dem_sha256 and digest != args.expect_dem_sha256:
        raise ValueError(f"dem array SHA-256 {digest} differs from the expected one")
    dem.raster_origin = origin
    dem.e0_px, dem.n0_px = e0_px, n0_px
    dem.map_crd = map_crd
    dem.files = names          # tile names, not local paths, so the cache does not depend on them
    dem._cache_file = str(output)
    out.mkdir(parents=True, exist_ok=True)
    dem.save_cache()

    check = DEM(cache_file=str(output))
    probe = (1656.22, 2031.83)
    if abs(float(check.interp_alt(*probe)) - float(DEM(cache_file=str(base_file)).interp_alt(*probe))) > 0:
        raise ValueError(f"{args.version} and {args.base} disagree at the antenna")

    package = Path(eigsep_terrain.__file__).resolve().parents[2]
    commit, dirty = git_state(package)
    gen_commit, gen_dirty = git_state(Path(__file__).resolve().parent)
    ny, nx = dem.data.shape
    res = float(dem.res)
    manifest = {
        "product": "dem", "version": args.version, "published": date.today().isoformat(),
        "description": (f"Float32 Marjum DEM, {len(east)} east x {len(north)} north 3DEP tiles "
                        f"12STJ{east[0]}-{east[-1]} x {north[0]}-{north[-1]}, on the DEM v0002 UTM "
                        f"raster grid (origin unchanged; {args.base} kept pixel for pixel)."),
        "files": {output.name: {"sha256": sha256(output), "bytes": output.stat().st_size}},
        "dem_array_sha256": digest,
        "input_sha256": input_hashes,
        "input_url_base": URL,
        "input_order_east_north": names.tolist(),
        f"{args.base}_sha256": sha256(base_file),
        "validation": {f"overlap_identical_to_{args.base}": True,
                       "shape_north_east": [ny, nx],
                       "pixel_size_m": res, "epsg": int(dem.crs.to_epsg()),
                       "raster_origin_utm_m": origin.tolist(),
                       "e0_px": e0_px, "n0_px": n0_px,
                       "extent_e_m": [-e0_px * res, (nx - 1 - e0_px) * res],
                       "extent_n_m": [-n0_px * res, (ny - 1 - n0_px) * res],
                       "survey_offset_subtracted_m": dem.survey_offset.tolist()},
        "builder": {"script": "data-analysis/scripts/marjum-2026-07/build_dem_sw_mosaic.py",
                    "data_analysis_commit": gen_commit, "data_analysis_dirty": gen_dirty,
                    "eigsep_terrain_commit": commit, "eigsep_terrain_dirty": dirty},
        "bulk_sync": "eigsep-sync",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"built and validated {output}: shape {dem.data.shape}, e0/n0 {e0_px}/{n0_px}, "
          f"dem sha256 {digest}")


if __name__ == "__main__":
    main()
