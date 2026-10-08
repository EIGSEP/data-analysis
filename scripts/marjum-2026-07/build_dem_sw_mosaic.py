"""Build derived/dem/v0003: the 4-east x 5-north float32 mosaic on the v0002 grid.

DEM v0002 covers the 3 x 4 USGS 3DEP tiles 91-93 east, 45-48 north. The
antenna's skyline at 87.5-91 m height reaches terrain beyond its west and
south edges, so horizon products need the larger mosaic that the retired
``terrain/marjum_dem_sw.npz`` held (tiles 90-93 east, 44-48 north). This
builds that mosaic from the staged GeoTIFFs and keeps the v0002 coordinate
frame exactly: ``raster_origin`` stays at v0002's southwest pixel centre and
the extra kilometre to the west and south is addressed through
``e0_px = n0_px = 2000``. It checks that the overlap equals v0002 pixel for
pixel.

Usage:
    python build_dem_sw_mosaic.py --out-dir <campaign>/derived/dem/v0003
(stage the 20 GeoTIFFs and the XML in <out-dir>/inputs first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

from eigsep_data.paths import get_campaign_root

FILE_BASE = "USGS_OPR_UT_WestEast_B22_12STJ"
EAST, NORTH = range(90, 94), range(44, 49)
SHIFT_PX = 2000          # one 1 km tile at 0.5 m


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import eigsep_terrain
    from eigsep_terrain.dem import DEM

    campaign = Path(get_campaign_root(required=True))
    v2_dir = campaign / "derived" / "dem" / "v0002"
    v2_manifest = json.loads((v2_dir / "manifest.json").read_text())
    v2_file = v2_dir / "marjum_dem.npz"
    if sha256(v2_file) != v2_manifest["files"]["marjum_dem.npz"]["sha256"]:
        raise ValueError("DEM v0002 differs from its manifest")

    out = args.out_dir
    output = out / "marjum_dem.npz"
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    inputs = out / "inputs"
    names = np.array([[f"{FILE_BASE}{e}{n}.tif" for n in NORTH] for e in EAST])
    xml = inputs / f"{FILE_BASE}9145.xml"
    paths = [xml, *(inputs / name for name in names.flat)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("stage these inputs first:\n" + "\n".join(missing))
    for name, digest in v2_manifest["input_sha256"].items():
        if sha256(inputs / name) != digest:
            raise ValueError(f"{name} differs from the copy DEM v0002 used")
    input_hashes = {p.name: sha256(p) for p in paths}

    dem = DEM()
    dem.load_tif(np.vectorize(str)(np.array([[inputs / name for name in row] for row in names])),
                 survey_offset=(0, 0, 3))
    dem.load_xml(str(xml))
    with np.load(v2_file) as v2:
        origin2 = np.asarray(v2["raster_origin"], float)
        heights2 = v2["dem"]
        if not np.array_equal(dem.raster_origin + SHIFT_PX * float(dem.res), origin2):
            raise ValueError("mosaic is not offset from v0002 by exactly one tile west and south")
        overlap = dem.data[SHIFT_PX:, SHIFT_PX:]
        if overlap.shape != heights2.shape or not np.array_equal(overlap, heights2):
            raise ValueError("mosaic overlap differs from DEM v0002")
        map_crd = {k: v2[k] for k in ("eastbc", "westbc", "northbc", "southbc")}
    dem.raster_origin = origin2
    dem.e0_px = dem.n0_px = SHIFT_PX
    dem.map_crd = map_crd
    dem._cache_file = str(output)
    out.mkdir(parents=True, exist_ok=True)
    dem.save_cache()

    check = DEM(cache_file=str(output))
    probe = (1656.22, 2031.83)
    if abs(float(check.interp_alt(*probe)) - float(DEM(cache_file=str(v2_file)).interp_alt(*probe))) > 0:
        raise ValueError("v0003 and v0002 disagree at the antenna")

    package = Path(eigsep_terrain.__file__).resolve().parents[2]
    commit, dirty = git_state(package)
    gen_commit, gen_dirty = git_state(Path(__file__).resolve().parent)
    manifest = {
        "product": "dem", "version": "v0003", "published": date.today().isoformat(),
        "description": ("Float32 Marjum DEM, 4 east x 5 north 3DEP tiles, on the DEM v0002 "
                        "UTM raster grid (origin unchanged; 1 km more to the west and south)."),
        "files": {output.name: {"sha256": sha256(output), "bytes": output.stat().st_size}},
        "input_sha256": input_hashes,
        "input_order_east_north": names.tolist(),
        "v0002_sha256": sha256(v2_file),
        "validation": {"overlap_identical_to_v0002": True,
                       "shape_north_east": list(dem.data.shape),
                       "pixel_size_m": float(dem.res), "epsg": int(dem.crs.to_epsg()),
                       "raster_origin_utm_m": origin2.tolist(),
                       "e0_px": SHIFT_PX, "n0_px": SHIFT_PX,
                       "survey_offset_subtracted_m": dem.survey_offset.tolist()},
        "builder": {"script": "data-analysis/scripts/marjum-2026-07/build_dem_sw_mosaic.py",
                    "data_analysis_commit": gen_commit, "data_analysis_dirty": gen_dirty,
                    "eigsep_terrain_commit": commit, "eigsep_terrain_dirty": dirty},
        "bulk_sync": "eigsep-sync",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("built and validated", output)


if __name__ == "__main__":
    main()
