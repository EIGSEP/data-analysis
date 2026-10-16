"""Check that terrain beyond a DEM mosaic cannot raise a horizon profile.

A horizon profile (``build_horizon_profiles.py`` format) is traced on a 0.5 m
mosaic out to its edge. This script asks, bearing by bearing, whether terrain
beyond that edge could lie above the traced skyline. It samples a coarse
regional DEM (USGS 3DEP 1 arc-second GeoTIFFs, NAD83 geographic) along the
same straight grid rays, from the mosaic edge out to the distance beyond which
no terrain lower than ``--z-bound`` can be seen above the skyline:

    r_cap(beta) = smallest r with  U + r tan(alpha) + r^2 / 2R  >=  z_bound + dz,

and requires every coarse sample, raised by ``dz``, to stay below the skyline:

    alpha_c(beta) = max_r atan2(z_coarse + dz - U - r^2 / 2R, r)  <  alpha(beta).

``dz`` covers the coarse DEM smoothing ridges and any datum offset. By default
it is calibrated on the mosaic itself: the largest uplift the coarse terrain
needs, on any bearing at 1 km or more, to reach the fine skyline line, plus
``DZ_MARGIN_M``. The per-pixel (fine - coarse) spread is reported too; its
maximum comes from cliff pixels, not skylines. ``z_bound`` is an assumed upper bound on terrain height
within ``r_cap``; the loaded coarse tiles must reach ``r_cap`` on every bearing
and their own maximum must be below ``z_bound``, or the script stops.

A bearing passes when alpha_c < alpha. The horizon the coarse rays predict,
max(alpha, alpha_c), is also written, so a failing bearing shows how much it
would rise. Diagnostic: the coarse DEM can show a fine mosaic is too small;
the replacement horizon still comes from the fine mosaic.

Geometry follows build_horizon_profiles.py: bearings atan2(dN, dE) on the
UTM grid (EPSG:6341 minus the DEM's raster origin), R = 6371 km, no
refraction, the march limited by the last interpolable fine sample.

Usage (horizon v0004 on DEM v0004):
    python coverage_certificate.py v0004 --dem-version v0004 --fetch
reads <campaign>/derived/dem/v0004/marjum_dem.npz and
<campaign>/curation/horizon_profiles_v0004.npz, stages the six 1 arc-second
tiles in <campaign>/derived/dem/v0004/inputs_1arcsec/ (downloaded with
--fetch) and writes <campaign>/curation/horizon_profiles_v0004_coverage.{json,npz}.
Exit status 1 if any bearing can rise.

Needs numpy, Pillow and pyproj.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

from eigsep_data.paths import get_campaign_root

EARTH_R_M = 6371e3
ERAS = ("30m", "87.5m", "91m")
STEP_M = 10.0            # coarse march step, a third of a 1 arc-second pixel
CORE_PX = 3600           # 1 arc-second tile without its 6-pixel buffer
BUFFER_PX = 6
COARSE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/"
COARSE_TILES = ("n39w113", "n39w114", "n39w115", "n40w113", "n40w114", "n40w115")
DZ_MARGIN_M = 5.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo):
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                    text=True, check=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain",
                                              "--untracked-files=no"))


def fetch_coarse(tiles_dir):
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for t in COARSE_TILES:
        path = tiles_dir / f"USGS_1_{t}.tif"
        if path.exists():
            continue
        print("download", path.name, flush=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(f"{COARSE_URL}{t}/{path.name}", timeout=120) as r, \
                open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)


def rel(path, campaign="marjum-2026-07"):
    """Path from the campaign directory on, as build_horizon_profiles.py records it."""
    parts = Path(path).resolve().parts
    return str(Path(*parts[parts.index(campaign):])) if campaign in parts else str(path)


def npy_shape(npz, name):
    """Shape of one array in an .npz without reading it."""
    with zipfile.ZipFile(npz) as z, z.open(f"{name}.npy") as f:
        major, _ = np.lib.format.read_magic(f)
        read = (np.lib.format.read_array_header_1_0 if major == 1
                else np.lib.format.read_array_header_2_0)
        shape, _, _ = read(f)
    return shape


class Mosaic:
    """Extent and edge range of a DEM cache, as build_horizon_profiles.Grid."""

    def __init__(self, path):
        with np.load(path) as z:
            self.res = float(z["res"])
            self.e0, self.n0 = int(z["e0_px"]), int(z["n0_px"])
            self.origin = np.asarray(z["raster_origin"], float)
            self.epsg = int(z["crs_epsg"])
        ny, nx = npy_shape(path, "dem")
        self.extent_e = ((0 - self.e0) * self.res, (nx - 1 - self.e0) * self.res)
        self.extent_n = ((0 - self.n0) * self.res, (ny - 1 - self.n0) * self.res)

    def edge_range(self, e, n, ux, uy):
        lim = []
        for u, p, (lo, hi) in ((ux, e, self.extent_e), (uy, n, self.extent_n)):
            hi = hi - self.res
            if u > 1e-12:
                lim.append((hi - p) / u)
            elif u < -1e-12:
                lim.append((lo - p) / u)
        return min(lim)

    def edge_side(self, e, n, ux, uy):
        """Which mosaic side ('E', 'W', 'N', 'S') a ray leaves through."""
        lim = {}
        for u, p, (lo, hi), (pos, neg) in ((ux, e, self.extent_e, "EW"),
                                           (uy, n, self.extent_n, "NS")):
            hi = hi - self.res
            if u > 1e-12:
                lim[pos] = (hi - p) / u
            elif u < -1e-12:
                lim[neg] = (lo - p) / u
        return min(lim, key=lim.get)


class Coarse:
    """A rectangular mosaic of 1 arc-second 3DEP tiles, bilinear in lon/lat."""

    def __init__(self, paths):
        tiles = {}
        for path in paths:
            with Image.open(path) as im:
                t = im.tag_v2
                sx, sy, _ = t[33550]
                keys = t[34735]
                geo = {keys[k]: tuple(keys[k + 1:k + 4]) for k in range(4, len(keys), 4)}
                if geo.get(2048, (0, 1, 0))[2] != 4269 or geo.get(1025, (0, 1, 1))[2] != 1:
                    raise ValueError(f"{path}: expected NAD83 geographic, pixel-is-area")
                if im.size != (CORE_PX + 2 * BUFFER_PX,) * 2:
                    raise ValueError(f"{path}: unexpected tile size")
                a = np.array(im, dtype=np.float32)
                nodata = float(t.get(42113, -999999))
                if np.any(a == nodata) or not np.isfinite(a).all():
                    raise ValueError(f"{path}: nodata samples")
                x0, y0 = t[33922][3], t[33922][4]
                west, north = round(x0 + BUFFER_PX * sx), round(y0 - BUFFER_PX * sy)
                tiles[west, north] = a[BUFFER_PX:-BUFFER_PX, BUFFER_PX:-BUFFER_PX]
        wests = sorted({w for w, _ in tiles})
        norths = sorted({n for _, n in tiles}, reverse=True)
        if len(tiles) != len(wests) * len(norths) or np.any(np.diff(wests) != 1) \
                or np.any(np.diff(norths) != -1):
            raise ValueError("coarse tiles must form a full rectangle")
        self.data = np.vstack([np.hstack([tiles[w, n] for w in wests]) for n in norths])
        self.step = 1.0 / CORE_PX
        self.lon0 = wests[0] + 0.5 * self.step          # centre of column 0
        self.lat0 = norths[0] - 0.5 * self.step         # centre of row 0 (north)
        self.lon_range = (wests[0], wests[-1] + 1)
        self.lat_range = (norths[-1] - 1, norths[0])
        self.zmax = float(self.data.max())

    def inside(self, lon, lat):
        x = (lon - self.lon0) / self.step
        y = (self.lat0 - lat) / self.step
        ny, nx = self.data.shape
        return (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)

    def interp(self, lon, lat):
        x = (lon - self.lon0) / self.step
        y = (self.lat0 - lat) / self.step
        x0 = np.clip(np.floor(x).astype(int), 0, self.data.shape[1] - 2)
        y0 = np.clip(np.floor(y).astype(int), 0, self.data.shape[0] - 2)
        fx, fy = x - x0, y - y0
        d = self.data
        return ((1 - fx) * (1 - fy) * d[y0, x0] + fx * (1 - fy) * d[y0, x0 + 1]
                + (1 - fx) * fy * d[y0 + 1, x0] + fx * fy * d[y0 + 1, x0 + 1])


def calibrate_dz(dem_path, mosaic, coarse, to_geo, stride=4):
    """Fine - coarse at every ``stride``-th fine pixel of the mosaic."""
    with np.load(dem_path) as z:
        dem = z["dem"]
    rows = np.arange(0, dem.shape[0], stride)
    cols = np.arange(0, dem.shape[1], stride)
    diffs = []
    for r in np.array_split(rows, 20):
        e = (cols[None, :] - mosaic.e0) * mosaic.res + 0 * r[:, None]
        n = (r[:, None] - mosaic.n0) * mosaic.res + 0 * cols[None, :]
        lon, lat = to_geo.transform(mosaic.origin[0] + e.ravel(), mosaic.origin[1] + n.ravel())
        diffs.append(dem[np.ix_(r, cols)].ravel() - coarse.interp(lon, lat))
    d = np.concatenate(diffs)
    return {"stride_px": stride, "samples": int(d.size),
            "mean_m": float(d.mean()), "std_m": float(d.std()),
            "p0.1_m": float(np.percentile(d, 0.1)), "p99.9_m": float(np.percentile(d, 99.9)),
            "max_m": float(d.max()), "min_m": float(d.min())}


def calibrate_skyline_dz(mosaic, coarse, to_geo, bearings, obs, r_min=1000.0):
    """How far the coarse DEM falls below the fine skyline, inside the mosaic.

    For each observer and bearing, the uplift the coarse terrain needs on
    [r_min, edge] to reach the fine horizon line:
        dz_j = -min_r (U + r tan(alpha) + r^2 / 2R - z_coarse(r)).
    Positive means the coarse DEM would have missed part of the real skyline
    by that height. Only bearings whose skyline line is reached within the
    mosaic count (dz_j > -inf); r_min keeps to distances like those beyond
    the edge.
    """
    beta = np.radians(bearings)
    vals = []
    for _, e, n, u, alpha in obs:
        for j, b in enumerate(beta):
            ux, uy = np.cos(b), np.sin(b)
            r_edge = mosaic.edge_range(e, n, ux, uy)
            r = np.arange(r_min, r_edge, STEP_M)
            lon, lat = to_geo.transform(mosaic.origin[0] + e + r * ux,
                                        mosaic.origin[1] + n + r * uy)
            gap = u + r * np.tan(alpha[j]) + r ** 2 / (2 * EARTH_R_M) - coarse.interp(lon, lat)
            vals.append(-gap.min())
    v = np.asarray(vals)
    return {"r_min_m": r_min, "observer_bearings": int(v.size),
            "max_m": float(v.max()), "p99.9_m": float(np.percentile(v, 99.9)),
            "p99_m": float(np.percentile(v, 99)), "median_m": float(np.median(v)),
            "positive_fraction": float(np.mean(v > 0))}


def observers(h, displacement_m):
    """(label, E, N, U, alpha[bearing]) for the three eras and the 8 displaced 91 m points."""
    e, n, _ = h["antenna_enu_m"]
    out = [(era, e, n, float(h[f"u_m_{era}"]), h[f"elev_rad_{era}"]) for era in ERAS]
    pert = h["horizontal_perturbation_elev_rad"]
    for k, t in enumerate(np.radians(np.arange(len(pert)) * 360.0 / len(pert))):
        out.append((f"91m_disp{k}", e + displacement_m * np.cos(t),
                    n + displacement_m * np.sin(t), float(h["u_m_91m"]), pert[k]))
    return out


def certify(mosaic, coarse, to_geo, bearings, obs, dz, z_bound):
    """alpha_c, r_cap and the beyond-edge argmax for every observer and bearing."""
    beta = np.radians(bearings)
    res = {}
    for label, e, n, u, alpha in obs:
        a_c = np.full(len(beta), -np.inf)
        r_at = np.full(len(beta), np.nan)
        r_edge = np.empty(len(beta))
        r_cap = np.empty(len(beta))
        side = np.empty(len(beta), "<U1")
        for j, b in enumerate(beta):
            ux, uy = np.cos(b), np.sin(b)
            r_edge[j] = mosaic.edge_range(e, n, ux, uy)
            side[j] = mosaic.edge_side(e, n, ux, uy)
            # smallest r beyond which terrain <= z_bound stays below alpha
            a2, b1, c0 = 1 / (2 * EARTH_R_M), np.tan(alpha[j]), u - z_bound - dz
            r_cap[j] = (-b1 + np.sqrt(b1 * b1 - 4 * a2 * c0)) / (2 * a2)
            if r_cap[j] <= r_edge[j]:
                continue
            r = np.arange(r_edge[j], r_cap[j] + STEP_M, STEP_M)
            lon, lat = to_geo.transform(mosaic.origin[0] + e + r * ux,
                                        mosaic.origin[1] + n + r * uy)
            if not coarse.inside(lon, lat).all():
                raise SystemExit(f"{label} bearing {bearings[j]}: coarse tiles end before "
                                 f"r_cap = {r_cap[j]:.0f} m; load more tiles")
            z = coarse.interp(lon, lat) + dz - r ** 2 / (2 * EARTH_R_M)
            a = np.arctan2(z - u, r)
            k = int(np.argmax(a))
            a_c[j], r_at[j] = a[k], r[k]
        res[label] = {"alpha": alpha, "alpha_c": a_c, "r_at": r_at, "r_edge": r_edge,
                      "r_cap": r_cap, "side": side}
    return res


def summarize(res):
    out = {}
    for label, d in res.items():
        margin = np.degrees(d["alpha"] - d["alpha_c"])
        fail = margin <= 0
        sides = {}
        for s in "ENWS":
            m = d["side"] == s
            if not m.any():
                continue
            rise = np.degrees(np.maximum(d["alpha_c"][m] - d["alpha"][m], 0))
            sides[s] = {"bearings": int(m.sum()), "fail": int(fail[m].sum()),
                        "min_margin_deg": round(float(margin[m].min()), 4),
                        "max_predicted_rise_deg": round(float(rise.max()), 4),
                        "r_cap_max_m": round(float(d["r_cap"][m].max()), 1)}
        out[label] = {"fail": int(fail.sum()), "min_margin_deg": round(float(margin.min()), 4),
                      "sides": sides}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="horizon_profiles version, e.g. v0004")
    ap.add_argument("--dem-version", required=True, help="DEM version the horizon was traced on")
    ap.add_argument("--dem", type=Path, default=None,
                    help="default: <campaign>/derived/dem/<dem-version>/marjum_dem.npz")
    ap.add_argument("--horizon", type=Path, default=None,
                    help="default: <campaign>/curation/horizon_profiles_<version>.npz")
    ap.add_argument("--coarse-dir", type=Path, default=None,
                    help="1 arc-second tiles; default <dem dir>/inputs_1arcsec")
    ap.add_argument("--fetch", action="store_true", help="download missing 1 arc-second tiles")
    ap.add_argument("--out", type=Path, default=None,
                    help="report JSON (an .npz of per-bearing arrays goes beside it); "
                         "default <campaign>/curation/horizon_profiles_<version>_coverage.json")
    ap.add_argument("--dz", type=float, default=None,
                    help="height margin (m); default: calibrated skyline uplift + DZ_MARGIN_M")
    ap.add_argument("--calibrate", action="store_true", help="report the calibrations even when --dz is given")
    ap.add_argument("--z-bound", type=float, default=4200.0,
                    help="assumed maximum terrain height within r_cap (m)")
    ap.add_argument("--displacement-m", type=float, default=None,
                    help="sensitivity-block displacement (default: the horizon JSON's, else 1.7)")
    args = ap.parse_args()
    campaign = Path(get_campaign_root(required=True))
    dem_dir = campaign / "derived" / "dem" / args.dem_version
    args.dem = args.dem or dem_dir / "marjum_dem.npz"
    args.horizon = args.horizon or campaign / "curation" / f"horizon_profiles_{args.version}.npz"
    args.coarse_dir = args.coarse_dir or dem_dir / "inputs_1arcsec"
    args.out = args.out or campaign / "curation" / f"horizon_profiles_{args.version}_coverage.json"
    if args.fetch:
        fetch_coarse(args.coarse_dir)
    args.coarse = [args.coarse_dir / f"USGS_1_{t}.tif" for t in COARSE_TILES]

    mosaic = Mosaic(args.dem)
    coarse = Coarse(args.coarse)
    if coarse.zmax >= args.z_bound:
        raise SystemExit(f"coarse tiles reach {coarse.zmax:.0f} m >= z_bound")
    to_geo = Transformer.from_crs(mosaic.epsg, 4269, always_xy=True)
    commit, dirty = git_state(Path(__file__).resolve().parent)
    report = {"product": "horizon_profiles coverage certificate", "version": args.version,
              "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "generator": "data-analysis/scripts/marjum-2026-07/coverage_certificate.py",
              "generator_commit": commit, "generator_dirty": dirty,
              "coarse_url_base": COARSE_URL,
              "inputs": {"dem": {"path": rel(args.dem), "sha256": sha256(args.dem)},
                         "horizon": {"path": rel(args.horizon), "sha256": sha256(args.horizon)},
                         "coarse": {p.name: sha256(p) for p in args.coarse}},
              "assumptions": [
                  "No terrain above z_bound_m within r_cap on any bearing.",
                  "Same geometry as build_horizon_profiles.py: straight rays on the UTM grid, "
                  "R = 6371 km, no refraction.",
                  "dz_m absorbs the coarse DEM's smoothing of ridges and the NAD83 / "
                  "NAD83(2011) and vertical datum offsets."],
              "mosaic_extent_m": {"e": mosaic.extent_e, "n": mosaic.extent_n},
              "coarse_box_deg": {"lon": coarse.lon_range, "lat": coarse.lat_range},
              "coarse_zmax_m": coarse.zmax, "z_bound_m": args.z_bound,
              "step_m": STEP_M, "earth_r_m": EARTH_R_M}
    hjson = args.horizon.with_suffix(".json")
    disp = args.displacement_m
    if disp is None:
        disp = 1.7
        if hjson.exists():
            disp = json.loads(hjson.read_text())["horizontal_displacement_sensitivity"][
                "displacement_m"]
    report["displacement_m"] = disp
    with np.load(args.horizon) as z:
        h = {k: z[k] for k in z.files}
    obs = observers(h, disp)

    if args.calibrate or args.dz is None:
        report["fine_minus_coarse_pixels"] = calibrate_dz(args.dem, mosaic, coarse, to_geo)
        print("fine - coarse, pixels:", report["fine_minus_coarse_pixels"], flush=True)
        report["skyline_uplift"] = calibrate_skyline_dz(mosaic, coarse, to_geo,
                                                        h["bearings_deg"], obs)
        print("coarse uplift to reach the fine skyline:", report["skyline_uplift"], flush=True)
    dz = args.dz if args.dz is not None else report["skyline_uplift"]["max_m"] + DZ_MARGIN_M
    report["dz_m"] = dz
    print(f"dz = {dz:.2f} m", flush=True)

    res = certify(mosaic, coarse, to_geo, h["bearings_deg"], obs, dz, args.z_bound)
    report["summary"] = summarize(res)
    report["pass"] = all(s["fail"] == 0 for s in report["summary"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"bearings_deg": h["bearings_deg"]}
    for label, d in res.items():
        for k, v in d.items():
            arrays[f"{k}_{label}"] = v
    np.savez(args.out.with_suffix(".npz"), **arrays)
    report["data_file"] = args.out.with_suffix(".npz").name
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for label, s in report["summary"].items():
        print(f"{label}: fail {s['fail']}, min margin {s['min_margin_deg']} deg;",
              {k: (v["fail"], v["min_margin_deg"], v["max_predicted_rise_deg"])
               for k, v in s["sides"].items()})
    total = sum(s["fail"] for s in report["summary"].values())
    print("PASS" if total == 0 else f"FAIL: {total} observer-bearings can rise")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
