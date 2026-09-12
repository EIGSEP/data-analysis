"""Build the Marjum 2026-07 pointing table (v0) for the beam-scan window.

Product: one row per correlator sample, keyed on ``header/times`` (UTC), with
fused azimuth/elevation, height, per-sample uncertainties, per-sensor raw
values for audit, and a quality bitmask.

Run::

    /home/aparsons/.local/share/mamba/envs/arp/bin/python3 build_table.py \
        --data ~/projects/eigsep/marjum-2026-07/data --out pointing_table_v0

Writes ``<out>.npz`` (arrays + header) and ``<out>.csv`` (decimated preview).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

VERSION = "v0"
SCHEMA_VERSION = 1

# Beam-scan window. End is the last *data* sample on disk (03:08:49), not the
# 03:22 figure in CAMPAIGN.md, which is the close time of a buffered burst.
WINDOW_START = dt.datetime(2026, 7, 17, 18, 50, tzinfo=dt.timezone.utc)
WINDOW_STOP = dt.datetime(2026, 7, 18, 3, 22, 59, tzinfo=dt.timezone.utc)

# Nominal platform height after the 07-17 18:50 lift, from CAMPAIGN.md.
# geometer owns the absolute value; this is a placeholder the table declares.
NOMINAL_HEIGHT_M = 91.0
NOMINAL_HEIGHT_SIGMA_M = 1.5


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


extract = _load("extract", "extract.py")
fuse = _load("fuse", "fuse.py")


def git_describe(repo):
    """Short SHA of ``repo`` at generation time; ``-dirty`` if the tree is not clean.

    Per the marjum-2026-07 provenance rules, a dirty stamp is not citable in a
    result -- commit the generator before producing a product you intend to cite.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


def sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build(data_dir):
    t0, t1 = WINDOW_START.timestamp(), WINDOW_STOP.timestamp()
    d = extract.load_window(data_dir, t0, t1)
    n = d["time"].size

    az, sigma_az, flags_az, az_offset = fuse.fuse_azimuth(
        d["motor_az_pos"], d["potmon_pot_az_angle"])
    el, sigma_el, flags_el = fuse.fuse_elevation(
        d["imu_el_el_deg"], d["motor_el_pos"])

    flags = flags_az | flags_el
    no_meta = d["motor_status"] == "absent"
    flags[no_meta] |= fuse.FLAG_NO_METADATA

    # Unvalidated motor-only elevation: quote the measured motor-vs-IMU
    # disagreement where both existed, rather than a fit residual.
    both = np.isfinite(d["imu_el_el_deg"]) & np.isfinite(d["motor_el_pos"])
    resid = fuse.wrap180(fuse.MOTOR_DEG_PER_STEP * d["motor_el_pos"][both]
                         - d["imu_el_el_deg"][both])
    motor_el_sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    fallback = ~np.isfinite(sigma_el) & np.isfinite(el)
    sigma_el[fallback] = motor_el_sigma

    # Samples with no usable sensor at all. v0 leaves these NaN by design.
    flags[~np.isfinite(az)] |= fuse.FLAG_NO_ESTIMATE
    flags[~np.isfinite(el)] |= fuse.FLAG_NO_ESTIMATE

    # Height: LIDAR ground returns where geometry allows, nominal elsewhere.
    lidar_h = fuse.lidar_height(d["lidar_distance_m"], d["imu_el_el_deg"])
    height = np.full(n, NOMINAL_HEIGHT_M)
    sigma_h = np.full(n, NOMINAL_HEIGHT_SIGMA_M)
    measured = np.isfinite(lidar_h)
    height[measured] = lidar_h[measured]
    # Spread of the ground returns is the honest per-sample LIDAR error.
    if measured.sum() > 10:
        sigma_h[measured] = float(1.4826 * np.median(
            np.abs(lidar_h[measured] - np.median(lidar_h[measured]))))
    flags[~measured] |= fuse.FLAG_HEIGHT_ASSUMED

    table = {
        "time_utc": d["time"],
        "az_deg": az,
        "el_deg": el,
        "height_m": height,
        "sigma_az_deg": sigma_az,
        "sigma_el_deg": sigma_el,
        "sigma_height_m": sigma_h,
        "flags": flags,
        # Raw streams retained for audit / re-fusion by consumers.
        "motor_az_deg": fuse.MOTOR_DEG_PER_STEP * d["motor_az_pos"],
        "motor_el_deg": fuse.wrap180(fuse.MOTOR_DEG_PER_STEP * d["motor_el_pos"]),
        "pot_az_deg": d["potmon_pot_az_angle"],
        "imu_el_deg": d["imu_el_el_deg"],
        "lidar_dist_m": d["lidar_distance_m"],
        "az_motor_pot_offset_deg": az_offset,
        "file_index": d["file_index"],
        "file_close_time": d["file_close_time"],
    }
    return table, d["_files"], motor_el_sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="pointing_table_v0")
    args = ap.parse_args()

    table, files, motor_el_sigma = build(os.path.expanduser(args.data))
    n = table["time_utc"].size

    commit = git_describe(os.path.join(HERE, "..", "..", "..", ".."))
    generated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    provenance = {
        "product": "pointing_table",
        "campaign": "marjum-2026-07",
        "version": VERSION,
        "generated_utc": generated.replace("+00:00", "Z"),
        "generator": "eigsep_data/notebooks/arp/marjum-2026-07/pointing/build_table.py",
        "generator_commit": commit,
        "generator_repo": "eigsep_data",
        "inputs": [{"path": f"marjum-2026-07/data/{os.path.basename(f)}",
                    "sha256": sha256(f)} for f in files],
        "params": {
            "window_start_utc": WINDOW_START.isoformat().replace("+00:00", "Z"),
            "window_stop_utc": WINDOW_STOP.isoformat().replace("+00:00", "Z"),
            "az_half_window_samples": 56,
            "az_slip_jump_deg": 1.0,
            "el_stuck_imu_ptp_deg": 3.0,
            "el_stuck_motor_ptp_deg": 30.0,
            "lidar_valid_m": [fuse.LIDAR_MIN_VALID, fuse.LIDAR_MAX_VALID],
            "nominal_height_m": NOMINAL_HEIGHT_M,
        },
    }
    compact = f"marjum-2026-07/pointing_table@{VERSION}+{commit}"

    header = {
        "product": "marjum-2026-07 pointing table",
        "provenance": provenance,
        "provenance_compact": compact,
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "generated_by": "pointing-analyst (B9)",
        "code_commit": commit,
        "window_start_utc": extract.utc(table["time_utc"][0]),
        "window_stop_utc": extract.utc(table["time_utc"][-1]),
        "n_samples": int(n),
        "n_files": len(files),
        "cadence_s": float(np.median(np.diff(table["time_utc"]))),
        "time_key": ("header/times, UTC epoch seconds. Correlator FILENAMES are "
                     "file CLOSE times (median fname - times[-1] = -0.94 s); "
                     "do not use them as sample times."),
        "frame": ("az: degrees, 0-360, topocentric convention inherited from the "
                  "potentiometer calibration; absolute zero point NOT yet tied to "
                  "true north -- awaiting geometer. el: degrees, -180..180, "
                  "gravity-referenced via imu_elevation_deg convention "
                  "(R = R_el @ R_az maps receiver -> topocentric)."),
        "height_note": (f"Nominal {NOMINAL_HEIGHT_M} m from CAMPAIGN.md 07-17 18:50 "
                        "lift; LIDAR ground returns override where available. "
                        "geometer owns the absolute height."),
        "sensor_sigmas_deg": {
            "pot_az": fuse.POT_AZ_SIGMA,
            "imu_el": fuse.IMU_EL_SIGMA,
            "motor_el_unvalidated": round(motor_el_sigma, 3),
            "motor_quantisation": round(fuse.MOTOR_QUANT_SIGMA, 5),
            "az_floor_sway": round(fuse.AZ_FLOOR_SIGMA, 3),
        },
        "flag_bits": {int(k): v for k, v in fuse.FLAG_NAMES.items()},
        "extrapolation_policy": fuse.EXTRAPOLATION_POLICY,
        "caveats": [
            "imu_az returns status='error' for 100% of samples in this window; "
            "azimuth rests entirely on the potentiometer.",
            "Azimuth zero point is uncalibrated against true north.",
            "Motor counts are relative and slip; never use them as absolute angles.",
            "The commanded 5 deg scan grid is NOT the achieved grid: measured "
            "steps are 4.435 deg median against 5.0018 deg commanded, because "
            "azimuth slips ~34 deg/hr during the scan. Do not assume the plan.",
            "EL_POST_FAILURE marks everything from the first proven EL stall; "
            "elevation there is parked, not scanned.",
        ],
        "files": [os.path.basename(f) for f in files],
    }

    out = args.out
    np.savez_compressed(out + ".npz", header=json.dumps(header, indent=2), **table)

    # Decimated CSV preview (every 20th sample) for eyeballing.
    step = 20
    cols = ["time_utc", "az_deg", "el_deg", "height_m",
            "sigma_az_deg", "sigma_el_deg", "flags"]
    with open(out + ".csv", "w") as fh:
        fh.write("# " + compact + "\n")
        fh.write("# " + json.dumps({k: header[k] for k in
                 ("product", "version", "schema_version", "generated_utc",
                  "window_start_utc", "window_stop_utc", "n_samples")}) + "\n")
        fh.write("utc_iso," + ",".join(cols) + "\n")
        for i in range(0, n, step):
            vals = [extract.utc(table["time_utc"][i])]
            for c in cols:
                v = table[c][i]
                vals.append(f"{v:.0f}" if c == "flags" else f"{v:.4f}")
            fh.write(",".join(vals) + "\n")

    print(f"wrote {out}.npz  ({n} samples, {len(files)} files)")
    print(f"wrote {out}.csv  (every {step}th sample)")
    print(f"provenance: {compact}")
    if commit.endswith("-dirty"):
        print("  WARNING: generator tree is dirty -- this stamp is not citable.")
    return table, header


if __name__ == "__main__":
    main()
