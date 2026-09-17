"""Build the Marjum 2026-07 pointing table.

Product: one row per correlator sample with fused azimuth/elevation, height,
per-sample uncertainties, the raw per-sensor values for audit, and quality
flags.  Written as Parquet with the provenance block in the file's key-value
metadata, per the registry in ``marjum-2026-07/INDEX.md``.

Version history
---------------
v0  beam-scan window only (07-17 18:50 -> 07-18 03:22), npz + csv.
v1  **meaning changes** -- whole campaign; join key is ``t_utc`` in int64 ns;
    adds ``hdr_time_bad`` and ``t_utc_sigma_s`` because 646 files (12.6%)
    have untrustworthy header clocks and their timestamps are filename
    anchors good to ~10 min, not to a second; adds ``quality``,
    ``height_era`` and ``phase``; height now varies by era instead of being
    a single nominal.
v1.1 adds flag bit 9 ``AZ_SLIP_RAMP``.  ``AZ_SLIP_EVENT`` detects *steps* in
    the motor-vs-pot offset and is blind to *ramps*, so it flagged neither of
    the two events that actually moved the 07-17 scan off its commanded grid.
    Purely additive: every other column and flag bit is unchanged, so a v1
    consumer reading columns by name is unaffected.
v1.2 adds flag bit 10 ``EL_SOLUTION_GLITCH`` (beam-analyst finding, via
    ``beam_metric_outliers_checkpoint.ipynb``): in the post-EL-failure wrap
    cluster the antenna is parked at el ~ +/-180 and the IMU elevation
    solver intermittently emits one spurious sample at |el| ~ 0 or ~59-60
    before returning to the park -- all previously passed as
    ``quality == "ok"`` with no flag set.  Criterion is an adjacent in-file
    elevation slew > 20 deg/s (~4x the ~5 deg/s commanded scan rate); see
    ``fuse.detect_el_solution_glitch``.  The wrap-park *mechanism* is
    established only for the post-EL-failure era -- campaign-wide, pre-
    failure firings of the same criterion are a different, less-understood
    population (most already ``quality == "suspect"`` for other reasons), so
    the bit is defined by its criterion, not the mechanism.  Purely additive.

Run::

    /home/aparsons/.local/share/mamba/envs/arp/bin/python3 build_table.py \
        --data ~/projects/eigsep/marjum-2026-07/data \
        --out ~/projects/eigsep/marjum-2026-07/curation/pointing_table
    # or one era at a time:
    ... --era '~30m/C'
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

VERSION = "v1.2"
SCHEMA_VERSION = 4

# Nominal platform height per curation height_era, and its uncertainty.
# geometer owns the absolute values; these are the campaign-note figures the
# table declares so a consumer can see exactly what was assumed.
ERA_HEIGHT_M = {
    "~2m": (2.0, 1.0),
    "~30m": (30.0, 3.0),
    "~87.5m": (87.5, 2.0),
    "~91m": (91.0, 1.5),
}

QUALITY_OK, QUALITY_SUSPECT, QUALITY_GAP = "ok", "suspect", "gap"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


extract = _load("extract", "extract.py")
fuse = _load("fuse", "fuse.py")


def git_describe(repo, scope=None):
    """Short SHA at generation time; ``-dirty`` if the generator is not clean.

    Per the campaign provenance rules a dirty stamp is not citable.  The
    dirtiness test is scoped to ``scope`` (the generator's own directory)
    rather than the whole repository, because this is a **shared** repo:
    other agents have unrelated work in flight, their uncommitted files
    cannot affect this generator -- it imports only its own ``extract.py``
    and ``fuse.py`` -- and committing them to clean the tree would sweep in
    someone else's work.  Dirt elsewhere is reported separately in the
    provenance block rather than silently dropped, so the weaker test is
    visible to anyone auditing the stamp.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown", []
    def porcelain(*paths):
        try:
            return subprocess.check_output(
                ["git", "-C", repo, "status", "--porcelain", "--"] + list(paths),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    scoped = porcelain(scope) if scope else porcelain()
    whole = porcelain()
    outside = [ln.strip() for ln in whole.splitlines()
               if ln.strip() and ln.strip() not in scoped.splitlines()]
    return (f"{sha}-dirty" if scoped else sha), outside


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_eras(campaign_dir):
    """Era/phase intervals from data-archivist's mode table."""
    path = os.path.join(campaign_dir, "curation", "mode_table.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append((
            dt.datetime.strptime(r["t_start_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc).timestamp(),
            dt.datetime.strptime(r["t_end_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc).timestamp(),
            r.get("height_era"), r.get("phase")))
    return out


def label_eras(times, eras):
    """Per-sample height_era / phase, from the mode table's intervals."""
    n = times.size
    era = np.array([""] * n, dtype=object)
    phase = np.array([""] * n, dtype=object)
    for t0, t1, h, p in eras:
        sel = (times >= t0) & (times <= t1)
        if sel.any():
            era[sel] = h or ""
            phase[sel] = p or ""
    return era, phase


def build(data_dir, t_start, t_stop, campaign_dir, index=None):
    d = extract.load_window(data_dir, t_start, t_stop, index=index)
    n = d["time"].size

    az, sigma_az, flags_az, az_offset = fuse.fuse_azimuth(
        d["motor_az_pos"], d["potmon_pot_az_angle"])
    el, sigma_el, flags_el = fuse.fuse_elevation(
        d["imu_el_el_deg"], d["motor_el_pos"])

    flags = flags_az | flags_el
    flags[d["motor_status"] == "absent"] |= fuse.FLAG_NO_METADATA
    flags[fuse.detect_az_slip_ramp(az, d["motor_az_pos"], d["time"])] |= \
        fuse.FLAG_AZ_SLIP_RAMP
    flags[fuse.detect_el_solution_glitch(el, d["time"], d["file_index"])] |= \
        fuse.FLAG_EL_SOLUTION_GLITCH

    # Unvalidated motor-only elevation: quote the measured motor-vs-IMU
    # disagreement where both existed, not a fit residual.
    both = np.isfinite(d["imu_el_el_deg"]) & np.isfinite(d["motor_el_pos"])
    if both.sum() > 100:
        resid = fuse.wrap180(fuse.MOTOR_DEG_PER_STEP * d["motor_el_pos"][both]
                             - d["imu_el_el_deg"][both])
        motor_el_sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    else:
        motor_el_sigma = float("nan")
    sigma_el[~np.isfinite(sigma_el) & np.isfinite(el)] = motor_el_sigma

    flags[~np.isfinite(az)] |= fuse.FLAG_NO_ESTIMATE
    flags[~np.isfinite(el)] |= fuse.FLAG_NO_ESTIMATE

    eras = load_eras(campaign_dir)
    era, phase = label_eras(d["time"], eras)

    # Height: era nominal, overridden by LIDAR ground returns where geometry
    # allows the rangefinder to actually see the ground.
    height = np.full(n, np.nan)
    sigma_h = np.full(n, np.nan)
    for key, (h, s) in ERA_HEIGHT_M.items():
        sel = era == key
        height[sel], sigma_h[sel] = h, s
    lidar_h = fuse.lidar_height(d["lidar_distance_m"], d["imu_el_el_deg"])
    measured = np.isfinite(lidar_h)
    height[measured] = lidar_h[measured]
    if measured.sum() > 10:
        sigma_h[measured] = float(1.4826 * np.median(
            np.abs(lidar_h[measured] - np.median(lidar_h[measured]))))
    flags[~measured] |= fuse.FLAG_HEIGHT_ASSUMED

    hdr_bad = d["hdr_time_bad"].astype(bool)
    t_sigma = np.where(hdr_bad, extract.FNAME_ANCHOR_SIGMA_S, 0.0)

    # quality describes confidence in the reported az/el VALUES, and nothing
    # else.  Drive state is a separate axis and lives in `flags`.
    #
    # This distinction matters: when the EL drive stalls and is then left
    # uncommanded, the antenna is genuinely parked and the gravity-referenced
    # IMU measures where it points perfectly well.  The pointing is sound; it
    # simply is not *scanning*.  Folding EL_STUCK / EL_POST_FAILURE into
    # `quality` conflated "we don't know where it pointed" with "it wasn't
    # doing anything interesting", and discarded hours of good pointing.
    # A consumer selecting *scanning* data filters the flags; a consumer
    # asking "where was it pointing at time t" uses `quality`.
    #
    # 'interp' is never emitted -- this product does not interpolate; absent
    # pointing is a gap, not a smoothed value.
    quality = np.full(n, QUALITY_OK, dtype=object)
    value_suspect = (
        hdr_bad                                        # t_utc only ~+/-10 min
        | ((flags & (fuse.FLAG_EL_NO_IMU               # el from motor alone
                     | fuse.FLAG_AZ_NO_POT             # az has no absolute ref
                     | fuse.FLAG_UNCOMMANDED)) != 0))  # swinging within a sample
    quality[value_suspect] = QUALITY_SUSPECT
    quality[(flags & fuse.FLAG_NO_ESTIMATE) != 0] = QUALITY_GAP

    table = {
        "t_utc": (d["time"] * 1e9).astype("int64"),
        "t_utc_s": d["time"],
        "t_utc_sigma_s": t_sigma,
        "file": d["file_name"],
        "sample_idx": d["sample_idx"].astype("int32"),
        "hdr_time_bad": hdr_bad,
        "az_deg": az,
        "el_deg": el,
        "az_sigma_deg": sigma_az,
        "el_sigma_deg": sigma_el,
        "height_m": height,
        "height_sigma_m": sigma_h,
        "quality": quality,
        "flags": flags.astype("int32"),
        "height_era": era,
        "phase": phase,
        # Raw per-sensor values, retained so a consumer can re-fuse.
        "motor_az_deg": fuse.MOTOR_DEG_PER_STEP * d["motor_az_pos"],
        "motor_el_deg": fuse.wrap180(fuse.MOTOR_DEG_PER_STEP * d["motor_el_pos"]),
        "pot_az_deg": d["potmon_pot_az_angle"],
        "imu_el_deg": d["imu_el_el_deg"],
        "imu_az_yaw_deg": d["imu_az_yaw"],
        "lidar_dist_m": d["lidar_distance_m"],
        "az_motor_pot_offset_deg": az_offset,
        "file_close_time_s": d["file_close_time"],
    }
    return table, d["_files"], motor_el_sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--era", default=None,
                    help="restrict to one height_era, e.g. '~30m'")
    ap.add_argument("--phase", default=None, help="restrict to one phase")
    ap.add_argument("--start", default=None, help="UTC ISO start override")
    ap.add_argument("--stop", default=None, help="UTC ISO stop override")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = os.path.expanduser(args.data)
    campaign_dir = os.path.dirname(data_dir.rstrip("/"))

    eras = load_eras(campaign_dir)
    if args.start or args.stop:
        t0 = dt.datetime.fromisoformat(args.start).replace(
            tzinfo=dt.timezone.utc).timestamp() if args.start else 0.0
        t1 = dt.datetime.fromisoformat(args.stop).replace(
            tzinfo=dt.timezone.utc).timestamp() if args.stop else 2e9
    elif args.era:
        sel = [e for e in eras if e[2] == args.era
               and (args.phase is None or e[3] == args.phase)]
        if not sel:
            raise SystemExit(f"no mode-table blocks for era {args.era!r}")
        t0, t1 = min(e[0] for e in sel), max(e[1] for e in sel)
    else:
        t0, t1 = 0.0, 2e9

    print(f"indexing {data_dir} ...")
    index = extract.file_time_index(data_dir)
    print(f"  {len(index)} files, "
          f"{sum(r['hdr_time_bad'] for r in index)} with bad header clocks")

    table, files, motor_el_sigma = build(data_dir, t0, t1, campaign_dir, index=index)
    n = table["t_utc"].size
    commit, dirt_outside = git_describe(
        os.path.join(HERE, "..", "..", "..", ".."), scope=HERE)
    compact = f"marjum-2026-07/pointing_table@{VERSION}+{commit}"
    generated = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")

    provenance = {
        "product": "pointing_table",
        "campaign": "marjum-2026-07",
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "generated_by": "pointing-analyst",
        "generator": "eigsep_data/notebooks/arp/marjum-2026-07/pointing/build_table.py",
        "generator_commit": commit,
        "generator_repo": "eigsep_data",
        "generator_clean_scope": (
            "notebooks/arp/marjum-2026-07/pointing -- the '-dirty' suffix "
            "reflects this generator's own files, not the whole shared repo. "
            "Uncommitted work elsewhere belongs to other agents and cannot "
            "affect this product: the generator imports only its own "
            "extract.py and fuse.py."),
        "repo_dirty_outside_generator": dirt_outside,
        "compact": compact,
        "n_samples": int(n),
        "n_files": len(files),
        "join_key": "t_utc (int64 UTC nanoseconds)",
        "time_note": (
            "t_utc comes from header/times where the header clock is sound. "
            "For hdr_time_bad rows it is a FILENAME anchor: the filename is the "
            "file close time, and its write lag is not constant across the "
            "campaign (sub-second on 07-16..07-18, median ~625-631 s on 07-12 "
            "and 07-15), so those rows carry t_utc_sigma_s=600. Filter on "
            "hdr_time_bad for time-critical work."),
        "extrapolation_policy": fuse.EXTRAPOLATION_POLICY,
        "sensor_sigmas_deg": {
            "pot_az": fuse.POT_AZ_SIGMA,
            "imu_el": fuse.IMU_EL_SIGMA,
            "motor_el_unvalidated": round(motor_el_sigma, 3),
            "motor_quantisation": round(fuse.MOTOR_QUANT_SIGMA, 5),
            "az_floor_sway": round(fuse.AZ_FLOOR_SIGMA, 3),
        },
        "flag_bits": {int(k): v for k, v in fuse.FLAG_NAMES.items()},
        "caveats": [
            "imu_az is dead from 07-16 onward (0% finite) and intermittent "
            "before; where dead, azimuth has no independent cross-check.",
            "imu_az yaw, where alive, drifts (+188 deg/hr vs pot -71 deg/hr) "
            "and is NOT an absolute azimuth reference.",
            "Azimuth zero point is not tied to true north; awaiting geometer.",
            "Motor counts are relative and slip; never use as absolute angles.",
            "Achieved azimuth steps are smaller than commanded (4.435 vs "
            "5.0018 deg in the 07-17 scan), accumulating 28.9 deg over the "
            "62-min block. Do not assume the scan plan.",
            "That slip is a discrete episode, not continuous drift: 27.4 deg "
            "lost in 12.3 min (07-17 20:41:24-20:53:43, -133 deg/hr), with "
            "the command tracked to within a few degrees either side.",
            "EL_SOLUTION_GLITCH's wrap-park mechanism (spurious sample "
            "escaping an el ~ +/-180 park) is established for the "
            "post-EL-failure era only; pre-failure firings of the same "
            ">20 deg/s slew criterion are a different, less-understood "
            "population and should not be assumed to be the same glitch.",
        ],
        "params": {
            "window_start_utc": extract.utc(table["t_utc_s"][0]),
            "window_stop_utc": extract.utc(table["t_utc_s"][-1]),
            "era_filter": args.era,
            "phase_filter": args.phase,
            "az_half_window_samples": 56,
            "az_slip_jump_deg": 1.0,
            "el_stuck_imu_ptp_deg": 3.0,
            "el_stuck_motor_ptp_deg": 30.0,
            "lidar_valid_m": [fuse.LIDAR_MIN_VALID, fuse.LIDAR_MAX_VALID],
            "hdr_time_tol_s": extract.HDR_TIME_TOL_S,
            "era_height_m": {k: v[0] for k, v in ERA_HEIGHT_M.items()},
        },
        "inputs": [{"path": f"marjum-2026-07/data/{os.path.basename(f)}",
                    "sha256": sha256(f)} for f in files],
    }

    arrays = {k: pa.array(v) for k, v in table.items()}
    at = pa.table(arrays)
    at = at.replace_schema_metadata({
        "eigsep_provenance": json.dumps(provenance),
        "eigsep_provenance_compact": compact,
    })
    out = args.out + ".parquet"
    pq.write_table(at, out, compression="zstd")

    schema_path = args.out + ".schema.json"
    with open(schema_path, "w") as fh:
        json.dump({
            "product": "pointing_table",
            "version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "join_key": "t_utc",
            "columns": {name: str(at.schema.field(name).type)
                        for name in at.schema.names},
            "quality_values": [QUALITY_OK, QUALITY_SUSPECT, QUALITY_GAP],
            "quality_note": (
                "quality rates confidence in the az/el VALUES only. Drive "
                "state (EL_STUCK, EL_POST_FAILURE) is separate and lives in "
                "flags: a parked-but-IMU-measured antenna is quality='ok' with "
                "EL_POST_FAILURE set, because we know exactly where it pointed "
                "even though it was not scanning. Select scanning data via "
                "flags, not via quality. 'interp' is never emitted: this "
                "product does not interpolate; absent pointing is a gap."),
            "flag_bits": {int(k): v for k, v in fuse.FLAG_NAMES.items()},
        }, fh, indent=2)

    print(f"wrote {out}  ({n} rows, {len(files)} files, "
          f"{os.path.getsize(out)/1e6:.1f} MB)")
    print(f"wrote {schema_path}")
    print(f"provenance: {compact}")
    if commit.endswith("-dirty"):
        print("  WARNING: generator tree is dirty -- this stamp is not citable.")


if __name__ == "__main__":
    main()
