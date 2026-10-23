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
v2.0 **meaning changes** -- ``az_deg`` is the *commanded* azimuth: motor
    counts plus one potentiometer-anchored offset per anchor segment
    (``fuse.commanded_azimuth``). On 07-17 the platform followed the motor
    through a 27 deg potentiometer slip: the unfitted HFSS beam's held-out
    test RMS is 0.071 with commanded az against 0.197 with the v1 fused az
    (data-analysis single_ch_beam_fit.ipynb at e0ffa78), and the az and el drives
    never run together (Aaron, 2026-09-23/24). Anchor rule, option (b)
    (Aaron, 2026-09-24): a new segment starts at every motor re-home, data
    gap and big slew (> 20 deg); its offset is the median pot - motor over
    its first 10 min after the slew. The v1 fused azimuth is kept as
    ``az_fused_deg``. New columns ``az_commanded_offset_deg`` and
    ``az_segment``; new flag bits 11 ``AZ_POT_DIVERGES``, 12 ``AZ_SLEW``, and
    13 ``AZ_FUSED_FALLBACK`` (no motor or no anchor; v1 fused az used).
    ``AZ_SLIP_EVENT`` and ``AZ_SLIP_RAMP`` keep their criteria but now mean
    pot/motor divergence, not platform slip. ``quality`` no longer marks
    commanded rows suspect for a missing per-sample pot; AZ_SLEW rows are
    suspect.
v3 ``EL_SOLUTION_GLITCH`` (memo 003, Known defects): the detector wraps the
    elevation difference into +/-180 deg, so passes through nadir no longer
    fire it (126 rows of v2.0, 31 in the 07-17 raster), and the flag now sets
    ``quality == "suspect"`` on both members of a flagged pair, since the
    rate does not say which one is bad (231 rows that v2.0 left "ok"). On
    the 07-17 raster, 60 flagged rows are solver jumps of +/-120 or 180 deg
    in ``el_deg`` that v2.0 marked "ok" with ``el_sigma_deg`` 0.24.
    ``height_era`` and ``phase`` are labelled by source file, not by the
    mode-table block's time interval, which left the edges of every block
    blank (167,585 rows, 13.6%); every row now has an era, and its nominal
    height where LIDAR does not measure one. The provenance records the mode
    table's SHA-256. Every other column is unchanged.
v4  **meaning changes** -- ``height_m`` is the antenna's height above the DEM
    surface under it, per height era, read from ``curation/horizon_profiles_v0005``
    (the same heights the horizon masks use): the geometry v0005 antenna for
    ``~91m``, with the S11 terrain-echo height changes for ``~87.5m`` (-1.66 m)
    and the new era ``~91m-reraise`` (+0.57 m, after the 07-17 battery
    re-raise; derived/height_eras/v0001, memo 002 Heights). v3 carried the
    field labels (87.5, 91 m) as heights. ``height_sigma_m`` now includes
    memo 002's +/-3.5 m held-out bound on the v0005 antenna; that part is one
    offset common to every
    suspended era, whose differences are known to +/-0.2 m (``~30m`` excepted:
    its offset is the field-note difference, +/-3 m). LIDAR ranges no longer override
    ``height_m``: the rangefinder is on the box, below the antenna by an
    unmodelled lever arm, and on the 509 rows it did override (the 07-17 dwell)
    it read 92.3 m with outliers at 29 and 198 m. ``FLAG_HEIGHT_ASSUMED`` is
    now set on every row; the raw range stays in ``lidar_dist_m``.

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
import glob
import hashlib
import importlib.util
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

VERSION = "v4"
SCHEMA_VERSION = 5

# Antenna height per curation height_era (v4): metres of the antenna above the
# DEM surface under it, read from the horizon product so that the table and the
# horizon masks place every era at the same height. There the 91 m era is the
# geometry release's antenna, the 87.5 m era and the configuration after the
# 07-17 re-raise are offset from it by the S11 terrain-echo height changes
# (derived/height_eras; memo 002, Heights), and the 30 m era by its field-note
# difference. The field labels (87.5 m from the tether anchor, 91 m with no
# datum) are other quantities and are not used as heights.
HORIZON_PROFILES = "horizon_profiles_v0005"
HORIZON_ERA = {"~30m": "30m", "~87.5m": "87.5m", "~91m": "91m",
               "~91m-reraise": "91m-reraise"}
# 1-sigma of the geometry release's antenna height, shared by every suspended
# era: memo 002's +/-3.5 m bound on the v0005 antenna (worst leave-one-out ray
# miss; the release's own posterior sigma_U, 0.15 m, is conditional on its
# unverified vertical frame). Era offsets add their own sigma in quadrature; an
# unmeasured (field-note) offset adds 3 m.
ANTENNA_SIGMA_M = 3.5
FIELDNOTE_OFFSET_SIGMA_M = 3.0
# Eras with no antenna position: nominal height and uncertainty.
NOMINAL_ERA_HEIGHT_M = {"~2m": (2.0, 1.0)}


def era_heights(campaign_dir):
    """{height_era: (height_m, sigma_m)} and the horizon product's JSON path."""
    path = os.path.join(campaign_dir, "curation", HORIZON_PROFILES + ".json")
    with open(path) as f:
        eras = json.load(f)["eras"]
    out = dict(NOMINAL_ERA_HEIGHT_M)
    for label, key in HORIZON_ERA.items():
        e = eras[key]
        if e["offset_source"].startswith("anchor"):
            off = 0.0
        else:
            off = e.get("offset_sigma_m", FIELDNOTE_OFFSET_SIGMA_M)
        out[label] = (float(e["agl_m_vs_ground_under_antenna"]),
                      float(np.hypot(ANTENNA_SIGMA_M, off)))
    return out, path

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
    """Era/phase blocks from data-archivist's mode table: each block's time
    interval, era and phase, and its first and last file and file count."""
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
            r.get("height_era"), r.get("phase"),
            r["file_first"], r["file_last"], int(r["n_files"])))
    return out


def label_eras(file_names, data_dir, eras, pattern="corr_*.h5"):
    """Per-sample height_era / phase by source file (v3).

    A mode-table block covers the ``n_files`` files from ``file_first`` to
    ``file_last`` in name order. v2.0 labelled by the block's time interval,
    from its first file's timestamp to its last file's, which left samples at
    the edges of every block unlabelled -- a one-file block covered almost
    none of its own samples -- so 167,585 rows (13.6%) had no era (memo 003).
    """
    names = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(data_dir, pattern)))
    pos = {f: i for i, f in enumerate(names)}
    by_file = {}
    for _, _, h, p, first, last, n_files in eras:
        i0, i1 = pos.get(first), pos.get(last)
        if i0 is None or i1 is None:
            continue                          # block outside this data directory
        if i1 - i0 + 1 != n_files:
            raise ValueError(f"mode-table block {first}..{last} lists {n_files} "
                             f"files; the data directory has {i1 - i0 + 1}")
        for f in names[i0:i1 + 1]:
            if f in by_file:
                raise ValueError(f"{f} is in two mode-table blocks")
            by_file[f] = (h or "", p or "")
    uniq, inv = np.unique(np.asarray(file_names), return_inverse=True)
    lab = [by_file.get(f, ("", "")) for f in uniq]
    era = np.array([e for e, _ in lab], dtype=object)[inv]
    phase = np.array([p for _, p in lab], dtype=object)[inv]
    return era, phase


def build(data_dir, t_start, t_stop, campaign_dir, index=None):
    d = extract.load_window(data_dir, t_start, t_stop, index=index)
    n = d["time"].size

    az_fused, sigma_fused, flags_az, az_offset = fuse.fuse_azimuth(
        d["motor_az_pos"], d["potmon_pot_az_angle"])
    az, sigma_cmd, flags_cmd, cmd_offset, az_segment = fuse.commanded_azimuth(
        d["motor_az_pos"], d["potmon_pot_az_angle"], d["time"], az_fused)
    commanded = (flags_cmd & fuse.FLAG_AZ_FUSED_FALLBACK) == 0
    sigma_az = np.where(commanded & np.isfinite(az), sigma_cmd, sigma_fused)
    el, sigma_el, flags_el = fuse.fuse_elevation(
        d["imu_el_el_deg"], d["motor_el_pos"])

    flags = flags_az | flags_el | flags_cmd
    flags[d["motor_status"] == "absent"] |= fuse.FLAG_NO_METADATA
    flags[fuse.detect_az_slip_ramp(az_fused, d["motor_az_pos"], d["time"])] |= \
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
    era, phase = label_eras(d["file_name"], data_dir, eras)

    # Height: the era's antenna height (era_heights). v4 no longer overrides it
    # with LIDAR ground returns: the rangefinder sits on the box, below the
    # antenna by an unmodelled lever arm (about the antenna coax length), so its
    # range is not the antenna height (memo 002: the dwell reads 2.2 m short of
    # the v0004 prediction). The raw range stays in lidar_dist_m.
    height = np.full(n, np.nan)
    sigma_h = np.full(n, np.nan)
    for key, (h, s) in era_heights(campaign_dir)[0].items():
        sel = era == key
        height[sel], sigma_h[sel] = h, s
    flags |= fuse.FLAG_HEIGHT_ASSUMED

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
    fallback_no_pot = (((flags & fuse.FLAG_AZ_FUSED_FALLBACK) != 0)
                       & ((flags & fuse.FLAG_AZ_NO_POT) != 0))
    value_suspect = (
        hdr_bad                                        # t_utc only ~+/-10 min
        | fallback_no_pot                              # az has no absolute ref
        | ((flags & (fuse.FLAG_EL_NO_IMU               # el from motor alone
                     | fuse.FLAG_AZ_SLEW               # platform may lag/slip
                     | fuse.FLAG_UNCOMMANDED           # swinging within a sample
                     | fuse.FLAG_EL_SOLUTION_GLITCH)) != 0))  # el solver jump
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
        "az_fused_deg": az_fused,
        "az_commanded_offset_deg": cmd_offset,
        "az_segment": az_segment.astype("int64"),
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
    heights, height_path = era_heights(campaign_dir)
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
        "generated_by": ("Claude Code sessions for C. H. Bye (v3, v4); Claude Code "
                         "session for Aaron (v2.0); pointing-analyst (v1.x)"),
        "generator": "data-analysis/notebooks/arp/marjum-2026-07/pointing/build_table.py",
        "generator_commit": commit,
        "generator_repo": "data-analysis",
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
            "v2.0 az_deg is commanded: motor counts give the motion, and the "
            "pot sets one offset per anchor segment (first 10 min after a "
            "re-home, gap or big slew). Absolute az therefore inherits one "
            "pot reading per segment; segments after a big slew may differ "
            "from each other by platform slip during the slew.",
            "The 07-17 20:41-20:53 'slip' in v1 fused az is a potentiometer "
            "slip; the platform followed the motor (beam-fit evidence in "
            "data-analysis single_ch_beam_fit.ipynb at e0ffa78). az_fused_deg "
            "keeps the v1 value for audit.",
            "Commanded az assumes the platform follows the motor between big "
            "slews; this is validated on the 07-17 beam-scan raster only.",
            "EL_SOLUTION_GLITCH's wrap-park mechanism (spurious sample "
            "escaping an el ~ +/-180 park) is established for the "
            "post-EL-failure era only; pre-failure firings of the same "
            ">20 deg/s slew criterion are a different, less-understood "
            "population and should not be assumed to be the same glitch.",
            "v3: EL_SOLUTION_GLITCH wraps the elevation difference (nadir "
            "passes no longer fire it) and sets quality 'suspect' on both "
            "members of a flagged pair; the rate does not say which is bad. "
            "height_era and phase are labelled by source file.",
            "v4: height_m is the era's antenna height above the DEM v0004 "
            "surface under the antenna (geometry v0005, with the S11 "
            "height changes for ~87.5m and the new ~91m-reraise era); LIDAR "
            "ranges no longer override it (unmodelled lever arm).",
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
            "era_height_m": {k: round(v[0], 2) for k, v in heights.items()},
            "era_height_sigma_m": {k: round(v[1], 2) for k, v in heights.items()},
            "commanded_az": {
                "stretch_gap_s": fuse.STRETCH_GAP_S,
                "rehome_jump_deg": fuse.REHOME_JUMP_DEG,
                "big_slew_deg": fuse.BIG_SLEW_DEG,
                "anchor_window_s": fuse.ANCHOR_WINDOW_S,
                "anchor_min_samples": fuse.ANCHOR_MIN_SAMPLES,
                "pot_diverge_deg": fuse.AZ_DIVERGE_DEG},
        },
        "height_source": {
            "path": f"marjum-2026-07/curation/{HORIZON_PROFILES}.json",
            "sha256": sha256(height_path)},
        "era_source": {
            "path": "marjum-2026-07/curation/mode_table.jsonl",
            "sha256": sha256(os.path.join(campaign_dir, "curation",
                                          "mode_table.jsonl"))},
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
