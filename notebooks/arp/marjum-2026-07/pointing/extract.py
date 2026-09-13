"""Extract the per-sample pointing metadata streams from correlator files.

The Marjum 2026-07 correlator files carry the pointing sensors inline: each
``metadata/<sensor>`` dataset is a JSON list with one record per entry in
``header/times``, so pointing is natively aligned to the spectra and no
cross-clock matching is needed.

Timestamp convention
--------------------
Everything here is keyed on ``header/times`` (UTC seconds, ~0.537 s cadence).
The *filename* timestamp is the file **close** time, not the start time
(median ``fname - times[-1]`` = -0.94 s across 07-17/18).  The filename is
carried alongside so consumers can join either way, but it must not be used
as a sample time.

Usage
-----
Loaded by path; ``import eigsep_data`` is broken in the arp env pending the
picohost rename fix (B12)::

    import importlib.util as u
    s = u.spec_from_file_location("extract", ".../pointing/extract.py")
    m = u.module_from_spec(s); s.loader.exec_module(m)
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re

import h5py
import numpy as np

# Sensor record fields we pull through, per stream.
FIELDS = {
    "imu_az": ("yaw", "pitch", "roll", "accel_x", "accel_y", "accel_z", "el_deg"),
    "imu_el": ("yaw", "pitch", "roll", "accel_x", "accel_y", "accel_z", "el_deg"),
    "motor": ("az_pos", "az_target_pos", "el_pos", "el_target_pos", "boot_id"),
    "potmon": ("pot_az_voltage", "pot_az_angle", "pot_az_cal_slope",
               "pot_az_cal_intercept", "pot_az_near_rail"),
    "lidar": ("distance_m", "laser_firing", "standby"),
}

FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")


def filename_timestamp(path):
    """UTC epoch seconds encoded in a correlator filename (= file close time)."""
    m = FNAME_RE.search(os.path.basename(path))
    if m is None:
        return np.nan
    stamp = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return stamp.replace(tzinfo=dt.timezone.utc).timestamp()


def _decode(raw):
    return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def _records_to_arrays(records, fields, n):
    """Turn a JSON record list into float arrays plus a status array.

    Records in ``status != 'update'`` (typically 'error') carry all-None
    payloads; those become NaN so downstream code sees a gap, not a zero.
    """
    out = {f: np.full(n, np.nan) for f in fields}
    status = np.array(["missing"] * n, dtype=object)
    for i, rec in enumerate(records[:n]):
        if not isinstance(rec, dict):
            continue
        status[i] = rec.get("status", "missing")
        for f in fields:
            v = rec.get(f)
            if v is None or isinstance(v, str):
                continue
            out[f][i] = float(v)
    return out, status


# A file's header clock is judged bad when it disagrees with the filename
# (its close time) by more than this.  The campaign's bad-clock population
# sits days to weeks away, so this is a generous threshold.
HDR_TIME_TOL_S = 3600.0

# The two correlator cadences in this campaign: corr_acc_len doubled at
# 2026-07-15 15:54:59 UTC, taking the sample spacing from 0.2684 s to
# 0.5369 s.  A file whose implied cadence matches neither is not trustworthy
# regardless of what its endpoints say.
KNOWN_CADENCES_S = (0.2684354782104492, 0.5368709564208984)
CADENCE_TOL = 0.02

# The filename is the file CLOSE time, but the write lag is NOT a small
# constant campaign-wide: it is sub-second on 07-16..07-18 yet has a median
# of ~625-631 s on 07-12 and 07-15 (buffered writes), with 32.5% of all
# good-clock files above 10 s.  So a filename anchor is good to ~+/-10 min,
# not to a second, and t_utc on a clock-bad row inherits that.
FNAME_ANCHOR_SIGMA_S = 600.0


def file_time_index(data_dir, pattern="corr_*.h5"):
    """Per-file timing summary for every correlator file, clock-bad or not.

    ``header/times`` is authoritative for *when data was taken* wherever it is
    valid; the filename is authoritative for file identity and ordering and is
    the fallback.  This returns both plus an ``hdr_time_bad`` verdict, so a
    caller can select on a trustworthy axis either way.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, pattern))):
        fts = filename_timestamp(path)
        try:
            with h5py.File(path, "r") as f:
                if "header/times" not in f:
                    continue
                times = f["header/times"][()]
        except OSError:
            continue
        if times.size == 0:
            continue
        t0, t1, n = float(times[0]), float(times[-1]), int(times.size)
        cadence = (t1 - t0) / max(n - 1, 1)
        # Three independent ways a header clock can be untrustworthy. Endpoint
        # agreement alone is not enough: some files agree with their filename
        # yet run backwards internally (negative implied cadence).
        off_filename = not np.isfinite(fts) or abs(fts - t1) > HDR_TIME_TOL_S
        non_monotonic = bool(np.any(np.diff(times) <= 0))
        odd_cadence = not any(abs(cadence - c) < CADENCE_TOL
                              for c in KNOWN_CADENCES_S)
        bad = off_filename or non_monotonic or odd_cadence
        out.append({"path": path, "fname_ts": fts, "hdr_t0": t0, "hdr_t1": t1,
                    "n": n, "hdr_time_bad": bool(bad), "cadence": cadence,
                    "bad_reason": ("off_filename" if off_filename else
                                   "non_monotonic" if non_monotonic else
                                   "odd_cadence" if odd_cadence else "")})
    # Clock-bad files get their cadence from the nearest sound neighbour,
    # since their own header span is meaningless.
    good = [r for r in out if not r["hdr_time_bad"]]
    if good:
        gts = np.array([r["fname_ts"] for r in good])
        gcad = np.array([r["cadence"] for r in good])
        for r in out:
            if r["hdr_time_bad"]:
                r["cadence"] = float(gcad[np.argmin(np.abs(gts - r["fname_ts"]))])
    for r in out:
        if r["hdr_time_bad"]:
            # Filename anchor only -- good to ~FNAME_ANCHOR_SIGMA_S, not to a
            # second. Every sample from such a file is flagged accordingly.
            r["t_start"] = r["fname_ts"] - (r["n"] - 1) * r["cadence"]
            r["t_stop"] = r["fname_ts"]
        else:
            r["t_start"], r["t_stop"] = r["hdr_t0"], r["hdr_t1"]
    return out


def sample_times(rec):
    """UTC epoch seconds for every sample in one file.

    Uses ``header/times`` verbatim when the clock is sound; otherwise
    reconstructs a uniform grid anchored on the filename close time.
    """
    if not rec["hdr_time_bad"]:
        with h5py.File(rec["path"], "r") as f:
            return f["header/times"][()].astype(float)
    return rec["fname_ts"] - (rec["n"] - 1 - np.arange(rec["n"])) * rec["cadence"]


def select_window(data_dir, t_start, t_stop, pattern="corr_*.h5", index=None):
    """Correlator files whose data span overlaps [t_start, t_stop].

    Uses ``header/times`` where the clock is sound and the filename-derived
    span where it is not, so bad-clock files are still selectable rather than
    silently invisible -- the failure mode that hides 619 files of the ~2 m
    Phase-A era from a naive header-only selection.
    """
    if index is None:
        index = file_time_index(data_dir, pattern)
    out = []
    for r in index:
        if r["t_stop"] >= t_start and r["t_start"] <= t_stop:
            out.append((r["path"], r["t_start"], r["t_stop"], r["n"], r))
    return out


def load_window(data_dir, t_start, t_stop, pattern="corr_*.h5", verbose=False,
                index=None):
    """Load every pointing stream over a time window into flat arrays.

    Returns a dict of concatenated per-sample arrays, sorted by time, with
    ``time`` (UTC epoch s), ``file_index``, ``file_close_time``,
    ``hdr_time_bad``, ``sample_idx`` and one entry per sensor field named
    ``<stream>_<field>`` plus ``<stream>_status``.
    """
    files = select_window(data_dir, t_start, t_stop, pattern, index=index)
    if not files:
        raise ValueError("no correlator files overlap the requested window")

    chunks = []
    for idx, (path, _, _, n, rec) in enumerate(files):
        times = sample_times(rec)
        with h5py.File(path, "r") as f:
            n = times.size
            chunk = {
                "time": times.astype(float),
                "file_index": np.full(n, idx),
                "file_close_time": np.full(n, filename_timestamp(path)),
                "hdr_time_bad": np.full(n, rec["hdr_time_bad"]),
                "sample_idx": np.arange(n),
            }
            for stream, fields in FIELDS.items():
                key = f"metadata/{stream}"
                if key in f:
                    arrs, status = _records_to_arrays(_decode(f[key][()]), fields, n)
                else:
                    arrs = {fld: np.full(n, np.nan) for fld in fields}
                    status = np.array(["absent"] * n, dtype=object)
                for fld, arr in arrs.items():
                    chunk[f"{stream}_{fld}"] = arr
                chunk[f"{stream}_status"] = status
            chunks.append(chunk)
        if verbose and idx % 25 == 0:
            print(f"  {idx+1}/{len(files)} {os.path.basename(path)}")

    keys = chunks[0].keys()
    data = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    order = np.argsort(data["time"], kind="stable")
    data = {k: v[order] for k, v in data.items()}
    data["_files"] = [f[0] for f in files]
    # Per-sample source filename, for the product's `file` column.
    names = np.array([os.path.basename(f[0]) for f in files])
    data["file_name"] = names[data["file_index"].astype(int)]
    return data


def utc(ts):
    """Format a UTC epoch second as an ISO-ish string (for printing)."""
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
