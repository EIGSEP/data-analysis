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


def select_window(data_dir, t_start, t_stop, pattern="corr_*.h5"):
    """Correlator files whose *data* span overlaps [t_start, t_stop].

    Selection uses ``header/times``, never the filename, so buffered bursts
    at the end of a run are placed at their true observation time.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, pattern))):
        with h5py.File(path, "r") as f:
            if "header/times" not in f:
                continue
            times = f["header/times"][()]
        if times.size and times[-1] >= t_start and times[0] <= t_stop:
            out.append((path, float(times[0]), float(times[-1]), int(times.size)))
    return out


def load_window(data_dir, t_start, t_stop, pattern="corr_*.h5", verbose=False):
    """Load every pointing stream over a time window into flat arrays.

    Returns a dict of concatenated per-sample arrays, sorted by time, with
    ``time`` (UTC epoch s), ``file_index``, ``file_close_time`` and one entry
    per sensor field named ``<stream>_<field>`` plus ``<stream>_status``.
    """
    files = select_window(data_dir, t_start, t_stop, pattern)
    if not files:
        raise ValueError("no correlator files overlap the requested window")

    chunks = []
    for idx, (path, _, _, n) in enumerate(files):
        with h5py.File(path, "r") as f:
            times = f["header/times"][()]
            n = times.size
            chunk = {
                "time": times.astype(float),
                "file_index": np.full(n, idx),
                "file_close_time": np.full(n, filename_timestamp(path)),
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
    return data


def utc(ts):
    """Format a UTC epoch second as an ISO-ish string (for printing)."""
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
