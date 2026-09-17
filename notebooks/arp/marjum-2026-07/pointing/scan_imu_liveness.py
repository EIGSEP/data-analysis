"""Survey IMU sensor liveness across the whole Marjum 2026-07 campaign.

Answers the question the mode table cannot: the curation ``az_alive`` column
is ``motor_az_med.notna()`` -- *motor* telemetry presence -- and says nothing
about whether the azimuth IMU produced usable data.

In the 07-17/18 beam-scan window ``imu_az`` returns ``status='error'`` for
100% of samples, leaving azimuth with no independent cross-check.  If it is
alive anywhere else, that window can be used to characterise the
potentiometer against an independent sensor and bound the single-sensor
uncertainty elsewhere.

Run::

    /home/aparsons/.local/share/mamba/envs/arp/bin/python3 scan_imu_liveness.py \
        --data ~/projects/eigsep/marjum-2026-07/data --out imu_liveness.csv
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import re

import h5py
import numpy as np

FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")


def filename_ts(path):
    m = FNAME_RE.search(os.path.basename(path))
    if m is None:
        return np.nan
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=dt.timezone.utc).timestamp()


def scan_file(path):
    """Per-file liveness counts for each pointing sensor."""
    row = {"file": os.path.basename(path), "fname_ts": filename_ts(path)}
    with h5py.File(path, "r") as f:
        times = f["header/times"][()] if "header/times" in f else np.array([])
        row["n"] = int(times.size)
        row["hdr_t0"] = float(times[0]) if times.size else np.nan
        row["hdr_t1"] = float(times[-1]) if times.size else np.nan
        for stream in ("imu_az", "imu_el", "motor", "potmon", "lidar"):
            key = f"metadata/{stream}"
            if key not in f:
                row[f"{stream}_absent"] = row["n"]
                row[f"{stream}_ok"] = 0
                row[f"{stream}_finite"] = 0
                continue
            raw = f[key][()]
            recs = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            status = collections.Counter(
                r.get("status", "missing") if isinstance(r, dict) else "bad"
                for r in recs)
            row[f"{stream}_absent"] = 0
            row[f"{stream}_ok"] = status.get("update", 0)
            # A status of 'update' can still carry nulls; count real numbers.
            probe = {"imu_az": "accel_y", "imu_el": "accel_y", "motor": "az_pos",
                     "potmon": "pot_az_angle", "lidar": "distance_m"}[stream]
            row[f"{stream}_finite"] = sum(
                1 for r in recs
                if isinstance(r, dict) and isinstance(r.get(probe), (int, float)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="imu_liveness.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.data), "corr_*.h5")))
    print(f"scanning {len(paths)} files")
    rows = []
    for i, p in enumerate(paths):
        try:
            rows.append(scan_file(p))
        except Exception as exc:  # a corrupt file should not kill the survey
            rows.append({"file": os.path.basename(p), "error": str(exc)[:80]})
        if i % 500 == 0:
            print(f"  {i}/{len(paths)}")

    cols = ["file", "fname_ts", "n", "hdr_t0", "hdr_t1"]
    for s in ("imu_az", "imu_el", "motor", "potmon", "lidar"):
        cols += [f"{s}_absent", f"{s}_ok", f"{s}_finite"]
    with open(args.out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
