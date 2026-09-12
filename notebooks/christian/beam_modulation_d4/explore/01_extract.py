"""Pull the chosen rotation window off the external disk into a self-contained sidecar.

Deployment-4 files carry per-integration metadata as JSON blobs.  Of those only
metadata/rfswitch is alive: metadata/motor is frozen at 2025-07-11 with az=el=0 for the
whole deployment, and metadata/imu_antenna froze on 07-18.  So the switch state is taken
from the file and the rotation angle has to come from the data itself.
"""
import json
import h5py, numpy as np
from common import DISK, FILES, KEYS, SIDECAR

times, sw, blocks = [], [], {k: [] for k in KEYS}
for name in FILES:
    with h5py.File(DISK / name, "r") as f:
        t = np.asarray(json.loads(f["header/times"][()].decode()), float)
        s = np.array([m.get("sw_state", -1)
                      for m in json.loads(f["metadata/rfswitch"][()].decode())])
        cols = {k: f[f"data/{k}"][:] for k in KEYS}
        dfreq = f["header"].attrs["dfreq"]
        nchan = f["header"].attrs["nchan"]
    n = min(len(t), len(s), *(c.shape[0] for c in cols.values()))
    times.append(t[:n]); sw.append(s[:n])
    for k in KEYS:
        blocks[k].append(cols[k][:n].astype(np.float32))

t = np.concatenate(times)
assert np.all(np.diff(t) > 0), "times not monotonic"
gaps = np.diff(t)
print(f"{len(t)} integrations, {(t[-1] - t[0]) / 60:.1f} min, "
      f"sample spacing {np.median(gaps):.4f} s (max {gaps.max():.2f} s)")

with h5py.File(SIDECAR, "w") as f:
    f.attrs["source"] = "Samsung_T5/eigsep_data/deployment4/corr_data"
    f.attrs["files"] = ",".join(FILES)
    f.attrs["note"] = ("deployment 4 has no motor or IMU telemetry; sw_state is the only "
                       "live per-integration metadata")
    f["times"] = t
    f["sw_state"] = np.concatenate(sw)
    f["freqs"] = np.arange(nchan) * dfreq
    for k in KEYS:
        f.create_dataset(f"data/{k}", data=np.concatenate(blocks[k]),
                         compression="gzip", compression_opts=4)
print("wrote", SIDECAR)
