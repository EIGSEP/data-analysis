import os
os.environ["MPLCONFIGDIR"] = "/tmp"

from pathlib import Path
import sys
import numpy as np
import hashlib
import json

scratch = Path("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch")
campaign = Path("/home/aparsons/projects/eigsep/marjum-2026-07")
gp_dir = campaign / "derived/geometry_posterior/v0005"
gp_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(scratch))
from test_schur_mcmc import build_model
model = build_model()
keys_clean = [str(k) for k in model.keys]

chain_npz_path = scratch / "posterior_v5_hmc_chains.npz"
z_chains = np.load(chain_npz_path)
traces = z_chains["traces"]

st_mode = np.load(scratch / "state_v5_joint_mode.npz")["state"]
cam_mode, ant_mode, tx_mode, bias_mode, ex_mode, txe_mode, pts_mode = model.unpack(st_mode)
exact_lp_mode = float(model.logp(st_mode))

param_names = []
for c_i in range(model.nc):
    k = keys_clean[c_i]
    for p in ["e", "n", "u", "th", "ph", "ti", "logf"]:
        param_names.append(f"cam_{k}_{p}")
param_names.extend(["ant_e", "ant_n", "ant_u", "tx_e", "tx_n", "tx_u", "bias_e", "bias_n", "extra", "tx_extra"])

# Save posterior_samples.npz
np.savez_compressed(
    gp_dir / "posterior_samples.npz",
    traces=traces,
    keys=np.array(keys_clean),
    param_names=np.array(param_names),
    camera=cam_mode,
    antenna=ant_mode,
    transmitter=tx_mode,
    gps_bias=bias_mode,
    antenna_extra_px=float(ex_mode),
    transmitter_extra_px=float(txe_mode),
    landmarks=st_mode[model.ng:].reshape(-1, 3),
    state=st_mode,
    logp=exact_lp_mode,
)

# Save provisional_geometry.npz
np.savez_compressed(
    gp_dir / "provisional_geometry.npz",
    keys=np.array(keys_clean),
    camera=cam_mode,
    antenna=ant_mode,
    transmitter=tx_mode,
    gps_bias=bias_mode,
    antenna_extra_px=float(ex_mode),
    transmitter_extra_px=float(txe_mode),
    landmarks=st_mode[model.ng:].reshape(-1, 3),
    state=st_mode,
    logp=exact_lp_mode,
)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("posterior_samples.npz sha256:", sha256(gp_dir / "posterior_samples.npz"))
print("provisional_geometry.npz sha256:", sha256(gp_dir / "provisional_geometry.npz"))

# Check manifest
man = json.loads((gp_dir / "manifest.json").read_text())
print("Expected posterior_samples.npz:", man["files"]["posterior_samples.npz"]["sha256"])
print("Expected provisional_geometry.npz:", man["files"]["provisional_geometry.npz"]["sha256"])
