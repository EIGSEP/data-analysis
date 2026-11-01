"""Self-cal phase on one frequency half (lo: ν < ν_mid, hi: ν >= ν_mid), same recipe as selfcal_phase.py."""
import sys, time, numpy as np
half = sys.argv[1]
sys.path.insert(0, "."); sys.argv = [sys.argv[0], "."]
import geomfit2 as gf
from eigsep_data.beam_mapping import vector_beam as vb
z, ch = gf.load("inputs.npz")
nu = np.array([171.875 + (c["d"] - 60) * 1.953125 for c in ch]); mid = np.median(nu)
keep = [k for k in range(len(ch)) if (nu[k] < mid) == (half == "lo")]
chans = [dict(Ex=ch[k]["Ex"], Ey=ch[k]["Ey"], valid=ch[k]["v"], Ex_measured=z["Ex_meas"][:, ch[k]["d"]] & ch[k]["v"],
              Ey_measured=z["Ey_meas"][:, ch[k]["d"]] & ch[k]["v"]) for k in keep]
print(half, "channels", len(keep), "nu", nu[keep].min(), nu[keep].max(), "mean", nu[keep].mean(), flush=True)
t0 = time.time()
phase, info = vb.self_calibrate_phase(z["pxs"], z["r_hat"], z["E1v"], z["E2v"], chans, block=20, verbose=True)
np.savez(f"selfcal_{half}.npz", phase=phase, times=z["times"], nu=nu[keep], updates_deg=np.array(info["updates_deg"]))
print(f"{half}: phase on {np.isfinite(phase).sum()} of {len(phase)}, {time.time()-t0:.0f} s", flush=True)
