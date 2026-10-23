"""Run pipeline.load() + Phase 2 selection B once; save everything to npz."""
import sys, numpy as np
from eigsep_data.beam_mapping import vector_beam as vb
cap = {}
_orig = vb.reduce_cross
def spy(*a, **k):
    out = _orig(*a, **k); cap["out"] = out; return out
vb.reduce_cross = spy
import pipeline, run_phase2 as R, geometry_v0013
inp = pipeline.load()
cache = np.load(pipeline.CACHE, allow_pickle=True)
j = R.join_pointing(inp["times"], str(cache["data_dir"]))
ok = np.isfinite(j.az_deg.to_numpy()) & np.isfinite(j.el_deg.to_numpy())
az = np.where(ok, j.az_deg.to_numpy(), 0.0); el = np.where(ok, j.el_deg.to_numpy(), 0.0)
sel = ok & (j.quality == "ok").to_numpy() & (j.rfswitch == "RFANT").to_numpy()
idx = np.flatnonzero(sel); gl = np.zeros(len(j), bool); gl[idx] = R.reversal_glitches(el[idx]); sel &= ~gl
nside = int(np.sqrt(inp["beam"].shape[-1] / 12))
pxs, r_hat, E1v, E2v = vb.fields_from_model(geometry_v0013.model(), geometry_v0013.PARAMS, az, el, nside)
Exh, Eyh, bad_time, Exm, Eym = cap["out"]
d = np.array([c["d_idx"] for c in inp["channels"]])
np.savez(sys.argv[1], pxs=pxs, r_hat=r_hat, E1v=E1v, E2v=E2v, sel=sel, az=az, el=el,
         Ex_all=Exh, Ey_all=Eyh, Ex_meas=Exm, Ey_meas=Eym, bad_time=bad_time,
         d_idx=d, h_idx=np.array([c["h_idx"] for c in inp["channels"]]),
         freq=np.array([c["freq_mhz"] for c in inp["channels"]]),
         valid=np.array([c["valid"] for c in inp["channels"]]),
         beam=inp["beam"], times=inp["times"])
print("saved", Exh.shape, sel.sum())
