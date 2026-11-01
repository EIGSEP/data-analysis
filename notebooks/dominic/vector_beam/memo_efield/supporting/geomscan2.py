import sys, json, time, numpy as np
sys.path.insert(0, sys.argv[1]); import geomfit2 as gf
from eigsep_data.beam_mapping import vector_beam as vb
S = sys.argv[1]; z, ch = gf.load(S + "/inputs.npz")
gs = json.load(open(S + "/gs_v0013.json")) if False else None
# g per channel: closure g at v0013 with the packaged self-cal phase (held fixed)
ph = np.load("../../selfcal_phase.npz")["phase"]
geo = vb.fields_from_model(gf.MODEL, gf.P0, z["az"], z["el"], 32); rot = np.exp(-1j * np.where(np.isfinite(ph), ph, 0))
gs = [vb.fit_closure_g(*geo, c["Ex"] * rot, c["Ey"] * rot, c["v"] & np.isfinite(ph)) for c in ch]
t0 = time.time(); e0 = gf.evaluate(gf.P0, [0, 0, 0], z, ch, gs); print(f"v0013, p=0: {e0:.4f}  ({time.time()-t0:.0f} s/eval)", flush=True)
names = ["polarization_delta_deg", "az_zero_delta_deg", "el_zero_delta_deg", "tx_east_delta_m", "tx_north_delta_m"]
grids = {0: [-4, -2, 0, 2, 4], 1: [-4, -2, -1, 0, 1, 2, 4], 2: [-1, 0, 0.5, 1, 1.5, 2, 3], 3: [-3, -2, -1, 0, 1, 2, 3], 4: [-3, -2, -1, 0, 1, 2, 3]}
out = dict(base=e0, g=[[float(abs(g)), float(np.angle(g))] for g in gs])
for i, grid in grids.items():
    rows = []
    for v in grid:
        p = gf.P0.copy(); p[i] = v; rows.append((float(v), gf.evaluate(p, [0, 0, 0], z, ch, gs)))
    out[names[i]] = rows
    print(f"{names[i]:24s} (v0013 {gf.P0[i]:+.3f}): " + "  ".join(f"{v:+.2f}:{e:.4f}" for v, e in rows), flush=True)
rows = [(float(pz), gf.evaluate(gf.P0, [0, 0, pz], z, ch, gs)) for pz in (0, 0.25, 0.5, 0.75, 1.0)]
out["phase_centre_z_m"] = rows
print("phase centre z (m)      : " + "  ".join(f"{v:.2f}:{e:.4f}" for v, e in rows), flush=True)
json.dump(out, open(S + "/geomscan2.json", "w"), indent=1)
