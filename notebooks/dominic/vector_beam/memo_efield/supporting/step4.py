"""Step 4: error budget of the HFSS-vs-data residual (memo 005 metric), corrections applied in turn,
at two axle tilts; plus a free-beam check with the E-field solver."""
import sys, json, time, numpy as np, healpy as hp
S = sys.argv[1]; sys.path.insert(0, S)
import tilt, geomfit2 as gf
from eigsep_base.rotations import vector_to_spherical
from eigsep_data.beam_mapping import vector_beam as vb
from eigsep_data.beam_mapping.tx_fit import stripe_split
z, ch = gf.load(S + "/inputs.npz"); split = stripe_split(z["az"]); P0 = gf.P0.copy(); t0 = time.time()
SURVEY = (-0.29, 2.26)                               # v0005 tx-minus-antenna relative to v0004 (m)
def geo(params, t1=0.0, t2=0.0):
    rhat, E1, E2 = tilt.geometry(params, z["az"], z["el"], t1, t2)
    th, ph = vector_to_spherical(rhat); return np.asarray(hp.ang2pix(32, th, ph)), rhat, E1, E2
def hfss_per_tooth(g, return_pred=False):
    pxs, _, E1, E2 = g; out, preds = [], []
    for k, c in enumerate(ch):
        D = np.abs(c["Ex"] if c["arm"] == 0 else c["Ey"]) ** 2; m = c["meas"]
        uh = z["beam"][z["h_idx"][k]][:, pxs].T; e = E1 if c["arm"] == 0 else E2
        cH = np.abs(np.einsum("ij,ij->i", e, np.conj(uh))) ** 2; tr, te = m & (split == 0), m & (split == 2)
        gain = D[tr] @ cH[tr] / (cH[tr] @ cH[tr]); pred = gain * cH
        out.append((np.linalg.norm(D[tr] - pred[tr]) / np.linalg.norm(D[tr]), np.linalg.norm(D[te] - pred[te]) / np.linalg.norm(D[te])))
        preds.append(pred)
    out = np.array(out)
    return (out, preds) if return_pred else out
# white-noise floor per tooth (successive-sample differences, as memo 005)
floor = []
for c in ch:
    D = np.abs(c["Ex"] if c["arm"] == 0 else c["Ey"]) ** 2; m = c["meas"] & (split == 2); i = np.flatnonzero(m)
    j = i[np.flatnonzero(np.diff(i) == 1)]; floor.append(np.std(D[j + 1] - D[j]) / np.sqrt(2) / np.sqrt(np.mean(D[m] ** 2)))
floor = np.array(floor); print(f"white-noise floor (test stripes): median {np.median(floor):.4f}", flush=True)
res = dict(floor=floor.tolist())
for label, T1 in (("tilt 3.3 (HFSS-free)", -3.3), ("tilt 5.7 (HFSS-preferred)", -5.7)):
    steps = {}
    steps["S0 v0013"] = hfss_per_tooth(geo(P0))[:, 1]
    p1 = np.r_[P0[:3], SURVEY]
    steps["S1 survey v0005"] = hfss_per_tooth(geo(p1))[:, 1]
    steps["S2 + axle tilt"] = hfss_per_tooth(geo(p1, T1))[:, 1]
    steps["S3 + roll-axis 0.5"] = hfss_per_tooth(geo(p1, T1, 0.5))[:, 1]
    # S4: per-tooth polarization rotation, chosen on TRAINING stripes
    da = np.arange(-10, 10.01, 1.0); tr_rms = []; te_rms = []
    for d in da:
        p = p1.copy(); p[0] = P0[0] + d; o = hfss_per_tooth(geo(p, T1, 0.5)); tr_rms.append(o[:, 0]); te_rms.append(o[:, 1])
    tr_rms, te_rms = np.array(tr_rms), np.array(te_rms); best = np.argmin(tr_rms, 0)
    steps["S4 + per-tooth polarization"] = te_rms[best, np.arange(len(ch))]
    dalpha = da[best]
    print(f"\n== {label} ==", flush=True)
    prev = None
    for k, v in steps.items():
        print(f"  {k:30s}: median {np.median(v):.4f}" + (f"   change {np.median(v) - prev:+.4f}" if prev is not None else ""), flush=True); prev = np.median(v)
    print(f"  per-tooth polarization rotation chosen (deg): {dalpha.astype(int).tolist()}", flush=True)
    print(f"  polarization rotation vs frequency: corr with freq {np.corrcoef(dalpha, [c['d'] for c in ch])[0,1]:+.2f}, median {np.median(dalpha):+.1f} deg", flush=True)
    # free-beam check at the S3 geometry: self-cal phase recomputed there, solver trained on training stripes
    g3 = geo(p1, T1, 0.5); pxs, rhat, E1, E2 = g3
    chans = [dict(Ex=np.where(c["v"], c["Ex"], np.nan), Ey=np.where(c["v"], c["Ey"], np.nan), valid=c["v"],
                  Ex_measured=c["meas"] if c["arm"] == 0 else np.zeros_like(c["meas"]), Ey_measured=c["meas"] if c["arm"] == 1 else np.zeros_like(c["meas"])) for c in ch]
    phase, info = vb.self_calibrate_phase(pxs, rhat, E1, E2, chans, block=20)
    print(f"  self-cal at this geometry: {np.isfinite(phase).sum()} integrations, {info['n_iter']} iters  [{time.time()-t0:.0f} s]", flush=True)
    _, hp_pred = hfss_per_tooth(geo(p1, T1, 0.5), return_pred=True)
    sol_r, hf_r = [], []
    for k, c in enumerate(ch):
        rot = np.exp(-1j * np.where(np.isfinite(phase), phase, 0)); Ex, Ey = c["Ex"] * rot, c["Ey"] * rot
        v = c["v"] & np.isfinite(phase); trn = v & (split == 0)
        g = vb.fit_closure_g(pxs, rhat, E1, E2, Ex, Ey, trn); sol = vb.solve_sampled_beam(pxs, rhat, E1, E2, Ex, Ey, trn, g=g)
        lut = {p: i for i, p in enumerate(sol["pix"])}; te = np.flatnonzero(v & c["meas"] & (split == 2))
        pr = np.array([np.nan if lut.get(pxs[t]) is None else (abs(E1[t] @ sol["u"][lut[pxs[t]]]) ** 2 if c["arm"] == 0 else abs(g * (E2[t] @ sol["u"][lut[pxs[t]]])) ** 2) for t in te])
        ok = np.isfinite(pr); D = np.abs(Ex if c["arm"] == 0 else Ey)[te][ok] ** 2
        sol_r.append(np.linalg.norm(D - pr[ok]) / np.linalg.norm(D)); hf_r.append(np.linalg.norm(D - hp_pred[k][te][ok]) / np.linalg.norm(D))
    print(f"  free-beam check (test samples the solver can predict): solver {np.median(sol_r):.4f}  vs HFSS {np.median(hf_r):.4f}  (solver better on {np.sum(np.array(sol_r) < np.array(hf_r))}/29)", flush=True)
    res[label] = dict(steps={k: v.tolist() for k, v in steps.items()}, dalpha=dalpha.tolist(), solver=sol_r, hfss_same=hf_r)
json.dump(res, open(S + "/step4.json", "w"), indent=1)
