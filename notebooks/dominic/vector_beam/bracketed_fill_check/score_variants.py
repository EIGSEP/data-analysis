import numpy as np, sys, json
from eigsep_data import vector_beam as vb
from eigsep_data.rfi import build_frequency_trust_mask
z = np.load(sys.argv[1]); out = sys.argv[2]
sel = z["sel"]; geo = (z["pxs"], z["r_hat"], z["E1v"], z["E2v"])
nc = z["Ex_all"].shape[1]
fm = 171.875 + (np.arange(nc) - 60) * 1.953125
trust = build_frequency_trust_mask(fm, min_freq=87.0, fm_low=87.0, fm_high=108.0, max_freq=230.0)
def unbracketed(M):
    anchor = M & trust[None, :]
    lo = np.zeros_like(M); hi = np.zeros_like(M)
    lo[:, 1:] = anchor[:, :-1]; hi[:, :-1] = anchor[:, 1:]
    return ~M & ~(lo & hi)
Ub = {a: unbracketed(z[a + "_meas"]) for a in ("Ex", "Ey")}
res = {}; keep = {}
for variant in ("baseline", "bracketed"):
    rows = []
    for k, d in enumerate(z["d_idx"]):
        Ex = z["Ex_all"][:, d].copy(); Ey = z["Ey_all"][:, d].copy()
        if variant == "bracketed":
            Ex[Ub["Ex"][:, d]] = np.nan; Ey[Ub["Ey"][:, d]] = np.nan
        valid = z["valid"][k] & sel & np.isfinite(Ex) & np.isfinite(Ey)
        row = dict(d_idx=int(d), freq=float(z["freq"][k]), n_valid=int(valid.sum()))
        if valid.sum() < 50:
            rows.append(row); print(variant, row, flush=True); continue
        args = (*geo, Ex, Ey, valid)
        g = vb.fit_closure_g(*args); sol = vb.solve_sampled_beam(*args, g=g)
        truth = z["beam"][z["h_idx"][k]][:, sol["pix"]].T
        xyz = vb.compare_to_truth(sol["u"], truth); loo = vb.leave_one_out(*args, g)
        row.update(g_mag=abs(g), g_phase=float(np.angle(g, deg=True)), n_pix=xyz["n"],
                   **{c: xyz[c] for c in ("power", "x", "y", "z")},
                   loo_n=len(loo["power_pred"]),
                   loo_power=float(np.corrcoef(loo["power_pred"], loo["power_true"])[0, 1]))
        if d in (77, 82, 85, 87):
            keep[f"{variant}_d{d}_pred"] = loo["power_pred"]; keep[f"{variant}_d{d}_true"] = loo["power_true"]
            keep[f"{variant}_d{d}_pix"] = sol["pix"]; keep[f"{variant}_d{d}_u"] = sol["u"]
            keep[f"{variant}_d{d}_truth"] = truth
        rows.append(row)
        print(variant, f"d{d} n={row['n_valid']} power {row['power']:.3f} y {row['y']:.3f} z {row['z']:.3f} loo {row['loo_power']:.3f}", flush=True)
    res[variant] = rows
json.dump(res, open(out + ".json", "w"), indent=1); np.savez(out + "_loo.npz", **keep)
