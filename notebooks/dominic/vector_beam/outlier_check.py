import numpy as np, run_phase2 as R, pipeline, geometry_v0013
from scipy.stats import spearmanr
from eigsep_data.beam_mapping import vector_beam as vb
inp = pipeline.load(); cache=np.load(pipeline.CACHE, allow_pickle=True)
j = R.join_pointing(inp["times"], str(cache["data_dir"]))
ok = np.isfinite(j.az_deg.to_numpy())
az, el = np.where(ok, j.az_deg, 0.0), np.where(ok, j.el_deg, 0.0)
sel = ok & (j.quality=="ok").to_numpy(); idx=np.flatnonzero(sel); g_=np.zeros(len(j),bool); g_[idx]=R.reversal_glitches(el[idx]); sel&=~g_
nside=int(np.sqrt(inp["beam"].shape[-1]/12))
geoB = vb.fields_from_model(geometry_v0013.model(), geometry_v0013.PARAMS, az, el, nside)
for d in (84, 85, 86):
    ch = next(c for c in inp["channels"] if c["d_idx"]==d)
    args=(*geoB, ch["Ex"], ch["Ey"], ch["valid"]&sel); g=vb.fit_closure_g(*args); o=vb.leave_one_out(*args, g)
    pp, pt = o["power_pred"], o["power_true"]; err=np.abs(pp-pt)
    k=np.argsort(err)[::-1]
    keep=np.ones(len(pp),bool); keep[k[:max(1,len(pp)//100)]]=False
    print(f"d{d}: n={len(pp)} pearson={np.corrcoef(pp,pt)[0,1]:.3f} spearman={spearmanr(pp,pt)[0]:.3f} pearson_drop_top1%={np.corrcoef(pp[keep],pt[keep])[0,1]:.3f}")
    print(f"   worst 3 pred/true: {[(f'{pp[i]:.3g}', f'{pt[i]:.3g}') for i in k[:3]]}, median true {np.median(pt):.3g}")
