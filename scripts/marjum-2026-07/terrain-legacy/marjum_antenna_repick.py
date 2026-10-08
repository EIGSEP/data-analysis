"""Robust camera-fixed antenna triangulation from distortion-aware pixel picks."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from marjum_camera import project, rays
from marjum_mcmc import digest


def solve_position(state, observations, start, indices=None, sigma_px=3.):
    keys=[str(k) for k in state["keys"]]
    use=np.arange(len(keys)) if indices is None else np.asarray(indices,int)
    def residual(position):
        value=[]
        for i in use:
            predicted,depth=project(state["cameras"][i],state["shapes"][i],
                                    position,state["distortion"][i])
            value.extend((predicted[0]-observations[i])/sigma_px)
            value.append(min(float(depth[0])-1.,0.)/.1)
        return np.asarray(value)
    return least_squares(residual,np.asarray(start,float),loss="soft_l1",f_scale=2.,
                         x_scale=10.,max_nfev=300,ftol=1e-11,xtol=1e-12,gtol=1e-11)


def diagnostics(state,observations,position):
    rows=[]
    for i,key in enumerate(state["keys"].astype(str)):
        predicted,depth=project(state["cameras"][i],state["shapes"][i],
                                position,state["distortion"][i])
        direction=rays(state["cameras"][i],state["shapes"][i],
                       [observations[i]],state["distortion"][i])[0]
        offset=position-state["cameras"][i,:3]
        miss=np.linalg.norm(offset-direction*np.dot(offset,direction))
        rows.append(dict(key=key,error_px=float(np.linalg.norm(predicted[0]-observations[i])),
                         ray_miss_m=float(miss),depth_m=float(depth[0])))
    return rows


def run(source,output="cv_antenna_repick_v1",meta_file="meta.json",sigma_px=3.):
    source_path=Path(source);meta_path=Path(meta_file);out=Path(output);out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError("Use a new output directory")
    state=dict(np.load(source_path));keys=[str(k) for k in state["keys"]]
    meta=json.loads(meta_path.read_text())
    missing=[k for k in keys if "ant_px" not in meta.get(k,{})]
    if missing:raise KeyError(f"Missing ant_px for {missing}")
    observed=np.array([meta[k]["ant_px"] for k in keys],float)
    initial=np.asarray(state["antenna"],float)
    fit=solve_position(state,observed,initial,sigma_px=sigma_px)
    before=diagnostics(state,observed,initial);after=diagnostics(state,observed,fit.x)
    loo=[]
    for held in range(len(keys)):
        use=np.delete(np.arange(len(keys)),held)
        trial=solve_position(state,observed,fit.x,use,sigma_px)
        row=diagnostics({k:(v[[held]] if k in ("keys","cameras","distortion","shapes","groups") else v)
                         for k,v in state.items()},observed[[held]],trial.x)[0]
        loo.append({**row,"fit_position":trial.x.tolist()})
    result=dict(state);result["antenna"]=fit.x
    np.savez_compressed(out/"fit_antenna.npz",**result)
    errors=np.array([r["error_px"] for r in after]);held=np.array([r["error_px"] for r in loo])
    report=dict(source=str(source_path),keys=keys,sigma_px=sigma_px,
                camera_parameters_fixed=True,distortion_used=True,
                initial_antenna=initial.tolist(),antenna=fit.x.tolist(),
                shift_m=(fit.x-initial).tolist(),optimizer=dict(cost=float(fit.cost),nfev=int(fit.nfev),status=int(fit.status),message=fit.message),
                residual_median_px=float(np.median(errors)),residual_rms_px=float(np.sqrt(np.mean(errors**2))),residual_max_px=float(errors.max()),
                loo_median_px=float(np.median(held)),loo_rms_px=float(np.sqrt(np.mean(held**2))),loo_max_px=float(held.max()),
                before=before,after=after,leave_one_out=loo,
                input_sha256={str(source_path):digest(source_path),str(meta_path):digest(meta_path)})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source",required=True)
    parser.add_argument("--output",default="cv_antenna_repick_v1");parser.add_argument("--meta-file",default="meta.json")
    parser.add_argument("--sigma-px",type=float,default=3.);run(**vars(parser.parse_args()))
