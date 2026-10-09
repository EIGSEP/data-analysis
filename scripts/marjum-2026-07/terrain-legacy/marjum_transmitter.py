"""Assemble new transmitter-conditioned cameras and fit a fixed-camera transmitter."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from marjum_camera import project, rays
from marjum_guided import lens_metadata
from marjum_mcmc import digest


def assemble(solution_file="transmitter_camera_solutions_v1.json",
             output="cv_transmitter_v1"):
    solution_path=Path(solution_file);spec=json.loads(solution_path.read_text())
    source=Path(spec["source"]);state=dict(np.load(source));base_n=len(state["keys"])
    out=Path(output);out.mkdir(exist_ok=True)
    target=out/"fit_extended.npz"
    if target.exists():raise FileExistsError(target)
    new_keys=list(spec["cameras"]);metadata=lens_metadata(new_keys)
    group_k={g:np.median(state["distortion"][state["groups"]==g],axis=0)
             for g in np.unique(state["groups"])}
    new_cameras=np.array([spec["cameras"][k] for k in new_keys],float)
    new_groups=np.array([int(metadata[k]["group"]=="ultrawide") for k in new_keys])
    new_distortion=np.array([group_k[g] for g in new_groups])
    shapes=[]
    for key in new_keys:
        with np.load(f"cv_features/sift_{key}.npz") as feature:shapes.append(feature["shape"])
    result=dict(state)
    result.update(keys=np.r_[state["keys"],new_keys],
                  cameras=np.vstack([state["cameras"],new_cameras]),
                  distortion=np.vstack([state["distortion"],new_distortion]),
                  groups=np.r_[state["groups"],new_groups],
                  shapes=np.vstack([state["shapes"],shapes]))
    np.savez_compressed(target,**result)
    for field in ("keys","cameras","distortion","groups","shapes"):
        if not np.array_equal(state[field],result[field][:base_n]):
            raise RuntimeError(f"Existing camera field changed: {field}")
    manifest=dict(source=str(source),solution_file=str(solution_path),new_keys=new_keys,
                  existing_camera_count=base_n,existing_cameras_byte_identical=True,
                  transmitter_conditioned=bool(spec["transmitter_conditioned"]),
                  rejected=spec["rejected"],input_sha256={str(source):digest(source),str(solution_path):digest(solution_path)})
    (out/"camera_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return result


def fit_transmitter(camera_fit="cv_transmitter_v1/fit_extended.npz",
                    output="cv_transmitter_v1/fit_transmitter.npz",
                    meta_file="meta.json",conditioned=("2159","2171","2199","2203"),
                    independent_keys=("2210","2211")):
    camera_path=Path(camera_fit);state=dict(np.load(camera_path));keys=[str(k) for k in state["keys"]]
    meta=json.loads(Path(meta_file).read_text());used=[k for k in keys if "transmitter_px" in meta.get(k,{})]
    index=np.array([keys.index(k) for k in used]);observed=np.array([meta[k]["transmitter_px"] for k in used])
    # Initialize from the two pre-existing camera rays only. These are the only
    # transmitter observations whose camera poses were not conditioned on it.
    independent=[j for j,k in enumerate(used) if k in independent_keys and k not in conditioned]
    if len(independent)<2:
        raise ValueError('At least two independently registered camera rays are required')
    A=np.zeros((3,3));b=np.zeros(3)
    for j in independent:
        i=index[j];direction=rays(state["cameras"][i],state["shapes"][i],
                                  [observed[j]],state["distortion"][i])[0]
        normal=np.eye(3)-np.outer(direction,direction);A+=normal;b+=normal@state["cameras"][i,:3]
    if np.linalg.cond(A)>1e8:
        raise ValueError('Independent transmitter rays have degenerate geometry')
    start=np.linalg.solve(A,b)
    def residual(position):
        value=[]
        for j in independent:
            i,xy=index[j],observed[j]
            predicted,depth=project(state["cameras"][i],state["shapes"][i],position,state["distortion"][i])
            value.extend((predicted[0]-xy)/3.);value.append(min(float(depth[0])-1.,0.)/.1)
        return np.asarray(value)
    fit=least_squares(residual,start,loss="soft_l1",f_scale=2.,x_scale=10.,max_nfev=300)
    rows=[]
    for i,key,xy in zip(index,used,observed):
        predicted,depth=project(state["cameras"][i],state["shapes"][i],fit.x,state["distortion"][i])
        direction=rays(state["cameras"][i],state["shapes"][i],[xy],state["distortion"][i])[0]
        delta=fit.x-state["cameras"][i,:3];miss=np.linalg.norm(delta-direction*np.dot(delta,direction))
        rows.append(dict(key=key,conditioned=key in conditioned,used_in_fit=key in [used[j] for j in independent],error_px=float(np.linalg.norm(predicted[0]-xy)),ray_miss_m=float(miss),depth_m=float(depth[0])))
    result=dict(state);result["transmitter"]=fit.x
    result["transmitter_fit_keys"]=np.array([used[j] for j in independent])
    result["transmitter_conditioned_keys"]=np.array(conditioned)
    np.savez_compressed(output,**result)
    report=dict(camera_fit=str(camera_path),transmitter=fit.x.tolist(),initializer=start.tolist(),
                camera_parameters_fixed=True,used_keys=used,conditioned_keys=list(conditioned),
                independent_keys=[used[j] for j in independent],
                diagnostic_only_keys=[k for k in used if k not in [used[j] for j in independent]],
                note="Only explicitly independently registered cameras enter this fit. Transmitter-conditioned and provisional manual-camera rays are diagnostics. Camera uncertainty is not a posterior covariance.",
                residuals=rows,optimizer=dict(cost=float(fit.cost),nfev=int(fit.nfev),status=int(fit.status),message=fit.message),
                input_sha256={str(camera_path):digest(camera_path),str(meta_file):digest(Path(meta_file))})
    Path(output).with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def assemble_refined(output='cv_transmitter_v3'):
    """Append the best 2198 candidate to the terrain-refined 2172 state."""
    out=Path(output);out.mkdir(exist_ok=False)
    source=Path('cv_transmitter_v2/fit_extended.npz')
    report_path=Path('cv_transmitter_2198_v2/report.json')
    state=dict(np.load(source));report=json.loads(report_path.read_text())
    best=min(report['candidates'],key=lambda r:r['cost'])
    if '2198' in state['keys']:raise ValueError('2198 is already present')
    state['keys']=np.concatenate([state['keys'],['2198']])
    state['cameras']=np.vstack([state['cameras'],best['camera']])
    state['distortion']=np.vstack([state['distortion'],best['distortion']])
    state['groups']=np.r_[state['groups'],1]
    with np.load('cv_features/sift_2198.npz') as f:
        state['shapes']=np.vstack([state['shapes'],f['shape']])
    conditioned=('2159','2171','2199','2203','2198')
    state['camera_provenance']=np.array([
        'transmitter-conditioned' if k in conditioned else
        'manual-seed terrain-only refinement' if k=='2172' else
        'established fixed camera' for k in state['keys']])
    np.savez_compressed(out/'fit_extended.npz',**state)
    return fit_transmitter(out/'fit_extended.npz',out/'fit_transmitter.npz',conditioned=conditioned)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",default="cv_transmitter_v1")
    parser.add_argument("--solutions",default="transmitter_camera_solutions_v1.json");args=parser.parse_args()
    assemble(args.solutions,args.output);fit_transmitter(f"{args.output}/fit_extended.npz",f"{args.output}/fit_transmitter.npz")
