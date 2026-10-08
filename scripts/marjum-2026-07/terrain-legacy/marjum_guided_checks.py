"""Common-subset multi-view checks and a conservative release status."""
from pathlib import Path
import json
import argparse
import numpy as np
from marjum_bundle import build_tracks
from marjum_camera import project
from marjum_guided import triangulate
from marjum_mcmc import digest


def check(output='cv_distortion_guided_v2',overwrite=False):
    root=Path(output)
    destination=root/'additional_checks.json'
    if destination.exists() and not overwrite:raise FileExistsError(destination)
    states={name:dict(np.load(root/f'{name}.npz')) for name in ['initial','guided','refined','joint']}
    keys=list(states['initial']['keys'].astype(str))
    features={}
    for key in keys:
        with np.load(f'cv_features/sift_{key}.npz') as f:features[key]=f['xy']
    with np.load(root/'holdout.npz') as f:tracks,_=build_tracks([(*k.split('_'),f[k]) for k in f.files])
    errors={name:[] for name in states};core=[]
    for track in tracks:
        if len(track)<3:continue
        ci=[keys.index(k) for k,fid in track];xy=np.array([features[k][fid] for k,fid in track])
        in_core=not any(k in ['2216','2221','2235','2238'] for k,fid in track)
        for j,i in enumerate(ci):
            core.append(in_core);use=[v for v in range(len(ci)) if v!=j]
            for name,s in states.items():
                point=triangulate(s['cameras'],s['shapes'],s['distortion'],[ci[v] for v in use],xy[use])
                error=np.nan
                if point is not None:
                    pred=[project(s['cameras'][v],s['shapes'][v],point,s['distortion'][v]) for v in ci]
                    if all(d[0]>1 for _,d in pred):error=float(np.linalg.norm(pred[j][0][0]-xy[j]))
                errors[name].append(error)
    arrays={name:np.array(v) for name,v in errors.items()};common=np.logical_and.reduce([np.isfinite(v) for v in arrays.values()]);core=np.array(core)
    result={'common_valid_count':int(common.sum()),'attempted':len(core),'core_common_valid_count':int((common&core).sum()),'stages':{}}
    for name,s in states.items():
        v=arrays[name];mask=common&core
        k=s['distortion'];near=np.any(abs(k)>=np.array([.15,.04])*.98,axis=1)
        result['stages'][name]=dict(common_three_view_median_px=float(np.median(v[common])),
             core_common_three_view_median_px=float(np.median(v[mask])) if mask.any() else None,
             invalid=int((~np.isfinite(v)).sum()),distortion_near_bound_images=[keys[i] for i in np.flatnonzero(near)])
    improved=result['stages']['guided']['common_three_view_median_px']<result['stages']['initial']['common_three_view_median_px']
    raster=json.loads((root/'validation_joint'/'report.json').read_text())
    medians={name:float(np.median([v[name]['horizon_rms_px'] for v in raster['images']])) for name in ['initial','joint']}
    result['raster_horizon_median_px']=medians
    result['release_status']=dict(relative_alignment_improved=bool(improved),production_fit_replaced=False,mcmc_ready=False,
        absolute_horizon_regressed=medians['joint']>medians['initial'],
        distortion_near_bound=bool(result['stages']['joint']['distortion_near_bound_images']),
        reason='Experimental initialization only; lens calibration and absolute terrain registration require validation before posterior sampling.')
    result['source_sha256']=digest(__file__)
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_distortion_guided_v2');p.add_argument('--overwrite',action='store_true');check(**vars(p.parse_args()))
