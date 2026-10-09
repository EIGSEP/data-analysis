"""Score position candidates with the library's tree-adjusted sky probabilities.

This is a model cross-check, not independent validation: these segmentation
maps also supplied the boundary samples. Reserved image correspondences remain
the independent geometric check. No segmentation inference is rerun.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from marjum_camera import rays,project
from marjum_mcmc import digest


def sky_nll(probability,model_sky,effective_samples):
    """Library-style Bernoulli score, normalized for correlated samples."""
    p=np.asarray(probability).clip(1e-3,1-1e-3)
    return float(-np.where(model_sky,np.log(p),np.log1p(-p)).mean()*effective_samples)


def correspondence_checks(states):
    from scipy.spatial import cKDTree
    from marjum_camera import epipolar_error
    from marjum_bundle import build_tracks
    from marjum_guided import triangulate
    keys=list(next(iter(states.values()))['keys'].astype(str));features={}
    for key in keys:
        with np.load(f'cv_features/sift_{key}.npz') as z:features[key]=z['xy']
    root=Path('cv_distortion_guided_v2')
    with np.load(root/'holdout.npz') as z:pairs=[(*k.split('_'),z[k]) for k in z.files]
    originals=[dict(np.load(root/f'{s}.npz')) for s in ['initial','pinhole','radial','guided','refined','joint']]
    trees={k:cKDTree(np.concatenate([s['obs_xy'][s['obs_cam']==i] for s in originals])) for i,k in enumerate(keys)}
    strict=[]
    for a,b,ids in pairs:
        keep=(trees[a].query(features[a][ids[:,0]])[0]>64)&(trees[b].query(features[b][ids[:,1]])[0]>64)
        if keep.sum()>=5:strict.append((a,b,ids[keep]))
    result={name:{} for name in states}
    for name,s in states.items():
        errors=[]
        for a,b,ids in strict:
            i,j=keys.index(a),keys.index(b)
            err=epipolar_error(s['cameras'][i],s['shapes'][i],s['distortion'][i],s['cameras'][j],s['shapes'][j],s['distortion'][j],features[a][ids[:,0]],features[b][ids[:,1]])
            errors.append(float(np.median(err)))
        result[name].update(strict_pair_count=len(strict),strict_pair_median_px=float(np.median(errors)))
    tracks,_=build_tracks(pairs);errors={name:[] for name in states}
    for track in tracks:
        if len(track)<3:continue
        ci=[keys.index(k) for k,fid in track];xy=np.array([features[k][fid] for k,fid in track])
        for j,i in enumerate(ci):
            use=[v for v in range(len(ci)) if v!=j]
            for name,s in states.items():
                point=triangulate(s['cameras'],s['shapes'],s['distortion'],[ci[v] for v in use],xy[use]);error=np.nan
                if point is not None:
                    pred=[project(s['cameras'][v],s['shapes'][v],point,s['distortion'][v]) for v in ci]
                    if all(d[0]>1 for _,d in pred):error=float(np.linalg.norm(pred[j][0][0]-xy[j]))
                errors[name].append(error)
    arrays={name:np.array(v) for name,v in errors.items()};common=np.logical_and.reduce([np.isfinite(v) for v in arrays.values()])
    for name,v in arrays.items():result[name].update(common_three_view_count=int(common.sum()),common_three_view_median_px=float(np.median(v[common])),invalid=int((~np.isfinite(v)).sum()))
    return result


def validate(output='cv_position_absolute',stages=('baseline','positions','orientations','poses','alternative','focal')):
    from eigsep_terrain.img import HorizonImage
    from eigsep_terrain.marjum_dem import MarjumDEM
    from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
    import eigsep_terrain.img as image_module
    import eigsep_terrain.utils as utils_module
    root=Path(output);target=root/'probability_validation.json'
    if target.exists():raise FileExistsError(target)
    states={s:dict(np.load(root/f'{s}.npz')) for s in stages}
    keys=list(states[stages[0]]['keys'].astype(str))
    dem=MarjumDEM(cache_file='marjum_dem_sw.npz');e,n=dem.get_en()
    meta=json.loads(Path('meta.json').read_text())
    report=dict(note=__doc__,images=[],input_sha256={str(p):digest(p) for p in
        [Path(__file__),Path(image_module.__file__),Path(utils_module.__file__)]+[root/f'{s}.npz' for s in stages]})
    report['correspondence_checks']=correspondence_checks(states)
    for i,key in enumerate(keys):
        seg=Path(f'img_seg_IMG_{key}.npz');report['input_sha256'][str(seg)]=digest(seg)
        image=HorizonImage(f'marjum-2026-07/IMG_{key}.HEIC',meta=meta)
        # A fixed regular grid, identical for every candidate. Native pixel
        # spacing is well below the library's 100-pixel smoothing scale.
        rr,cc=np.mgrid[6:image.npix_y:12,6:image.npix_x:12]
        use=image.horizon_mask[rr,cc];rr,cc=rr[use],cc[use]
        xy=np.c_[cc,rr];p=image.psky[rr,cc].clip(1e-3,1-1e-3)
        neff=float(np.clip(np.ptp(cc)/image.px_smooth,1,len(cc)))
        row=dict(key=key,samples=len(p),effective_samples=neff,stages={})
        for label,s in states.items():
            cam=s['cameras'][i];k=s['distortion'][i]
            direction=rays(cam,s['shapes'][i],xy,k)
            distance=ray_distance_coarse_to_fine_numba(e,n,dem.data,np.asarray(cam[:3],np.float32),np.asarray(direction.T,np.float32))
            sky=np.isnan(distance)
            nll=sky_nll(p,sky,neff)
            q,depth=project(cam,s['shapes'][i],s['antenna'],k)
            row['stages'][label]=dict(sky_nll=float(nll),antenna_px=float(np.linalg.norm(q[0]-meta[key]['ant_px'])),antenna_depth_m=float(depth[0]))
        report['images'].append(row);print(row,flush=True)
        del image
    report['total_sky_nll']={s:sum(r['stages'][s]['sky_nll'] for r in report['images']) for s in stages}
    target.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_position_absolute');p.add_argument('--stages',nargs='+',default=['baseline','positions','orientations','poses','alternative','focal'])
    validate(**vars(p.parse_args()))
