"""Independent raster and withheld-track checks for distortion-aware states."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from marjum_bundle import Terrain, build_tracks
from marjum_camera import rays,project,radial_support,epipolar_error
from marjum_guided import triangulate


def validate(output='cv_distortion_guided_v2',stage='refined'):
    from eigsep_terrain.marjum_dem import MarjumDEM
    from eigsep_terrain.imageio import load_image
    from eigsep_terrain.img import HorizonImage
    from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
    from marjum_mcmc import digest
    root=Path(output);out=root/f'validation_{stage}'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    manifest=json.loads((root/'manifest.json').read_text())
    changed=[str(k) for k,v in manifest['input_sha256'].items() if not Path(k).exists() or digest(k)!=v]
    if changed:raise ValueError(f'Changed experiment inputs: {changed}')
    if stage=='joint':
        provenance=json.loads((root/'joint_manifest.json').read_text())
        if any(digest(k)!=v for k,v in provenance['input_sha256'].items()):
            raise ValueError('Changed joint-stage inputs')
    states={label:dict(np.load(root/f'{label}.npz')) for label in ['initial',stage]}
    keys=list(states[stage]['keys'].astype(str));features={}
    for key in keys:
        with np.load(f'cv_features/sift_{key}.npz') as f:features[key]=f['xy']
    with np.load(root/'holdout.npz') as f:pairs=[(*name.split('_'),f[name]) for name in f.files]
    tracks,_=build_tracks(pairs)
    terrain=Terrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
    report={'stage':stage,'withheld_three_view':{},'images':[]}
    # A stricter post-fit audit removes proximity to ANY fitted keypoint, using
    # positions only (never errors) and the same subset for every comparison.
    from scipy.spatial import cKDTree
    all_states={name:dict(np.load(root/f'{name}.npz')) for name in ['initial','pinhole','radial','guided','refined']}
    if stage not in all_states:all_states[stage]=states[stage]
    used={k:cKDTree(np.concatenate([s['obs_xy'][s['obs_cam']==i] for s in all_states.values()])) for i,k in enumerate(keys)}
    strict=[]
    for a,b,ids in pairs:
        good=(used[a].query(features[a][ids[:,0]])[0]>64)&(used[b].query(features[b][ids[:,1]])[0]>64)
        if good.sum()>=5:strict.append((a,b,ids[good]))
    report['strict_64px_holdout']={}
    core=set(['2213','2215','2217','2218','2220','2222','2223','2224','2226','2231','2234','2237','2239'])
    for label,s in all_states.items():
        rows=[]
        for a,b,ids in strict:
            ia,ib=keys.index(a),keys.index(b)
            e=epipolar_error(s['cameras'][ia],s['shapes'][ia],s['distortion'][ia],s['cameras'][ib],s['shapes'][ib],s['distortion'][ib],features[a][ids[:,0]],features[b][ids[:,1]])
            if np.isfinite(e).any():rows.append(dict(a=a,b=b,count=len(ids),median_px=float(np.nanmedian(e))))
        report['strict_64px_holdout'][label]=dict(pairs=rows,pair_count=len(rows),
             median_px=float(np.median([r['median_px'] for r in rows])) if rows else None,
             core_median_px=float(np.median([r['median_px'] for r in rows if r['a'] in core and r['b'] in core])) if any(r['a'] in core and r['b'] in core for r in rows) else None)
    for label,s in states.items():
        errors=[];invalid=0;attempted=0
        for track in tracks:
            if len(track)<3:continue
            ci=[keys.index(k) for k,fid in track];xy=np.array([features[k][fid] for k,fid in track])
            for j,i in enumerate(ci):
                attempted+=1;use=[a for a in range(len(ci)) if a!=j]
                point=triangulate(s['cameras'],s['shapes'],s['distortion'],[ci[a] for a in use],xy[use])
                if point is None:invalid+=1;continue
                pred,depth=project(s['cameras'][i],s['shapes'][i],point,s['distortion'][i])
                if depth[0]<=1:invalid+=1;continue
                errors.append(float(np.linalg.norm(pred[0]-xy[j])))
        report['withheld_three_view'][label]=dict(attempted=attempted,invalid=invalid,valid=len(errors),
                      median_px=float(np.median(errors)) if errors else None,p90_px=float(np.quantile(errors,.9)) if errors else None)
    meta=json.loads(Path('meta.json').read_text())
    for i,key in enumerate(keys):
        rgb=np.flipud(load_image(f'marjum-2026-07/IMG_{key}.HEIC'))
        h,w=rgb.shape[:2];step=12;small=rgb[::step,::step];del rgb
        with np.load(f'img_seg_IMG_{key}.npz') as seg:sky=np.flipud(seg['skymask'])[::step,::step].astype(bool)
        rr,cc=np.meshgrid(np.arange(0,h,step),np.arange(0,w,step),indexing='ij');xy=np.c_[cc.ravel(),rr.ravel()]
        helper=HorizonImage.__new__(HorizonImage)
        actual=helper._raster_boundary(sky);actual_valid=np.isfinite(actual)&(actual<sky.shape[0]-1)
        row={'key':key};fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,(label,s) in zip(axes,states.items()):
            p=s['cameras'][i];k=s['distortion'][i]
            direction=rays(p,(h,w),xy,k)
            distance=ray_distance_coarse_to_fine_numba(terrain.e,terrain.n,terrain.dem.data,np.asarray(p[:3],np.float32),np.asarray(direction.T,np.float32))
            model=np.isnan(distance).reshape(rr.shape)
            pred=helper._raster_boundary(model);valid=actual_valid&np.isfinite(pred)&(pred<sky.shape[0]-1)
            error=(pred[valid]-actual[valid])*step
            row[label]=dict(horizon_rms_px=float(np.sqrt(np.mean(error**2))),matched_columns=int(valid.sum()),
                            missing_columns=int((actual_valid&~valid).sum()),actual_columns=int(actual_valid.sum()),
                            radial_min_derivative=float(radial_support(p,(h,w),k).min()))
            ax.imshow(small,origin='lower',extent=(0,w,0,h))
            ax.contour(cc,rr,sky.astype(float),levels=[.5],colors='lime',linewidths=.7)
            ax.contour(cc,rr,model.astype(float),levels=[.5],colors='red',linewidths=.7)
            if 'ant_px' in meta.get(key,{}):
                pick=np.array(meta[key]['ant_px']);ap=project(p,(h,w),s['antenna'],k)[0][0]
                ax.plot(*pick,'m+',ms=9);ax.plot(*ap,'cx',ms=8)
                row[label]['antenna_px']=float(np.linalg.norm(ap-pick))
            ax.set(xlim=(0,w),ylim=(0,h),title=f'{key}: {label}',xticks=[],yticks=[])
        fig.suptitle('HELD-OUT horizon: green observed, red DEM. Antenna: magenta pick, cyan triangulated prediction.')
        fig.tight_layout();fig.savefig(out/f'{key}.png',dpi=120);plt.close(fig)
        report['images'].append(row);print(row,flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_distortion_guided_v2');p.add_argument('--stage',default='refined')
    validate(**vars(p.parse_args()))
