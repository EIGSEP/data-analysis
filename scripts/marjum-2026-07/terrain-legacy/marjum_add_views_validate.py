"""Full-raster and invariance checks for appended Marjum camera views."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from marjum_camera import rays,radial_support
from marjum_mcmc import digest


def validate(output='cv_position_add_views_v2',source='cv_position_absolute/focal.npz',step=12,overwrite=False,targets=None):
    from eigsep_terrain.img import HorizonImage
    from eigsep_terrain.marjum_dem import MarjumDEM
    from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
    root=Path(output);out=root/'validation';out.mkdir(exist_ok=overwrite)
    if (out/'report.json').exists() and not overwrite:raise FileExistsError(out/'report.json')
    fit_path=root/'fit_extended.npz'
    if not fit_path.exists():fit_path=root/'fit_transmitter.npz'
    old=dict(np.load(source));new=dict(np.load(fit_path));nref=len(old['keys'])
    unchanged=np.array([str(k) not in (targets or []) for k in old['keys']])
    # Adding views must be a pure append operation for established geometry.
    for name in ['keys','cameras','distortion','groups','shapes','antenna','points','obs_cam','obs_point','obs_xy','obs_fid']:
        a=old[name];b=new[name] if name in ['antenna','points','obs_cam','obs_point','obs_xy','obs_fid'] else new[name][:nref]
        if targets and name in ['cameras','distortion','groups','shapes']:
            a=a[unchanged];b=b[unchanged]
        if not np.array_equal(a,b):raise ValueError(f'established fit changed: {name}')
    dem=MarjumDEM(cache_file='marjum_dem_sw.npz');e,n=dem.get_en();rows=[]
    selected=[i for i,k in enumerate(new['keys']) if (str(k) in targets if targets else i>=nref)]
    for i in selected:
        key=new['keys'][i]
        # Use HorizonImage defaults, matching marjum_position_validate.py's
        # established-view probability protocol exactly.
        key=str(key);image=HorizonImage(f'marjum-2026-07/IMG_{key}.HEIC')
        h,w=new['shapes'][i];rr,cc=np.mgrid[0:h:step,0:w:step];xy=np.c_[cc.ravel(),rr.ravel()]
        direction=rays(new['cameras'][i],(h,w),xy,new['distortion'][i])
        distance=ray_distance_coarse_to_fine_numba(e,n,dem.data,np.asarray(new['cameras'][i,:3],np.float32),np.asarray(direction.T,np.float32))
        model=np.isnan(distance).reshape(rr.shape);actual=image.sky_mask[::step,::step]
        observed=image._raster_boundary(actual);predicted=image._raster_boundary(model)
        valid=np.isfinite(observed)&np.isfinite(predicted);error=(predicted[valid]-observed[valid])*step
        near=image.horizon_mask[::step,::step];probability=image.psky[::step,::step][near].clip(1e-3,1-1e-3)
        model_near=model[near];columns=cc[near];neff=float(np.clip(np.ptp(columns)/image.px_smooth,1,len(columns)))
        row=dict(key=key,raster_horizon_rms_px=float(np.sqrt(np.mean(error**2))),raster_horizon_median_abs_px=float(np.median(abs(error))),
            matched_columns=int(valid.sum()),radial_min_derivative=float(radial_support(new['cameras'][i],(h,w),new['distortion'][i]).min()),
            camera_clearance_m=float(new['cameras'][i,2]-np.asarray(dem.interp_alt(np.array([new['cameras'][i,0]]),np.array([new['cameras'][i,1]])))[0]),
            tree_adjusted_sky_nll=float(-np.where(model_near,np.log(probability),np.log1p(-probability)).mean()*neff),
            tree_adjusted_mean_cross_entropy=float(-np.where(model_near,np.log(probability),np.log1p(-probability)).mean()),
            probability_samples=int(len(probability)),probability_effective_samples=neff)
        rows.append(row);print(row,flush=True)
        fig,ax=plt.subplots(figsize=(8,6));ax.imshow(image.img[::step,::step],origin='lower')
        ax.contour(actual.astype(float),levels=[.5],colors='lime',linewidths=.7)
        ax.contour(model.astype(float),levels=[.5],colors='red',linewidths=.7)
        ax.set(title=f'{key}: green segmentation, red distortion-aware DEM ({row["raster_horizon_rms_px"]:.1f}px RMS)',xticks=[],yticks=[])
        fig.tight_layout();fig.savefig(out/f'{key}.png',dpi=150);plt.close(fig);del image
    positions={str(new['keys'][i]):new['cameras'][i,:3].tolist() for i in selected}
    separations={f'{a}_{b}':float(np.linalg.norm(np.array(positions[a])-np.array(positions[b]))) for j,a in enumerate(positions) for b in list(positions)[j+1:]}
    report=dict(established_geometry_unchanged=True,images=rows,positions_grid_m=positions,camera_separations_m=separations,
        allowed_refined_keys=targets or [],
        input_sha256={str(p):digest(p) for p in [Path(__file__),Path(source),fit_path]+[Path(f'img_seg_IMG_{k}.npz') for k in positions]})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_position_add_views_v2');p.add_argument('--source',default='cv_position_absolute/focal.npz');p.add_argument('--step',type=int,default=12);p.add_argument('--overwrite',action='store_true')
    validate(**vars(p.parse_args()))
