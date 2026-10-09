"""Terrain-only refinement from the manually restored 2172 pose.

The transmitter pick never enters camera fitting. Existing cameras are fixed.
Distortion sensitivity is tested against a fixed shared-lens calibration.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from marjum_camera import rays, radial_support
from marjum_add_views import horizon_error
from marjum_bundle import boundary_pixels
from marjum_position import RefinedTerrain
from marjum_mcmc import digest


def run(output='cv_transmitter_v2'):
    from eigsep_terrain.marjum_dem import MarjumDEM
    out=Path(output);out.mkdir(exist_ok=False)
    source=Path('cv_transmitter_v1/fit_transmitter_manual_2172.npz')
    state=dict(np.load(source));i=list(state['keys']).index('2172')
    base=state['cameras'][i].copy();k0=state['distortion'][i].copy();shape=state['shapes'][i]
    terrain=RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
    with np.load('img_seg_IMG_2172.npz') as seg:
        horizon=boundary_pixels(np.flipud(seg['skymask']).astype(bool),np.flipud(seg['ptree']),spacing=60)
    # Reserve contiguous 180-pixel image regions, not adjacent boundary samples.
    held=(horizon[:,0]//180).astype(int)%3==1;train=~held
    reports=[];solutions=[]
    for free in [False,True]:
        def unpack(q):return np.r_[base[:6]+q[:6],base[6]*np.exp(q[6])],k0+q[7:9]
        def residual(q):
            p,k=unpack(q)
            d=rays(p,shape,horizon[train],k)
            az=np.arctan2(d[:,1],d[:,0]);alt=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
            err=(alt-terrain.skyline(p[:3],az,count=2048,peaks=8,refine=33))*base[6]
            return np.r_[err/10.,q[:3]/[10,10,5],q[3:6]/.1,q[6]/.05,
                         q[7:]/[.025,.025],min(p[2]-terrain.height(*p[:2])-.5,0)/.2,
                         np.minimum(radial_support(p,shape,k)-.25,0)*100]
        active=np.arange(9 if free else 7)
        def expand(x):
            q=np.zeros(9);q[active]=x;return q
        bounds=np.array([15,15,8,.15,.15,.1,.1,.08,.08])[active]
        fit=least_squares(lambda x:residual(expand(x)),np.zeros(len(active)),
                          bounds=(-bounds,bounds),x_scale=np.array([3,3,2,.01,.01,.01,.02,.01,.01])[active],
                          loss='soft_l1',f_scale=2,max_nfev=100,ftol=1e-6)
        p,k=unpack(expand(fit.x));err=horizon_error(p,shape,k,horizon,terrain)
        report=dict(free_distortion=free,camera=p.tolist(),distortion=k.tolist(),
                    train_rms_px=float(np.sqrt(np.mean(err[train]**2))),
                    heldout_rms_px=float(np.sqrt(np.mean(err[held]**2))),
                    nfev=fit.nfev,success=bool(fit.success),cost=float(fit.cost))
        reports.append(report);solutions.append((p,k));print(report,flush=True)
    # Retain the shared calibration unless per-view distortion clearly improves
    # the reserved horizon; do not use the transmitter residual to select a fit.
    selected=int(reports[1]['heldout_rms_px'] < .9*reports[0]['heldout_rms_px'])
    p,k=solutions[selected];state['cameras'][i]=p;state['distortion'][i]=k
    baseerr=horizon_error(base,shape,k0,horizon,terrain)
    if reports[selected]['heldout_rms_px'] > np.sqrt(np.mean(baseerr[held]**2)):
        state['cameras'][i]=base;state['distortion'][i]=k0;selected=-1
    np.savez_compressed(out/'fit_extended.npz',**state)
    report=dict(source=str(source),input_sha256=digest(source),selected=selected,
                transmitter_used_for_camera_fit=False,manual_seed_may_have_used_transmitter_overlay=True,
                baseline_heldout_rms_px=float(np.sqrt(np.mean(baseerr[held]**2))),
                baseline_train_rms_px=float(np.sqrt(np.mean(baseerr[train]**2))),
                train_count=int(train.sum()),heldout_count=int(held.sum()),candidates=reports)
    (out/'2172_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':run()
