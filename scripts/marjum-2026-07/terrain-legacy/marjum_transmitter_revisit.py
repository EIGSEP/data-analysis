"""Refine three provisional views; anchor 2171 to fixed 2172 DEM features."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from marjum_camera import project,rays,radial_support
from marjum_bundle import boundary_pixels
from marjum_add_views import dem_landmarks,spatial_split,horizon_error
from marjum_position import RefinedTerrain

def run():
    from eigsep_terrain.marjum_dem import MarjumDEM
    out=Path('cv_transmitter_v4');out.mkdir(exist_ok=False)
    state=dict(np.load('cv_transmitter_v3/fit_transmitter.npz'));keys=list(state['keys'])
    terrain=RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz'));reports={}
    for key in ['2171','2159','2203']:
        i=keys.index(key);base=state['cameras'][i].copy();k0=state['distortion'][i].copy();shape=state['shapes'][i]
        xyz=np.empty((0,3));xy=np.empty((0,2));heldfeat=np.zeros(0,bool)
        if key=='2171':
            j=keys.index('2172');features={k:dict(np.load(f'cv_features/sift_{k}.npz')) for k in [key,'2172']}
            m=dem_landmarks(key,features,[dict(key='2172',camera=state['cameras'][j],distortion=state['distortion'][j])],terrain)
            distance=np.linalg.norm(m['xyz']-state['cameras'][j,:3],axis=1)
            good=distance>80;xyz=m['xyz'][good];xy=m['xy'][good]
            trainfeat,heldfeat=spatial_split(xy)
            base[:3]=state['cameras'][j,:3];k0=state['distortion'][j].copy()
            # Initialize orientation from distant, mutually matched scene points.
            def rr(q):
                p=base.copy();p[3:6]+=q[:3];p[6]*=np.exp(q[3])
                return (project(p,shape,xyz[trainfeat],k0)[0]-xy[trainfeat]).ravel()/5
            fit=least_squares(rr,np.zeros(4),loss='soft_l1',f_scale=2,max_nfev=100)
            base[3:6]+=fit.x[:3];base[6]*=np.exp(fit.x[3])
            error=np.linalg.norm(project(base,shape,xyz,k0)[0]-xy,axis=1)
            use=(error<30)&trainfeat
            print(key,'distant matches',len(xy),'training inliers',int(use.sum()),flush=True)
        with np.load(f'img_seg_IMG_{key}.npz') as seg:
            h=boundary_pixels(np.flipud(seg['skymask']).astype(bool),np.flipud(seg['ptree']),60)
        held=(h[:,0]//180).astype(int)%3==1;train=~held
        candidates=[]
        for free in [False,True]:
            def unpack(q):return np.r_[base[:6]+q[:6],base[6]*np.exp(q[6])],k0+q[7:]
            def residual(q):
                p,k=unpack(q);d=rays(p,shape,h[train],k)
                az=np.arctan2(d[:,1],d[:,0]);alt=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
                err=(alt-terrain.skyline(p[:3],az,count=2048,peaks=8,refine=33))*base[6]
                r=np.r_[err/10,q[:3]/[15,15,15],q[3:6]/.2,q[6]/.1,q[7:]/.03,
                        min(p[2]-terrain.height(*p[:2])-.5,0)/.2,np.minimum(radial_support(p,shape,k)-.25,0)*100]
                if key=='2171' and use.sum()>=8:
                    pred,depth=project(p,shape,xyz[use],k)
                    r=np.r_[r,(pred-xy[use]).ravel()/5,np.minimum(depth-1,0)]
                return r
            count=9 if free else 7
            def expand(x):return np.r_[x,np.zeros(9-count)]
            bound=np.array([30,30,30,.3,.3,.2,.2,.08,.08])[:count]
            fit=least_squares(lambda x:residual(expand(x)),np.zeros(count),bounds=(-bound,bound),
                              x_scale=np.array([5,5,5,.02,.02,.02,.03,.01,.01])[:count],loss='soft_l1',f_scale=2,max_nfev=150,ftol=1e-6)
            p,k=unpack(expand(fit.x));err=horizon_error(p,shape,k,h,terrain)
            row=dict(camera=p.tolist(),distortion=k.tolist(),free_distortion=free,
                     train_rms_px=float(np.sqrt(np.mean(err[train]**2))),heldout_rms_px=float(np.sqrt(np.mean(err[held]**2))))
            if len(xy):
                error=np.linalg.norm(project(p,shape,xyz,k)[0]-xy,axis=1)
                row.update(feature_heldout_median_px=float(np.median(error[heldfeat])),feature_heldout_count=int(heldfeat.sum()),feature_train_count=int(use.sum()))
            candidates.append(row);print(key,row,flush=True)
        olderr=horizon_error(state['cameras'][i],shape,state['distortion'][i],h,terrain)
        best=min(candidates,key=lambda r:r['heldout_rms_px'])
        accepted=best['heldout_rms_px']<np.sqrt(np.mean(olderr[held]**2))
        if key=='2171':accepted &= best.get('feature_heldout_median_px',np.inf)<20
        if accepted:
            state['cameras'][i]=best['camera'];state['distortion'][i]=best['distortion']
        reports[key]=dict(accepted=bool(accepted),baseline_heldout_rms_px=float(np.sqrt(np.mean(olderr[held]**2))),candidates=candidates)
        (out/'refinement.json').write_text(json.dumps(reports,indent=2)+'\n')
    np.savez_compressed(out/'fit_transmitter.npz',**state)
    return reports
if __name__=='__main__':run()
