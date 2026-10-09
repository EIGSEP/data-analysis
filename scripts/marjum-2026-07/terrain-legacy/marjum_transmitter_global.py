"""Broad terrain-only initialization for the two platform photographs."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution,least_squares
from marjum_camera import rays,project
from marjum_bundle import boundary_pixels
from marjum_position import RefinedTerrain
from marjum_add_views import horizon_error

def run(key,landmark=False):
    from eigsep_terrain.marjum_dem import MarjumDEM
    s=np.load('cv_transmitter_v3/fit_transmitter.npz');i=list(s['keys']).index(key);base=s['cameras'][i];k=s['distortion'][i];shape=s['shapes'][i]
    t=RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
    with np.load(f'img_seg_IMG_{key}.npz') as seg:
        h=boundary_pixels(np.flipud(seg['skymask']).astype(bool),np.flipud(seg['ptree']),60)
    held=(h[:,0]//180).astype(int)%3==1;train=~held
    def unpack(q):return np.r_[base[:6]+q[:6],base[6]*np.exp(q[6])]
    pick=np.array(json.loads(Path('meta.json').read_text())[key]['transmitter_px'])
    def residual(q,full=False):
        p=unpack(q);d=rays(p,shape,h[train],k);az=np.arctan2(d[:,1],d[:,0]);alt=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
        horizon=(alt-t.skyline(p[:3],az,count=8192 if full else 1024,peaks=16 if full else 8,refine=65 if full else 33))*base[6]
        r=np.r_[horizon/10,q[:3]/60,q[6]/.2,min(p[2]-t.height(*p[:2])-1,0)/.3]
        if landmark:
            pred,depth=project(p,shape,s['transmitter'],k)
            r=np.r_[r,(pred[0]-pick)/2,min(depth[0]-1,0)/.1]
        return r
    bound=np.array([90,90,65,.15,.2,.15,.3]);bounds=list(zip(-bound,bound))
    def cost(q):
        r=residual(q);return np.sum(4*(np.sqrt(1+(r/2)**2)-1))
    f=differential_evolution(cost,bounds,seed=17,maxiter=90,popsize=7,polish=False,x0=np.zeros(7),tol=.001)
    print(key,'global',f.fun,unpack(f.x),flush=True)
    fit=least_squares(lambda q:residual(q,True),f.x,bounds=(-bound,bound),loss='soft_l1',f_scale=2,
                      x_scale=[10,10,10,.03,.03,.02,.05],max_nfev=150,ftol=1e-6)
    p=unpack(fit.x);err=horizon_error(p,shape,k,h,t)
    report=dict(camera=p.tolist(),distortion=k.tolist(),transmitter_conditioned=landmark,
                transmitter_residual_px=float(np.linalg.norm(project(p,shape,s['transmitter'],k)[0][0]-pick)),train_rms_px=float(np.sqrt(np.mean(err[train]**2))),
                heldout_rms_px=float(np.sqrt(np.mean(err[held]**2))),cost=float(fit.cost),global_cost=float(f.fun))
    suffix='_conditioned' if landmark else ''
    Path(f'cv_transmitter_v4/global_{key}{suffix}.json').write_text(json.dumps(report,indent=2)+'\n');print(key,report,flush=True)
if __name__=='__main__':run(sys.argv[1],landmark='--landmark' in sys.argv)
