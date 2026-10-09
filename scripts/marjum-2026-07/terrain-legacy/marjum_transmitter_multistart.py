"""Position multistarts using the fully refined skyline, without transmitter picks."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from marjum_bundle import boundary_pixels
from marjum_camera import rays
from marjum_add_views import horizon_error
from marjum_position import RefinedTerrain

def run(key):
    from eigsep_terrain.marjum_dem import MarjumDEM
    s=np.load('cv_transmitter_v3/fit_transmitter.npz');i=list(s['keys']).index(key)
    base=s['cameras'][i].copy();k=s['distortion'][i];shape=s['shapes'][i]
    dem=MarjumDEM(cache_file='marjum_dem_sw.npz');t=RefinedTerrain(dem)
    with np.load(f'img_seg_IMG_{key}.npz') as seg:
        h=boundary_pixels(np.flipud(seg['skymask']).astype(bool),np.flipud(seg['ptree']),60)
    held=(h[:,0]//180).astype(int)%3==1;train=~held
    rows=[]
    for shift in [(0,0),(15,0),(-15,0),(0,15),(0,-15)]:
        def unpack(q):return np.r_[base[:6]+q[:6],base[6]*np.exp(q[6])]
        def residual(q):
            p=unpack(q);err=horizon_error(p,shape,k,h[train],t)*base[6]/p[6]
            return np.r_[err/8,q[:3]/30,q[6]/.15,min(p[2]-t.height(*p[:2])-1.,0)/.3]
        q=np.zeros(7);q[:2]=shift;bound=np.array([60,60,45,.4,.4,.3,.25])
        fit=least_squares(residual,q,bounds=(-bound,bound),x_scale=[10,10,10,.03,.03,.02,.05],
                          loss='soft_l1',f_scale=2,max_nfev=100,ftol=1e-6)
        p=unpack(fit.x);e=horizon_error(p,shape,k,h,t)
        row=dict(shift=shift,camera=p.tolist(),distortion=k.tolist(),cost=float(fit.cost),
                 train_rms_px=float(np.sqrt(np.mean(e[train]**2))),heldout_rms_px=float(np.sqrt(np.mean(e[held]**2))))
        rows.append(row);print(key,row,flush=True)
        Path(f'cv_transmitter_v4/multistart_{key}.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows
if __name__=='__main__':run(sys.argv[1])
