"""Relative 2171 registration using epipolar matches to fixed 2172 plus terrain."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from marjum_camera import epipolar_error,project
from marjum_bundle import boundary_pixels
from marjum_add_views import horizon_error,spatial_split
from marjum_position import RefinedTerrain

def run():
    from eigsep_terrain.marjum_dem import MarjumDEM
    s=np.load('cv_transmitter_v3/fit_transmitter.npz');keys=list(s['keys']);i=keys.index('2171');j=keys.index('2172')
    z=np.load('cv_transmitter_v4/shared_2171.npz');base=z['camera'];k=z['distortion'];ref=s['cameras'][j];shape=s['shapes'][i]
    matches=np.load('cv_transmitter_v4/relative_2171.npz');xy=matches['xy'];refxy=matches['ref_xy']
    tr,te=spatial_split(xy);tr &= matches['inlier'];te &= matches['inlier']
    dem=MarjumDEM(cache_file='marjum_dem_sw.npz');terrain=RefinedTerrain(dem)
    with np.load('img_seg_IMG_2171.npz') as seg:
        h=boundary_pixels(np.flipud(seg['skymask']).astype(bool),np.flipud(seg['ptree']),60)
    held=(h[:,0]//180).astype(int)%3==1
    def unpack(q):return np.r_[base[:6]+q[:6],base[6]*np.exp(q[6])]
    def residual(q):
        p=unpack(q)
        e=epipolar_error(ref,s['shapes'][j],s['distortion'][j],p,shape,k,refxy[tr],xy[tr])
        height=max(terrain.height(*p[:2]),float(np.asarray(dem.interp_alt(np.array([p[0]]),np.array([p[1]])))[0]))
        return np.r_[e/2,horizon_error(p,shape,k,h[~held],terrain)/8,q[:3]/2,q[6]/.05,
                     min(p[2]-height-.8,0)/.1, min(np.linalg.norm(p[:3]-ref[:3])-.1,0)/.05]
    q=np.zeros(7);q[:3]=matches['direction']*.5
    bound=np.array([5,5,3,.1,.1,.1,.1])
    f=least_squares(residual,q,bounds=(-bound,bound),loss='soft_l1',f_scale=2,x_scale=[1,1,1,.01,.01,.01,.02],max_nfev=150,ftol=1e-6)
    p=unpack(f.x);he=horizon_error(p,shape,k,h,terrain)
    ee=epipolar_error(ref,s['shapes'][j],s['distortion'][j],p,shape,k,refxy[te],xy[te])
    report=dict(camera=p.tolist(),distortion=k.tolist(),heldout_horizon_rms_px=float(np.sqrt(np.mean(he[held]**2))),
                heldout_epipolar_median_px=float(np.median(ee)),heldout_epipolar_count=int(te.sum()),baseline_m=float(np.linalg.norm(p[:3]-ref[:3])),
                note='RANSAC used all matches for hypothesis screening; heldout matches excluded from nonlinear fitting. Transmitter pick unused.')
    Path('cv_transmitter_v4/pair_2171.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
if __name__=='__main__':run()
