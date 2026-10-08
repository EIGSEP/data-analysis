"""Position-focused refinement and DEM-anchored absolute antenna candidates.

Distortion and focal lengths are initially held fixed. Reserved image matches
never enter the fit. One third of the cached horizon observations is withheld.
Multiple starts and sensitivity fits diagnose stability, not posterior coverage.
"""
from functools import lru_cache
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import hstack,vstack,lil_matrix,csr_matrix
from marjum_bundle import Terrain
from marjum_camera import rays,project,epipolar_error
from marjum_guided_joint import AntennaFit
from marjum_guided import evaluate,graph
from marjum_mcmc import digest
from marjum_grid import GridCoordinates


class RefinedTerrain(Terrain):
    def skyline(self,cam,azimuth,count=8192,peaks=16,refine=65):
        """Deterministic coarse skyline with local radial-maximum refinement."""
        az=np.atleast_1d(azimuth);ce=np.cos(az);sn=np.sin(az)
        def edge(c,lo,hi,d):
            safe=np.where(abs(d)>1e-12,d,1e-12)
            return np.where(d>=0,(hi-c)/safe,(lo-c)/safe)
        end=np.maximum(np.minimum(edge(cam[0],self.e[0],self.e[-1],ce),edge(cam[1],self.n[0],self.n[-1],sn))-self.res,2.)
        logend=np.log(end);fraction=np.linspace(0,1,count)
        distance=np.exp(logend[:,None]*fraction)
        def angles(dist):
            z=self.height(cam[0]+ce[:,None]*dist,cam[1]+sn[:,None]*dist)
            a=np.arctan2(z-cam[2],dist)
            return np.where(np.isfinite(a),a,-np.pi/2)
        a=angles(distance)
        local=(a>=np.roll(a,1,axis=1))&(a>=np.roll(a,-1,axis=1));local[:,[0,-1]]=True
        score=np.where(local,a,-np.inf)
        idx=np.argpartition(score,-peaks,axis=1)[:,-peaks:]
        fractions=np.clip((idx[:,:,None]+np.linspace(-1,1,refine))/(count-1),0,1)
        fine=np.exp(logend[:,None,None]*fractions).reshape(len(az),-1)
        return np.maximum(a.max(axis=1),angles(fine).max(axis=1))


class PositionFit(AntennaFit):
    def __init__(self,*args,horizon_sigma=10.,terrain_sigma=5.,**kwargs):
        super().__init__(*args,**kwargs)
        self.base_size=len(self.x0);self.base_rows=self.sparsity.shape[0]
        self.grid=GridCoordinates(self.terrain.dem)
        self.gps,self.alt,self.raw_gps=self.grid.gps(self.keys)
        self.horizon_sigma=horizon_sigma;self.terrain_sigma=terrain_sigma
        self.horizons=[np.asarray(self.features[k]['horizon'],float) for k in self.keys]
        self.htrain=[np.arange(len(h))%3!=1 for h in self.horizons]
        self.x0=np.r_[self.x0,[0.,0.,0.]];self.lower=np.r_[self.lower,[-150.,-150.,-150.]];self.upper=np.r_[self.upper,[150.,150.,150.]]
        self.scale=np.r_[self.scale,[20.,20.,30.]]
        mat=hstack([self.sparsity,csr_matrix((self.base_rows,3))]).tolil()
        gps0=3*len(self.obs['xy'])+self.np
        mat[gps0:gps0+2*self.nc,self.base_size:self.base_size+2]=1
        mat[gps0+2*self.nc:gps0+3*self.nc,self.base_size+2]=1
        rows=[]
        rows.extend([[self.base_size+j] for j in range(3)])
        for i,mask in enumerate(self.htrain):rows.extend([list(range(7*i,7*i+7))]*int(mask.sum()))
        extra=lil_matrix((len(rows),len(self.x0)),dtype=int)
        for i,cols in enumerate(rows):extra[i,cols]=1
        self.sparsity=vstack([mat,extra]).tocsr()

    @lru_cache(maxsize=8192)
    def horizon_one(self,i,parameters):
        p=np.array(parameters);d=rays(p,self.shapes[i],self.horizons[i],self.ks[i])
        az=np.arctan2(d[:,1],d[:,0]);observed=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
        predicted=self.terrain.skyline(p[:3],az)
        # Fixed reference focal length avoids changing likelihood scale when f moves.
        return (observed-predicted)*self.base[i,6]

    def residuals(self,x):
        r=AntennaFit.residuals(self,x[:self.base_size])
        cams,ks,points=self.unpack(x);bias=x[self.base_size:]
        n=len(self.obs['xy']);r[3*n:3*n+self.np]*=5./self.terrain_sigma
        gps0=3*n+self.np
        r[gps0:gps0+2*self.nc]=((cams[:,:2]+bias[:2]-self.gps)/self.gps_sigma[:,None]).ravel()
        r[gps0+2*self.nc:gps0+3*self.nc]=(cams[:,2]+bias[2]-self.alt)/30.
        horizons=np.concatenate([self.horizon_one(i,tuple(p))[mask] for i,(p,mask) in enumerate(zip(cams,self.htrain))])
        return np.r_[r,bias/np.array([30.,30.,50.]),horizons/self.horizon_sigma]

    def loss(self,z):
        rho=super().loss(z)
        start=self.base_rows+3;t=1+z[start:]
        rho[:,start:]=[2*(np.sqrt(t)-1),1/np.sqrt(t),-.5*t**(-1.5)]
        return rho

    def cost(self,x):
        r=self.residuals(x)
        return float(2*self.loss((r/2)**2)[0].sum())

    def active(self,mode):
        slots={'positions':[0,1,2],'orientations':[3,4,5],'poses':[0,1,2,3,4,5],'focal':[0,1,2,3,4,5,6]}[mode]
        return np.r_[[7*i+j for i in range(self.nc) for j in slots],np.arange(7*self.nc,len(self.x0))].astype(int)

    def solve_mode(self,start,mode,max_nfev=100):
        active=self.active(mode);fixed=start.copy();t=time.monotonic()
        def expand(q):
            value=fixed.copy();value[active]=q;return value
        fit=least_squares(lambda q:self.residuals(expand(q)),start[active],
             jac_sparsity=self.sparsity[:,active],bounds=(self.lower[active],self.upper[active]),
             x_scale=self.scale[active],loss=self.loss,f_scale=2,max_nfev=max_nfev,ftol=2e-6,xtol=1e-7,gtol=1e-5)
        value=expand(fit.x)
        print(mode,'cost',fit.cost,'nfev',fit.nfev,'seconds',time.monotonic()-t,flush=True)
        return value,dict(cost=float(fit.cost),nfev=fit.nfev,status=fit.status,message=fit.message,mode=mode)

    def shifted(self,shift):
        x=self.x0.copy()
        for i in range(self.nc):x[7*i:7*i+3]+=shift
        x[self.ng:self.geometry_size]+=np.tile(shift,self.np)
        x[self.geometry_size:self.geometry_size+3]+=shift
        cams=self.unpack(x)[0];w=1/self.gps_sigma**2
        x[self.base_size:self.base_size+2]=-np.sum((cams[:,:2]-self.gps)*w[:,None],axis=0)/(w.sum()+1/30.**2)
        x[self.base_size+2]=-np.sum((cams[:,2]-self.alt)/30.**2)/(self.nc/30.**2+1/50.**2)
        return x


def load_inputs(root='cv_distortion_guided_v2'):
    root=Path(root);s=dict(np.load(root/'joint.npz'));keys=list(s['keys'].astype(str));features={}
    for k in keys:
        with np.load(f'cv_features/sift_{k}.npz') as f:features[k]={name:f[name] for name in f.files}
    with np.load(root/'holdout.npz') as f:holdout=[(*k.split('_'),f[k]) for k in f.files]
    obs=dict(points=s['points'],oc=s['obs_cam'],op=s['obs_point'],xy=s['obs_xy'],fid=s['obs_fid'])
    return s,keys,features,obs,holdout


def save_state(out,label,model,x,source,features,holdout,extra=None):
    cams,ks,points=model.unpack(x);ant=model.ant0+x[model.geometry_size:model.geometry_size+3]
    report=evaluate(model.keys,features,cams,ks,model.terrain,holdout,{**model.obs,'points':points})
    # evaluate() independently retriangulates the antenna. Report the antenna
    # actually optimized here, rather than silently substituting that estimate.
    ant_error=np.array([np.linalg.norm(project(cams[i],model.shapes[i],ant,ks[i])[0][0]-xy)
                        for i,xy in zip(model.ai,model.axy)])
    report.update(antenna_median_px=float(np.median(ant_error)),antenna_max_px=float(ant_error.max()),
                  antenna_images=[dict(key=model.keys[i],error_px=float(e)) for i,e in zip(model.ai,ant_error)])
    report.update(cost=model.cost(x),gps_bias_m=x[model.base_size:].tolist(),antenna_enu=ant.tolist(),
                  position_changes_m=np.linalg.norm(cams[:,:3]-model.base[:,:3],axis=1).tolist(),
                  heldout_horizon={},training_horizon={})
    for i,key in enumerate(model.keys):
        error=model.horizon_one(i,tuple(cams[i]));mask=model.htrain[i]
        report['heldout_horizon'][key]=float(np.sqrt(np.mean(error[~mask]**2)))
        report['training_horizon'][key]=float(np.sqrt(np.mean(error[mask]**2)))
    report['heldout_horizon_median_px']=float(np.median(list(report['heldout_horizon'].values())))
    if extra:report.update(extra)
    np.savez_compressed(out/f'{label}.npz',**{**source,'cameras':cams,'distortion':ks,'points':points,'antenna':ant},
                        x=x,gps_bias=x[model.base_size:])
    (out/f'{label}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(label,{k:report[k] for k in ['antenna_median_px','antenna_max_px','holdout_pair_median_px','heldout_horizon_median_px','antenna_enu']},flush=True)
    return report


def run(output='cv_position_absolute',max_nfev=100):
    from eigsep_terrain.marjum_dem import MarjumDEM
    out=Path(output);out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use a new output directory')
    source,keys,features,obs,holdout=load_inputs();terrain=RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
    model=PositionFit(keys,features,source['cameras'],source['distortion'],source['groups'],obs,terrain,
                      False,antenna=source['antenna'])
    files=[Path(__file__),Path('marjum_grid.py'),Path('marjum_camera.py'),Path('marjum_guided.py'),Path('marjum_guided_joint.py'),
           Path('marjum_bundle.py'),Path('meta.json'),Path('marjum_dem_sw.npz'),Path('marjum_2026_07_exif.npz'),
           Path('cv_distortion_guided_v2/joint.npz'),Path('cv_distortion_guided_v2/holdout.npz')]
    files.extend(Path(f'cv_features/sift_{k}.npz') for k in keys)
    files.extend(Path(f'marjum-2026-07/IMG_{k}.HEIC') for k in keys)
    files.extend(Path(str(p)) for p in np.asarray(terrain.dem.files).ravel())
    manifest=dict(input_sha256={str(p):digest(p) for p in files},horizon_sigma_px=10.,terrain_sigma_m=5.,
         common_gps_bias_sigma_m=[30.,30.,50.],distortion_fixed=True,skyline_coarse_samples=8192,skyline_peak_refinement=65,
         grid_epsg=model.grid.epsg,grid_origin_projected_m=model.grid.origin.tolist(),legacy_survey_offset_not_applied=model.grid.legacy_survey_offset,
         legacy_antenna_epoch=2025,legacy_antenna_position_used_as_constraint=False,
         antenna_initialization='2026 distortion-guided joint triangulation; no coordinate prior',
         gps_bias_convention='measured HEIC GPS = fitted camera position + common bias',
         horizon_training_rule='index % 3 != 1',keys=keys,max_nfev=max_nfev)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez_compressed(out/'horizon_split.npz',**{f'xy_{k}':h for k,h in zip(keys,model.horizons)},
                         **{f'train_{k}':m for k,m in zip(keys,model.htrain)})
    save_state(out,'baseline',model,model.x0,source,features,holdout)
    # Same objective, two restricted tests: positions versus orientations.
    position,info=model.solve_mode(model.x0,'positions',max_nfev)
    save_state(out,'positions',model,position,source,features,holdout,info)
    orientation,info=model.solve_mode(model.x0,'orientations',max_nfev)
    save_state(out,'orientations',model,orientation,source,features,holdout,info)
    full,info=model.solve_mode(position,'poses',max_nfev)
    save_state(out,'poses',model,full,source,features,holdout,info)
    # Coherent translations preserve every image-to-image projection at the start.
    grid=[]
    for e in [-12.,0.,12.]:
        for n in [-12.,0.,12.]:
            for u in [-6.,0.,6.]:
                shift=np.array([e,n,u]);x=model.shifted(shift)
                grid.append((model.cost(x),shift))
    grid.sort(key=lambda v:v[0]);(out/'translation_grid.json').write_text(json.dumps([dict(cost=c,shift_m=s.tolist()) for c,s in grid],indent=2)+'\n')
    alternative=next(s for c,s in grid if np.linalg.norm(s)>6)
    print('Alternative coherent translation',alternative,flush=True)
    other,info=model.solve_mode(model.shifted(alternative),'poses',max_nfev)
    save_state(out,'alternative',model,other,source,features,holdout,info)
    best=min([full,other],key=model.cost)
    focal,info=model.solve_mode(best,'focal',max_nfev)
    save_state(out,'focal',model,focal,source,features,holdout,info)
    return model,focal


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_position_absolute');p.add_argument('--max-nfev',type=int,default=100)
    run(**vars(p.parse_args()))
