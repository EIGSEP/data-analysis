"""Horizon-preserving joint camera/distortion/transmitter polish from a supplied NPZ.

Only the six transmitter-only cameras are free. Shared-scene constraints do
not assume features lie on the DEM. The source NPZ is never overwritten.
"""
import argparse
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from marjum_bundle import boundary_pixels
from marjum_camera import project, rays, radial_support, epipolar_error
from marjum_add_views import horizon_error, spatial_split
from marjum_position import RefinedTerrain
from marjum_transmitter_bridge import mutual
from marjum_mcmc import digest

FREE=('2159','2171','2172','2198','2199','2203')


class Polish:
    def __init__(self,source,output):
        from eigsep_terrain.marjum_dem import MarjumDEM
        self.source=Path(source);self.out=Path(output);self.out.mkdir(exist_ok=False)
        self.state=dict(np.load(source));self.keys=list(self.state['keys'].astype(str))
        self.meta=json.loads(Path('meta.json').read_text())
        self.ids=[self.keys.index(k) for k in FREE]
        self.base=self.state['cameras'][self.ids].copy();self.k0=self.state['distortion'][self.ids].copy()
        self.shapes=self.state['shapes'][self.ids];self.tx0=self.state['transmitter'].copy()
        self.terrain=RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
        self.h=[];self.train=[]
        for key in FREE:
            with np.load(f'img_seg_IMG_{key}.npz') as seg:
                excluded=np.flipud(seg['horizon_exclude']) if 'horizon_exclude' in seg else None
                h=boundary_pixels(np.flipud(seg['skymask']).astype(bool),np.flipud(seg['ptree']),60,exclude=excluded)
            self.h.append(h);self.train.append((h[:,0]//180).astype(int)%3!=1)
        self.active={k:i for i,k in enumerate(FREE)}
        self.txkeys=[k for k in self.keys if 'transmitter_px' in self.meta.get(k,{})]
        self.pairs=[];features={k:dict(np.load(f'cv_features/sift_{k}.npz')) for k in self.txkeys}
        cv2.setRNGSeed(37)
        for a,b in combinations(self.txkeys,2):
            if a not in FREE and b not in FREE:continue
            m=mutual(features[a]['descriptors'],features[b]['descriptors'])
            if len(m)<30:continue
            ids=np.array(list(m.items()));x=features[a]['xy'][ids[:,0]];y=features[b]['xy'][ids[:,1]]
            tr,te=spatial_split(x)
            if tr.sum()<24 or te.sum()<8:continue
            F,mask=cv2.findFundamentalMat(x[tr],y[tr],cv2.FM_RANSAC,4.,.999)
            if F is None or F.shape!=(3,3) or mask is None or mask.sum()<24:continue
            train_ids=np.flatnonzero(tr)[mask.ravel()>0]
            # Holdout selection uses the training-estimated model, not a refit.
            xx=np.c_[x,np.ones(len(x))];yy=np.c_[y,np.ones(len(y))]
            fx=xx@F.T;fy=yy@F
            e=abs(np.sum(yy*fx,axis=1))/np.sqrt(np.sum(fx[:,:2]**2+fy[:,:2]**2,axis=1))
            test_ids=np.flatnonzero(te&(e<6))
            if len(test_ids)<8:continue
            train_ids=train_ids[np.linspace(0,len(train_ids)-1,min(180,len(train_ids))).astype(int)]
            self.pairs.append(dict(a=a,b=b,x=x[train_ids],y=y[train_ids],test_x=x[test_ids],test_y=y[test_ids]))
        print('feature pairs',[(p['a'],p['b'],len(p['x']),len(p['test_x'])) for p in self.pairs],flush=True)
        self.scale=np.tile([5,5,3,.02,.02,.02,.03,.01,.01],len(FREE))
        self.bound=np.tile([15,15,10,.22,.22,.18,.18,.08,.08],len(FREE))
        self.report=dict(source=str(source),input_sha256={str(self.source):digest(self.source),'meta.json':digest(Path('meta.json'))},baseline=self.metrics(np.zeros(54),self.tx0))

    def unpack(self,q):
        q=np.asarray(q[:54]).reshape(6,9)
        return np.c_[self.base[:,:6]+q[:,:6],self.base[:,6]*np.exp(q[:,6])],self.k0+q[:,7:9]

    @lru_cache(maxsize=4096)
    def horizon(self,i,p,k):
        p=np.asarray(p);return horizon_error(p,self.shapes[i],k,self.h[i],self.terrain)*self.base[i,6]/p[6]

    def one(self,i,q,p,k):
        e=self.horizon(i,tuple(p),tuple(k))[self.train[i]]
        height=self.terrain.height(*p[:2])
        return np.r_[e/8,q[:3]/[5,5,3],q[3:6]/.08,q[6]/.07,q[7:]/.025,
                     min(p[2]-height-.5,0)/.2,np.minimum(radial_support(p,self.shapes[i],k)-.3,0)*100]

    def camera(self,key,cams,ks):
        if key in self.active:
            i=self.active[key];return cams[i],self.shapes[i],ks[i]
        i=self.keys.index(key);return self.state['cameras'][i],self.state['shapes'][i],self.state['distortion'][i]

    def metrics(self,q,tx):
        cams,ks=self.unpack(q);rows={}
        for i,key in enumerate(FREE):
            h=self.horizon(i,tuple(cams[i]),tuple(ks[i]));tr=self.train[i]
            rows[key]=dict(train_rms_px=float(np.sqrt(np.mean(h[tr]**2))),heldout_rms_px=float(np.sqrt(np.mean(h[~tr]**2))),
                           camera=cams[i].tolist(),distortion=ks[i].tolist(),radial_min_derivative=float(radial_support(cams[i],self.shapes[i],ks[i]).min()))
        txrows={}
        for key in self.txkeys:
            p,s,k=self.camera(key,cams,ks);xy=self.meta[key]['transmitter_px'];pred,depth=project(p,s,tx,k)
            d=rays(p,s,[xy],k)[0];delta=tx-p[:3]
            txrows[key]=dict(error_px=float(np.linalg.norm(pred[0]-xy)),depth_m=float(depth[0]),
                             ray_miss_m=float(np.linalg.norm(delta-d*(delta@d))))
        pairs=[]
        for pair in self.pairs:
            p,s,k=self.camera(pair['a'],cams,ks);p2,s2,k2=self.camera(pair['b'],cams,ks)
            e=epipolar_error(p,s,k,p2,s2,k2,pair['test_x'],pair['test_y'])
            pairs.append(dict(a=pair['a'],b=pair['b'],heldout_median_px=float(np.nanmedian(e)),heldout_count=len(e)))
        return dict(horizons=rows,transmitter=np.asarray(tx).tolist(),transmitter_residuals=txrows,pairs=pairs)

    def run(self,max_nfev=100):
        q=np.zeros(54)
        for i,key in enumerate(FREE):
            def residual(x):
                p=np.r_[self.base[i,:6]+x[:6],self.base[i,6]*np.exp(x[6])];k=self.k0[i]+x[7:]
                return self.one(i,x,p,k)
            f=least_squares(residual,np.zeros(9),bounds=(-self.bound[:9],self.bound[:9]),x_scale=self.scale[:9],
                            loss='soft_l1',f_scale=2,max_nfev=max_nfev,ftol=2e-6)
            trial=q.copy();trial[9*i:9*i+9]=f.x
            row=self.metrics(trial,self.tx0)['horizons'][key]
            if row['heldout_rms_px']<=self.report['baseline']['horizons'][key]['heldout_rms_px']:
                q=trial
            print('terrain',key,row,flush=True)
        self.report['terrain_only']=self.metrics(q,self.tx0)
        self.save('terrain_only.npz',q,self.tx0)
        self.start=q.copy()
        self.limits=np.array([self.report['terrain_only']['horizons'][k]['train_rms_px']*1.1+3 for k in FREE])
        def residual(x,structure=False):
            cams,ks=self.unpack(x);tx=self.tx0+x[54:57];values=[];deps=[]
            def add(v,columns):
                v=np.atleast_1d(v);values.extend(v);deps.extend([columns]*len(v))
            for i in range(6):
                cols=list(range(9*i,9*i+9));add(self.one(i,x[9*i:9*i+9],cams[i],ks[i]),cols)
                e=self.horizon(i,tuple(cams[i]),tuple(ks[i]))[self.train[i]]
                add([max(np.sqrt(np.mean(e**2))-self.limits[i],0)/1.5],cols)
            for key in self.txkeys:
                p,s,k=self.camera(key,cams,ks);pred,depth=project(p,s,tx,k)
                cols=list(range(54,57))+(list(range(9*self.active[key],9*self.active[key]+9)) if key in FREE else [])
                # Fixed camera reprojection has appreciable registration error;
                # do not pretend it is limited by subpixel manual picking.
                sigma=3. if key in FREE else 15.
                add((pred[0]-self.meta[key]['transmitter_px'])/sigma,cols);add([min(depth[0]-.5,0)/.1],cols)
            for pair in self.pairs:
                a,b=pair['a'],pair['b'];p,s,k=self.camera(a,cams,ks);p2,s2,k2=self.camera(b,cams,ks)
                cols=[j for key in [a,b] if key in FREE for j in range(9*self.active[key],9*self.active[key]+9)]
                e=epipolar_error(p,s,k,p2,s2,k2,pair['x'],pair['y'])
                add(np.nan_to_num(e,nan=1000.)/3,cols)
            if structure:
                mat=lil_matrix((len(values),57),dtype=int)
                for i,cols in enumerate(deps):mat[i,cols]=1
                return mat.tocsr()
            return np.array(values)
        x0=np.r_[q,[0,0,0]];bounds=np.r_[self.bound,[10,10,10]]
        f=least_squares(residual,x0,jac_sparsity=residual(x0,True),bounds=(-bounds,bounds),x_scale=np.r_[self.scale,[2,2,2]],
                        loss='soft_l1',f_scale=2,max_nfev=max_nfev,ftol=2e-6,verbose=1)
        self.report['joint']=self.metrics(f.x[:54],self.tx0+f.x[54:]);self.report['optimizer']=dict(success=bool(f.success),nfev=f.nfev,cost=float(f.cost))
        self.save('fit_transmitter.npz',f.x[:54],self.tx0+f.x[54:])
        np.savez_compressed(self.out/'optimizer.npz',x=f.x,jac=f.jac.toarray(),residual=f.fun)
        self.report['horizon_guard_passed']={k:self.report['joint']['horizons'][k]['heldout_rms_px']<=self.report['terrain_only']['horizons'][k]['heldout_rms_px']*1.15+3 for k in FREE}
        self.report['note']='Joint deterministic fit, not a posterior. Free camera rays are jointly constrained and must not be counted again as independent fixed-camera measurements.'
        (self.out/'report.json').write_text(json.dumps(self.report,indent=2)+'\n')
        print(json.dumps(self.report['joint'],indent=2),flush=True)
        return self.report

    def save(self,name,q,tx):
        cams,ks=self.unpack(q);state=dict(self.state);state['cameras']=self.state['cameras'].copy();state['distortion']=self.state['distortion'].copy()
        state['cameras'][self.ids]=cams;state['distortion'][self.ids]=ks;state['transmitter']=np.asarray(tx)
        state['transmitter_fit_keys']=np.array(self.txkeys)
        state['transmitter_conditioned_keys']=np.array(FREE)
        provenance=list(state.get('camera_provenance',np.full(len(self.keys),'established')))
        for i in self.ids:provenance[i]='joint terrain/features/transmitter fit' if name=='fit_transmitter.npz' else 'terrain-only distortion refinement'
        state['camera_provenance']=np.array(provenance)
        fixed=[i for i,k in enumerate(self.keys) if k not in FREE]
        for field in ['cameras','distortion']:
            assert self.state[field][fixed].tobytes()==state[field][fixed].tobytes()
        np.savez_compressed(self.out/name,**state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='cv_transmitter_joint_v1/fit_transmitter.npz');p.add_argument('--output',default='cv_transmitter_joint_v2');p.add_argument('--max-nfev',type=int,default=100)
    a=p.parse_args();Polish(a.source,a.output).run(a.max_nfev)
