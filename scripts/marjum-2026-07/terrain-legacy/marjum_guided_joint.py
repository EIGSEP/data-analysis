"""Horizon-free shared-antenna follow-up to the label-free guided ablation.

This stage explicitly uses antenna labels as camera constraints. Its antenna
reprojection errors are training diagnostics, NOT independent validation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import hstack,vstack,lil_matrix,csr_matrix
from marjum_guided import Fit,evaluate,graph
from marjum_camera import project
from marjum_bundle import Terrain
from marjum_mcmc import digest


class AntennaFit(Fit):
    def __init__(self,*args,antenna,**kwargs):
        super().__init__(*args,**kwargs)
        self.geometry_size=len(self.x0);self.ant0=np.array(antenna)
        meta=json.loads(Path('meta.json').read_text())
        self.ai=[i for i,k in enumerate(self.keys) if 'ant_px' in meta.get(k,{})]
        self.axy=np.array([meta[self.keys[i]]['ant_px'] for i in self.ai])
        self.x0=np.r_[self.x0,[0.,0.,0.]]
        self.lower=np.r_[self.lower,[-100.,-100.,-100.]];self.upper=np.r_[self.upper,[100.,100.,100.]]
        self.scale=np.r_[self.scale,[10.,10.,10.]]
        mat=lil_matrix((3*len(self.ai),len(self.x0)),dtype=int)
        for j,i in enumerate(self.ai):
            cols=list(range(7*i,7*i+7))+list(range(self.geometry_size,self.geometry_size+3))
            if self.free:cols+=list(range(7*self.nc+2*self.group[i],7*self.nc+2*self.group[i]+2))
            mat[2*j:2*j+2,cols]=1;mat[2*len(self.ai)+j,cols]=1
        self.sparsity=vstack([hstack([self.sparsity,csr_matrix((self.sparsity.shape[0],3))]),mat]).tocsr()

    def unpack(self,x):
        return Fit.unpack(self,x[:self.geometry_size])

    def residuals(self,x):
        base=Fit.residuals(self,x[:self.geometry_size])
        cams,ks,points=self.unpack(x);ant=self.ant0+x[-3:]
        pred,depth=zip(*(project(cams[i],self.shapes[i],ant,ks[i]) for i in self.ai))
        return np.r_[base,(np.concatenate(pred)-self.axy).ravel()/3.,np.minimum(np.concatenate(depth)-1.,0)/.1]


def run(output='cv_distortion_guided_v2',source='refined',max_nfev=200):
    from eigsep_terrain.marjum_dem import MarjumDEM
    root=Path(output)
    if (root/'joint.npz').exists() or (root/'joint_report.json').exists():raise FileExistsError('Joint stage already exists')
    manifest=json.loads((root/'manifest.json').read_text())
    changed=[k for k,v in manifest['input_sha256'].items() if digest(k)!=v]
    if changed:raise ValueError(changed)
    state=dict(np.load(root/f'{source}.npz'));keys=list(state['keys'].astype(str));features={}
    for key in keys:
        with np.load(f'cv_features/sift_{key}.npz') as f:features[key]={k:f[k] for k in f.files}
    with np.load(root/'holdout.npz') as f:holdout=[(*k.split('_'),f[k]) for k in f.files]
    obs=dict(points=state['points'],oc=state['obs_cam'],op=state['obs_point'],xy=state['obs_xy'],fid=state['obs_fid'])
    terrain=Terrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
    model=AntennaFit(keys,features,state['cameras'],state['distortion'],state['groups'],obs,terrain,True,antenna=state['antenna'])
    provenance=dict(source=source,antenna_in_camera_objective=True,horizon_in_objective=False,
                    input_sha256={str(p):digest(p) for p in [Path(__file__),root/'manifest.json',root/f'{source}.npz']})
    (root/'joint_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    result=model.solve(max_nfev);cam,ks,points=model.unpack(result.x);ant=model.ant0+result.x[-3:]
    report=evaluate(keys,features,cam,ks,terrain,holdout,{**obs,'points':points})
    report.update(cost=float(result.cost),nfev=result.nfev,status=result.status,message=result.message,
                  antenna_in_camera_objective=True,horizon_in_objective=False,components=graph(keys,obs),
                  note='Antenna errors, including conditional leave-one-out triangulation, are NOT independent validation: camera fitting used labels.')
    np.savez_compressed(root/'joint.npz',**{**state,'cameras':cam,'distortion':ks,'points':points,'antenna':ant})
    (root/'joint_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:report[k] for k in ['training_median_px','holdout_pair_median_px','antenna_median_px','antenna_max_px','horizon_median_equivalent_px']},flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_distortion_guided_v2');p.add_argument('--source',default='refined');p.add_argument('--max-nfev',type=int,default=200)
    run(**vars(p.parse_args()))
