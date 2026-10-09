"""Horizon-free, distortion-aware DEM-guided matching experiment.

New artifacts only. The antenna is triangulated AFTER camera fitting, never used
as a camera constraint. Horizon observations are diagnostics only. This is an
initializer/ablation, not an MCMC posterior or independent survey.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.ndimage import map_coordinates
from marjum_bundle import Terrain, PRM_ORDER, extract_features, build_tracks, verify_pair
from marjum_camera import project, rays, radial_support, epipolar_error


def lens_metadata(keys):
    from PIL import Image, ExifTags
    import pillow_heif
    pillow_heif.register_heif_opener()
    result={}
    for key in keys:
        im=Image.open(f'marjum-2026-07/IMG_{key}.HEIC')
        ex=im.getexif()
        fields=dict(ex);fields.update(ex.get_ifd(34665))
        fields={ExifTags.TAGS.get(k,k):str(v) for k,v in fields.items()}
        result[key]={k:fields.get(k) for k in ['Model','LensModel','FocalLength','FocalLengthIn35mmFilm','DigitalZoomRatio','Orientation']}
        result[key]['group']='ultrawide' if float(fields['FocalLength'])<3 else 'wide'
        im.close()
    return result


def intersect(terrain,p,shape,xy,k):
    from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
    direction=rays(p,shape,xy,k)
    distance=ray_distance_coarse_to_fine_numba(terrain.e,terrain.n,terrain.dem.data,
                 np.asarray(p[:3],np.float32),np.asarray(direction.T,np.float32)).astype(float)
    return p[:3]+distance[:,None]*direction,distance


def reserve(features):
    """Reserve spatial cells before matching; exclude previous CV training picks."""
    rng=np.random.default_rng(20260910)
    with np.load('cv_initialization_v3/state_1.npz') as s:
        previous={str(k):s['obs_xy'][s['obs_cam']==i] for i,k in enumerate(s['keys'])}
    training,testing={},{}
    for key,f in features.items():
        cell=np.floor(f['xy']/120).astype(int)
        cells,idx=np.unique(cell,axis=0,return_inverse=True)
        held=(rng.random(len(cells))<.35)[idx]
        # Keep an 8px buffer around reserved features, including duplicate SIFT orientations.
        training[key]=~held
        if held.any():
            training[key] &= cKDTree(f['xy'][held]).query(f['xy'])[0]>8
        testing[key]=held.copy()
        if key in previous:
            testing[key] &= cKDTree(previous[key]).query(f['xy'])[0]>8
    return training,testing


def initial_pairs(features,training,testing):
    bf=cv2.BFMatcher(cv2.NORM_L2)
    pairs,holdout=[],[]
    cv2.setRNGSeed(42)
    for a,b in combinations(features,2):
        fa,fb=features[a],features[b]
        def one(x,y):
            return {m.queryIdx:m.trainIdx for v in bf.knnMatch(x,y,k=2) if len(v)==2
                    for m,n in [v] if m.distance<.75*n.distance}
        ab=one(fa['descriptors'],fb['descriptors']);ba=one(fb['descriptors'],fa['descriptors'])
        ids=np.array([(i,j) for i,j in ab.items() if ba.get(j)==i],int).reshape(-1,2)
        if len(ids)<12:continue
        good,_=verify_pair(fa['xy'][ids[:,0]],fb['xy'][ids[:,1]],threshold=6.)
        ids=ids[good]
        train=ids[training[a][ids[:,0]] & training[b][ids[:,1]]]
        test=ids[testing[a][ids[:,0]] & testing[b][ids[:,1]]]
        if len(train)>=8:pairs.append((a,b,train))
        if len(test)>=5:holdout.append((a,b,test))
    return pairs,holdout


def triangulate(cams,shapes,ks,ci,xy):
    origins=cams[ci,:3]
    ds=np.array([rays(cams[i],shapes[i],q,ks[i])[0] for i,q in zip(ci,xy)])
    mat=np.eye(3)[None]-ds[:,:,None]*ds[:,None,:]
    if np.linalg.cond(mat.sum(axis=0))>1e8:return None
    return np.linalg.solve(mat.sum(axis=0),np.einsum('nij,nj->i',mat,origins))


def observations(keys,features,pairs,cams,ks,terrain,max_tracks=2400):
    tracks,conflicts=build_tracks(pairs)
    # Favor multi-view tracks, but retain spatially diverse two-view information.
    rng=np.random.default_rng(23);rng.shuffle(tracks)
    tracks.sort(key=len,reverse=True)
    shapes=[features[k]['shape'] for k in keys]
    hits={}
    for i,key in enumerate(keys):
        ids=sorted({fid for track in tracks for k,fid in track if k==key})
        if ids:
            points,_=intersect(terrain,cams[i],shapes[i],features[key]['xy'][ids],ks[i])
            hits.update({(key,fid):p for fid,p in zip(ids,points)})
    points=[];oc=[];op=[];xy=[];fid_out=[];accepted=[]
    for track in tracks:
        ci=[keys.index(k) for k,fid in track];q=np.array([features[k]['xy'][fid] for k,fid in track])
        point=triangulate(cams,shapes,ks,ci,q)
        candidates=[hits[(k,fid)] for k,fid in track]
        if point is not None:candidates.append(point)
        best=None
        for p in candidates:
            if not np.isfinite(p).all():continue
            height=terrain.height(*p[:2])
            if not np.isfinite(height) or abs(p[2]-height)>80:continue
            pred=[project(cams[i],shapes[i],p,ks[i]) for i in ci]
            if any(d[0]<1 for _,d in pred):continue
            error=np.median(np.linalg.norm(np.concatenate([v for v,_ in pred])-q,axis=1))
            if best is None or error<best[0]:best=error,p
        if best is None or best[0]>100:continue
        j=len(points);points.append(best[1]);oc.extend(ci);op.extend([j]*len(ci));xy.extend(q)
        fid_out.extend([fid for k,fid in track]);accepted.append(track)
        if len(points)>=max_tracks:break
    return dict(points=np.array(points),oc=np.array(oc),op=np.array(op),xy=np.array(xy),
                fid=np.array(fid_out),tracks=accepted,conflicts=conflicts)


class Fit:
    def __init__(self,keys,features,cams,ks,groups,obs,terrain,free_distortion=True):
        self.keys=keys;self.features=features;self.base=cams.copy();self.ks=ks.copy()
        self.group=np.array(groups);self.terrain=terrain;self.obs=obs;self.free=free_distortion
        self.shapes=[features[k]['shape'] for k in keys];self.nc=len(keys);self.np=len(obs['points'])
        self.ng=self.nc*7+(4 if free_distortion else 0)
        self.x0=np.zeros(self.ng+3*self.np)
        if free_distortion:
            self.x0[self.nc*7:self.ng]=np.array([ks[np.flatnonzero(self.group==g)[0]] for g in [0,1]]).ravel()
        ex=np.load('marjum_2026_07_exif.npz');ids=[list(ex['keys']).index(k) for k in keys]
        self.gps=np.c_[ex['e_gps'][ids],ex['n_gps'][ids]];self.alt=ex['u_gps'][ids]
        self.gps_sigma=np.maximum(ex['h_err_m'][ids],15.)
        self.focal=np.array([np.hypot(*s)/np.hypot(36,24) for s in self.shapes])*ex['focal_35mm'][ids]
        self.lower=np.r_[np.tile([-80,-80,-80,-.6,-.6,-.4,-.5],self.nc),
                         np.tile([-.15,-.04],2) if free_distortion else [],np.full(3*self.np,-400.)]
        self.upper=-self.lower
        for offset,pts in [(0,cams[:,:3]),(self.ng,obs['points'])]:
            stride=7 if offset==0 else 3
            for j,p in enumerate(pts):
                sl=slice(offset+stride*j,offset+stride*j+2)
                self.lower[sl]=np.maximum(self.lower[sl],[terrain.e[0]+2-p[0],terrain.n[0]+2-p[1]])
                self.upper[sl]=np.minimum(self.upper[sl],[terrain.e[-1]-2-p[0],terrain.n[-1]-2-p[1]])
        self.scale=np.r_[np.tile([10,10,10,.03,.03,.02,.05],self.nc),
                         np.tile([.02,.01],2) if free_distortion else [],np.full(3*self.np,10.)]
        rows=[]
        def camera(i):
            return list(range(7*i,7*i+7))+(list(range(7*self.nc+2*self.group[i],7*self.nc+2*self.group[i]+2)) if free_distortion else [])
        def point(j):return list(range(self.ng+3*j,self.ng+3*j+3))
        for i,j in zip(obs['oc'],obs['op']):rows.extend([camera(i)+point(j)]*2)
        for i,j in zip(obs['oc'],obs['op']):rows.append(camera(i)+point(j))
        rows.extend(point(j) for j in range(self.np))
        for i in range(self.nc):rows.extend([camera(i)]*2)
        for _ in range(3):rows.extend(camera(i) for i in range(self.nc))
        if free_distortion:
            rows.extend([[7*self.nc+j] for j in range(4)])
            for i in range(self.nc):rows.extend([camera(i)]*24)
        mat=lil_matrix((len(rows),len(self.x0)),dtype=int)
        for i,cols in enumerate(rows):mat[i,cols]=1
        self.sparsity=mat.tocsr()

    def unpack(self,x):
        d=x[:7*self.nc].reshape(-1,7);cams=self.base+d;cams[:,6]=self.base[:,6]*np.exp(d[:,6])
        ks=x[7*self.nc:self.ng].reshape(2,2)[self.group] if self.free else self.ks
        return cams,ks,self.obs['points']+x[self.ng:].reshape(-1,3)

    def residuals(self,x):
        cam,ks,points=self.unpack(x);o=self.obs
        pred=np.empty_like(o['xy']);depth=np.empty(len(pred))
        for i in range(self.nc):
            use=o['oc']==i;pred[use],depth[use]=project(cam[i],self.shapes[i],points[o['op'][use]],ks[i])
        r=np.r_[(pred-o['xy']).ravel()/3.,np.minimum(depth-1,0)/.1,
                (points[:,2]-self.terrain.height(points[:,0],points[:,1]))/5.,
                ((cam[:,:2]-self.gps)/self.gps_sigma[:,None]).ravel(),
                (cam[:,2]-self.alt)/30.,np.log(cam[:,6]/self.focal)/.3,
                np.minimum(cam[:,2]-self.terrain.height(cam[:,0],cam[:,1])-.1,0)/1.]
        if self.free:
            r=np.r_[r,x[7*self.nc:self.ng]/np.tile([.05,.02],2),
                    np.concatenate([np.minimum(radial_support(p,s,k)-.2,0)*100 for p,s,k in zip(cam,self.shapes,ks)])]
        return r

    def loss(self,z):
        robust=np.zeros(len(z),bool);n=len(self.obs['xy']);robust[:2*n]=True;robust[3*n:3*n+self.np]=True
        rho=np.array([z,np.ones_like(z),np.zeros_like(z)]);t=1+z[robust]
        rho[:,robust]=[2*(np.sqrt(t)-1),1/np.sqrt(t),-.5*t**(-1.5)]
        return rho

    def solve(self,max_nfev):
        t=time.monotonic()
        result=least_squares(self.residuals,self.x0,jac_sparsity=self.sparsity,bounds=(self.lower,self.upper),
                             x_scale=self.scale,loss=self.loss,f_scale=2,max_nfev=max_nfev,ftol=1e-5,xtol=1e-6)
        print('fit',self.free,'cost',result.cost,'nfev',result.nfev,'seconds',time.monotonic()-t,flush=True)
        return result


def patch_score(p1,s1,k1,p2,s2,k2,a,b,gray1,gray2,terrain,refine=False):
    """DEM tangent-plane perspective warp; image intensities determine agreement."""
    point=triangulate(np.array([p1,p2]),[s1,s2],np.array([k1,k2]),[0,1],np.array([a,b]))
    if point is None or not np.isfinite(point).all():return -1.,np.zeros(2)
    h=terrain.height(*point[:2])
    if not np.isfinite(h):return -1.,np.zeros(2)
    gx=(terrain.height(point[0]+1,point[1])-terrain.height(point[0]-1,point[1]))/2
    gy=(terrain.height(point[0],point[1]+1)-terrain.height(point[0],point[1]-1))/2
    normal=np.array([-gx,-gy,1.]);normal/=np.linalg.norm(normal)
    delta=np.array(np.meshgrid(np.linspace(-24,24,7),np.linspace(-24,24,7))).reshape(2,-1).T
    aa=a+delta;ds=rays(p1,s1,aa,k1);den=ds@normal
    if np.min(abs(den))<.03:return -1.,np.zeros(2)
    distance=((point-p1[:3])@normal)/den
    if np.min(distance)<1:return -1.,np.zeros(2)
    bb,depth=project(p2,s2,p1[:3]+distance[:,None]*ds,k2)
    center=project(p2,s2,point,k2)[0][0];bb+=b-center
    if np.min(depth)<1:return -1.,np.zeros(2)
    def sample(gray,shape,q):
        factor=np.array([gray.shape[1]/shape[1],gray.shape[0]/shape[0]])
        v=(q+.5)*factor-.5
        return map_coordinates(gray,[v[:,1],v[:,0]],order=1,mode='constant',cval=np.nan,output=np.float64)
    va=sample(gray1,s1,aa)
    def unit(v):return (v-v.mean())/max(np.linalg.norm(v-v.mean()),1e-9)
    if not np.isfinite(va).all() or np.std(va)<2:return -1.,np.zeros(2)
    target=unit(va)
    def residual(shift):
        vb=sample(gray2,s2,bb+shift)
        return unit(vb)-target if np.isfinite(vb).all() else np.full(len(va),10.)
    shift=np.zeros(2)
    if refine:
        fit=least_squares(residual,shift,bounds=(-4.,4.),max_nfev=15,diff_step=.05)
        shift=fit.x
    err=residual(shift)
    return float(1-.5*(err@err)),shift


def guided_pairs(keys,features,training,cams,ks,terrain,gray):
    shapes=[features[k]['shape'] for k in keys];hits={};distances={};trees={}
    for i,key in enumerate(keys):
        hit,d=intersect(terrain,cams[i],shapes[i],features[key]['xy'],ks[i]);hits[key]=hit;distances[key]=d
        ids=np.flatnonzero(training[key]);trees[key]=(ids,cKDTree(features[key]['xy'][ids]))
    pairs=[];report=[]
    def direction(a,b,ia,ib):
        fa,fb=features[a],features[b];target_ids,tree=trees[b]
        source_ids=np.flatnonzero(training[a]&np.isfinite(distances[a]))
        ray=rays(cams[ia],shapes[ia],fa['xy'][source_ids],ks[ia]);distance=distances[a][source_ids]
        xyz=cams[ia,:3]+(distance[:,None,None]*np.array([.6,.8,1.,1.2,1.5])[None,:,None])*ray[:,None,:]
        q,depth=project(cams[ib],shapes[ib],xyz.reshape(-1,3),ks[ib]);q=q.reshape(-1,5,2);depth=depth.reshape(-1,5)
        neighborhoods=tree.query_ball_point(q.reshape(-1,2),40.)
        result={}
        for n,idx in enumerate(source_ids):
            valid=[neighborhoods[5*n+j] for j in range(5) if depth[n,j]>1]
            if not valid:continue
            candidates=np.unique(np.concatenate(valid)).astype(int)
            if len(candidates)<2:continue
            ids=target_ids[candidates]
            # Broad mutual-visibility check; the DEM is deliberately not exact.
            nominal=np.linalg.norm(hits[a][idx]-cams[ib,:3])
            ok=(~np.isfinite(distances[b][ids]))|(nominal<distances[b][ids]+np.maximum(30.,.2*distances[b][ids]))
            ids=ids[ok]
            if len(ids)<2:continue
            error=np.sum((fb['descriptors'][ids]-fa['descriptors'][idx])**2,axis=1)
            best=np.argsort(error)[:2]
            if error[best[0]]<.8**2*error[best[1]]:result[int(idx)]=int(ids[best[0]])
        return result
    for ia,ib in combinations(range(len(keys)),2):
        a,b=keys[ia],keys[ib];fa,fb=features[a],features[b]
        ab=direction(a,b,ia,ib);ba=direction(b,a,ib,ia)
        ids=np.array([(i,j) for i,j in ab.items() if ba.get(j)==i],int).reshape(-1,2)
        accepted=[]
        for i,j in ids:
            score,_=patch_score(cams[ia],shapes[ia],ks[ia],cams[ib],shapes[ib],ks[ib],fa['xy'][i],fb['xy'][j],gray[a],gray[b],terrain)
            if score>.65:accepted.append((i,j))
        if len(accepted)>=8:pairs.append((a,b,np.array(accepted,int)))
        report.append(dict(a=a,b=b,mutual=len(ids),photometric=len(accepted)))
        if len(accepted)>=8:print('guided',a,b,len(ids),'->',len(accepted),flush=True)
    return pairs,report


def refine_observations(obs,keys,features,cams,ks,terrain,gray):
    result={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in obs.items()}
    shifts=[]
    for j in range(len(obs['points'])):
        idx=np.flatnonzero(obs['op']==j);anchor=idx[0];ia=obs['oc'][anchor];a=keys[ia]
        for row in idx[1:]:
            ib=obs['oc'][row];b=keys[ib]
            score,shift=patch_score(cams[ia],features[a]['shape'],ks[ia],cams[ib],features[b]['shape'],ks[ib],
                                    obs['xy'][anchor],obs['xy'][row],gray[a],gray[b],terrain,refine=True)
            if score>.8 and np.max(abs(shift))<3.95:
                result['xy'][row]+=shift;shifts.append(np.linalg.norm(shift))
    return result,dict(adjusted=len(shifts),median_shift_px=float(np.median(shifts)) if shifts else None)


def graph(keys,obs):
    groups=[{i} for i in range(len(keys))]
    for j in range(len(obs['points'])):
        nodes=set(obs['oc'][obs['op']==j]);touch=[g for g in groups if g&nodes]
        merged=set().union(*touch);groups=[g for g in groups if not g&nodes]+[merged]
    return [[keys[i] for i in sorted(g)] for g in groups]


def evaluate(keys,features,cams,ks,terrain,holdout,obs):
    meta=json.loads(Path('meta.json').read_text());shapes=[features[k]['shape'] for k in keys]
    ai=[i for i,k in enumerate(keys) if 'ant_px' in meta.get(k,{})];axy=np.array([meta[keys[i]]['ant_px'] for i in ai])
    def ant_fit(indices,start):
        return least_squares(lambda a:np.concatenate([project(cams[ai[j]],shapes[ai[j]],a,ks[ai[j]])[0][0]-axy[j] for j in indices]),start,max_nfev=100).x
    from eigsep_terrain.fitio import load_fit
    _,start,_=load_fit('cv_initialization_v3/candidate_1.npz');ant=ant_fit(range(len(ai)),start)
    ant_errors=np.array([np.linalg.norm(project(cams[i],shapes[i],ant,ks[i])[0][0]-q) for i,q in zip(ai,axy)])
    loo=[]
    for j,i in enumerate(ai):
        point=ant_fit([v for v in range(len(ai)) if v!=j],ant)
        loo.append(float(np.linalg.norm(project(cams[i],shapes[i],point,ks[i])[0][0]-axy[j])))
    held=[]
    for a,b,ids in holdout:
        ia,ib=keys.index(a),keys.index(b)
        error=epipolar_error(cams[ia],shapes[ia],ks[ia],cams[ib],shapes[ib],ks[ib],features[a]['xy'][ids[:,0]],features[b]['xy'][ids[:,1]])
        if np.isfinite(error).any():held.append(dict(a=a,b=b,count=len(ids),median_px=float(np.nanmedian(error))))
    horizon=[]
    for i,key in enumerate(keys):
        d=rays(cams[i],shapes[i],features[key]['horizon'],ks[i]);az=np.arctan2(d[:,1],d[:,0])
        modeled=terrain.skyline(cams[i,:3],az,3072)
        r=(np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))-modeled)*cams[i,6]
        horizon.append(dict(key=key,rms_equivalent_px=float(np.sqrt(np.mean(r*r)))))
    pred=np.empty_like(obs['xy'])
    for i in range(len(keys)):
        use=obs['oc']==i;pred[use]=project(cams[i],shapes[i],obs['points'][obs['op'][use]],ks[i])[0]
    return dict(antenna_enu=ant.tolist(),antenna_median_px=float(np.median(ant_errors)),antenna_max_px=float(ant_errors.max()),
                antenna_loo_median_px=float(np.median(loo)),antenna_images=[dict(key=keys[i],error_px=float(e),loo_px=l) for i,e,l in zip(ai,ant_errors,loo)],
                holdout=held,holdout_pair_median_px=float(np.median([v['median_px'] for v in held])),horizon=horizon,
                horizon_median_equivalent_px=float(np.median([v['rms_equivalent_px'] for v in horizon])),
                training_median_px=float(np.median(np.linalg.norm(pred-obs['xy'],axis=1))),
                distortion={k:ks[i].tolist() for i,k in enumerate(keys)},
                camera_clearance_m=(cams[:,2]-terrain.height(cams[:,0],cams[:,1])).tolist(),
                radial_min_derivative=min(float(radial_support(p,s,k).min()) for p,s,k in zip(cams,shapes,ks)))


def run(output='cv_distortion_guided',max_nfev=160):
    from eigsep_terrain.fitio import load_fit
    from eigsep_terrain.marjum_dem import MarjumDEM
    from eigsep_terrain.imageio import load_image
    cv2.setNumThreads(2)
    out=Path(output);out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use a fresh output directory')
    p,_,_=load_fit('fit_bundle_v2.npz');selected,_,_=load_fit('cv_initialization_v3/candidate_1.npz');p.update(selected)
    keys=sorted(p);cams=np.array([[p[k][q] for q in PRM_ORDER] for k in keys]);ks=np.zeros((len(keys),2))
    metadata=lens_metadata(keys);groups=[int(metadata[k]['group']=='ultrawide') for k in keys]
    terrain=Terrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
    # Explicitly repair ONLY newly reintroduced below-DEM starts, recording the shifts.
    before=cams[:,2].copy();cams[:,2]=np.maximum(cams[:,2],terrain.height(cams[:,0],cams[:,1])+2.)
    features=extract_features(Path('.'),keys,Path('cv_features'))
    training,testing=reserve(features);pairs,holdout=initial_pairs(features,training,testing)
    np.savez_compressed(out/'holdout.npz',**{f'{a}_{b}':ids for a,b,ids in holdout})
    manifest=dict(keys=keys,lens_metadata=metadata,initial_altitude_repairs_m=(cams[:,2]-before).tolist(),
                  horizon_in_objective=False,antenna_in_camera_objective=False,max_nfev=max_nfev,
                  description='Processed-image shared radial distortion; guided matching + tangent-plane photometric checks. No ML weights required.')
    files=[Path(__file__),Path('marjum_camera.py'),Path('marjum_bundle.py'),Path('meta.json'),Path('marjum_dem_sw.npz'),Path('marjum_2026_07_exif.npz'),Path('fit_bundle_v2.npz'),Path('cv_initialization_v3/candidate_1.npz'),Path('cv_initialization_v3/state_1.npz')]
    files += [Path(f'cv_features/sift_{k}.npz') for k in keys]+[Path(f'marjum-2026-07/IMG_{k}.HEIC') for k in keys]
    from marjum_mcmc import digest
    manifest['input_sha256']={str(f):digest(f) for f in files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez_compressed(out/'reservations.npz',**{f'train_{k}':v for k,v in training.items()},**{f'test_{k}':v for k,v in testing.items()})
    obs=observations(keys,features,pairs,cams,ks,terrain)
    print('bootstrap tracks',len(obs['points']),'observations',len(obs['xy']),'holdout pairs',len(holdout),flush=True)
    reports={}
    def save(label,cam,k,o,result=None):
        report=evaluate(keys,features,cam,k,terrain,holdout,o);report['components']=graph(keys,o)
        report['tracks']=len(o['points']);report['observations']=len(o['xy'])
        if result is not None:report.update(cost=float(result.cost),nfev=result.nfev,status=result.status,message=result.message)
        reports[label]=report
        # NOT legacy fitio: distortion is essential and must not be silently discarded.
        np.savez_compressed(out/f'{label}.npz',keys=np.array(keys),cameras=cam,distortion=k,groups=groups,
             antenna=np.array(report['antenna_enu']),points=o['points'],obs_cam=o['oc'],obs_point=o['op'],obs_xy=o['xy'],obs_fid=o['fid'],shapes=np.array([features[q]['shape'] for q in keys]))
        (out/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(label,{q:report[q] for q in ['training_median_px','holdout_pair_median_px','antenna_loo_median_px','horizon_median_equivalent_px','components']},flush=True)
    save('initial',cams,ks,obs)
    for name,free in [('pinhole',False),('radial',True)]:
        model=Fit(keys,features,cams,ks,groups,obs,terrain,free)
        fit=model.solve(max_nfev);cam,k,points=model.unpack(fit.x);o={**obs,'points':points};save(name,cam,k,o,fit)
        if free:radial_cam,radial_k=cam,k
    gray={}
    for key in keys:
        rgb=np.flipud(load_image(f'marjum-2026-07/IMG_{key}.HEIC'))
        h,w=rgb.shape[:2];fac=1600/max(h,w)
        gray[key]=cv2.cvtColor(cv2.resize(rgb,(round(w*fac),round(h*fac)),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
    new_pairs,guided_report=guided_pairs(keys,features,training,radial_cam,radial_k,terrain,gray)
    (out/'guided_matches.json').write_text(json.dumps(guided_report,indent=2)+'\n')
    # Put photometrically checked guided edges first in conflict resolution.
    combined=observations(keys,features,new_pairs+pairs,radial_cam,radial_k,terrain,max_tracks=3200)
    model=Fit(keys,features,radial_cam,radial_k,groups,combined,terrain,True)
    fit=model.solve(max_nfev);cam,k,points=model.unpack(fit.x);combined={**combined,'points':points};save('guided',cam,k,combined,fit)
    refined,refine_report=refine_observations(combined,keys,features,cam,k,terrain,gray)
    (out/'patch_refinement.json').write_text(json.dumps(refine_report,indent=2)+'\n')
    model=Fit(keys,features,cam,k,groups,refined,terrain,True)
    fit=model.solve(max_nfev);cam,k,points=model.unpack(fit.x);save('refined',cam,k,{**refined,'points':points},fit)
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_distortion_guided');p.add_argument('--max-nfev',type=int,default=160)
    run(**vars(p.parse_args()))
