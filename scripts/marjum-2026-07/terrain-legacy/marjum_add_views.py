"""Register additional Marjum views against the fixed 17-view 3D model.

The established cameras, landmarks, distortion coefficients, and antenna are
never changed. New views are initialized by robust 2D-to-3D landmark matching,
then their pose and focal length are refined using feature reprojection and a
tree-filtered horizon boundary. Reserved spatial cells never enter the fit.
"""
from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from marjum_bundle import PRM_ORDER, Terrain, boundary_pixels, extract_features
from marjum_camera import project, rays, radial_support
from marjum_guided import lens_metadata
from marjum_grid import GridCoordinates
from marjum_mcmc import digest
from marjum_position import RefinedTerrain


def segment_missing(keys,device='cuda'):
    """Create standard psky/ptree/skymask caches using the library segmenter."""
    missing=[k for k in keys if not Path(f'img_seg_IMG_{k}.npz').exists()]
    if not missing:return []
    from eigsep_terrain.seg import TiledSkyProbSegFormer
    from eigsep_terrain.utils import fill_psky_holes
    model=TiledSkyProbSegFormer(device=device)
    # The installed GPU wrapper constructs half-precision weights but leaves
    # processor tensors as float32. Keep both sides float32; b0 fits in memory
    # one 1024-pixel tile at a time and this also matches the established CPU path.
    if device.startswith('cuda'):model.model.float()
    for key in missing:
        photo=f'marjum-2026-07/IMG_{key}.HEIC'
        psky,ptree=model.p_sky_tiled(photo,tile=1024,overlap=256,batch=1)
        sky,_=fill_psky_holes(psky,.6,200**2,8,150)
        np.savez_compressed(f'img_seg_IMG_{key}.npz',skymask=sky,psky=psky,ptree=ptree)
        print('segmented',key,psky.shape,flush=True)
    return missing


def feature_landmarks(state,features,target):
    """Mutual ratio matches from one target to fitted landmark observations."""
    keys=list(state['keys'].astype(str));bf=cv2.BFMatcher(cv2.NORM_L2)
    candidates={}
    for i,key in enumerate(keys):
        rows=np.flatnonzero(state['obs_cam']==i)
        # Each fitted feature occurs at most once per camera; preserve that map.
        fid=state['obs_fid'][rows].astype(int);pid=state['obs_point'][rows].astype(int)
        unique,index=np.unique(fid,return_index=True);fid=fid[index];pid=pid[index]
        if len(fid)<8:continue
        da=features[target]['descriptors'];db=features[key]['descriptors'][fid]
        def ratio(a,b):
            return {m.queryIdx:m.trainIdx for pair in bf.knnMatch(a,b,k=2) if len(pair)==2
                    for m,n in [pair] if m.distance<.78*n.distance}
        ab=ratio(da,db);ba=ratio(db,da)
        for tfid,j in ab.items():
            if ba.get(j)!=tfid:continue
            point=int(pid[j]);distance=float(np.sum((da[tfid]-db[j])**2))
            row=candidates.setdefault(int(tfid),{}).setdefault(point,dict(votes=0,distance=np.inf,references=[]))
            row['votes']+=1;row['distance']=min(row['distance'],distance);row['references'].append(key)
    selected=[]
    for fid,choices in candidates.items():
        point,row=min(choices.items(),key=lambda v:(-v[1]['votes'],v[1]['distance']))
        selected.append((fid,point,row['votes'],row['distance'],row['references']))
    selected.sort(key=lambda r:(-r[2],r[3]))
    return selected


def dem_landmarks(target,features,references,terrain,ratio_threshold=.78):
    """Match to all reference pixels and ray-trace them onto the fitted DEM."""
    from marjum_guided import intersect
    bf=cv2.BFMatcher(cv2.NORM_L2);xyz=[];xy=[];source=[];source_fid=[];target_fid=[]
    def ratio(a,b):
        return {m.queryIdx:m.trainIdx for pair in bf.knnMatch(a,b,k=2) if len(pair)==2
                for m,n in [pair] if m.distance<ratio_threshold*n.distance}
    for ref in references:
        key=ref['key'];a=features[target];b=features[key]
        if key==target:continue
        ab=ratio(a['descriptors'],b['descriptors']);ba=ratio(b['descriptors'],a['descriptors'])
        ids=np.array([(i,j) for i,j in ab.items() if ba.get(j)==i],int).reshape(-1,2)
        if not len(ids):continue
        hit,distance=intersect(terrain,ref['camera'],b['shape'],b['xy'][ids[:,1]],ref['distortion'])
        good=np.isfinite(distance)&np.isfinite(hit).all(axis=1)
        xyz.extend(hit[good]);xy.extend(a['xy'][ids[good,0]]);source.extend([key]*int(good.sum()))
        source_fid.extend(ids[good,1]);target_fid.extend(ids[good,0])
    return dict(xyz=np.asarray(xyz,float).reshape(-1,3),xy=np.asarray(xy,float).reshape(-1,2),
                source=np.asarray(source),source_fid=np.asarray(source_fid,int),target_fid=np.asarray(target_fid,int))


def spatial_split(xy):
    cell=np.floor(np.asarray(xy)/120).astype(np.int64)
    code=(cell[:,0]*73856093)^(cell[:,1]*19349663)
    holdout=(code&3)==0
    # Always retain enough geometry in each side for a meaningful check.
    if holdout.sum()<8 or (~holdout).sum()<12:
        order=np.arange(len(xy));holdout=(order%4)==1
    return ~holdout,holdout


def pnp_pose(xyz,xy,shape,focal,k):
    h,w=shape
    # The project stores images bottom-up; OpenCV camera y points downward.
    # Flip rows for PnP so the body-to-OpenCV transform is a proper rotation,
    # not a reflection. The half-pixel offset follows y_cv = h - 1 - y.
    xy_cv=np.asarray(xy,np.float64).copy();xy_cv[:,1]=h-1-xy_cv[:,1]
    matrix=np.array([[focal,0,w//2],[0,focal,h-1-h//2],[0,0,1.]],float)
    dist=np.array([k[0],k[1],0.,0.],float)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(np.asarray(xyz,np.float64),xy_cv,matrix,dist,
        iterationsCount=4000,reprojectionError=10.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<8:return None,None
    rcv=cv2.Rodrigues(rvec)[0];position=(-rcv.T@tvec).ravel()
    axis=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    body=rcv.T@axis
    ph,th,ti=Rotation.from_matrix(body).as_euler('ZYZ')
    pose=np.array([*position,th,ph,ti,focal])
    return pose,inliers.ravel()


def feature_refine(start,xyz,xy,shape,k,focal_prior,terrain,max_nfev=150):
    lo=np.array([terrain.e[0]+2,terrain.n[0]+2,terrain.data.min()-50,-np.pi,-20,-np.pi,np.log(.45)])
    hi=np.array([terrain.e[-1]-2,terrain.n[-1]-2,terrain.data.max()+150,np.pi,20,np.pi,np.log(1.8)])
    q0=np.r_[start[:6],np.log(start[6]/focal_prior)]
    def unpack(q):return np.r_[q[:6],focal_prior*np.exp(q[6])]
    def residual(q):
        p=unpack(q);pred,depth=project(p,shape,xyz,k)
        return np.r_[(pred-xy).ravel()/3.,np.minimum(depth-1,0)/.1,q[6]/.30,
                     min(p[2]-terrain.height(p[0],p[1])-1.,0.)/.2]
    fit=least_squares(residual,q0,bounds=(lo,hi),x_scale=[10,10,10,.03,.03,.02,.05],
                      loss='soft_l1',f_scale=2,max_nfev=max_nfev,ftol=1e-7,xtol=1e-8)
    return unpack(fit.x),fit


def horizon_error(p,shape,k,boundary,terrain):
    direction=rays(p,shape,boundary,k);az=np.arctan2(direction[:,1],direction[:,0])
    observed=np.arctan2(direction[:,2],np.hypot(direction[:,0],direction[:,1]))
    return (observed-terrain.skyline(p[:3],az))*p[6]


def joint_refine(start,xyz,xy,shape,k,focal_prior,terrain,boundary,htrain,
                 gps=None,antenna=None,antenna_xy=None,antenna_sigma=3.,max_nfev=120):
    lo=np.array([terrain.e[0]+2,terrain.n[0]+2,terrain.data.min()-50,-np.pi,-20,-np.pi,np.log(.45)])
    hi=np.array([terrain.e[-1]-2,terrain.n[-1]-2,terrain.data.max()+150,np.pi,20,np.pi,np.log(1.8)])
    q0=np.r_[start[:6],np.log(start[6]/focal_prior)]
    def unpack(q):return np.r_[q[:6],focal_prior*np.exp(q[6])]
    def residual(q):
        p=unpack(q);pred,depth=project(p,shape,xyz,k)
        r=[((pred-xy)/3.).ravel(),np.minimum(depth-1,0)/.1,[q[6]/.30],horizon_error(p,shape,k,boundary[htrain],terrain)/10.]
        clearance=p[2]-terrain.height(p[0],p[1]);r.append([min(clearance-1.,0.)/.2])
        if gps is not None:r.extend([(p[:2]-gps[:2])/15.,[(p[2]-gps[2])/30.]])
        if antenna is not None and antenna_xy is not None:
            ant_pred,ant_depth=project(p,shape,antenna,k)
            r.extend([((ant_pred[0]-antenna_xy)/antenna_sigma).ravel(),
                      [min(ant_depth[0]-1.,0.)/.1]])
        return np.concatenate(r)
    fit=least_squares(residual,q0,bounds=(lo,hi),x_scale=[10,10,10,.03,.03,.02,.05],
                      loss='soft_l1',f_scale=2,max_nfev=max_nfev,ftol=2e-7,xtol=1e-8)
    return unpack(fit.x),fit


def probability_score(key,p,shape,k,dem,step=12):
    from eigsep_terrain.img import HorizonImage
    from eigsep_terrain.ray_numba import ray_distance_coarse_to_fine_numba
    image=HorizonImage(f'marjum-2026-07/IMG_{key}.HEIC',px_smooth=150,px_dist=30)
    rr,cc=np.mgrid[step//2:shape[0]:step,step//2:shape[1]:step]
    use=image.horizon_mask[rr,cc];rr,cc=rr[use],cc[use]
    direction=rays(p,shape,np.c_[cc,rr],k)
    distance=ray_distance_coarse_to_fine_numba(*dem.get_en(),dem.data,np.asarray(p[:3],np.float32),np.asarray(direction.T,np.float32))
    probability=image.psky[rr,cc].clip(1e-3,1-1e-3);model=np.isnan(distance)
    neff=float(np.clip(np.ptp(cc)/image.px_smooth,1,len(cc)))
    return dict(samples=len(cc),effective_samples=neff,sky_nll=float(-np.where(model,np.log(probability),np.log1p(-probability)).mean()*neff))


def run(output='cv_position_add_views',targets=('2209','2210','2211'),source='cv_position_absolute/focal.npz',device='cuda',max_nfev=150,fixed_references=False,pixel_field='ant_px',position_field='antenna',pixel_sigma=3.):
    from eigsep_terrain.marjum_dem import MarjumDEM
    root=Path(output);root.mkdir(exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Use a new output directory')
    generated=segment_missing(targets,device)
    all_features=extract_features(Path('.'),list(targets),Path('cv_features'))
    state=dict(np.load(source));refkeys=list(state['keys'].astype(str))
    for key in refkeys:
        with np.load(f'cv_features/sift_{key}.npz') as z:all_features[key]={n:z[n] for n in z.files}
    dem=MarjumDEM(cache_file='marjum_dem_sw.npz');terrain=RefinedTerrain(dem);grid=GridCoordinates(dem)
    groups=state['groups']
    group_k={g:np.median(state['distortion'][groups==g],axis=0)
             for g in np.unique(groups)}
    lens=lens_metadata(list(targets));ex=np.load('marjum_2026_07_exif.npz');exkeys=list(ex['keys'].astype(str))
    meta=json.loads(Path('meta.json').read_text());cameras={};camera_distortion={};camera_group={};reports={}
    references=[dict(key=k,camera=state['cameras'][i],distortion=state['distortion'][i]) for i,k in enumerate(refkeys)]
    # 2209 has the strongest direct DEM bridge; after it is registered, 2211
    # and then 2210 gain hundreds of close-view matches while remaining tied
    # to the fixed 17-view coordinate system through 2209.
    registration_order=[k for k in ['2209','2211','2210'] if k in targets]+[k for k in targets if k not in ['2209','2211','2210']]
    if all(k in targets for k in ['2209','2210']):registration_order.append('2209')
    for key in registration_order:
        feature=all_features[key];shape=tuple(feature['shape']);matches=dem_landmarks(key,all_features,references,terrain)
        xy=matches['xy'];xyz=matches['xyz']
        if len(xy)<12:raise RuntimeError(f'{key}: only {len(xy)} DEM bridge matches')
        train,holdout=spatial_split(xy);f35=float(lens[key]['FocalLengthIn35mmFilm'])
        focal_prior=np.hypot(*shape)/np.hypot(36,24)*f35
        group=int(lens[key]['group']=='ultrawide')
        if group not in group_k:raise RuntimeError(f'{key}: no fitted distortion for lens group {group}')
        k=group_k[group].copy()
        starts=[]
        for scale in [.7,.85,1.,1.15,1.3]:
            # RANSAC initialization sees all matches; reserved cells are excluded
            # from every nonlinear refinement and reported as diagnostics, not as
            # a fully independent test of the selected PnP hypothesis.
            p,inlier=pnp_pose(xyz,xy,shape,focal_prior*scale,k)
            if p is None:continue
            initial_error=np.linalg.norm(project(p,shape,xyz,k)[0]-xy,axis=1)
            starts.append((-len(inlier),float(np.median(initial_error[inlier])),p,inlier))
        if not starts:raise RuntimeError(f'{key}: no PnP initialization')
        _,_,p,inlier=min(starts,key=lambda v:(v[0],v[1]))
        good=np.array([j for j in inlier if train[j]],int)
        if len(good)<6:good=np.asarray(inlier,int)
        p,fit=feature_refine(p,xyz[good],xy[good],shape,k,focal_prior,terrain,max_nfev)
        pred,depth=project(p,shape,xyz,k);error=np.linalg.norm(pred-xy,axis=1)
        selected=np.flatnonzero(train&(depth>1)&(error<15.))
        if len(selected)>=6:
            good=selected;p,fit=feature_refine(p,xyz[good],xy[good],shape,k,focal_prior,terrain,max_nfev)
        boundary=np.asarray(feature['horizon'],float);htrain=np.arange(len(boundary))%3!=1
        gps=None;i=exkeys.index(key) if key in exkeys else None
        if i is not None and bool(ex['has_gps'][i]):
            gps_xy,alt,_=grid.gps([key]);bias=state['gps_bias'] if 'gps_bias' in state else np.zeros(3)
            gps=np.r_[gps_xy[0],alt[0]]-bias
        landmark_xy=meta.get(key,{}).get(pixel_field)
        landmark_position=state.get(position_field)
        p,jfit=joint_refine(p,xyz[good],xy[good],shape,k,focal_prior,terrain,
                            boundary,htrain,gps,landmark_position,landmark_xy,
                            antenna_sigma=pixel_sigma,
                            max_nfev=max_nfev)
        train_error=np.linalg.norm(project(p,shape,xyz[good],k)[0]-xy[good],axis=1)
        held_error=np.linalg.norm(project(p,shape,xyz[holdout],k)[0]-xy[holdout],axis=1)
        he=horizon_error(p,shape,k,boundary,terrain)
        source_counts={str(v):int(np.sum(matches['source']==v)) for v in np.unique(matches['source'])}
        landmark_error=None
        if landmark_xy is not None and landmark_position is not None:
            landmark_error=float(np.linalg.norm(
                project(p,shape,landmark_position,k)[0][0]-landmark_xy))
        report=dict(raw_dem_bridge_matches=len(xy),bridge_source_counts=source_counts,fit_landmarks=len(good),
            training_reprojection_median_px=float(np.median(train_error)),heldout_raw_reprojection_median_px=float(np.median(held_error)),
            heldout_within_10px=int((held_error<10).sum()),heldout_count=int(holdout.sum()),
            horizon_training_rms_px=float(np.sqrt(np.mean(he[htrain]**2))),horizon_heldout_rms_px=float(np.sqrt(np.mean(he[~htrain]**2))),
            focal_px=float(p[6]),focal_prior_px=float(focal_prior),pose=p.tolist(),distortion=k.tolist(),
            landmark_pixel_field=pixel_field,landmark_position_field=position_field,
            landmark_constraint_used=landmark_xy is not None and landmark_position is not None,
            landmark_residual_px=landmark_error,
            antenna_constraint_used=(pixel_field=='ant_px' and landmark_xy is not None),
            antenna_residual_px=landmark_error if pixel_field=='ant_px' else None,
            camera_clearance_m=float(p[2]-terrain.height(p[0],p[1])),gps_prior_grid_m=None if gps is None else gps.tolist(),
            probability=probability_score(key,p,shape,k,dem),optimizer=dict(cost=float(jfit.cost),nfev=int(jfit.nfev),status=int(jfit.status),message=jfit.message))
        reports[key]=report;cameras[key]=p;camera_distortion[key]=k;camera_group[key]=group
        if not fixed_references:
            references.append(dict(key=key,camera=p,distortion=k))
        print(key,json.dumps(report,indent=2),flush=True)
    extended=dict(state);extended.update(keys=np.r_[state['keys'],np.array(targets)],cameras=np.vstack([state['cameras'],[cameras[k] for k in targets]]),
        distortion=np.vstack([state['distortion'],[camera_distortion[k] for k in targets]]),groups=np.r_[groups,[camera_group[k] for k in targets]],
        shapes=np.vstack([state['shapes'],[all_features[k]['shape'] for k in targets]]))
    # Optimization vector belongs only to the 17-view fit and cannot describe appended cameras.
    extended.pop('x',None)
    np.savez_compressed(root/'fit_extended.npz',**extended)
    inputs=[Path(__file__),Path(source),Path('marjum_camera.py'),Path('marjum_position.py'),Path('marjum_grid.py'),Path('meta.json'),Path('marjum_dem_sw.npz')]
    inputs += [Path(f'marjum-2026-07/IMG_{k}.HEIC') for k in targets]+[Path(f'img_seg_IMG_{k}.npz') for k in targets]+[Path(f'cv_features/sift_{k}.npz') for k in targets]
    summary=dict(source=source,targets=list(targets),generated_segmentations=generated,reference_cameras_fixed=True,
        reference_landmarks_fixed=True,candidate_views_used_as_references=not fixed_references,
        landmark_position_fixed=True,landmark_pixel_field=pixel_field,
        landmark_position_field=position_field,landmark_pixel_sigma=pixel_sigma,
        distortion_assignment='EXIF physical lens group; median fitted coefficients per group',
        distortion_by_group={str(g):k.tolist() for g,k in group_k.items()},reports=reports,
        input_sha256={str(p):digest(p) for p in inputs})
    (root/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='cv_position_add_views');p.add_argument('--targets',nargs='+',default=['2209','2210','2211']);p.add_argument('--source',default='cv_position_absolute/focal.npz');p.add_argument('--device',default='cuda');p.add_argument('--max-nfev',type=int,default=150);p.add_argument('--fixed-references',action='store_true',help='Do not let unvalidated targets become references for later targets');p.add_argument('--pixel-field',default='ant_px');p.add_argument('--position-field',default='antenna');p.add_argument('--pixel-sigma',type=float,default=3.)
    run(**vars(p.parse_args()))
