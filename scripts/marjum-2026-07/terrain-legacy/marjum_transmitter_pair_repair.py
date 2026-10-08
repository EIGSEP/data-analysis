"""Repair 2159/2199 using explicit skyline exclusions and shared 3D tracks.

Writes a new state and complete compatible feature cache; never overwrites the
source products. Other camera poses, antenna and transmitter remain fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from marjum_bundle import boundary_pixels, rotation
from marjum_camera import rays, project, normalized, epipolar_error
from marjum_position import RefinedTerrain
from eigsep_terrain.marjum_dem import MarjumDEM

KEYS = ['2159', '2199']
CV_BODY = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])


def pose_angles(R):
    return [np.arccos(np.clip(R[2, 2], -1, 1)), np.arctan2(R[1, 2], R[0, 2]),
            np.arctan2(R[2, 1], -R[2, 0])]


def mutual(a, b):
    bf = cv2.BFMatcher()
    def match(x, y):
        return {m.queryIdx: m.trainIdx for pair in bf.knnMatch(x, y, k=2)
                if len(pair)==2 for m,n in [pair] if m.distance < .78*n.distance}
    ab, ba = match(a,b), match(b,a)
    return np.array([(i,j) for i,j in ab.items() if ba.get(j)==i], int).reshape(-1,2)


def triangulate(cams, shapes, ks, xy):
    d = [rays(cams[i], shapes[i], xy[i], ks[i]) for i in range(2)]
    points, positive = [], []
    for a,b in zip(*d):
        t = np.linalg.lstsq(np.c_[a,-b], cams[1,:3]-cams[0,:3], rcond=None)[0]
        points.append((cams[0,:3]+t[0]*a+cams[1,:3]+t[1]*b)/2)
        positive.append(min(t)>0)
    return np.array(points), np.array(positive)


def run(args):
    cv2.setRNGSeed(2159)
    pillow_heif.register_heif_opener()
    root, out = args.terrain.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out/'cv_features';cache.mkdir(exist_ok=True)
    masks = json.loads(args.masks.read_text())
    state_path = root/'cv_transmitter_refit_2159B/fit_transmitter.npz'
    state = dict(np.load(state_path))
    ids = [list(state['keys']).index(k) for k in KEYS]
    base, shapes, ks = [state[k][ids].copy() for k in ['cameras','shapes','distortion']]
    tx = state['transmitter']
    meta = json.loads((root/'meta.json').read_text())
    terrain = RefinedTerrain(MarjumDEM(cache_file=str(root/'marjum_dem.npz')))
    features, horizons, images, exclusions = [], [], [], []
    for key in state['keys']:
        if str(key) not in KEYS:
            shutil.copyfile(root/f'cv_features/sift_{key}.npz', cache/f'sift_{key}.npz')
    for key in KEYS:
        seg = dict(np.load(root/f'img_seg_IMG_{key}.npz'))
        image = np.array(ImageOps.exif_transpose(Image.open(root.parent/f'marjum-2026-07/imgs/IMG_{key}.HEIC')).convert('RGB'))
        h,w = image.shape[:2]
        exclude = np.zeros((h,w),np.uint8)
        cv2.fillPoly(exclude,[np.array(masks[key],np.int32)],1)
        seg['horizon_exclude'] = exclude.astype(bool)
        np.savez_compressed(out/f'img_seg_IMG_{key}.npz', **seg)
        horizon = boundary_pixels(np.flipud(seg['skymask']).astype(bool),np.flipud(seg['ptree']),spacing=45,exclude=np.flipud(exclude))
        # Exclude the near antenna and low foreground from this terrain tie set.
        mask = ((~seg['skymask'].astype(bool)) & (seg['ptree']<.15) & (~exclude.astype(bool))).astype(np.uint8)*255
        mask[2100:] = 0
        small = cv2.resize(image,(1512,2016),interpolation=cv2.INTER_AREA)
        smallmask = cv2.erode(cv2.resize(mask,(1512,2016),interpolation=cv2.INTER_NEAREST),np.ones((5,5),np.uint8))
        kp,des = cv2.SIFT_create(nfeatures=18000,contrastThreshold=.015).detectAndCompute(cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),smallmask)
        xy=(np.array([p.pt for p in kp])+.5)*2-.5;xy[:,1]=h-1-xy[:,1]
        f=dict(xy=xy, descriptors=des, shape=np.array([h,w]), horizon=horizon,
               signature=np.array([-1],np.int64))  # Dedicated immutable fit cache, not a generic extractor cache.
        np.savez_compressed(cache/f'sift_{key}.npz',**f)
        features.append(f);horizons.append(horizon);images.append(image);exclusions.append(exclude)
    pairs = mutual(features[0]['descriptors'],features[1]['descriptors'])
    x,y = [features[i]['xy'][pairs[:,i]] for i in range(2)]
    # Split spatial cells BEFORE geometric verification. Held-out matches never
    # estimate the outlier model, relative pose, or the camera/landmark bundle.
    train = ((x[:,0]//240 + 2*(x[:,1]//240)).astype(int)%4)!=0
    q = [normalized(base[i],shapes[i],a,ks[i])*3000 for i,a in enumerate([x,y])]
    F,mask = cv2.findFundamentalMat(q[0][train],q[1][train],cv2.FM_RANSAC,3.,.999)
    if F is None or F.shape!=(3,3):raise RuntimeError('Pair lacks a fundamental model')
    xx=np.c_[q[0],np.ones(len(x))];yy=np.c_[q[1],np.ones(len(y))]
    fx=xx@F.T;fy=yy@F
    err=np.abs(np.sum(yy*fx,axis=1))/np.sqrt(np.sum(fx[:,:2]**2+fy[:,:2]**2,axis=1))
    good=err<4
    tr=np.flatnonzero(train & good);te=np.flatnonzero((~train)&good)
    # Spatial thinning reduces correlated features from the same rock patch.
    cells=(x[tr]/100).astype(int);_,sel=np.unique(cells,axis=0,return_index=True);tr=tr[np.sort(sel)]
    if len(tr)<20 or len(te)<8:raise RuntimeError(f'Insufficient verified matches {len(tr)}, {len(te)}')
    tr=tr[np.linspace(0,len(tr)-1,min(len(tr),180)).astype(int)]
    obs=[x[tr],y[tr]]
    qcv=[normalized(base[i],shapes[i],obs[i],ks[i]) for i in range(2)]
    for a in qcv:a[:,1]*=-1
    E,emask=cv2.findEssentialMat(qcv[0],qcv[1],np.eye(3),method=cv2.RANSAC,prob=.999,threshold=3/3000)
    _,R,t,posemask=cv2.recoverPose(E[:3],qcv[0],qcv[1],np.eye(3),mask=emask)
    W2=rotation(base[1])@CV_BODY
    init=base.copy();init[0,3:6]=pose_angles(W2@R@CV_BODY.T)
    baseline=np.linalg.norm(base[0,:3]-base[1,:3])
    init[0,:3]=base[1,:3]+W2@t.ravel()*baseline
    if args.initializer == 'source':
        init=base.copy()
    pts,positive=triangulate(init,shapes,ks,obs)
    tr=tr[positive];obs=[a[positive] for a in obs];pts=pts[positive]
    if len(pts)<20:raise RuntimeError('Too few positive-depth points')
    # Reserve spatial blocks across the whole skyline, including the short
    # restored right-hand rock segment (180-pixel blocks withheld almost all
    # of that segment and therefore supplied virtually no fitting constraint).
    htrain=[((a[:,0]//90).astype(int)%3)!=1 for a in horizons]
    n=len(pts);npar=14+3*n
    def unpack(v):
        c=init.copy();c[:,:6]+=v[:14].reshape(2,7)[:,:6];c[:,6]*=np.exp(v[:14].reshape(2,7)[:,6])
        return c,pts+v[14:].reshape(n,3)
    def herror(c,i,which,count=2048):
        d=rays(c,shapes[i],horizons[i][which],ks[i]);az=np.arctan2(d[:,1],d[:,0]);el=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
        return (el-terrain.skyline(c[:3],az,count=count,peaks=8 if count==2048 else 16,refine=33 if count==2048 else 65))*base[i,6]
    deps=[]
    def residual(v,structure=False):
        c,p=unpack(v);parts=[]
        def add(value,cols):
            a=np.asarray(value).ravel();parts.extend(a)
            if structure:deps.extend([cols]*len(a))
        for i in range(2):
            cc=list(range(7*i,7*i+7))
            pred,depth=project(c[i],shapes[i],p,ks[i])
            for j in range(n):add((pred[j]-obs[i][j])/3,cc+list(range(14+3*j,17+3*j)))
            add(herror(c[i],i,htrain[i])/10,cc)
            add((c[i,:3]-base[i,:3])/[10,10,5],cc)
            add((c[i,3:6]-init[i,3:6])/.15,cc)
            add([np.log(c[i,6]/base[i,6])/.07],cc)
            predtx,dt=project(c[i],shapes[i],tx,ks[i])
            add((predtx[0]-meta[KEYS[i]]['transmitter_px'])/6,cc)
            add([min(c[i,2]-terrain.height(*c[i,:2])-.6,0)/.2],cc)
            for j in range(n):add(min(depth[j]-.5,0)/.1,cc+list(range(14+3*j,17+3*j)))
        for j in range(n):add((p[j,2]-terrain.height(*p[j,:2]))/5,list(range(14+3*j,17+3*j)))
        return np.array(parts)
    v0=np.zeros(npar);residual(v0,True)
    sparsity=lil_matrix((len(deps),npar),dtype=int)
    for i,cols in enumerate(deps):sparsity[i,cols]=1
    print('features',*[len(f['xy']) for f in features],'mutual',len(pairs),'train',n,'holdout',len(te),flush=True)
    print('initial poses',init,flush=True)
    bounds=np.r_[np.tile([20,20,12,.35,.35,.35,.2],2),np.full(3*n,150.)]
    scale=np.r_[np.tile([2,2,2,.02,.02,.02,.03],2),np.full(3*n,5.)]
    fit=least_squares(residual,v0,jac_sparsity=sparsity.tocsr(),bounds=(-bounds,bounds),x_scale=scale,loss='soft_l1',f_scale=2,max_nfev=args.max_nfev,ftol=1e-6,verbose=1)
    cams,points=unpack(fit.x)
    def metrics(c):
        return dict(horizon={key:dict(train_rms_px=float(np.sqrt(np.mean(herror(c[i],i,htrain[i])**2))),heldout_rms_px=float(np.sqrt(np.mean(herror(c[i],i,~htrain[i])**2))),restored_right_count=int(np.sum(horizons[i][:,0]>=2300)),restored_right_rms_px=float(np.sqrt(np.mean(herror(c[i],i,horizons[i][:,0]>=2300)**2))),camera=c[i].tolist()) for i,key in enumerate(KEYS)},
                    tie_holdout_median_px=float(np.median(epipolar_error(c[0],shapes[0],ks[0],c[1],shapes[1],ks[1],x[te],y[te]))),
                    tie_train_median_px=float(np.median(epipolar_error(c[0],shapes[0],ks[0],c[1],shapes[1],ks[1],obs[0],obs[1]))))
    trace_path=root.parent/'marjum-2026-07/derived/geometry_memo_inputs/v0001/legacy_combined.npz'
    with np.load(trace_path) as trace:
        tids=[list(trace['keys']).index(k) for k in KEYS]
        tail=trace['draws'][:,-3000:]
        trace_cams=np.array([tail[:,:,7*i:7*i+7].mean((0,1)) for i in tids])
        trace_cams[:,6]=np.exp(trace_cams[:,6])
    report=dict(status='deterministic repair; not a posterior',initializer=args.initializer,mutual_matches=len(pairs),train_tracks=n,heldout_matches=len(te),before=metrics(base),old_trace_mean=metrics(trace_cams),after=metrics(cams),optimizer=dict(success=bool(fit.success),nfev=fit.nfev,cost=float(fit.cost)),mask_definition=masks)
    comparison_cams=trace_cams
    comparison_label='Old trace mean'
    if args.comparison_state is not None:
        with np.load(args.comparison_state) as prior:
            pids=[list(prior['keys']).index(k) for k in KEYS]
            comparison_cams=prior['cameras'][pids].copy()
            comparison_label='Previous broad-mask fit'
            report['previous_broad_mask_fit']=metrics(comparison_cams)
    report['skyline_validation']={}
    for i,key in enumerate(KEYS):
        right=horizons[i][:,0]>=2300
        error=herror(cams[i],i,slice(None),count=8192)
        report['skyline_validation'][key]=dict(
            xy_bottom_up=horizons[i].tolist(), train=htrain[i].tolist(),
            residual_px=error.tolist(),
            restored_train_count=int(np.sum(right & htrain[i])),
            restored_heldout_count=int(np.sum(right & ~htrain[i])),
            heldout_rms_px=float(np.sqrt(np.mean(error[~htrain[i]]**2))),
            restored_right_rms_px=float(np.sqrt(np.mean(error[right]**2))))
    # Install new tracks in the actual state consumed by the sampler. These two
    # cameras had no existing observations; never silently orphan old feature IDs.
    if np.isin(state['obs_cam'],ids).any():raise RuntimeError('Source has existing observations for repaired cameras; feature-ID merge required')
    offset=len(state['points'])
    state['points']=np.concatenate([state['points'],points])
    state['obs_cam']=np.r_[state['obs_cam'],np.repeat(ids,n)]
    state['obs_point']=np.r_[state['obs_point'],np.tile(np.arange(n)+offset,2)]
    state['obs_xy']=np.concatenate([state['obs_xy'],*obs])
    state['obs_fid']=np.r_[state['obs_fid'],pairs[tr,0],pairs[tr,1]]
    state['cameras'][ids]=cams
    prov=state['camera_provenance'].astype(object)
    for i in ids:prov[i]='2159/2199 paired terrain repair; explicit vegetation exclusions and shared 3D tracks'
    state['camera_provenance']=np.array(prov,dtype=str)
    np.savez_compressed(out/'fit_transmitter.npz',**state)
    np.savez_compressed(out/'pair_validation.npz',train_x=obs[0],train_y=obs[1],test_x=x[te],test_y=y[te],train_fids=pairs[tr],points=points)
    report['source_sha256']=hashlib.sha256(state_path.read_bytes()).hexdigest()
    inputs=[state_path,root/'marjum_dem.npz',root/'meta.json',trace_path,args.masks,Path(__file__),
            Path(__file__).with_name('marjum_bundle.py'),Path(__file__).with_name('marjum_camera.py'),
            Path(__file__).with_name('marjum_position.py')]
    inputs += [root/f'img_seg_IMG_{k}.npz' for k in KEYS]
    inputs += [root.parent/f'marjum-2026-07/imgs/IMG_{k}.HEIC' for k in KEYS]
    if args.comparison_state is not None:inputs.append(args.comparison_state.resolve())
    report['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    report['validation'] = dict(all_new_points_positive_depth=bool(all(np.all(project(cams[i],shapes[i],points,ks[i])[1]>0) for i in range(2))),
                               excluded_skyline_samples_absent=bool(all(not np.any(np.flipud(exclusions[i])[np.floor(h[:,1]).astype(int),h[:,0].astype(int)]) for i,h in enumerate(horizons))),
                               remaining_cameras_unchanged=True,
                               note='Fit/cache must be consumed together. Old chains are not updated by this deterministic repair.')
    report['code_commit']=subprocess.check_output(['git','-C',str(Path(__file__).parent),'rev-parse','HEAD'],text=True).strip()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    render(out,images,exclusions,horizons,comparison_cams,cams,shapes,ks,terrain,obs,comparison_label)
    print(json.dumps(report,indent=2),flush=True)


def render(out,images,exclusions,horizons,base,cams,shapes,ks,terrain,obs,comparison_label='Old trace mean'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for i,key in enumerate(KEYS):
        for col,(label,p) in enumerate([(comparison_label,base[i]),('Repaired initializer',cams[i])]):
            ax=axs[i,col];ax.imshow(images[i]);ax.set(xlim=(0,3024),ylim=(1500,0),title=f'{key} — {label}',aspect='auto')
            ax.contourf(exclusions[i],levels=[.5,1.5],colors=['red'],alpha=.16)
            h=horizons[i];ax.plot(h[:,0],4031-h[:,1],'.',ms=3,label='Accepted image skyline')
            # Parameterize terrain horizon by azimuth, then project far unit
            # directions. Camera position is included in terrain ray tracing.
            az=np.linspace(p[4]-.7,p[4]+.7,420)
            el=terrain.skyline(p[:3],az,count=8192)
            d=np.c_[np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)]
            xy,depth=project(p,shapes[i],p[:3]+1000*d,ks[i])
            body=d@rotation(p)
            radius=np.linalg.norm(body[:,:2],axis=1)/np.maximum(body[:,2],1e-9)
            maxradius=np.linalg.norm(normalized(p,shapes[i],[[0,0],[3023,4031]],ks[i]),axis=1).max()
            ok=(depth>0)&(xy[:,0]>=0)&(xy[:,0]<3024)&(radius<=maxradius)
            xy[~ok]=np.nan
            ax.plot(xy[:,0],4031-xy[:,1],color='cyan',lw=1.2,label='DEM horizon')
            ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'horizon_comparison.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,7))
    for i in range(2):
        axs[i].imshow(images[i]);axs[i].scatter(obs[i][:,0],4031-obs[i][:,1],s=7,c=np.arange(len(obs[i])),cmap='turbo')
        axs[i].set(ylim=(2200,0),title=f'{KEYS[i]}: shared training tracks')
    fig.tight_layout();fig.savefig(out/'tie_points.png',dpi=120);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--terrain',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masks',type=Path,default=Path(__file__).with_name('horizon_masks.json'))
    p.add_argument('--max-nfev',type=int,default=180)
    p.add_argument('--initializer',choices=['source','essential'],default='source')
    p.add_argument('--comparison-state',type=Path,help='Optional previous fit, evaluated on the same restored skyline')
    run(p.parse_args())
