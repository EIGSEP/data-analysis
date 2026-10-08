"""Check fitted epipolar geometry on SIFT correspondences absent from the fit."""
import json
from pathlib import Path
from itertools import combinations
import cv2
import numpy as np
from scipy.spatial import cKDTree
from eigsep_terrain.fitio import load_fit
from marjum_bundle import PRM_ORDER, rotation, verify_pair


def epipolar_error(p1, shape1, p2, shape2, x1, x2):
    """Sampson distance in native image pixels, with translation explicitly used."""
    def matrix(p, shape):
        h, w = shape
        return rotation(p) @ np.array([[0.,-1.,h//2],[-1.,0.,w//2],[0.,0.,p[6]]])
    b = p2[:3]-p1[:3]
    if np.linalg.norm(b) < .01:
        return np.full(len(x1), np.nan)  # no informative epipolar geometry
    b = b/np.linalg.norm(b)
    bx = np.array([[0.,-b[2],b[1]],[b[2],0.,-b[0]],[-b[1],b[0],0.]])
    f = matrix(p2,shape2).T @ bx @ matrix(p1,shape1)
    x, y = np.c_[x1,np.ones(len(x1))], np.c_[x2,np.ones(len(x2))]
    fx, fty = x @ f.T, y @ f
    return abs(np.sum(y*fx,axis=1))/np.sqrt(np.sum(fx[:,:2]**2+fty[:,:2]**2,axis=1))


def validate(before, after, state, output):
    p0, _, _ = load_fit(before)
    p1, _, _ = load_fit(after)
    with np.load(state) as z:
        used = {str(k):z['obs_xy'][z['obs_cam']==i] for i,k in enumerate(z['keys'])}
    features = {}
    for k in p1:
        with np.load(f'cv_features/sift_{k}.npz') as z:
            features[k] = {name:z[name] for name in ('xy','shape','descriptors')}
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    cv2.setRNGSeed(42)
    for a,b in combinations(p1,2):
        fa, fb = features[a], features[b]
        def matches(da,db):
            return {m.queryIdx:m.trainIdx for pair in bf.knnMatch(da,db,k=2)
                    if len(pair)==2 for m,n in [pair] if m.distance < .75*n.distance}
        ab, ba = matches(fa['descriptors'],fb['descriptors']), matches(fb['descriptors'],fa['descriptors'])
        ids = np.array([(i,j) for i,j in ab.items() if ba.get(j)==i],int).reshape(-1,2)
        if len(ids)<12:
            continue
        x,y = fa['xy'][ids[:,0]], fb['xy'][ids[:,1]]
        keep, model = verify_pair(x,y)
        # Exclude a 5 px neighbourhood around ANY training observation, including
        # duplicate SIFT orientations and reuse via other image pairs.
        for key,xy in [(a,x),(b,y)]:
            if len(used[key]):
                keep &= cKDTree(used[key]).query(xy)[0] > 5.
        x,y = x[keep], y[keep]
        if len(x)<12:
            continue
        row = dict(a=a,b=b,count=len(x),verification=model)
        for label,poses in [('before',p0),('after',p1)]:
            v1,v2 = [np.array([poses[k][q] for q in PRM_ORDER]) for k in [a,b]]
            error = epipolar_error(v1,fa['shape'],v2,fb['shape'],x,y)
            row[label+'_median_px'] = float(np.nanmedian(error)) if np.isfinite(error).any() else None
        rows.append(row)
    Path(output).write_text(json.dumps(rows,indent=2)+'\n')
    print('held-out pairs:',len(rows))
    print('median of pair medians:',{label:float(np.median([r[label+'_median_px'] for r in rows if r[label+'_median_px'] is not None])) for label in ['before','after']})
    return rows


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',default='fit_bundle_v5_2231_polished.npz')
    p.add_argument('--after',default='cv_initialization_v3/fit_cv.npz')
    p.add_argument('--state',default='cv_initialization_v3/state_0.npz')
    p.add_argument('--output',default='cv_initialization_v3/holdout.json')
    validate(**vars(p.parse_args()))
