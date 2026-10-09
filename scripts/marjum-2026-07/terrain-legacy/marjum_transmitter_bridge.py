"""Triangulated feature bridges for new transmitter views; no DEM snapping."""
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from marjum_camera import project, rays
from marjum_add_views import pnp_pose, spatial_split, feature_refine
from marjum_guided import lens_metadata
from marjum_position import RefinedTerrain


def mutual(a, b):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def one(x, y):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(x, y, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < .75*n.distance}
    ab, ba = one(a, b), one(b, a)
    return {i: j for i, j in ab.items() if ba.get(j) == i}


def run(target='2198', output='cv_transmitter_bridge_v2'):
    from eigsep_terrain.marjum_dem import MarjumDEM
    out = Path(output); out.mkdir(exist_ok=False)
    source = 'cv_transmitter_v1/fit_transmitter_manual_2172.npz'
    state = dict(np.load(source)); keys = list(state['keys'].astype(str))
    refs = ['2159', '2171', '2172', '2199', '2203', '2210', '2211']
    features = {k: dict(np.load(f'cv_features/sift_{k}.npz')) for k in refs+[target]}
    tm = {k: mutual(features[target]['descriptors'], features[k]['descriptors']) for k in refs}
    candidates = {}; counts = {}
    for a, b in combinations(refs, 2):
        matches = mutual(features[a]['descriptors'], features[b]['descriptors'])
        ia, ib = keys.index(a), keys.index(b)
        ca, cb = state['cameras'][ia], state['cameras'][ib]
        accepted = 0
        for fid in tm[a].keys() & tm[b].keys():
            fa, fb = tm[a][fid], tm[b][fid]
            if matches.get(fa) != fb: continue
            qa, qb = features[a]['xy'][fa], features[b]['xy'][fb]
            da = rays(ca, state['shapes'][ia], [qa], state['distortion'][ia])[0]
            db = rays(cb, state['shapes'][ib], [qb], state['distortion'][ib])[0]
            angle = np.degrees(np.arccos(np.clip(da@db, -1, 1)))
            if angle < 1 or angle > 100: continue
            na, nb = np.eye(3)-np.outer(da, da), np.eye(3)-np.outer(db, db)
            point = np.linalg.solve(na+nb, na@ca[:3]+nb@cb[:3])
            pa, za = project(ca, state['shapes'][ia], point, state['distortion'][ia])
            pb, zb = project(cb, state['shapes'][ib], point, state['distortion'][ib])
            error = max(np.linalg.norm(pa[0]-qa), np.linalg.norm(pb[0]-qb))
            if min(za[0], zb[0]) < 1 or error > 8: continue
            accepted += 1
            if fid not in candidates or error < candidates[fid][0]:
                candidates[fid] = (float(error), point, a+'_'+b)
        counts[a+'_'+b] = accepted
    print('consistent three-view tracks', len(candidates), counts, flush=True)
    report = dict(target=target, source=source, pair_counts=counts, tracks=len(candidates))
    if len(candidates) >= 12:
        ids = sorted(candidates); xy = features[target]['xy'][ids]
        xyz = np.array([candidates[i][1] for i in ids]); train, test = spatial_split(xy)
        shape = features[target]['shape']; lens = lens_metadata([target])[target]
        group = int(lens['group']=='ultrawide')
        distortion = np.median(state['distortion'][state['groups']==group], axis=0)
        focal = np.hypot(*shape)/np.hypot(36,24)*float(lens['FocalLengthIn35mmFilm'])
        terrain = RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz'))
        starts = []
        cv2.setRNGSeed(47)
        for scale in [.7, .85, 1., 1.15, 1.3]:
            p, inliers = pnp_pose(xyz[train], xy[train], shape, focal*scale, distortion)
            if p is None: continue
            try:
                p, fit = feature_refine(p, xyz[train][inliers], xy[train][inliers], shape,
                                       distortion, focal, terrain)
            except ValueError: continue
            err = np.linalg.norm(project(p, shape, xyz, distortion)[0]-xy, axis=1)
            starts.append((int(np.sum(err[train]<10)), p, err))
        if starts:
            _, p, err = max(starts, key=lambda x:x[0])
            report.update(camera=p.tolist(), distortion=distortion.tolist(), group=group,
                          train_median_px=float(np.median(err[train])),
                          heldout_median_px=float(np.median(err[test])),
                          heldout_inlier_fraction=float(np.mean(err[test]<15)),
                          heldout_count=int(test.sum()))
            np.savez_compressed(out/'bridge.npz', xyz=xyz, xy=xy, train=train, camera=p,
                                distortion=distortion, shape=shape)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run()
