"""Optional SuperPoint + LightGlue front end for marjum_bundle.

Requires the official https://github.com/cvg/LightGlue package and its weights.
The default SIFT workflow needs neither. This module does not install packages.
Weights may download on first use. All returned pixels follow the notebooks'
bottom-up convention. Learned matches still undergo geometric verification.
"""
from itertools import combinations
from pathlib import Path
import cv2
import numpy as np

from marjum_bundle import boundary_pixels, verify_pair


def learned_matches(root, keys, device='cpu', max_size=1600, nfeatures=2048, per_pair=60):
    import torch
    try:
        from lightglue import SuperPoint, LightGlue
    except ImportError as exc:
        raise ImportError('Optional backend requires the official cvg/LightGlue package; use SIFT otherwise.') from exc
    from eigsep_terrain.imageio import load_image
    root = Path(root)
    extractor = SuperPoint(max_num_keypoints=nfeatures).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    features, tensors = {}, {}
    with torch.inference_mode():
        for key in keys:
            rgb = np.flipud(load_image(str(root/f'marjum-2026-07/IMG_{key}.HEIC'))).copy()
            h, w = rgb.shape[:2]
            factor = min(1., max_size/max(h, w))
            small = cv2.resize(rgb, (round(w*factor), round(h*factor)), interpolation=cv2.INTER_AREA)
            with np.load(root/f'img_seg_IMG_{key}.npz') as z:
                sky = np.flipud(z['skymask']).astype(bool)
                tree = np.flipud(z['ptree'])
            tensor = torch.from_numpy(small.transpose(2, 0, 1).copy()).float().to(device)/255
            f = extractor.extract(tensor, resize=None)
            xy_small = f['keypoints'][0].cpu().numpy()
            scale = np.array([w/small.shape[1], h/small.shape[0]])
            xy = (xy_small+.5)*scale-.5
            xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, w-1)
            yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, h-1)
            keep = (~sky[yi, xi]) & (tree[yi, xi] < .15)
            select = torch.from_numpy(keep).to(device)
            # Feature IDs remain consistent across every pair for track merging.
            tensors[key] = {k: v[:, select] if k in ('keypoints','descriptors','keypoint_scores') else v
                            for k, v in f.items()}
            features[key] = dict(xy=xy[keep], shape=np.array([h,w]), horizon=boundary_pixels(sky, tree))
            del rgb, sky, tree, small, tensor
        pairs, diagnostics = [], []
        cv2.setRNGSeed(42)
        for a, b in combinations(keys, 2):
            if min(len(features[a]['xy']), len(features[b]['xy'])) < 12:
                continue
            match = matcher({'image0':tensors[a], 'image1':tensors[b]})
            ids = match['matches'][0].cpu().numpy()
            keep, model = verify_pair(features[a]['xy'][ids[:, 0]], features[b]['xy'][ids[:, 1]])
            diagnostics.append(dict(a=a,b=b,matches=len(ids),inliers=int(keep.sum()),model=model))
            ids = ids[keep]
            if len(ids) < 12:
                continue
            cells = np.floor(features[a]['xy'][ids[:,0]]/180).astype(int)
            _, select = np.unique(cells, axis=0, return_index=True)
            ids = ids[np.sort(select)]
            if len(ids) > per_pair:
                ids = ids[np.linspace(0, len(ids)-1, per_pair).astype(int)]
            pairs.append((a,b,ids))
    return features, pairs, diagnostics
