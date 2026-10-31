import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.environ["EIGSEP_CAMPAIGN_ROOT"] = "/home/aparsons/projects/eigsep/marjum-2026-07"

import sys
import copy
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize

ROOT = "/home/aparsons/projects/eigsep"
CAMPAIGN = os.path.join(ROOT, "marjum-2026-07")
V4 = os.path.join(CAMPAIGN, "derived/geometry_posterior/v0004")
V4_IN = os.path.join(V4, "inputs/model_v0002")
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src"))
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src/eigsep_terrain/marjum_geometry"))
sys.path.insert(0, os.path.join(ROOT, "data-analysis/scripts/marjum-2026-07/geometry"))

import marjum_mcmc_b21 as b
from marjum_bundle import build_tracks
from marjum_camera import project, rays
import make_product as mp
import geometry_checks as gc
import cv2
from itertools import combinations


def gn_polish_camera(model, i, cam_i, ant, points, bias, extra, tx, tx_extra, steps=5):
    pose = cam_i.copy()
    step = b.CAMERA_STEP * 0.2

    def res_vec(x):
        p = b.camera_pose(x)
        r1 = model.camera_residuals(i, p, ant, points, bias, extra, tx=tx, tx_extra=tx_extra or 0.0)
        if r1 is None:
            return None
        r2 = model.horizon_residuals(i, p)
        r = np.r_[r1, r2]
        return r if np.all(np.isfinite(r)) else None

    for _ in range(steps):
        x = b.camera_coordinates(pose)
        r0 = res_vec(x)
        if r0 is None:
            break
        cost0 = 0.5 * np.dot(r0, r0)
        J = np.empty((len(r0), 7))
        ok = True
        for j in range(7):
            up, dn = x.copy(), x.copy()
            up[j] += step[j]
            dn[j] -= step[j]
            rp, rm = res_vec(up), res_vec(dn)
            if rp is not None and rm is not None:
                J[:, j] = (rp - rm) / (2 * step[j])
            elif rp is not None:
                J[:, j] = (rp - r0) / step[j]
            elif rm is not None:
                J[:, j] = (r0 - rm) / step[j]
            else:
                ok = False
                break
        if not ok:
            break
        # Levenberg-Marquardt damped step in scaled coordinates
        S = b.CAMERA_SCALE
        Js = J * S
        H = Js.T @ Js + np.eye(7) * 1.0
        g = Js.T @ r0
        dx = -S * np.linalg.solve(H, g)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
            cand = b.camera_pose(x + alpha * dx)
            rc = res_vec(b.camera_coordinates(cand))
            if rc is not None and 0.5 * np.dot(rc, rc) < cost0:
                pose = cand
                improved = True
                break
        if not improved:
            break
    return pose


def relax_landmarks(model, cam, points, steps=3):
    pts = points.copy()
    for _ in range(steps):
        fac = b.landmark_geometry(model, cam, pts, step=0.02)
        base = model.point_logp(cam=cam, points=pts)
        grad = np.zeros_like(pts)
        eps = 0.01
        for ax in range(3):
            p_up = pts.copy(); p_up[:, ax] += eps
            p_dn = pts.copy(); p_dn[:, ax] -= eps
            grad[:, ax] = (model.point_logp(cam=cam, points=p_up) - model.point_logp(cam=cam, points=p_dn)) / (2 * eps)
        delta = np.einsum("jab,jbc,jc->ja", fac, fac, grad)
        delta = np.clip(delta, -5.0, 5.0)
        for alpha in (1.0, 0.5, 0.2, 0.05):
            cand = pts + alpha * delta
            lp = model.point_logp(cam=cam, points=cand)
            improve = lp > base
            pts[improve] = cand[improve]
            base[improve] = lp[improve]
    return pts


def main():
    _, model_v4 = mp.load_model()
    v4_geo = np.load(os.path.join(V4, "provisional_geometry.npz"))
    v4_state = np.asarray(v4_geo["state"], float)
    cam_v4, ant_v4, tx_v4, bias_v4, extra_v4, txe_v4, pts_v4 = model_v4.unpack(v4_state)

    ft = np.load(os.path.join(V4_IN, "fit_transmitter.npz"), allow_pickle=True)
    keys = [str(k) for k in ft["keys"]]
    sift_dir = os.path.join(V4_IN, "cv_features")
    features = {}
    for k in keys:
        with np.load(os.path.join(sift_dir, f"sift_{k}.npz")) as z:
            features[k] = {name: z[name] for name in z.files}

    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    cv2.setRNGSeed(42)
    for a, b_key in combinations(features, 2):
        fa, fb = features[a], features[b_key]
        if min(len(fa["xy"]), len(fb["xy"])) < 2:
            continue
        def good(d1, d2, ratio=0.78):
            return {m.queryIdx: m for v in bf.knnMatch(d1, d2, k=2)
                    if len(v) == 2 for m, n in [v] if m.distance < ratio * n.distance}
        ab = good(fa["descriptors"], fb["descriptors"])
        ba = good(fb["descriptors"], fa["descriptors"])
        match = [m for i, m in ab.items() if m.trainIdx in ba and ba[m.trainIdx].trainIdx == i]
        if len(match) < 8:
            continue
        ids = np.array([(m.queryIdx, m.trainIdx) for m in match], dtype=int).reshape(-1, 2)
        _, hm = cv2.findHomography(fa["xy"][ids[:, 0]], fb["xy"][ids[:, 1]], cv2.USAC_MAGSAC, 4.5)
        _, fm = cv2.findFundamentalMat(fa["xy"][ids[:, 0]], fb["xy"][ids[:, 1]], cv2.USAC_MAGSAC, 4.5, 0.999, 10000)
        hm = np.zeros(len(ids), bool) if hm is None else hm.ravel().astype(bool)
        fm = np.zeros(len(ids), bool) if fm is None else fm.ravel().astype(bool)
        keep = hm if hm.sum() >= max(8, 0.85 * fm.sum()) else fm
        ids = ids[keep]
        if len(ids) < 8:
            continue
        xy = fa["xy"][ids[:, 0]]
        cells = np.floor(xy / 150).astype(int)
        _, selected = np.unique(cells, axis=0, return_index=True)
        ids = ids[np.sort(selected)]
        if len(ids) > 80:
            ids = ids[np.linspace(0, len(ids) - 1, 80).astype(int)]
        pairs.append((a, b_key, ids))

    new_tracks, _ = build_tracks(pairs)

    obs_cam_orig = ft["obs_cam"]
    obs_pt_orig = ft["obs_point"]
    obs_xy_orig = ft["obs_xy"]
    obs_fid_orig = ft["obs_fid"]

    keep_pts = []
    kept_oc, kept_op, kept_xy, kept_fid = [], [], [], []
    existing_used = set()
    terrain = model_v4.terrain

    for j in range(len(pts_v4)):
        mask = obs_pt_orig == j
        ci = obs_cam_orig[mask]
        xy = obs_xy_orig[mask]
        fid = obs_fid_orig[mask]
        pt = pts_v4[j].copy()
        dirs = np.array([rays(cam_v4[c], model_v4.shapes[c], np.atleast_2d(q), model_v4.distortion[c])[0]
                         for c, q in zip(ci, xy)])
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
        max_ang = float(np.degrees(np.arccos(dots)).max())
        if len(ci) == 2 and max_ang < 1.5:
            hit = terrain.intersect(cam_v4[ci[0]], model_v4.shapes[ci[0]], np.atleast_2d(xy[0]))[0]
            if np.all(np.isfinite(hit)):
                pt = hit
            else:
                continue
        new_j = len(keep_pts)
        keep_pts.append(pt)
        for c, q, f_id in zip(ci, xy, fid):
            kept_oc.append(int(c))
            kept_op.append(new_j)
            kept_xy.append(q)
            kept_fid.append(int(f_id))
            existing_used.add((int(c), int(f_id)))

    obs_counts = np.bincount(kept_oc, minlength=len(keys))
    sparse_cams = {i for i, c in enumerate(obs_counts) if c < 120}

    for track in new_tracks:
        t_idx = [(keys.index(k), int(fid)) for k, fid in track if k in keys]
        if len(t_idx) < 2:
            continue
        if not any(ci in sparse_cams for ci, _ in t_idx):
            continue
        if any((ci, fid) in existing_used for ci, fid in t_idx):
            continue
        ci = [c for c, _ in t_idx]
        xy = [features[keys[c]]["xy"][fid] for c, fid in t_idx]
        origins = cam_v4[ci, :3]
        dirs = np.array([rays(cam_v4[c], model_v4.shapes[c], np.atleast_2d(q), model_v4.distortion[c])[0]
                         for c, q in zip(ci, xy)])
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
        max_angle = float(np.degrees(np.arccos(dots)).max())
        mat = np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]
        candidates = []
        if max_angle >= 1.5 and np.linalg.cond(mat.sum(axis=0)) < 1e5:
            candidates.append(np.linalg.solve(mat.sum(axis=0), np.einsum("nij,nj->i", mat, origins)))
        for c, q in zip(ci, xy):
            hit = terrain.intersect(cam_v4[c], model_v4.shapes[c], np.atleast_2d(q))[0]
            if np.all(np.isfinite(hit)):
                candidates.append(hit)
        best = None
        for pt in candidates:
            if not (terrain.e[0] + 5 < pt[0] < terrain.e[-1] - 5 and terrain.n[0] + 5 < pt[1] < terrain.n[-1] - 5):
                continue
            h = float(terrain.height(pt[0], pt[1]))
            if not np.isfinite(h) or abs(pt[2] - h) > 20.0:
                continue
            if max_angle < 2.0:
                pt = np.array([pt[0], pt[1], h])
            errs = []
            valid = True
            for c, q in zip(ci, xy):
                pred, depth = project(cam_v4[c], model_v4.shapes[c], np.atleast_2d(pt), model_v4.distortion[c])
                if depth[0] <= 5.0:
                    valid = False
                    break
                errs.append(np.linalg.norm(pred[0] - q))
            if not valid:
                continue
            rms_px = float(np.sqrt(np.mean(np.square(errs))))
            if rms_px > 18.0:
                continue
            score = rms_px + 0.5 * abs(pt[2] - h)
            if best is None or score < best[0]:
                best = (score, pt, rms_px)
        if best is None:
            continue
        pt = best[1]
        new_j = len(keep_pts)
        keep_pts.append(pt)
        for (c, fid), q in zip(t_idx, xy):
            kept_oc.append(int(c))
            kept_op.append(new_j)
            kept_xy.append(q)
            kept_fid.append(int(fid))
            existing_used.add((int(c), int(fid)))

    aug_pts = np.array(keep_pts, float)
    aug_oc = np.array(kept_oc, int)
    aug_op = np.array(kept_op, int)
    aug_xy = np.array(kept_xy, float)
    aug_fid = np.array(kept_fid, int)

    aug_counts = np.bincount(aug_oc, minlength=len(keys))
    print(f"Total augmented landmarks: {len(aug_pts)}, total obs: {len(aug_oc)}")
    for i, k in enumerate(keys):
        if i in sparse_cams:
            print(f"  [{i:2d}] {k}: {int(obs_counts[i]):4d} -> {int(aug_counts[i]):4d} obs")

    aug_state_path = "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/fit_transmitter_aug.npz"
    ft_dict = {k: ft[k] for k in ft.files}
    ft_dict["cameras"] = cam_v4
    ft_dict["antenna"] = ant_v4
    ft_dict["transmitter"] = tx_v4
    ft_dict["points"] = aug_pts
    ft_dict["obs_cam"] = aug_oc
    ft_dict["obs_point"] = aug_op
    ft_dict["obs_xy"] = aug_xy
    ft_dict["obs_fid"] = aug_fid
    np.savez_compressed(aug_state_path, **ft_dict)

    cfg = b.Config(
        skyline_samples=4096,
        antenna_extra_prior_px=1.5,
        log_f_sigma_by_camera=model_v4.config.log_f_sigma_by_camera,
    )
    model_aug = b.Posterior(
        state_file=aug_state_path,
        dem_file=str(mp.DEM),
        meta_file=mp.INPUT / "meta.json",
        exif_file=mp.HERE / "inputs/exif_utm.npz",
        feature_dir=mp.INPUT / "cv_features",
        config=cfg,
    )

    cam_cur = cam_v4.copy()
    ant_cur = ant_v4.copy()
    tx_cur = tx_v4.copy()
    bias_cur = bias_v4.copy()
    extra_cur = 1.5
    txe_cur = 2.0
    pts_cur = relax_landmarks(model_aug, cam_cur, aug_pts, steps=3)

    state_cur = model_aug.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur)
    print(f"Initial augmented logp: {model_aug.logp(state_cur):.2f}")

    for rnd in range(4):
        t0 = time.time()
        for i in range(model_aug.nc):
            cam_cur[i] = gn_polish_camera(
                model_aug, i, cam_cur[i], ant_cur, pts_cur, bias_cur, extra_cur, tx_cur, txe_cur, steps=4
            )
        f_tmp = gc.Fit(model_aug, model_aug.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur))
        f_tmp.refit_antenna()
        f_tmp.refit_transmitter()
        ant_cur, tx_cur = f_tmp.ant, f_tmp.tx
        pts_cur = relax_landmarks(model_aug, cam_cur, pts_cur, steps=2)
        use = model_aug.has_gps
        w = 1.0 / model_aug.gps_sigma[use] ** 2
        prec = w.sum() + 1.0 / model_aug.config.gps_common_sigma_m ** 2
        bias_cur = ((model_aug.gps[use] - cam_cur[use, :2]) * w[:, None]).sum(axis=0) / prec
        st = model_aug.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur)
        print(f"Round {rnd+1}: logp={model_aug.logp(st):.2f}, ant={np.round(ant_cur, 3)}, tx={np.round(tx_cur, 3)} [{time.time()-t0:.1f}s]", flush=True)

    state_pol = model_aug.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur)
    np.savez_compressed(
        "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/polished_aug.npz",
        state=state_pol,
    )

    # Fast LOO antenna check using gn_polish_camera + Powell refinement
    print("\nRunning LOO antenna check on polished augmented state...", flush=True)
    rows = []
    for i in np.flatnonzero(model_aug.has_ant_label):
        t0 = time.time()
        m = copy.copy(model_aug)
        m.has_ant_label = model_aug.has_ant_label.copy()
        m.has_ant_label[i] = False
        m.n_ant_label = int(m.has_ant_label.sum())
        c_out = gn_polish_camera(m, i, cam_cur[i], ant_cur, pts_cur, bias_cur, extra_cur, tx_cur, txe_cur, steps=8)
        f_out = gc.Fit(m, m.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur))
        f_out.cam[i] = c_out
        f_out.refit_antenna()
        xy = model_aug.axy[i]
        d_in = gc.ray(model_aug, cam_cur[i], i, xy)
        d_out = gc.ray(model_aug, f_out.cam[i], i, xy)
        in_m = gc.miss(cam_cur[i, :3], d_in, ant_cur)
        out_m = gc.miss(f_out.cam[i, :3], d_out, f_out.ant)
        pshift = float(np.linalg.norm(f_out.cam[i, :3] - cam_cur[i, :3]))
        rows.append((model_aug.keys[i], in_m, out_m, pshift))
        print(f"  {model_aug.keys[i]}: in={in_m:.2f}m, LOO={out_m:.2f}m, pose_shift={pshift:.2f}m [{time.time()-t0:.1f}s]", flush=True)

    in_misses = [r[1] for r in rows]
    out_misses = [r[2] for r in rows]
    print(f"\nSummary LOO Antenna (Gauss-Newton):")
    print(f"  In-sample ray miss: median={np.median(in_misses):.2f} m, max={np.max(in_misses):.2f} m")
    print(f"  Held-out ray miss:  median={np.median(out_misses):.2f} m, 90th={np.percentile(out_misses, 90):.2f} m, max={np.max(out_misses):.2f} m")


if __name__ == "__main__":
    main()
