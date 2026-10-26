import os
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.environ["EIGSEP_CAMPAIGN_ROOT"] = "/home/aparsons/projects/eigsep/marjum-2026-07"

import sys
import copy
import time
from itertools import combinations
import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

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
from marjum_mcmc import student_residual
import make_product as mp
import geometry_checks as gc
from test_polish_and_loo import gn_polish_camera, relax_landmarks

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

cam_init = cam_v4.copy()
pts_init = pts_v4.copy()

# Match SIFT features across all 29 cameras
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
    cells = np.floor(xy / 120).astype(int)
    _, selected = np.unique(cells, axis=0, return_index=True)
    ids = ids[np.sort(selected)]
    if len(ids) > 90:
        ids = ids[np.linspace(0, len(ids) - 1, 90).astype(int)]
    pairs.append((a, b_key, ids))

obs_cam_orig = ft["obs_cam"]
obs_pt_orig = ft["obs_point"]
obs_xy_orig = ft["obs_xy"]
obs_fid_orig = ft["obs_fid"]

keep_pts = []
kept_oc, kept_op, kept_xy, kept_fid = [], [], [], []
fid_to_pt = {}
pt_cams = {}
terrain = model_v4.terrain

for j in range(len(pts_init)):
    mask = obs_pt_orig == j
    ci = obs_cam_orig[mask]
    xy = obs_xy_orig[mask]
    fid = obs_fid_orig[mask]
    pt = pts_init[j].copy()
    good_obs = []
    for c, q, f_id in zip(ci, xy, fid):
        pred, depth = project(cam_init[c], model_v4.shapes[c], np.atleast_2d(pt), model_v4.distortion[c])
        err = float(np.linalg.norm(pred[0] - q))
        if depth[0] > 2.0 and (err <= 16.0 or len(ci) <= 2):
            good_obs.append((int(c), q, int(f_id)))
    if len(good_obs) < 2:
        continue
    ci_g = [x[0] for x in good_obs]
    xy_g = [x[1] for x in good_obs]
    dirs = np.array([rays(cam_init[c], model_v4.shapes[c], np.atleast_2d(q), model_v4.distortion[c])[0]
                     for c, q in zip(ci_g, xy_g)])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    max_ang = float(np.degrees(np.arccos(dots)).max())
    if len(good_obs) == 2 and max_ang < 1.5:
        hit = terrain.intersect(cam_init[ci_g[0]], model_v4.shapes[ci_g[0]], np.atleast_2d(xy_g[0]))[0]
        if np.all(np.isfinite(hit)):
            pt = hit
        else:
            continue
    new_j = len(keep_pts)
    keep_pts.append(pt)
    pt_cams[new_j] = set()
    for c, q, f_id in good_obs:
        kept_oc.append(c)
        kept_op.append(new_j)
        kept_xy.append(q)
        kept_fid.append(f_id)
        fid_to_pt[(c, f_id)] = new_j
        pt_cams[new_j].add(c)

# Attach matched keypoints in any camera directly to existing 3D landmarks if consistent within 12 px
attached_existing = 0
for a, b_key, ids in pairs:
    ca, cb = keys.index(a), keys.index(b_key)
    for ia, ib in ids:
        ia, ib = int(ia), int(ib)
        for (c1, f1, c2, f2) in [((ca, ia, cb, ib)), ((cb, ib, ca, ia))]:
            if (c1, f1) in fid_to_pt and (c2, f2) not in fid_to_pt:
                j_pt = fid_to_pt[(c1, f1)]
                if c2 in pt_cams[j_pt]:
                    continue
                pt = keep_pts[j_pt]
                q2 = features[keys[c2]]["xy"][f2]
                pred, depth = project(cam_init[c2], model_v4.shapes[c2], np.atleast_2d(pt), model_v4.distortion[c2])
                if depth[0] > 5.0 and np.linalg.norm(pred[0] - q2) <= 12.0:
                    kept_oc.append(c2)
                    kept_op.append(j_pt)
                    kept_xy.append(q2)
                    kept_fid.append(f2)
                    fid_to_pt[(c2, f2)] = j_pt
                    pt_cams[j_pt].add(c2)
                    attached_existing += 1

# Add new tracks (including DEM-intersected tracks for close pairs like 2209-2210 and 2171-2172)
new_tracks, _ = build_tracks(pairs)
obs_counts = np.bincount(kept_oc, minlength=len(keys))
need_more = {i for i, c in enumerate(obs_counts) if c < 250}
added_new_pts = 0
for track in new_tracks:
    t_idx = [(keys.index(k), int(fid)) for k, fid in track if k in keys and (keys.index(k), int(fid)) not in fid_to_pt]
    seen_c = set()
    t_clean = []
    for c, fid in t_idx:
        if c not in seen_c:
            seen_c.add(c)
            t_clean.append((c, fid))
    if len(t_clean) < 2:
        continue
    if not any(c in need_more for c, _ in t_clean):
        continue
    ci = [c for c, _ in t_clean]
    xy = [features[keys[c]]["xy"][fid] for c, fid in t_clean]
    origins = cam_init[ci, :3]
    dirs = np.array([rays(cam_init[c], model_v4.shapes[c], np.atleast_2d(q), model_v4.distortion[c])[0]
                     for c, q in zip(ci, xy)])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    max_angle = float(np.degrees(np.arccos(dots)).max())
    mat = np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]
    candidates = []
    if max_angle >= 1.5 and np.linalg.cond(mat.sum(axis=0)) < 1e5:
        candidates.append(np.linalg.solve(mat.sum(axis=0), np.einsum("nij,nj->i", mat, origins)))
    for c, q in zip(ci, xy):
        hit = terrain.intersect(cam_init[c], model_v4.shapes[c], np.atleast_2d(q))[0]
        if np.all(np.isfinite(hit)):
            candidates.append(hit)
    best = None
    for pt in candidates:
        if not (terrain.e[0] + 5 < pt[0] < terrain.e[-1] - 5 and terrain.n[0] + 5 < pt[1] < terrain.n[-1] - 5):
            continue
        h = float(terrain.height(pt[0], pt[1]))
        if not np.isfinite(h) or abs(pt[2] - h) > 15.0:
            continue
        if max_angle < 2.0:
            pt = np.array([pt[0], pt[1], h])
        errs = []
        valid = True
        for c, q in zip(ci, xy):
            pred, depth = project(cam_init[c], model_v4.shapes[c], np.atleast_2d(pt), model_v4.distortion[c])
            if depth[0] <= 5.0:
                valid = False
                break
            errs.append(np.linalg.norm(pred[0] - q))
        if not valid:
            continue
        rms_px = float(np.sqrt(np.mean(np.square(errs))))
        if rms_px > 15.0:
            continue
        score = rms_px + 0.5 * abs(pt[2] - h)
        if best is None or score < best[0]:
            best = (score, pt, rms_px)
    if best is None:
        continue
    pt = best[1]
    new_j = len(keep_pts)
    keep_pts.append(pt)
    pt_cams[new_j] = set(ci)
    added_new_pts += 1
    for (c, fid), q in zip(t_clean, xy):
        kept_oc.append(c)
        kept_op.append(new_j)
        kept_xy.append(q)
        kept_fid.append(fid)
        fid_to_pt[(c, fid)] = new_j

aug_pts = np.array(keep_pts, float)
aug_oc = np.array(kept_oc, int)
aug_op = np.array(kept_op, int)
aug_xy = np.array(kept_xy, float)
aug_fid = np.array(kept_fid, int)

final_counts = np.bincount(aug_oc, minlength=len(keys))
print(f"Total landmarks: {len(aug_pts)} (attached={attached_existing}, new_pts={added_new_pts}), total obs: {len(aug_oc)}")
for i, k in enumerate(keys):
    print(f"  [{i:2d}] {k}: {int(np.sum(obs_cam_orig==i)):4d} -> {int(final_counts[i]):4d} obs")

aug_state_path = "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/fit_transmitter_v5.npz"
ft_dict = {k: ft[k] for k in ft.files}
ft_dict["cameras"] = cam_init
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
model = b.Posterior(
    state_file=aug_state_path,
    dem_file=str(mp.DEM),
    meta_file=mp.INPUT / "meta.json",
    exif_file=mp.HERE / "inputs/exif_utm.npz",
    feature_dir=mp.INPUT / "cv_features",
    config=cfg,
)

nc = model.nc
npt = model.npoint
n_glob = 7 * nc + 3 + 3 + 2
n_var = n_glob + 3 * npt

def compute_horizon_linearization(cam_ref):
    r0_list, J_list = [], []
    step = b.CAMERA_STEP * 0.2
    for i in range(nc):
        x0 = b.camera_coordinates(cam_ref[i])
        r0 = model.horizon_residuals(i, cam_ref[i])
        J = np.zeros((len(r0), 7))
        for j in range(7):
            up, dn = x0.copy(), x0.copy()
            up[j] += step[j]; dn[j] -= step[j]
            rp = model.horizon_residuals(i, b.camera_pose(up))
            rm = model.horizon_residuals(i, b.camera_pose(dn))
            J[:, j] = (rp - rm) / (2 * step[j])
        Q, R = np.linalg.qr(J, mode="reduced")
        r0_list.append(Q.T @ r0)
        J_list.append(R)
    return r0_list, J_list

ant_cams = np.flatnonzero(model.has_ant_label)
tx_cams = np.flatnonzero(model.has_tx_label)
gps_cams = np.flatnonzero(model.has_gps)
alt_cams = np.flatnonzero(model.has_alt)
hdg_cams = np.flatnonzero(model.has_heading)
n_obs = len(model.xy)

# Include depth barrier rows for antenna (len(ant_cams)) and transmitter (len(tx_cams))!
n_rows = (
    2 * n_obs
    + npt
    + 3 * len(ant_cams)
    + 3 * len(tx_cams)
    + 7 * nc
    + 2 * len(gps_cams)
    + len(alt_cams)
    + len(hdg_cams)
    + 4 * nc
    + 2
)

sp = lil_matrix((n_rows, n_var), dtype=int)
row = 0
for c_i, p_j in zip(model.oc, model.op):
    cols = list(range(7 * c_i, 7 * c_i + 7)) + list(range(n_glob + 3 * p_j, n_glob + 3 * p_j + 3))
    sp[row, cols] = 1; sp[row + 1, cols] = 1; row += 2
for p_j in range(npt):
    sp[row, range(n_glob + 3 * p_j, n_glob + 3 * p_j + 3)] = 1; row += 1
for c_i in ant_cams:
    cols = list(range(7 * c_i, 7 * c_i + 7)) + list(range(7 * nc, 7 * nc + 3))
    sp[row, cols] = 1; sp[row + 1, cols] = 1; sp[row + 2, cols] = 1; row += 3
for c_i in tx_cams:
    cols = list(range(7 * c_i, 7 * c_i + 7)) + list(range(7 * nc + 3, 7 * nc + 6))
    sp[row, cols] = 1; sp[row + 1, cols] = 1; sp[row + 2, cols] = 1; row += 3
for c_i in range(nc):
    cols = list(range(7 * c_i, 7 * c_i + 7))
    for _ in range(7):
        sp[row, cols] = 1; row += 1
for c_i in gps_cams:
    cols = [7 * c_i, 7 * c_i + 1, 7 * nc + 6, 7 * nc + 7]
    sp[row, cols] = 1; sp[row + 1, cols] = 1; row += 2
for c_i in alt_cams:
    sp[row, 7 * c_i + 2] = 1; row += 1
for c_i in hdg_cams:
    sp[row, 7 * c_i + 4] = 1; row += 1
for c_i in range(nc):
    sp[row, 7 * c_i + 3] = 1; row += 1
    sp[row, 7 * c_i + 5] = 1; row += 1
    sp[row, 7 * c_i + 6] = 1; row += 1
    sp[row, [7 * c_i, 7 * c_i + 1, 7 * c_i + 2]] = 1; row += 1  # camera above ground barrier
sp[row, 7 * nc + 6] = 1; row += 1
sp[row, 7 * nc + 7] = 1; row += 1
assert row == n_rows
sp = sp.tocsr()

cam_cur = cam_init.copy()
wrapped = model.heading + (cam_cur[:, 4] - model.heading + np.pi) % (2 * np.pi) - np.pi
cam_cur[:, 4] = np.where(model.has_heading, wrapped, cam_cur[:, 4])
ant_cur = ant_v4.copy()
tx_cur = tx_v4.copy()
bias_cur = bias_v4.copy()
extra_cur = 1.5
txe_cur = 2.0
pts_cur = relax_landmarks(model, cam_cur, aug_pts, steps=3)

st0 = model.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur)
print(f"Initial v5 logp: {model.logp(st0):.2f}", flush=True)

sigma_ant = np.hypot(cfg.antenna_label_sigma_px, extra_cur)
sigma_tx = np.hypot(cfg.antenna_label_sigma_px, txe_cur)

for outer in range(4):
    t_out = time.time()
    r0_hor, J_hor = compute_horizon_linearization(cam_cur)
    x_ref_cam = np.array([b.camera_coordinates(cam_cur[i]) for i in range(nc)])

    def res_fun(dx):
        d_cam = dx[:7 * nc].reshape(nc, 7)
        x_cam = x_ref_cam + d_cam
        cams = np.array([b.camera_pose(x_cam[i]) for i in range(nc)])
        ant = ant_cur + dx[7 * nc : 7 * nc + 3]
        tx = tx_cur + dx[7 * nc + 3 : 7 * nc + 6]
        bias = bias_cur + dx[7 * nc + 6 : 7 * nc + 8]
        pts = pts_cur + dx[n_glob:].reshape(npt, 3)

        pred = np.empty_like(model.xy)
        for i, idx in enumerate(model.by_camera):
            pred[idx], _ = project(cams[i], model.shapes[i], pts[model.op[idx]], model.distortion[i])
        tie = student_residual((pred - model.xy) / cfg.tie_sigma_px, cfg.student_df, 2).ravel()

        h_terr = model.terrain.height(pts[:, 0], pts[:, 1])
        terr = student_residual((pts[:, 2] - h_terr) / cfg.terrain_sigma_m, cfg.student_df)

        ant_res = []
        for i in ant_cams:
            ap, adepth = project(cams[i], model.shapes[i], ant, model.distortion[i])
            ant_res.extend((ap[0] - model.axy[i]) / sigma_ant)
            ant_res.append(min(0.0, float(adepth[0]) - 1.0) / 0.05)
        ant_res = np.asarray(ant_res)

        tx_res = []
        for i in tx_cams:
            tp, tdepth = project(cams[i], model.shapes[i], tx, model.distortion[i])
            tx_res.extend((tp[0] - model.txy[i]) / sigma_tx)
            tx_res.append(min(0.0, float(tdepth[0]) - 0.6) / 0.02)
        tx_res = np.asarray(tx_res)

        hor_res = np.concatenate([r0_hor[i] + J_hor[i] @ d_cam[i] for i in range(nc)])
        gps_res = ((cams[gps_cams, :2] + bias - model.gps[gps_cams]) / model.gps_sigma[gps_cams, None]).ravel()
        alt_res = (cams[alt_cams, 2] - model.alt[alt_cams]) / cfg.altitude_sigma_m
        hdg_res = (cams[hdg_cams, 4] - model.heading[hdg_cams]) / cfg.heading_sigma_rad
        cam_h = model.terrain.height(cams[:, 0], cams[:, 1])
        below_res = np.minimum(0.0, cams[:, 2] - (cam_h + 0.3)) / 0.05
        pri_res = np.column_stack([
            (cams[:, 3] - np.pi / 2) / cfg.elevation_sigma_rad,
            cams[:, 5] / cfg.roll_sigma_rad,
            (x_cam[:, 6] - np.log(model.focal)) / model.focal_sigma,
            below_res,
        ]).ravel()
        bias_res = bias / cfg.gps_common_sigma_m

        out = np.concatenate([tie, terr, ant_res, tx_res, hor_res, gps_res, alt_res, hdg_res, pri_res, bias_res])
        return np.where(np.isfinite(out), out, 1e3)

    x_scale = np.r_[np.tile([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01], nc), np.ones(8), np.ones(3 * npt)]
    bounds_lo = np.r_[np.tile([-8.0, -8.0, -6.0, -0.1, -0.1, -0.1, -0.1], nc), np.full(8, -6.0), np.full(3 * npt, -30.0)]
    bounds_hi = -bounds_lo

    sol = least_squares(
        res_fun, np.zeros(n_var), jac_sparsity=sp, x_scale=x_scale,
        bounds=(bounds_lo, bounds_hi), max_nfev=25, ftol=1e-4, xtol=1e-4,
    )
    dx = sol.x
    # Line search on exact model.logp!
    best_lp = model.logp(model.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur))
    for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
        d_cam = alpha * dx[:7 * nc].reshape(nc, 7)
        cam_cand = np.array([b.camera_pose(x_ref_cam[i] + d_cam[i]) for i in range(nc)])
        wrapped = model.heading + (cam_cand[:, 4] - model.heading + np.pi) % (2 * np.pi) - np.pi
        cam_cand[:, 4] = np.where(model.has_heading, wrapped, cam_cand[:, 4])
        ant_cand = ant_cur + alpha * dx[7 * nc : 7 * nc + 3]
        tx_cand = tx_cur + alpha * dx[7 * nc + 3 : 7 * nc + 6]
        bias_cand = bias_cur + alpha * dx[7 * nc + 6 : 7 * nc + 8]
        pts_cand = pts_cur + alpha * dx[n_glob:].reshape(npt, 3)
        pts_cand[:, 0] = np.clip(pts_cand[:, 0], terrain.e[0] + 3.0, terrain.e[-1] - 3.0)
        pts_cand[:, 1] = np.clip(pts_cand[:, 1], terrain.n[0] + 3.0, terrain.n[-1] - 3.0)
        st_cand = model.pack(cam_cand, ant_cand, tx_cand, bias_cand, extra_cur, txe_cur, pts_cand)
        lp_cand = model.logp(st_cand)
        if np.isfinite(lp_cand) and lp_cand > best_lp:
            cam_cur, ant_cur, tx_cur, bias_cur, pts_cur = cam_cand, ant_cand, tx_cand, bias_cand, pts_cand
            best_lp = lp_cand
            break
    # Also do 1 round of exact per-camera GN polish on the 4096-ray horizon
    for i in range(nc):
        cam_cur[i] = gn_polish_camera(model, i, cam_cur[i], ant_cur, pts_cur, bias_cur, extra_cur, tx_cur, txe_cur, steps=2)
    pts_cur = relax_landmarks(model, cam_cur, pts_cur, steps=2)
    # Optimize extra_cur and txe_cur at current state
    from scipy.optimize import minimize_scalar
    res_ex = minimize_scalar(lambda e: -model.scatter_logp(cam_cur, ant_cur, e), bounds=(0.01, 15.0), method="bounded")
    extra_cur = float(res_ex.x)
    res_txe = minimize_scalar(lambda e: -model.tx_scatter_logp(cam_cur, tx_cur, e), bounds=(0.01, 15.0), method="bounded")
    txe_cur = float(res_txe.x)
    sigma_ant = np.hypot(cfg.antenna_label_sigma_px, extra_cur)
    sigma_tx = np.hypot(cfg.antenna_label_sigma_px, txe_cur)
    st_now = model.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur)
    print(f"Outer {outer+1}: logp={model.logp(st_now):.2f}, extra={extra_cur:.2f}px, txe={txe_cur:.2f}px, ant={np.round(ant_cur, 3)}, tx={np.round(tx_cur, 3)} [{time.time()-t_out:.1f}s]", flush=True)

state_v5 = model.pack(cam_cur, ant_cur, tx_cur, bias_cur, extra_cur, txe_cur, pts_cur)
# Update fit_transmitter_v5.npz with the polished MAP state so start_vector() starts right at MAP!
ft_dict["cameras"] = cam_cur
ft_dict["antenna"] = ant_cur
ft_dict["transmitter"] = tx_cur
ft_dict["points"] = pts_cur
ft_dict["gps_bias"] = bias_cur
np.savez_compressed(aug_state_path, **ft_dict)
np.savez_compressed(
    "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/state_v5_map.npz",
    state=state_v5,
)

# Run LOO antenna and TX pair checks on state_v5!
rows = []
for i in np.flatnonzero(model.has_ant_label):
    m = copy.copy(model)
    m.has_ant_label = model.has_ant_label.copy()
    m.has_ant_label[i] = False
    m.n_ant_label = int(m.has_ant_label.sum())
    c_out = gn_polish_camera(m, i, cam_cur[i], ant_cur, pts_cur, bias_cur, extra_cur, tx_cur, txe_cur, steps=8)
    f_out = gc.Fit(m, state_v5)
    f_out.cam[i] = c_out
    f_out.refit_antenna()
    xy = model.axy[i]
    d_in = gc.ray(model, cam_cur[i], i, xy)
    d_out = gc.ray(model, f_out.cam[i], i, xy)
    in_m = gc.miss(cam_cur[i, :3], d_in, ant_cur)
    out_m = gc.miss(f_out.cam[i, :3], d_out, f_out.ant)
    pshift = float(np.linalg.norm(f_out.cam[i, :3] - cam_cur[i, :3]))
    rows.append((model.keys[i], in_m, out_m, pshift))
    print(f"  {model.keys[i]}: in={in_m:.2f}m, LOO={out_m:.2f}m, pose_shift={pshift:.2f}m", flush=True)

in_misses = [r[1] for r in rows]
out_misses = [r[2] for r in rows]
print(f"\nSummary LOO Antenna (v5 MAP):")
print(f"  In-sample ray miss: median={np.median(in_misses):.2f} m, max={np.max(in_misses):.2f} m")
print(f"  Held-out ray miss:  median={np.median(out_misses):.2f} m, 90th={np.percentile(out_misses, 90):.2f} m, max={np.max(out_misses):.2f} m")
