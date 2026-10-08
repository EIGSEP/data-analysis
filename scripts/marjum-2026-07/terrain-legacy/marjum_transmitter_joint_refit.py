"""Comprehensive joint refit of transmitter-only cameras and transmitter position.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import cv2

from marjum_bundle import rotation, boundary_pixels
from marjum_camera import rays, project
from marjum_position import RefinedTerrain
from eigsep_terrain.marjum_dem import MarjumDEM

LENS_WIDE_DIST = np.array([0.04477526, -0.03992051])
LENS_ULTRAWIDE_DIST = np.array([-0.02152495, 0.01223792])

def run_joint_refit():
    v5_path = Path("cv_transmitter_v5/fit_transmitter.npz")
    v5 = dict(np.load(v5_path))
    keys = list(v5["keys"])
    with open("meta.json") as f:
        meta = json.load(f)

    terrain = RefinedTerrain(MarjumDEM(cache_file="marjum_dem_sw.npz"))

    free_keys = ["2159", "2171", "2172", "2198", "2199", "2203"]
    fixed_keys = ["2210", "2211"]
    all_tx_keys = free_keys + fixed_keys

    init_cams = {}
    init_shapes = {}
    init_dist = {}

    init_cams["2159"] = np.array([1650.47, 2028.27, 1684.53, np.radians(97.92), np.radians(-95.18), np.radians(0.56), 3024.0])
    init_shapes["2159"] = v5["shapes"][keys.index("2159")]
    init_dist["2159"] = LENS_WIDE_DIST.copy()

    init_cams["2171"] = np.array([1629.79, 2000.95, 1681.96, np.radians(69.1), np.radians(64.0), np.radians(1.6), 4495.8])
    init_shapes["2171"] = v5["shapes"][keys.index("2171")]
    init_dist["2171"] = LENS_WIDE_DIST.copy()

    init_cams["2172"] = np.array([1638.41, 1995.37, 1682.50, np.radians(76.8), np.radians(65.4), np.radians(1.0), 4556.0])
    init_shapes["2172"] = v5["shapes"][keys.index("2172")]
    init_dist["2172"] = LENS_WIDE_DIST.copy()

    init_cams["2198"] = np.array([1699.26, 2043.25, 1733.74, np.radians(113.3), np.radians(-185.5), np.radians(-19.7), 1176.0])
    init_shapes["2198"] = v5["shapes"][keys.index("2198")]
    init_dist["2198"] = LENS_ULTRAWIDE_DIST.copy()

    init_cams["2199"] = np.array([1650.67, 2027.51, 1684.77, np.radians(104.1), np.radians(-98.7), np.radians(-1.5), 2957.5])
    init_shapes["2199"] = v5["shapes"][keys.index("2199")]
    init_dist["2199"] = LENS_WIDE_DIST.copy()

    init_cams["2203"] = np.array([1650.60, 2027.50, 1684.70, np.radians(107.8), np.radians(-87.5), np.radians(3.0), 1600.0])
    init_shapes["2203"] = v5["shapes"][keys.index("2203")]
    init_dist["2203"] = LENS_ULTRAWIDE_DIST.copy()

    for k in fixed_keys:
        idx = keys.index(k)
        init_cams[k] = v5["cameras"][idx].copy()
        init_shapes[k] = v5["shapes"][idx].copy()
        init_dist[k] = v5["distortion"][idx].copy()

    init_r_tx = np.array([1652.41, 2025.24, 1684.74])

    feat_data = {k: np.load(f"cv_features/sift_{k}.npz") for k in all_tx_keys}
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    pairwise_matches = []
    for i, k1 in enumerate(all_tx_keys):
        for j, k2 in enumerate(all_tx_keys):
            if j <= i or (k1 in fixed_keys and k2 in fixed_keys): continue
            des1 = feat_data[k1]["descriptors"].astype(np.float32)
            des2 = feat_data[k2]["descriptors"].astype(np.float32)
            xy1 = feat_data[k1]["xy"]
            xy2 = feat_data[k2]["xy"]
            m = matcher.knnMatch(des1, des2, k=2)
            good = [pair for pair in m if pair[0].distance < 0.75 * pair[1].distance]
            if len(good) >= 8:
                pts1 = xy1[[p[0].queryIdx for p in good]]
                pts2 = xy2[[p[0].trainIdx for p in good]]
                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
                if mask is not None:
                    inliers = mask.ravel().astype(bool)
                    if inliers.sum() >= 7:
                        idx_in = np.where(inliers)[0]
                        if len(idx_in) > 40:
                            idx_in = np.random.RandomState(42).choice(idx_in, 40, replace=False)
                        pairwise_matches.append({
                            "k1": k1, "k2": k2,
                            "pts1": pts1[idx_in], "pts2": pts2[idx_in],
                        })
    print(f"Constructed {len(pairwise_matches)} pairwise feature sets.")

    horizon_data = {}
    for k in ["2171", "2172", "2198"]:
        seg = np.load(f"img_seg_IMG_{k}.npz")
        skymask = np.flipud(seg["skymask"]).astype(bool)
        ptree = np.flipud(seg["ptree"]) if "ptree" in seg else None
        pts = boundary_pixels(skymask, ptree, spacing=80)
        if len(pts) > 0:
            horizon_data[k] = pts
            print(f"Loaded {len(pts)} horizon points for {k}")

    def unpack(x):
        r_tx = x[:3]
        cams = {}
        for idx, k in enumerate(free_keys):
            q = x[3 + 7*idx : 3 + 7*(idx+1)]
            base = init_cams[k]
            cams[k] = np.r_[base[:6] + q[:6], base[6] * np.exp(q[6])]
        for k in fixed_keys:
            cams[k] = init_cams[k]
        return r_tx, cams

    def residual(x):
        r_tx, cams = unpack(x)
        res = []

        # 1. Transmitter Reprojection Residuals (all 8 cameras)
        for k in all_tx_keys:
            cam = cams[k]
            shape = init_shapes[k]
            dist = init_dist[k]
            tx_px_obs = np.array(meta[k]["transmitter_px"])
            proj_px, depth = project(cam, shape, r_tx, dist)
            depth_penalty = max(0.0, 0.5 - float(depth[0])) * 500.0
            res.extend((proj_px[0] - tx_px_obs) / 1.0)
            res.append(depth_penalty)

        # 2. Pairwise Epipolar Constraints
        for pair in pairwise_matches:
            k1, k2 = pair["k1"], pair["k2"]
            p1, s1, d1 = cams[k1], init_shapes[k1], init_dist[k1]
            p2, s2, d2 = cams[k2], init_shapes[k2], init_dist[k2]
            r1 = rays(p1, s1, pair["pts1"], d1)
            r2 = rays(p2, s2, pair["pts2"], d2)
            b = p2[:3] - p1[:3]
            b_norm = np.linalg.norm(b)
            if b_norm < 1e-4:
                b_unit = np.array([1.0, 0.0, 0.0])
            else:
                b_unit = b / b_norm
            f_avg = 0.5 * (p1[6] + p2[6])
            triple = np.einsum("ij,j->i", np.cross(r1, r2), b_unit)
            res.extend(triple * f_avg / 2.5)

        # 3. Horizon Skyline Constraints
        for k, hpts in horizon_data.items():
            cam = cams[k]
            shape = init_shapes[k]
            dist = init_dist[k]
            d = rays(cam, shape, hpts, dist)
            az = np.arctan2(d[:, 1], d[:, 0])
            alt = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
            sky_alt = terrain.skyline(cam[:3], az, count=1024, peaks=6, refine=17)
            err_px = (alt - sky_alt) * cam[6]
            res.extend(err_px / 15.0)

        # 4. DEM ground height regularizer & priors on free camera parameters
        for idx, k in enumerate(free_keys):
            q = x[3 + 7*idx : 3 + 7*(idx+1)]
            cam = cams[k]
            h_terrain = terrain.height(cam[0], cam[1])
            dz = cam[2] - h_terrain
            res.append(max(0.0, 1.0 - dz) / 0.1)
            res.append(max(0.0, dz - 2.5) / 0.2)

            res.extend(q[:2] / 5.0)
            res.append(q[2] / 2.0)
            res.extend(q[3:6] / 0.08)
            res.append(q[6] / 0.05)

        res.extend((r_tx - init_r_tx) / 1.0)

        return np.array(res, dtype=float)

    x0 = np.zeros(3 + 7 * len(free_keys))
    x0[:3] = init_r_tx

    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    lb[:3] = init_r_tx - [5.0, 5.0, 3.0]
    ub[:3] = init_r_tx + [5.0, 5.0, 3.0]

    for idx, k in enumerate(free_keys):
        lb[3 + 7*idx : 3 + 7*(idx+1)] = [-15.0, -15.0, -5.0, -0.2, -0.2, -0.15, -0.15]
        ub[3 + 7*idx : 3 + 7*(idx+1)] = [ 15.0,  15.0,  5.0,  0.2,  0.2,  0.15,  0.15]

    print("Starting joint optimization...")
    res = least_squares(
        residual, x0, bounds=(lb, ub),
        loss="soft_l1", f_scale=2.0, max_nfev=150, ftol=1e-6, verbose=2
    )
    print(f"Optimization finished: success={res.success}, cost={res.cost:.4f}, nfev={res.nfev}")

    r_tx_opt, cams_opt = unpack(res.x)
    print(f"Optimized Transmitter Position: {r_tx_opt.round(3)}")

    print("\n--- Transmitter Pixel Residuals ---")
    reproj_errors = {}
    for k in all_tx_keys:
        cam = cams_opt[k]
        shape = init_shapes[k]
        dist = init_dist[k]
        tx_px_obs = np.array(meta[k]["transmitter_px"])
        proj_px, depth = project(cam, shape, r_tx_opt, dist)
        px_err = float(np.linalg.norm(proj_px[0] - tx_px_obs))
        reproj_errors[k] = {
            "observed_px": tx_px_obs.tolist(),
            "projected_px": proj_px[0].tolist(),
            "depth_m": float(depth[0]),
            "pixel_error": px_err,
        }
        print(f"{k}: obs={tx_px_obs.round(1)}, proj={proj_px[0].round(1)}, depth={depth[0]:.2f}m, err={px_err:.3f} px")

    print("\n--- Camera Poses ---")
    for k in free_keys:
        cam = cams_opt[k]
        h_g = terrain.height(cam[0], cam[1])
        print(f"{k}: pos={cam[:3].round(2)} (clearance={cam[2]-h_g:.2f}m), angles(deg)={np.degrees(cam[3:6]).round(2)}, f={cam[6]:.1f}")

    for k in free_keys:
        idx = keys.index(k)
        v5["cameras"][idx] = cams_opt[k]
        v5["distortion"][idx] = init_dist[k]

    np.savez_compressed(v5_path, **v5)
    print(f"Saved updated joint state to {v5_path}")

    summary = {
        "transmitter_position": r_tx_opt.tolist(),
        "transmitter_reprojection_errors": reproj_errors,
        "cameras": {k: cams_opt[k].tolist() for k in all_tx_keys},
        "lens_distortion": {k: init_dist[k].tolist() for k in all_tx_keys},
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "success": bool(res.success),
    }
    with open("cv_transmitter_v5/fit_transmitter.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved cv_transmitter_v5/fit_transmitter.json")

if __name__ == "__main__":
    run_joint_refit()
