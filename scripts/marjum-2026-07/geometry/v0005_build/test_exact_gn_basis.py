import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.environ["EIGSEP_CAMPAIGN_ROOT"] = "/home/aparsons/projects/eigsep/marjum-2026-07"

import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/aparsons/projects/eigsep"
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src"))
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src/eigsep_terrain/marjum_geometry"))
sys.path.insert(0, os.path.join(ROOT, "data-analysis/scripts/marjum-2026-07/geometry"))
sys.path.insert(0, "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch")

import marjum_mcmc_b21 as b
from marjum_camera import project
from marjum_mcmc import student_residual
from test_ess_mh_5chains import build_model, prepare_calibrated_system, fast_terrain_height
from test_surrogate_mh import unpack_211


def surrogate_residuals(model, g, extra, txe, data):
    """Return vector r(g) such that surrogate_logp(g) == -0.5 * dot(r, r) + norm_const(extra, txe)."""
    x_cam = g[:203].reshape(model.nc, 7)
    cam = np.empty((model.nc, 7))
    cam[:, :6] = x_cam[:, :6]
    cam[:, 6] = np.exp(x_cam[:, 6])
    ant = g[203:206]
    tx = g[206:209]
    bias = g[209:211]

    dg_cam = g[:203] - data["g_bar"][:203]
    pts = data["pts_bar"] + (data["A_flat"] @ dg_cam).reshape(-1, 3)

    th, ph, ti = cam[:, 3], cam[:, 4], cam[:, 5]
    cth, sth = np.cos(th), np.sin(th)
    cph, sph = np.cos(ph), np.sin(ph)
    cti, sti = np.cos(ti), np.sin(ti)
    R_all = np.empty((model.nc, 3, 3))
    R_all[:, 0, 0] = cph * cth * cti - sph * sti
    R_all[:, 0, 1] = -cph * cth * sti - sph * cti
    R_all[:, 0, 2] = cph * sth
    R_all[:, 1, 0] = sph * cth * cti + cph * sti
    R_all[:, 1, 1] = -sph * cth * sti + cph * cti
    R_all[:, 1, 2] = sph * sth
    R_all[:, 2, 0] = -sth * cti
    R_all[:, 2, 1] = sth * sti
    R_all[:, 2, 2] = cth

    t = model.terrain
    c = model.config
    oc, op = model.oc, model.op
    v = np.einsum("ni,nij->nj", pts[op] - cam[oc, :3], R_all[oc])
    q = -v[:, [1, 0]] / np.maximum(v[:, 2:3], 0.1)
    r2_obs = np.sum(q * q, axis=1)
    dist_factor = (1.0 + data["obs_k0"] * r2_obs + data["obs_k1"] * r2_obs * r2_obs) * cam[oc, 6]
    pred = q * dist_factor[:, None] + data["obs_center"]
    tie = student_residual((pred - model.xy) / c.tie_sigma_px, c.student_df, 2).ravel()
    h_pts = fast_terrain_height(t, pts[:, 0], pts[:, 1])
    terr = student_residual((pts[:, 2] - h_pts) / c.terrain_sigma_m, c.student_df)

    # Horizon residuals
    h_idx = data["hor_cam_idx"]
    d_vec = data["hor_d_base"] / cam[h_idx, 6:7]
    rd = np.linalg.norm(d_vec, axis=1)
    r = data["r_init"] * (data["cam_bar"][h_idx, 6] / cam[h_idx, 6])
    k0, k1 = data["hor_k0"], data["hor_k1"]
    for _ in range(4):
        r2 = r * r
        val = r * (1.0 + k0 * r2 + k1 * r2 * r2) - rd
        der = 1.0 + 3.0 * k0 * r2 + 5.0 * k1 * r2 * r2
        r = r - val / np.maximum(der, 0.05)
    q_h = d_vec * (r / np.maximum(rd, 1e-30))[:, None]
    body = np.c_[-q_h[:, 1], -q_h[:, 0], np.ones(len(q_h))]
    world = np.einsum("ni,nij->nj", body, R_all[h_idx].transpose(0, 2, 1))
    world /= np.linalg.norm(world, axis=1, keepdims=True)
    el = np.arctan2(world[:, 2], np.hypot(world[:, 0], world[:, 1]))
    daz = (np.arctan2(world[:, 1], world[:, 0]) - data["az0_all"] + np.pi) % (2 * np.pi) - np.pi

    offsets = data["offsets"]
    d_off = offsets[1] - offsets[0]
    pos_idx = (daz - offsets[0]) / d_off
    idx0 = np.clip(np.floor(pos_idx).astype(int), 0, len(offsets) - 2)
    frac = np.clip(pos_idx - idx0, 0.0, 1.0)
    rows = np.arange(len(daz))
    sky_base = (1.0 - frac) * data["s0_smooth"][rows, idx0] + frac * data["s0_smooth"][rows, idx0 + 1]
    dpos = cam[h_idx, :3] - data["cam_bar"][h_idx, :3]
    sky = sky_base + np.sum(dpos * data["ds_dpos_all"], axis=1)
    r_h = student_residual((el - sky) * data["hor_focal_exif"] / c.horizon_sigma_angular_px, c.student_df)

    sigma_ant = np.hypot(c.antenna_label_sigma_px, extra)
    sigma_tx = np.hypot(c.antenna_label_sigma_px, txe)
    res_other = []
    for i in range(model.nc):
        ci = cam[i]
        if model.has_ant_label[i]:
            ap, _ = project(ci, model.shapes[i], ant, model.distortion[i])
            res_other.append((ap[0] - model.axy[i]) / sigma_ant)
        if model.has_tx_label[i]:
            tp, _ = project(ci, model.shapes[i], tx, model.distortion[i])
            res_other.append((tp[0] - model.txy[i]) / sigma_tx)
        if model.has_gps[i]:
            res_other.append((ci[:2] + bias - model.gps[i]) / model.gps_sigma[i])
        if model.has_alt[i]:
            res_other.append([(ci[2] - model.alt[i]) / c.altitude_sigma_m])
        if model.has_heading[i]:
            delta = (ci[4] - model.heading[i] + np.pi) % (2 * np.pi) - np.pi
            res_other.append([delta / c.heading_sigma_rad])
        res_other.append([(ci[3] - np.pi / 2) / c.elevation_sigma_rad,
                          ci[5] / c.roll_sigma_rad,
                          (x_cam[i, 6] - np.log(model.focal[i])) / model.focal_sigma[i]])
    res_other.append(bias / c.gps_common_sigma_m)
    return np.concatenate([tie, terr, r_h, np.concatenate(res_other)])


def main():
    t0 = time.time()
    model = build_model()
    z_star = np.load("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/state_v5_mode.npz")["state"]
    data, _, _ = prepare_calibrated_system(model, z_star)
    data["A_flat"] = np.ascontiguousarray(data["A_full"].reshape(-1, 203))
    # Smooth s0_mat_all slightly along azimuth (sigma=2 grid steps = 0.0005 rad = 0.03 deg) so derivatives are smooth!
    data["s0_smooth"] = gaussian_filter1d(data["s0_mat_all"], sigma=2.0, axis=1)

    g_cur = data["g_bar"].copy()
    ex, txe = data["extra_bar"], data["txe_bar"]
    step_g = np.r_[np.tile([0.1, 0.1, 0.1, 1e-3, 1e-3, 1e-3, 2e-3], model.nc), [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.2, 0.2]]

    for gn_it in range(3):
        r0 = surrogate_residuals(model, g_cur, ex, txe, data)
        J = np.empty((len(r0), 211))
        for k in range(211):
            gp = g_cur.copy(); gp[k] += step_g[k]
            gm = g_cur.copy(); gm[k] -= step_g[k]
            J[:, k] = (surrogate_residuals(model, gp, ex, txe, data) - surrogate_residuals(model, gm, ex, txe, data)) / (2 * step_g[k])
        H_gn = J.T @ J
        grad = J.T @ r0
        # Levenberg-Marquardt damped step
        dg = -np.linalg.solve(H_gn + 1e-3 * np.diag(np.diag(H_gn)), grad)
        for alpha in [1.0, 0.5, 0.25, 0.1]:
            g_try = g_cur + alpha * dg
            r_try = surrogate_residuals(model, g_try, ex, txe, data)
            if np.dot(r_try, r_try) < np.dot(r0, r0):
                g_cur = g_try
                break
        st_cur = model.pack(*unpack_211(model, g_cur, ex, txe, data["g_bar"], data["pts_bar"], data["A_full"]))
        print(f"GN iter {gn_it}: 0.5|r|^2 = {0.5*np.dot(r_try, r_try):.2f}, |grad| = {np.linalg.norm(grad):.3f}, exact logp = {model.logp(st_cur):.2f}", flush=True)

    # Now build L_211 directly from the exact composite Gauss-Newton Hessian H_gn = J.T @ J!
    scale_211 = step_g
    H_scaled = (H_gn * scale_211[:, None]) * scale_211[None, :]
    H_scaled = 0.5 * (H_scaled + H_scaled.T)
    w, v = np.linalg.eigh(H_scaled)
    print(f"Exact composite H_scaled eigenvalues: min={w.min():.5f}, med={np.median(w):.3f}, max={w.max():.1f}", flush=True)
    L_gn = scale_211[:, None] * (v / np.sqrt(np.maximum(w, 1e-6)))

    # Test quadratic accuracy of -0.5 * |r(g_cur + L_gn @ u)|^2 across random draws u ~ N(0, I_211)!
    r_star = surrogate_residuals(model, g_cur, ex, txe, data)
    lp_star = -0.5 * float(np.dot(r_star, r_star))
    st_star = model.pack(*unpack_211(model, g_cur, ex, txe, data["g_bar"], data["pts_bar"], data["A_full"]))
    ex_star = model.logp(st_star)
    np.savez("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/state_v5_gn.npz", state=st_star)

    rng = np.random.default_rng(42)
    for scale in [0.25, 0.5, 1.0]:
        quad_res = []
        exact_diffs = []
        for _ in range(20):
            u = scale * rng.standard_normal(211)
            g = g_cur + L_gn @ u
            r_u = surrogate_residuals(model, g, ex, txe, data)
            lp_u = -0.5 * float(np.dot(r_u, r_u))
            st_u = model.pack(*unpack_211(model, g, ex, txe, data["g_bar"], data["pts_bar"], data["A_full"]))
            ex_u = model.logp(st_u)
            if np.isfinite(ex_u):
                quad_res.append((lp_u - lp_star) + 0.5 * float(np.dot(u, u)))
                exact_diffs.append((ex_u - ex_star) - (lp_u - lp_star))
        print(f"Scale {scale} ({len(exact_diffs)}/20 finite): non-Gaussian residual (lp_u - lp_star + 0.5|u|^2) "
              f"mean={np.mean(quad_res):+.2f}, SD={np.std(quad_res):.2f}; (exact - surr) SD={np.std(exact_diffs):.3f}", flush=True)


if __name__ == "__main__":
    main()
