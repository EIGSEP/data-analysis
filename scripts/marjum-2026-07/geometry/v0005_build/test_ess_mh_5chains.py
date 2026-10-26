import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.environ["EIGSEP_CAMPAIGN_ROOT"] = "/home/aparsons/projects/eigsep/marjum-2026-07"

import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize_scalar
import arviz as az

ROOT = "/home/aparsons/projects/eigsep"
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src"))
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src/eigsep_terrain/marjum_geometry"))
sys.path.insert(0, os.path.join(ROOT, "data-analysis/scripts/marjum-2026-07/geometry"))
sys.path.insert(0, "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch")

import marjum_mcmc_b21 as b
from marjum_bundle import rotation
from marjum_camera import project, rays
from marjum_mcmc import student_residual
from test_schur_mcmc import build_model, compute_schur_couplings
from test_surrogate_mh import unpack_211


def fast_terrain_height(terrain, e, n):
    r = (n - terrain.n[0]) / terrain.res
    c = (e - terrain.e[0]) / terrain.res
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    dr = r - r0
    dc = c - c0
    d = terrain.data
    H, W = d.shape
    r0c = np.clip(r0, 0, H - 2)
    c0c = np.clip(c0, 0, W - 2)
    z00 = d[r0c, c0c].astype(np.float64)
    z01 = d[r0c, c0c + 1].astype(np.float64)
    z10 = d[r0c + 1, c0c].astype(np.float64)
    z11 = d[r0c + 1, c0c + 1].astype(np.float64)
    return (1.0 - dr) * ((1.0 - dc) * z00 + dc * z01) + dr * ((1.0 - dc) * z10 + dc * z11)


def prepare_calibrated_system(model, z_star):
    cam_bar, ant_bar, tx_bar, bias_bar, extra_bar, txe_bar, pts_bar = model.unpack(z_star)
    extra_bar = float(minimize_scalar(lambda e: -model.scatter_logp(cam_bar, ant_bar, e), bounds=(0.05, 20.0), method="bounded").x)
    txe_bar = float(minimize_scalar(lambda e: -model.tx_scatter_logp(cam_bar, tx_bar, e), bounds=(0.05, 20.0), method="bounded").x)

    x_bar, pts_bar, A_obs, _ = compute_schur_couplings(
        model, cam_bar, pts_bar, ant_bar, tx_bar, bias_bar, extra_bar, txe_bar
    )

    nc, npt = model.nc, model.npoint
    c = model.config
    n_obs = len(model.xy)
    eps_p = 0.02
    eps_c = b.CAMERA_STEP * 0.2

    J_pt = np.zeros((n_obs, 2, 3))
    J_cam = np.zeros((n_obs, 2, 7))
    for ax in range(3):
        p_up = pts_bar.copy(); p_up[:, ax] += eps_p
        p_dn = pts_bar.copy(); p_dn[:, ax] -= eps_p
        pr_up, pr_dn = np.empty_like(model.xy), np.empty_like(model.xy)
        for i, idx in enumerate(model.by_camera):
            pr_up[idx], _ = project(cam_bar[i], model.shapes[i], p_up[model.op[idx]], model.distortion[i])
            pr_dn[idx], _ = project(cam_bar[i], model.shapes[i], p_dn[model.op[idx]], model.distortion[i])
        J_pt[:, :, ax] = (student_residual((pr_up - model.xy) / c.tie_sigma_px, c.student_df, 2) -
                          student_residual((pr_dn - model.xy) / c.tie_sigma_px, c.student_df, 2)) / (2 * eps_p)

    for i, idx in enumerate(model.by_camera):
        if not len(idx):
            continue
        x0 = x_bar[i]
        pts_i = pts_bar[model.op[idx]]
        xy_i = model.xy[idx]
        for k_ax in range(7):
            up, dn = x0.copy(), x0.copy()
            up[k_ax] += eps_c[k_ax]; dn[k_ax] -= eps_c[k_ax]
            pr_u, _ = project(b.camera_pose(up), model.shapes[i], pts_i, model.distortion[i])
            pr_d, _ = project(b.camera_pose(dn), model.shapes[i], pts_i, model.distortion[i])
            J_cam[idx, :, k_ax] = (student_residual((pr_u - xy_i) / c.tie_sigma_px, c.student_df, 2) -
                                   student_residual((pr_d - xy_i) / c.tie_sigma_px, c.student_df, 2)) / (2 * eps_c[k_ax])

    J_terr = np.zeros((npt, 3))
    for ax in range(3):
        p_up = pts_bar.copy(); p_up[:, ax] += eps_p
        p_dn = pts_bar.copy(); p_dn[:, ax] -= eps_p
        tu = student_residual((p_up[:, 2] - model.terrain.height(p_up[:, 0], p_up[:, 1])) / c.terrain_sigma_m, c.student_df)
        td = student_residual((p_dn[:, 2] - model.terrain.height(p_dn[:, 0], p_dn[:, 1])) / c.terrain_sigma_m, c.student_df)
        J_terr[:, ax] = (tu - td) / (2 * eps_p)

    H_ll = np.einsum("na,nb->nab", J_terr, J_terr) + np.eye(3)[None] * 1e-4
    jtj_obs = np.einsum("nra,nrb->nab", J_pt, J_pt)
    for a in range(3):
        for b_ax in range(3):
            H_ll[:, a, b_ax] += np.bincount(model.op, weights=jtj_obs[:, a, b_ax], minlength=npt)
    H_ll_inv = np.linalg.inv(H_ll)

    S = np.zeros((211, 211))
    H_lc = np.zeros((npt, 3, 203))
    H_lc_obs = np.einsum("nra,nrb->nab", J_pt, J_cam)
    J_cc_obs = np.einsum("nra,nrb->nab", J_cam, J_cam)

    for n in range(n_obs):
        ci = model.oc[n]
        pj = model.op[n]
        S[7 * ci : 7 * ci + 7, 7 * ci : 7 * ci + 7] += J_cc_obs[n]
        H_lc[pj, :, 7 * ci : 7 * ci + 7] += H_lc_obs[n]

    H_ll_inv_H_lc = np.einsum("jab,jbc->jac", H_ll_inv, H_lc)
    S[:203, :203] -= np.einsum("jab,jac->bc", H_lc, H_ll_inv_H_lc)

    sigma_ant = np.hypot(c.antenna_label_sigma_px, extra_bar)
    sigma_tx = np.hypot(c.antenna_label_sigma_px, txe_bar)

    for i in range(nc):
        x0 = x_bar[i]
        r0_h = model.horizon_residuals(i, cam_bar[i])
        J_h = np.zeros((len(r0_h), 7))
        for k_ax in range(7):
            up, dn = x0.copy(), x0.copy()
            up[k_ax] += eps_c[k_ax]; dn[k_ax] -= eps_c[k_ax]
            J_h[:, k_ax] = (model.horizon_residuals(i, b.camera_pose(up)) - model.horizon_residuals(i, b.camera_pose(dn))) / (2 * eps_c[k_ax])
        S[7 * i : 7 * i + 7, 7 * i : 7 * i + 7] += J_h.T @ J_h

        S[7 * i + 3, 7 * i + 3] += 1.0 / c.elevation_sigma_rad ** 2
        S[7 * i + 5, 7 * i + 5] += 1.0 / c.roll_sigma_rad ** 2
        S[7 * i + 6, 7 * i + 6] += 1.0 / model.focal_sigma[i] ** 2
        if model.has_alt[i]:
            S[7 * i + 2, 7 * i + 2] += 1.0 / c.altitude_sigma_m ** 2
        if model.has_heading[i]:
            S[7 * i + 4, 7 * i + 4] += 1.0 / c.heading_sigma_rad ** 2
        if model.has_gps[i]:
            w_g = 1.0 / model.gps_sigma[i] ** 2
            for ax in range(2):
                S[7 * i + ax, 7 * i + ax] += w_g
                S[209 + ax, 209 + ax] += w_g
                S[7 * i + ax, 209 + ax] += w_g
                S[209 + ax, 7 * i + ax] += w_g

        if model.has_ant_label[i]:
            J_ca = np.zeros((2, 10))
            for k_ax in range(7):
                up, dn = x0.copy(), x0.copy()
                up[k_ax] += eps_c[k_ax]; dn[k_ax] -= eps_c[k_ax]
                pu, _ = project(b.camera_pose(up), model.shapes[i], ant_bar, model.distortion[i])
                pd, _ = project(b.camera_pose(dn), model.shapes[i], ant_bar, model.distortion[i])
                J_ca[:, k_ax] = (pu[0] - pd[0]) / (2 * eps_c[k_ax] * sigma_ant)
            for a_ax in range(3):
                au, ad = ant_bar.copy(), ant_bar.copy()
                au[a_ax] += 0.02; ad[a_ax] -= 0.02
                pu, _ = project(cam_bar[i], model.shapes[i], au, model.distortion[i])
                pd, _ = project(cam_bar[i], model.shapes[i], ad, model.distortion[i])
                J_ca[:, 7 + a_ax] = (pu[0] - pd[0]) / (0.04 * sigma_ant)
            idx_ca = list(range(7 * i, 7 * i + 7)) + [203, 204, 205]
            S[np.ix_(idx_ca, idx_ca)] += J_ca.T @ J_ca

        if model.has_tx_label[i]:
            J_ct = np.zeros((2, 10))
            for k_ax in range(7):
                up, dn = x0.copy(), x0.copy()
                up[k_ax] += eps_c[k_ax]; dn[k_ax] -= eps_c[k_ax]
                pu, _ = project(b.camera_pose(up), model.shapes[i], tx_bar, model.distortion[i])
                pd, _ = project(b.camera_pose(dn), model.shapes[i], tx_bar, model.distortion[i])
                J_ct[:, k_ax] = (pu[0] - pd[0]) / (2 * eps_c[k_ax] * sigma_tx)
            for a_ax in range(3):
                tu, td = tx_bar.copy(), tx_bar.copy()
                tu[a_ax] += 0.02; td[a_ax] -= 0.02
                pu, _ = project(cam_bar[i], model.shapes[i], tu, model.distortion[i])
                pd, _ = project(cam_bar[i], model.shapes[i], td, model.distortion[i])
                J_ct[:, 7 + a_ax] = (pu[0] - pd[0]) / (0.04 * sigma_tx)
            idx_ct = list(range(7 * i, 7 * i + 7)) + [206, 207, 208]
            S[np.ix_(idx_ct, idx_ct)] += J_ct.T @ J_ct

    for ax in range(2):
        S[209 + ax, 209 + ax] += 1.0 / c.gps_common_sigma_m ** 2

    # Diagonalize S directly in natural prior-scaled units so no mode is artificially clamped!
    scale_211 = np.r_[np.tile(b.CAMERA_SCALE, nc), np.ones(8)]
    S_scaled = 0.5 * ((S * scale_211[:, None]) * scale_211[None, :] + ((S * scale_211[:, None]) * scale_211[None, :]).T)
    w, v = np.linalg.eigh(S_scaled)
    w_reg = np.maximum(w, 1e-4)  # DO NOT clamp at 0.5! Minimum true eigenvalue is ~0.005!
    L_raw = scale_211[:, None] * (v / np.sqrt(w_reg))

    A_full = -H_ll_inv_H_lc
    g_bar = np.r_[x_bar.ravel(), ant_bar, tx_bar, bias_bar]

    # Build vectorized horizon & observation arrays
    offsets = np.linspace(-0.08, 0.08, 321)  # 0.0005 rad spacing, wide enough so daz never clips
    hor_cam_idx, hor_d_base, hor_k0, hor_k1, hor_focal_exif = [], [], [], [], []
    az0_all, s0_mat_all, ds_dpos_all = [], [], []

    for i in range(nc):
        xy = model.horizon[i]
        h, w_im = model.shapes[i]
        d_raw = (xy - [w_im // 2, h // 2])
        hor_cam_idx.append(np.full(len(xy), i, dtype=int))
        hor_d_base.append(d_raw)
        hor_k0.append(np.full(len(xy), model.distortion[i, 0]))
        hor_k1.append(np.full(len(xy), model.distortion[i, 1]))
        hor_focal_exif.append(np.full(len(xy), model.focal[i]))

        d0 = rays(cam_bar[i], model.shapes[i], xy, model.distortion[i])
        az0 = np.arctan2(d0[:, 1], d0[:, 0])
        az_mat = az0[:, None] + offsets[None, :]
        s0_mat = model.terrain.skyline(cam_bar[i, :3], az_mat.ravel(), model.config.skyline_samples).reshape(az_mat.shape)
        ds_dpos = np.zeros((len(az0), 3))
        eps_pos = 0.4
        for ax in range(3):
            pu, pd = cam_bar[i, :3].copy(), cam_bar[i, :3].copy()
            pu[ax] += eps_pos; pd[ax] -= eps_pos
            ds_dpos[:, ax] = (model.terrain.skyline(pu, az0, model.config.skyline_samples) -
                              model.terrain.skyline(pd, az0, model.config.skyline_samples)) / (2 * eps_pos)
        az0_all.append(az0)
        s0_mat_all.append(s0_mat)
        ds_dpos_all.append(ds_dpos)

    hor_cam_idx = np.concatenate(hor_cam_idx)
    hor_d_base = np.vstack(hor_d_base)
    hor_k0 = np.concatenate(hor_k0)
    hor_k1 = np.concatenate(hor_k1)
    hor_focal_exif = np.concatenate(hor_focal_exif)
    az0_all = np.concatenate(az0_all)
    s0_mat_all = np.vstack(s0_mat_all)
    ds_dpos_all = np.vstack(ds_dpos_all)

    d_init = hor_d_base / cam_bar[hor_cam_idx, 6:7]
    r_init = np.linalg.norm(d_init, axis=1)
    for _ in range(25):
        r2 = r_init * r_init
        val = r_init * (1 + hor_k0 * r2 + hor_k1 * r2 * r2) - np.linalg.norm(d_init, axis=1)
        der = 1 + 3 * hor_k0 * r2 + 5 * hor_k1 * r2 * r2
        r_init = np.maximum(0, r_init - np.clip(val / np.maximum(der, 0.05), -0.25, 0.25))

    # Precompute vectorized tie-point arrays
    obs_k0 = model.distortion[model.oc, 0]
    obs_k1 = model.distortion[model.oc, 1]
    obs_center = np.column_stack((model.shapes[model.oc, 1] // 2, model.shapes[model.oc, 0] // 2))

    data = dict(
        g_bar=g_bar, extra_bar=extra_bar, txe_bar=txe_bar,
        pts_bar=pts_bar, A_full=A_full, L_211=L_raw, cam_bar=cam_bar,
        B_pts=np.ascontiguousarray(np.einsum("jac,ck->jak", A_full, L_raw[:203, :]).reshape(-1, 211)),
        offsets=offsets, hor_cam_idx=hor_cam_idx, hor_d_base=hor_d_base,
        hor_k0=hor_k0, hor_k1=hor_k1, hor_focal_exif=hor_focal_exif,
        az0_all=az0_all, s0_mat_all=s0_mat_all, ds_dpos_all=ds_dpos_all,
        r_init=r_init, obs_k0=obs_k0, obs_k1=obs_k1, obs_center=obs_center
    )

    # Calibrate exact 1D curvature at delta = 0.15 along each column of L_raw
    lp0 = fast_surrogate_logp(model, np.zeros(211), extra_bar, txe_bar, data)
    scales = np.ones(211)
    delta = 0.15
    for k in range(211):
        ek = np.zeros(211); ek[k] = delta
        lpp = fast_surrogate_logp(model, ek, extra_bar, txe_bar, data)
        lpm = fast_surrogate_logp(model, -ek, extra_bar, txe_bar, data)
        if np.isfinite(lpp) and np.isfinite(lpm):
            curv = -(lpp - 2 * lp0 + lpm) / (delta * delta)
            if curv > 0.05:
                scales[k] = np.clip(1.0 / np.sqrt(curv), 0.2, 5.0)
    L_cal = L_raw * scales[None, :]
    data["L_211"] = L_cal
    data["B_pts"] = np.ascontiguousarray(np.einsum("jac,ck->jak", A_full, L_cal[:203, :]).reshape(-1, 211))
    return data, scales, w


def fast_surrogate_logp(model, u, extra, txe, data):
    if extra <= 0 or txe <= 0:
        return -np.inf
    g = data["g_bar"] + data["L_211"] @ u
    x_cam = g[:203].reshape(model.nc, 7)
    cam = np.empty((model.nc, 7))
    cam[:, :6] = x_cam[:, :6]
    cam[:, 6] = np.exp(x_cam[:, 6])
    ant = g[203:206]
    tx = g[206:209]
    bias = g[209:211]

    t = model.terrain
    if not (t.e[0] + 2 < ant[0] < t.e[-1] - 2 and t.n[0] + 2 < ant[1] < t.n[-1] - 2):
        return -np.inf
    if not (t.e[0] + 2 < tx[0] < t.e[-1] - 2 and t.n[0] + 2 < tx[1] < t.n[-1] - 2):
        return -np.inf
    if not (float(fast_terrain_height(t, ant[0], ant[1])) < ant[2] < 4000.0):
        return -np.inf
    if not (float(fast_terrain_height(t, tx[0], tx[1])) - 5.0 < tx[2] < 4000.0):
        return -np.inf
    if np.any((cam[:, 3] <= 0) | (cam[:, 3] >= np.pi) | (np.abs(cam[:, 5]) >= np.pi) |
              (cam[:, 6] <= 100) | (cam[:, 6] >= 50000) |
              (cam[:, 0] <= t.e[0] + 2) | (cam[:, 0] >= t.e[-1] - 2) |
              (cam[:, 1] <= t.n[0] + 2) | (cam[:, 1] >= t.n[-1] - 2)):
        return -np.inf
    if np.any(cam[:, 2] <= fast_terrain_height(t, cam[:, 0], cam[:, 1]) + 0.1):
        return -np.inf

    pts = data["pts_bar"] + (data["B_pts"] @ u).reshape(-1, 3)
    if np.any((pts[:, 0] <= t.e[0] + 2) | (pts[:, 0] >= t.e[-1] - 2) |
              (pts[:, 1] <= t.n[0] + 2) | (pts[:, 1] >= t.n[-1] - 2)):
        return -np.inf

    # Vectorized rotation matrices for all 29 cameras
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

    # Vectorized tie-point projections across all 14,834 observations!
    c = model.config
    oc, op = model.oc, model.op
    v = np.einsum("ni,nij->nj", pts[op] - cam[oc, :3], R_all[oc])
    if np.any(v[:, 2] <= 0.1):
        return -np.inf
    q = -v[:, [1, 0]] / v[:, 2:3]
    r2_obs = np.sum(q * q, axis=1)
    dist_factor = (1.0 + data["obs_k0"] * r2_obs + data["obs_k1"] * r2_obs * r2_obs) * cam[oc, 6]
    pred = q * dist_factor[:, None] + data["obs_center"]
    tie = student_residual((pred - model.xy) / c.tie_sigma_px, c.student_df, 2)
    h_pts = fast_terrain_height(t, pts[:, 0], pts[:, 1])
    terr = student_residual((pts[:, 2] - h_pts) / c.terrain_sigma_m, c.student_df)
    total = -0.5 * float(np.sum(tie * tie) + np.sum(terr * terr))

    tx_part = model.tx_scatter_logp(cam, tx, txe)
    if not np.isfinite(tx_part):
        return -np.inf
    total += tx_part

    sigma = np.hypot(c.antenna_label_sigma_px, extra)

    # Vectorized horizon logp across all 29 cameras!
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
    sky_base = (1.0 - frac) * data["s0_mat_all"][rows, idx0] + frac * data["s0_mat_all"][rows, idx0 + 1]
    dpos = cam[h_idx, :3] - data["cam_bar"][h_idx, :3]
    sky = sky_base + np.sum(dpos * data["ds_dpos_all"], axis=1)
    r_h = student_residual((el - sky) * data["hor_focal_exif"] / c.horizon_sigma_angular_px, c.student_df)
    total -= 0.5 * float(np.sum(r_h * r_h))

    for i in range(model.nc):
        ci = cam[i]
        if model.has_ant_label[i]:
            ap, adepth = project(ci, model.shapes[i], ant, model.distortion[i])
            if adepth[0] <= 0.1:
                return -np.inf
            total -= 0.5 * float(np.sum(((ap[0] - model.axy[i]) / sigma) ** 2))
        if model.has_gps[i]:
            total -= 0.5 * float(np.sum(((ci[:2] + bias - model.gps[i]) / model.gps_sigma[i]) ** 2))
        if model.has_alt[i]:
            total -= 0.5 * float(((ci[2] - model.alt[i]) / c.altitude_sigma_m) ** 2)
        if model.has_heading[i]:
            delta = ci[4] - model.heading[i]
            if abs(delta) >= np.pi:
                return -np.inf
            total -= 0.5 * float((delta / c.heading_sigma_rad) ** 2)
        total -= 0.5 * float(((ci[3] - np.pi / 2) / c.elevation_sigma_rad) ** 2 +
                             (ci[5] / c.roll_sigma_rad) ** 2 +
                             ((x_cam[i, 6] - np.log(model.focal[i])) / model.focal_sigma[i]) ** 2)

    total -= model.n_ant_label * 2 * np.log(sigma / c.antenna_label_sigma_px)
    total -= 0.5 * (extra / c.antenna_extra_prior_px) ** 2
    total -= 0.5 * float(np.sum((bias / c.gps_common_sigma_m) ** 2))
    return total


def ess_block_step(model, u, extra, txe, surr_lp, cols, data, rng):
    """One reversible Elliptical Slice Sampling update on block `cols` of u."""
    u_sub = u[cols]
    nu = rng.standard_normal(len(cols))
    # Log-likelihood relative to N(0, I) prior on u_sub:
    log_y = (surr_lp + 0.5 * float(np.dot(u_sub, u_sub))) + np.log(rng.random())
    theta = rng.uniform(0.0, 2.0 * np.pi)
    theta_min = theta - 2.0 * np.pi
    theta_max = theta
    while True:
        u_try = u.copy()
        u_sub_try = u_sub * np.cos(theta) + nu * np.sin(theta)
        u_try[cols] = u_sub_try
        lp_try = fast_surrogate_logp(model, u_try, extra, txe, data)
        if np.isfinite(lp_try) and (lp_try + 0.5 * float(np.dot(u_sub_try, u_sub_try))) > log_y:
            return u_try, lp_try
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)


def run_ess_mh_chain(args):
    ch, u_start, ex_start, txe_start, data, tune, draws, inner_blocks_per_outer = args
    model = build_model()
    rng = np.random.default_rng(np.random.SeedSequence([20261017, ch]))

    u_cur = u_start.copy()
    ex_cur = float(ex_start)
    txe_cur = float(txe_start)

    blocks = np.array_split(np.arange(211), 7)  # 7 blocks of ~30 modes

    surr_cur = fast_surrogate_logp(model, u_cur, ex_cur, txe_cur, data)
    g_cur = data["g_bar"] + data["L_211"] @ u_cur
    state_cur = model.pack(*unpack_211(model, g_cur, ex_cur, txe_cur, data["g_bar"], data["pts_bar"], data["A_full"]))
    exact_cur = float(model.logp(state_cur))

    outer_acc = 0
    outer_n = 0
    records = []
    logp_trace = []

    for it in range(tune + draws):
        u_prop = u_cur.copy()
        ex_prop = ex_cur
        txe_prop = txe_cur
        surr_p = surr_cur

        # Random-scan Reversible Subspace Elliptical Slice Sampling + Nuisance Slice/MH
        for _ in range(inner_blocks_per_outer):
            b_i = int(rng.integers(0, len(blocks) + 2))
            if b_i < len(blocks):
                u_prop, surr_p = ess_block_step(model, u_prop, ex_prop, txe_prop, surr_p, blocks[b_i], data, rng)
            elif b_i == len(blocks):
                # Full 211-dimensional elliptical slice step!
                u_prop, surr_p = ess_block_step(model, u_prop, ex_prop, txe_prop, surr_p, np.arange(211), data, rng)
            else:
                # Reversible MH update on (extra, tx_extra)
                ex_cand = ex_prop + 0.35 * rng.standard_normal()
                if ex_cand > 0:
                    s_cand = fast_surrogate_logp(model, u_prop, ex_cand, txe_prop, data)
                    if np.isfinite(s_cand) and np.log(rng.random()) < s_cand - surr_p:
                        ex_prop, surr_p = ex_cand, s_cand
                txe_cand = txe_prop + 0.35 * rng.standard_normal()
                if txe_cand > 0:
                    s_cand = fast_surrogate_logp(model, u_prop, ex_prop, txe_cand, data)
                    if np.isfinite(s_cand) and np.log(rng.random()) < s_cand - surr_p:
                        txe_prop, surr_p = txe_cand, s_cand

        # Exact outer Metropolis-Hastings test on Posterior.logp!
        g_prop = data["g_bar"] + data["L_211"] @ u_prop
        state_prop = model.pack(*unpack_211(model, g_prop, ex_prop, txe_prop, data["g_bar"], data["pts_bar"], data["A_full"]))
        exact_prop = float(model.logp(state_prop))
        log_alpha = (exact_prop - surr_p) - (exact_cur - surr_cur)
        if it >= tune:
            outer_n += 1
        if np.isfinite(exact_prop) and np.log(rng.random()) < log_alpha:
            u_cur, ex_cur, txe_cur = u_prop, ex_prop, txe_prop
            surr_cur, exact_cur = surr_p, exact_prop
            state_cur = state_prop
            if it >= tune:
                outer_acc += 1

        if it >= tune:
            records.append(state_cur[:model.ng].copy())
            logp_trace.append((it - tune, exact_cur))

    return ch, np.array(records), np.array(logp_trace), outer_acc / max(outer_n, 1), state_cur


def main():
    t0 = time.time()
    model = build_model()
    z_star = np.load("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/state_v5_mode.npz")["state"]
    data, scales, w = prepare_calibrated_system(model, z_star)
    print(f"Prepared calibrated system in {time.time()-t0:.1f}s! w_min={w.min():.5f}, scales min={scales.min():.3f}, med={np.median(scales):.3f}, max={scales.max():.3f}", flush=True)

    t1 = time.time()
    for _ in range(100):
        lp0 = fast_surrogate_logp(model, np.zeros(211), data["extra_bar"], data["txe_bar"], data)
    print(f"Fully vectorized fast_surrogate_logp time: {(time.time()-t1)*10:.2f} ms/eval! lp0={lp0:.4f}", flush=True)

    rng = np.random.default_rng(42)
    ex0 = model.logp(z_star)
    for scale in [0.5, 1.0]:
        diffs = []
        quad_res = []
        for _ in range(20):
            u = scale * rng.standard_normal(211)
            lp_s = fast_surrogate_logp(model, u, data["extra_bar"], data["txe_bar"], data)
            g = data["g_bar"] + data["L_211"] @ u
            st = model.pack(*unpack_211(model, g, data["extra_bar"], data["txe_bar"], data["g_bar"], data["pts_bar"], data["A_full"]))
            lp_e = model.logp(st)
            if np.isfinite(lp_e) and np.isfinite(lp_s):
                diffs.append((lp_e - lp_s) - (ex0 - lp0))
                quad_res.append((lp_s - lp0) + 0.5 * np.sum(u * u))
        print(f"Scale {scale} ({len(diffs)}/20 finite): (exact - surr) SD = {np.std(diffs):.4f} (max {np.max(np.abs(diffs)):.4f}), "
              f"non-Gaussian residual (surr + 0.5|u|^2) SD = {np.std(quad_res):.2f}", flush=True)

    # Test 5 chains with 40 warmup + 300 retained outer sweeps, 10 inner ESS steps per outer sweep!
    chains = (0, 1, 5, 6, 7)
    starts = []
    for i, ch in enumerate(chains):
        if i == 0:
            starts.append((np.zeros(211), data["extra_bar"], data["txe_bar"]))
        else:
            r_c = np.random.default_rng(20261017 + ch)
            u_init = 0.5 * r_c.standard_normal(211)
            ex_init = max(0.2, data["extra_bar"] + 0.3 * r_c.standard_normal())
            txe_init = max(0.05, data["txe_bar"] + 0.1 * abs(r_c.standard_normal()))
            starts.append((u_init, ex_init, txe_init))

    tune, draws, inner = 40, 300, 10
    t_run = time.time()
    tasks = [(ch, starts[i][0], starts[i][1], starts[i][2], data, tune, draws, inner) for i, ch in enumerate(chains)]
    with ProcessPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(run_ess_mh_chain, tasks))
    print(f"5 ESS-MH chains ({tune}+{draws} outer x {inner} inner) finished in {time.time()-t_run:.1f}s!", flush=True)

    for ch, rec, lpt, acc, _ in results:
        print(f"  Chain {ch}: outer exact MH acceptance = {acc:.2%}, final logp = {lpt[-1, 1]:.2f}, max logp = {lpt[:, 1].max():.2f}")

    traces = np.stack([r[1] for r in results])
    coords = az.convert_to_dataset(traces)
    rhat = np.asarray(az.rhat(coords).to_array()).ravel()
    bulk = np.asarray(az.ess(coords, method="bulk").to_array()).ravel()
    tail = np.asarray(az.ess(coords, method="tail").to_array()).ravel()
    print(f"R-hat: max={np.nanmax(rhat):.4f}, median={np.nanmedian(rhat):.4f}, <=1.05: {np.sum(rhat <= 1.05)}/{len(rhat)}, <=1.02: {np.sum(rhat <= 1.02)}/{len(rhat)}, <=1.01: {np.sum(rhat <= 1.01)}/{len(rhat)}")
    print(f"ESS bulk: min={np.nanmin(bulk):.1f}, median={np.nanmedian(bulk):.1f}")
    print(f"ESS tail: min={np.nanmin(tail):.1f}, median={np.nanmedian(tail):.1f}")
    worst_idx = np.argsort(-rhat)[:10]
    for idx in worst_idx:
        if idx < 7 * model.nc:
            c_i, p_i = divmod(idx, 7)
            pname = f"cam_{model.keys[c_i]}_{['e','n','u','th','ph','ti','logf'][p_i]}"
        else:
            pname = ["ant_e","ant_n","ant_u","tx_e","tx_n","tx_u","bias_e","bias_n","extra","tx_extra"][idx - 7 * model.nc]
        print(f"  worst [{idx:3d}] {pname:16s}: rhat={rhat[idx]:.4f}, bulk={bulk[idx]:.1f}, tail={tail[idx]:.1f}")


if __name__ == "__main__":
    main()
