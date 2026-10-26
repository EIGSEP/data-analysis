import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.environ["EIGSEP_CAMPAIGN_ROOT"] = "/home/aparsons/projects/eigsep/marjum-2026-07"

import sys
import time
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import arviz as az

ROOT = "/home/aparsons/projects/eigsep"
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src"))
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src/eigsep_terrain/marjum_geometry"))
sys.path.insert(0, os.path.join(ROOT, "data-analysis/scripts/marjum-2026-07/geometry"))
sys.path.insert(0, "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch")

from test_ess_mh_5chains import build_model, fast_terrain_height
from build_vol_diffeo import map_g_vol_preserving
from test_surrogate_mh import unpack_211
from test_fast_vol_mcmc import eval_fast_surrogate, hit_and_run_slice_step, block_mh_step, update_extra_txe_exact
from fast_numba_kernels import tie_and_terrain_and_grad_numba, eval_crest_skyline_and_grad_numba, pullback_vol_diffeo_grad


def fast_terrain_height_and_grad(terrain, e, n):
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
    h = (1.0 - dr) * ((1.0 - dc) * z00 + dc * z01) + dr * ((1.0 - dc) * z10 + dc * z11)
    dh_de = ((1.0 - dr) * (z01 - z00) + dr * (z11 - z10)) / terrain.res
    dh_dn = ((1.0 - dc) * (z10 - z00) + dc * (z11 - z01)) / terrain.res
    return h, dh_de, dh_dn


def eval_surr_and_grad(model, g_lin, pts, extra, txe, data, v_buf, gv_buf, gdpts_buf):
    if extra <= 0 or txe <= 0:
        return -np.inf, None, None
    g_nl = map_g_vol_preserving(g_lin, data)
    x_cam = g_nl[:203].reshape(model.nc, 7)
    cam = np.empty((model.nc, 7))
    cam[:, :6] = x_cam[:, :6]
    cam[:, 6] = np.exp(x_cam[:, 6])
    ant = g_nl[203:206]
    tx = g_nl[206:209]
    bias = g_nl[209:211]
    t = model.terrain
    if not (t.e[0] + 2 < ant[0] < t.e[-1] - 2 and t.n[0] + 2 < ant[1] < t.n[-1] - 2):
        return -np.inf, None, None
    if not (t.e[0] + 2 < tx[0] < t.e[-1] - 2 and t.n[0] + 2 < tx[1] < t.n[-1] - 2):
        return -np.inf, None, None
    if not (float(fast_terrain_height(t, ant[0], ant[1])) < ant[2] < 4000.0):
        return -np.inf, None, None
    if not (float(fast_terrain_height(t, tx[0], tx[1])) - 5.0 < tx[2] < 4000.0):
        return -np.inf, None, None
    if np.any((cam[:, 3] <= 0) | (cam[:, 3] >= np.pi) | (np.abs(cam[:, 5]) >= np.pi) |
              (cam[:, 6] <= 100) | (cam[:, 6] >= 50000) |
              (cam[:, 0] <= t.e[0] + 2) | (cam[:, 0] >= t.e[-1] - 2) |
              (cam[:, 1] <= t.n[0] + 2) | (cam[:, 1] >= t.n[-1] - 2)):
        return -np.inf, None, None
    if np.any(cam[:, 2] <= fast_terrain_height(t, cam[:, 0], cam[:, 1]) + 0.1):
        return -np.inf, None, None
    if np.any((pts[:, 0] <= t.e[0] + 2) | (pts[:, 0] >= t.e[-1] - 2) |
              (pts[:, 1] <= t.n[0] + 2) | (pts[:, 1] >= t.n[-1] - 2)):
        return -np.inf, None, None

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

    c = model.config
    grad_cam = np.zeros((model.nc, 7))
    grad_R = np.zeros((model.nc, 3, 3))
    grad_pts = np.zeros_like(pts)
    lp_tie_terr, ok_tt = tie_and_terrain_and_grad_numba(
        cam, R_all, pts, data["op_s"], data["cam_ids"], data["cam_starts"], data["cam_ends"],
        data["k0_s"], data["k1_s"], data["ctr_s"], data["xy_s"],
        c.student_df * c.tie_sigma_px ** 2, c.student_df,
        t.data, float(t.e[0]), float(t.n[0]), 1.0 / float(t.res),
        c.student_df * c.terrain_sigma_m ** 2, c.student_df,
        grad_cam, grad_R, grad_pts
    )
    if not ok_tt:
        return -np.inf, None, None

    h_idx = data["hor_cam_idx"]
    f_h = cam[h_idx, 6]
    d_vec = data["hor_d_base"] / f_h[:, None]
    rd = np.linalg.norm(d_vec, axis=1)
    r = data["r_init"] * (data["cam_bar"][h_idx, 6] / f_h)
    k0, k1 = data["hor_k0"], data["hor_k1"]
    for _ in range(2):
        r2 = r * r
        val = r * (1.0 + k0 * r2 + k1 * r2 * r2) - rd
        der = 1.0 + 3.0 * k0 * r2 + 5.0 * k1 * r2 * r2
        r = r - val / np.maximum(der, 0.05)
    r2 = r * r
    der_final = np.maximum(1.0 + 3.0 * k0 * r2 + 5.0 * k1 * r2 * r2, 0.05)
    q_h = d_vec * (r / np.maximum(rd, 1e-30))[:, None]
    body = np.c_[-q_h[:, 1], -q_h[:, 0], np.ones(len(q_h))]
    world_raw = np.einsum("ni,nij->nj", body, R_all[h_idx].transpose(0, 2, 1))
    wnorm = np.linalg.norm(world_raw, axis=1)
    world = world_raw / wnorm[:, None]
    xy_hypot = np.hypot(world[:, 0], world[:, 1])
    el = np.arctan2(world[:, 2], xy_hypot)
    az = np.arctan2(world[:, 1], world[:, 0])
    sky, ds_dpos_cr, dsky_ddaz_cr = eval_crest_skyline_and_grad_numba(
        t.data, float(t.e[0]), float(t.e[-1]), float(t.n[0]), float(t.n[-1]), float(t.res),
        cam[h_idx, :3], az, data["crest_frac"], data["crest_log_end0"], data["crest_dist0"]
    )
    dh_ang = (el - sky) * data["hor_focal_exif"]
    denom_hor = c.student_df * c.horizon_sigma_angular_px ** 2
    lp_hor = -0.5 * (c.student_df + 1.0) * float(np.sum(np.log1p((dh_ang * dh_ang) / denom_hor)))

    w_hor = -(c.student_df + 1.0) * dh_ang * data["hor_focal_exif"] / (denom_hor + dh_ang * dh_ang)
    gdpos = -w_hor[:, None] * ds_dpos_cr
    for ax in range(3):
        grad_cam[:, ax] += np.bincount(h_idx, weights=gdpos[:, ax], minlength=model.nc)
    g_daz = -w_hor * dsky_ddaz_cr
    g_el = w_hor
    inv_xy2 = 1.0 / (xy_hypot * xy_hypot)
    gw = np.empty_like(world)
    gw[:, 0] = -g_el * world[:, 2] * world[:, 0] / xy_hypot - g_daz * world[:, 1] * inv_xy2
    gw[:, 1] = -g_el * world[:, 2] * world[:, 1] / xy_hypot + g_daz * world[:, 0] * inv_xy2
    gw[:, 2] = g_el * xy_hypot
    gw_raw = (gw - world * np.sum(gw * world, axis=1, keepdims=True)) / wnorm[:, None]
    g_R_hor = np.einsum("nj,ni->nji", gw_raw, body)
    for a in range(3):
        for b_ax in range(3):
            grad_R[:, a, b_ax] += np.bincount(h_idx, weights=g_R_hor[:, a, b_ax], minlength=model.nc)
    g_body = np.einsum("nj,nji->ni", gw_raw, R_all[h_idx])
    gf_hor = -((-g_body[:, 1]) * q_h[:, 0] + (-g_body[:, 0]) * q_h[:, 1]) / (f_h * der_final)
    grad_cam[:, 6] += np.bincount(h_idx, weights=gf_hor, minlength=model.nc)

    grad_ant = np.zeros(3)
    grad_tx = np.zeros(3)
    sigma_ant = np.hypot(c.antenna_label_sigma_px, extra)
    sigma_tx = np.hypot(c.antenna_label_sigma_px, txe)
    quad_target_sum = 0.0
    for target, idx_arr, xy_arr, sig_val, g_targ in [
        (ant, data["ant_idx"], model.axy, sigma_ant, grad_ant),
        (tx, data["tx_idx"], model.txy, sigma_tx, grad_tx),
    ]:
        diff_t = target[None, :] - cam[idx_arr, :3]
        vt = np.einsum("ni,nij->nj", diff_t, R_all[idx_arr])
        if np.any(vt[:, 2] <= 0.1):
            return -np.inf, None, None
        inv_vtz = 1.0 / vt[:, 2]
        qt0 = -vt[:, 1] * inv_vtz
        qt1 = -vt[:, 0] * inv_vtz
        r2t = qt0 * qt0 + qt1 * qt1
        k0t, k1t = data["cam_k0"][idx_arr], data["cam_k1"][idx_arr]
        polyt = 1.0 + k0t * r2t + k1t * r2t * r2t
        ft = cam[idx_arr, 6]
        dft = polyt * ft
        pred_t = np.c_[qt0 * dft, qt1 * dft] + data["cam_center_px"][idx_arr]
        res_t = (pred_t - xy_arr[idx_arr]) / sig_val
        quad_target_sum += float(np.sum(res_t * res_t))
        g_pred = -res_t / sig_val
        gqt_dot = g_pred[:, 0] * qt0 + g_pred[:, 1] * qt1
        grad_cam[idx_arr, 6] += gqt_dot * polyt
        gr2t = gqt_dot * ft * (k0t + 2.0 * k1t * r2t)
        gqt0 = g_pred[:, 0] * dft + 2.0 * qt0 * gr2t
        gqt1 = g_pred[:, 1] * dft + 2.0 * qt1 * gr2t
        gvt = np.c_[-gqt1 * inv_vtz, -gqt0 * inv_vtz, -(gqt0 * qt0 + gqt1 * qt1) * inv_vtz]
        g_diff = np.einsum("nj,nij->ni", gvt, R_all[idx_arr])
        g_targ += np.sum(g_diff, axis=0)
        grad_cam[idx_arr, :3] -= g_diff
        grad_R[idx_arr] += np.einsum("ni,nj->nij", diff_t, gvt)

    gi = data["gps_idx"]
    res_gps = (cam[gi, :2] + bias[None, :] - model.gps[gi]) / model.gps_sigma[gi, None]
    ali = data["alt_idx"]
    res_alt = (cam[ali, 2] - model.alt[ali]) / c.altitude_sigma_m
    hi = data["hdg_idx"]
    delta_h = cam[hi, 4] - model.heading[hi]
    if np.any(np.abs(delta_h) >= np.pi):
        return -np.inf, None, None
    res_hdg = delta_h / c.heading_sigma_rad
    res_el = (cam[:, 3] - np.pi / 2) / c.elevation_sigma_rad
    res_ro = cam[:, 5] / c.roll_sigma_rad
    res_fo = (x_cam[:, 6] - data["log_focal0"]) / model.focal_sigma
    res_bi = bias / c.gps_common_sigma_m

    quad_terms = -0.5 * (
        quad_target_sum + float(
            np.sum(res_gps * res_gps) + np.sum(res_alt * res_alt) + np.sum(res_hdg * res_hdg) +
            np.sum(res_el * res_el) + np.sum(res_ro * res_ro) + np.sum(res_fo * res_fo) +
            np.sum(res_bi * res_bi)
        )
    )
    norm_terms = (
        -model.n_ant_label * 2 * np.log(sigma_ant / c.antenna_label_sigma_px)
        - 0.5 * (extra / c.antenna_extra_prior_px) ** 2
        - model.n_tx_label * 2 * np.log(sigma_tx / c.antenna_label_sigma_px)
        - 0.5 * (txe / c.antenna_extra_prior_px) ** 2
    )

    grad_cam[gi, :2] -= res_gps / model.gps_sigma[gi, None]
    grad_bias = -np.sum(res_gps / model.gps_sigma[gi, None], axis=0) - res_bi / c.gps_common_sigma_m
    grad_cam[ali, 2] -= res_alt / c.altitude_sigma_m
    grad_cam[hi, 4] -= res_hdg / c.heading_sigma_rad
    grad_cam[:, 3] -= res_el / c.elevation_sigma_rad
    grad_cam[:, 5] -= res_ro / c.roll_sigma_rad

    dR_dth = np.empty((model.nc, 3, 3))
    dR_dth[:, 0, 0] = -cph * sth * cti; dR_dth[:, 0, 1] = cph * sth * sti; dR_dth[:, 0, 2] = cph * cth
    dR_dth[:, 1, 0] = -sph * sth * cti; dR_dth[:, 1, 1] = sph * sth * sti; dR_dth[:, 1, 2] = sph * cth
    dR_dth[:, 2, 0] = -cth * cti;       dR_dth[:, 2, 1] = cth * sti;       dR_dth[:, 2, 2] = -sth

    dR_dph = np.empty((model.nc, 3, 3))
    dR_dph[:, 0, :] = -R_all[:, 1, :]
    dR_dph[:, 1, :] = R_all[:, 0, :]
    dR_dph[:, 2, :] = 0.0

    dR_dti = np.empty((model.nc, 3, 3))
    dR_dti[:, :, 0] = R_all[:, :, 1]
    dR_dti[:, :, 1] = -R_all[:, :, 0]
    dR_dti[:, :, 2] = 0.0

    grad_cam[:, 3] += np.einsum("nij,nij->n", grad_R, dR_dth)
    grad_cam[:, 4] += np.einsum("nij,nij->n", grad_R, dR_dph)
    grad_cam[:, 5] += np.einsum("nij,nij->n", grad_R, dR_dti)

    grad_x_cam = grad_cam.copy()
    grad_x_cam[:, 6] = grad_cam[:, 6] * cam[:, 6] - res_fo / model.focal_sigma
    grad_g_nl = np.r_[grad_x_cam.ravel(), grad_ant, grad_tx, grad_bias]
    grad_g_lin = pullback_vol_diffeo_grad(g_lin, grad_g_nl, data)

    grad_u = data["L_cal_T"] @ grad_g_lin + data["A_u_T"] @ grad_pts.ravel()
    return lp_tie_terr + lp_hor + quad_terms + norm_terms, g_nl, grad_u


def hmc_leapfrog_surrogate(model, u_cur, g_lin_cur, pts_cur, ex_cur, txe_cur, surr_cur, g_nl_cur, grad_cur, n_steps, eps, data, v_buf, gv_buf, gdpts_buf, rng):
    """Symplectic reversible leapfrog HMC transition with exact involutive specular reflection at boundaries."""
    p = rng.standard_normal(211)
    ke_init = 0.5 * float(np.dot(p, p))

    u = u_cur.copy()
    g_lin = g_lin_cur.copy()
    pts = pts_cur.copy()
    lp = surr_cur
    g_nl = g_nl_cur.copy()
    grad = grad_cur.copy()
    p_t = p.copy()

    for _ in range(n_steps):
        p_half = p_t + 0.5 * eps * grad
        u_try = u + eps * p_half
        gl_try = data["g_center"] + data["L_cal"] @ u_try
        pt_try = data["pts_center"] + (data["A_u"] @ u_try).reshape(-1, 3)
        lp_try, gnl_try, gr_try = eval_surr_and_grad(model, gl_try, pt_try, ex_cur, txe_cur, data, v_buf, gv_buf, gdpts_buf)
        if gnl_try is None:
            # Exact involutive specular bounce off the boundary!
            p_t = -p_t
        else:
            u, g_lin, pts, lp, g_nl, grad = u_try, gl_try, pt_try, lp_try, gnl_try, gr_try
            p_t = p_half + 0.5 * eps * grad

    ke_final = 0.5 * float(np.dot(p_t, p_t))
    log_alpha = (lp - ke_final) - (surr_cur - ke_init)
    if np.log(rng.random()) < log_alpha:
        return u, g_lin, pts, lp, g_nl, grad, 1
    return u_cur, g_lin_cur, pts_cur, surr_cur, g_nl_cur, grad_cur, 0


def run_hmc_chain(args):
    ch, u_start, ex_start, txe_start, data, tune, draws = args
    model = build_model()
    rng = np.random.default_rng(np.random.SeedSequence([20261017, ch]))
    v_buf = np.empty((len(model.oc), 3))
    gv_buf = np.empty((len(model.oc), 3))
    gdpts_buf = np.empty((len(model.oc), 3))

    u_cur = u_start.copy()
    ex_cur = float(ex_start)
    txe_cur = float(txe_start)
    g_lin_cur = data["g_center"] + data["L_cal"] @ u_cur
    pts_cur = data["pts_center"] + (data["A_u"] @ u_cur).reshape(-1, 3)
    surr_cur, g_nl_cur, grad_cur = eval_surr_and_grad(model, g_lin_cur, pts_cur, ex_cur, txe_cur, data, v_buf, gv_buf, gdpts_buf)
    while g_nl_cur is None:
        u_cur *= 0.8
        g_lin_cur = data["g_center"] + data["L_cal"] @ u_cur
        pts_cur = data["pts_center"] + (data["A_u"] @ u_cur).reshape(-1, 3)
        surr_cur, g_nl_cur, grad_cur = eval_surr_and_grad(model, g_lin_cur, pts_cur, ex_cur, txe_cur, data, v_buf, gv_buf, gdpts_buf)

    cam_cur, ant_cur, tx_cur, bias_cur, _, _, _ = unpack_211(
        model, g_nl_cur, ex_cur, txe_cur, data["g_bar"], data["pts_bar"], data["A_full"]
    )
    st_cur = model.pack(cam_cur, ant_cur, tx_cur, bias_cur, ex_cur, txe_cur, pts_cur)
    ex_lp_cur = float(model.logp(st_cur))

    soft_half = [
        ("rw", np.arange(0, 3), 2.38 / np.sqrt(3)),
        ("rw", np.arange(3, 6), 2.38 / np.sqrt(3)),
        ("rw", np.arange(6, 9), 2.38 / np.sqrt(3)),
        ("rw", np.arange(9, 12), 2.38 / np.sqrt(3)),
        ("rw", np.arange(12, 18), 2.38 / np.sqrt(6)),
        ("rw", np.arange(18, 26), 2.38 / np.sqrt(8)),
        ("rw", np.arange(26, 36), 2.38 / np.sqrt(10)),
        ("slice", np.arange(0, 20), 2.5),
    ]
    soft_rev = list(reversed(soft_half))

    outer_acc = 0
    outer_n = 0
    hmc_acc = 0
    hmc_n = 0
    records = []
    logp_trace = []

    for it in range(tune + draws):
        u_p = u_cur.copy()
        g_lin_p = g_lin_cur.copy()
        pts_p = pts_cur.copy()
        surr_p = surr_cur
        g_nl_p = g_nl_cur.copy()

        # 1. Forward soft-mode half-sweep
        for kind, cols, param in soft_half:
            if kind == "rw":
                u_p, g_lin_p, pts_p, surr_p, g_nl_p, _ = block_mh_step(
                    model, u_p, g_lin_p, pts_p, ex_cur, txe_cur, surr_p, g_nl_p, cols, param, data, v_buf, rng
                )
            else:
                u_p, g_lin_p, pts_p, surr_p, g_nl_p = hit_and_run_slice_step(
                    model, u_p, g_lin_p, pts_p, ex_cur, txe_cur, surr_p, g_nl_p, cols, param, data, v_buf, rng
                )

        # 2. Specular-bounce HMC trajectory across all 211 modes (calibrated eps: ~75% outer acc!)
        surr_p, g_nl_p, grad_p = eval_surr_and_grad(model, g_lin_p, pts_p, ex_cur, txe_cur, data, v_buf, gv_buf, gdpts_buf)
        n_steps = int(rng.integers(6, 9))
        eps = float(rng.uniform(0.048, 0.062))
        u_p, g_lin_p, pts_p, surr_p, g_nl_p, grad_p, acc_h = hmc_leapfrog_surrogate(
            model, u_p, g_lin_p, pts_p, ex_cur, txe_cur, surr_p, g_nl_p, grad_p,
            n_steps, eps, data, v_buf, gv_buf, gdpts_buf, rng
        )
        if it >= tune:
            hmc_n += 1
            hmc_acc += acc_h

        # 3. Reverse soft-mode half-sweep (strictly self-adjoint)
        for kind, cols, param in soft_rev:
            if kind == "rw":
                u_p, g_lin_p, pts_p, surr_p, g_nl_p, _ = block_mh_step(
                    model, u_p, g_lin_p, pts_p, ex_cur, txe_cur, surr_p, g_nl_p, cols, param, data, v_buf, rng
                )
            else:
                u_p, g_lin_p, pts_p, surr_p, g_nl_p = hit_and_run_slice_step(
                    model, u_p, g_lin_p, pts_p, ex_cur, txe_cur, surr_p, g_nl_p, cols, param, data, v_buf, rng
                )

        # Outer exact Posterior.logp Metropolis-Hastings check (log_jac == 0 identically!)
        x_cam_p = g_nl_p[:203].reshape(model.nc, 7)
        cam_p = np.empty((model.nc, 7))
        cam_p[:, :6] = x_cam_p[:, :6]
        cam_p[:, 6] = np.exp(x_cam_p[:, 6])
        ant_p = g_nl_p[203:206]
        tx_p = g_nl_p[206:209]
        bias_p = g_nl_p[209:211]
        st_p = model.pack(cam_p, ant_p, tx_p, bias_p, ex_cur, txe_cur, pts_p)
        ex_lp_p = float(model.logp(st_p))
        log_alpha = (ex_lp_p - surr_p) - (ex_lp_cur - surr_cur)
        if it >= tune:
            outer_n += 1
        if np.isfinite(ex_lp_p) and np.log(rng.random()) < log_alpha:
            u_cur, g_lin_cur, pts_cur, surr_cur, g_nl_cur = u_p, g_lin_p, pts_p, surr_p, g_nl_p
            cam_cur, ant_cur, tx_cur, bias_cur, st_cur, ex_lp_cur = cam_p, ant_p, tx_p, bias_p, st_p, ex_lp_p
            if it >= tune:
                outer_acc += 1

        # Exact conditional update on (extra, tx_extra)
        ex_cur, txe_cur, surr_cur, ex_lp_cur = update_extra_txe_exact(
            model, cam_cur, ant_cur, tx_cur, ex_cur, txe_cur, surr_cur, ex_lp_cur, rng
        )
        st_cur[211] = ex_cur
        st_cur[212] = txe_cur

        if it >= tune:
            records.append(st_cur[:model.ng].copy())
            logp_trace.append((it - tune, ex_lp_cur))

    return ch, np.array(records), np.array(logp_trace), outer_acc / max(outer_n, 1), hmc_acc / max(hmc_n, 1), st_cur


def main():
    t0 = time.time()
    model = build_model()
    with open("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/diffeo_vol_system.pkl", "rb") as f:
        data = pickle.load(f)
    data["L_cal_T"] = np.ascontiguousarray(data["L_cal"].T)
    data["A_u_T"] = np.ascontiguousarray(data["A_u"].T)
    ex, txe = data["extra_bar"], data["txe_bar"]

    chains = (0, 1, 5, 6, 7)
    starts = []
    for i, ch in enumerate(chains):
        if i == 0:
            starts.append((np.zeros(211), ex, txe))
        else:
            r_c = np.random.default_rng(20261017 + ch)
            u_init = 0.5 * r_c.standard_normal(211)
            ex_init = max(0.2, ex + 0.3 * r_c.standard_normal())
            txe_init = max(0.05, txe + 0.1 * abs(r_c.standard_normal()))
            starts.append((u_init, ex_init, txe_init))

    tune, draws = 200, 2000
    t_run = time.time()
    tasks = [(ch, starts[i][0], starts[i][1], starts[i][2], data, tune, draws) for i, ch in enumerate(chains)]
    with ProcessPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(run_hmc_chain, tasks))
    print(f"5 Volume-Preserving HMC Chains ({tune}+{draws} outer) finished in {time.time()-t_run:.1f}s!", flush=True)

    for ch, rec, lpt, o_acc, h_acc, _ in results:
        print(f"  Chain {ch}: inner HMC acc = {h_acc:.2%}, outer exact MH acc = {o_acc:.2%}, final logp = {lpt[-1, 1]:.2f}, max logp = {lpt[:, 1].max():.2f}")

    traces = np.stack([r[1] for r in results])
    np.savez_compressed(
        "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/posterior_v5_hmc_chains.npz",
        traces=traces,
        chains=np.array(chains),
        logp_traces=np.array([r[2] for r in results]),
        final_states=np.array([r[5] for r in results]),
    )
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
