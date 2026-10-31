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
from fast_numba_kernels import eval_crest_skyline_numba, tie_and_terrain_numba


def eval_fast_surrogate(model, g_lin, pts, extra, txe, data, v_buf):
    if extra <= 0 or txe <= 0:
        return -np.inf, None
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
        return -np.inf, None
    if not (t.e[0] + 2 < tx[0] < t.e[-1] - 2 and t.n[0] + 2 < tx[1] < t.n[-1] - 2):
        return -np.inf, None
    if not (float(fast_terrain_height(t, ant[0], ant[1])) < ant[2] < 4000.0):
        return -np.inf, None
    if not (float(fast_terrain_height(t, tx[0], tx[1])) - 5.0 < tx[2] < 4000.0):
        return -np.inf, None
    if np.any((cam[:, 3] <= 0) | (cam[:, 3] >= np.pi) | (np.abs(cam[:, 5]) >= np.pi) |
              (cam[:, 6] <= 100) | (cam[:, 6] >= 50000) |
              (cam[:, 0] <= t.e[0] + 2) | (cam[:, 0] >= t.e[-1] - 2) |
              (cam[:, 1] <= t.n[0] + 2) | (cam[:, 1] >= t.n[-1] - 2)):
        return -np.inf, None
    if np.any(cam[:, 2] <= fast_terrain_height(t, cam[:, 0], cam[:, 1]) + 0.1):
        return -np.inf, None
    if np.any((pts[:, 0] <= t.e[0] + 2) | (pts[:, 0] >= t.e[-1] - 2) |
              (pts[:, 1] <= t.n[0] + 2) | (pts[:, 1] >= t.n[-1] - 2)):
        return -np.inf, None

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
    lp_tie_terr = tie_and_terrain_numba(
        cam, R_all, pts, data["op_s"], data["cam_ids"], data["cam_starts"], data["cam_ends"],
        data["k0_s"], data["k1_s"], data["ctr_s"], data["xy_s"],
        c.student_df * c.tie_sigma_px ** 2, -0.5 * (c.student_df + 2.0),
        t.data, float(t.e[0]), float(t.n[0]), 1.0 / float(t.res),
        c.student_df * c.terrain_sigma_m ** 2, -0.5 * (c.student_df + 1.0)
    )
    if not np.isfinite(lp_tie_terr):
        return -np.inf, None

    h_idx = data["hor_cam_idx"]
    d_vec = data["hor_d_base"] / cam[h_idx, 6:7]
    rd = np.linalg.norm(d_vec, axis=1)
    r = data["r_init"] * (data["cam_bar"][h_idx, 6] / cam[h_idx, 6])
    k0, k1 = data["hor_k0"], data["hor_k1"]
    for _ in range(2):
        r2 = r * r
        val = r * (1.0 + k0 * r2 + k1 * r2 * r2) - rd
        der = 1.0 + 3.0 * k0 * r2 + 5.0 * k1 * r2 * r2
        r = r - val / np.maximum(der, 0.05)
    q_h = d_vec * (r / np.maximum(rd, 1e-30))[:, None]
    body = np.c_[-q_h[:, 1], -q_h[:, 0], np.ones(len(q_h))]
    world = np.einsum("ni,nij->nj", body, R_all[h_idx].transpose(0, 2, 1))
    world /= np.linalg.norm(world, axis=1, keepdims=True)
    el = np.arctan2(world[:, 2], np.hypot(world[:, 0], world[:, 1]))
    az = np.arctan2(world[:, 1], world[:, 0])
    sky = eval_crest_skyline_numba(
        t.data, float(t.e[0]), float(t.e[-1]), float(t.n[0]), float(t.n[-1]), float(t.res),
        cam[h_idx, :3], az, data["crest_frac"], data["crest_log_end0"], data["crest_dist0"]
    )
    dh_ang = (el - sky) * data["hor_focal_exif"]
    lp_hor = -0.5 * (c.student_df + 1.0) * float(np.sum(np.log1p((dh_ang * dh_ang) / (c.student_df * c.horizon_sigma_angular_px ** 2))))

    ai = data["ant_idx"]
    va = np.einsum("ni,nij->nj", ant[None, :] - cam[ai, :3], R_all[ai])
    if np.any(va[:, 2] <= 0.1):
        return -np.inf, None
    qa = -va[:, [1, 0]] / va[:, 2:3]
    r2a = np.sum(qa * qa, axis=1)
    pred_a = qa * ((1.0 + data["cam_k0"][ai] * r2a + data["cam_k1"][ai] * r2a * r2a) * cam[ai, 6])[:, None] + data["cam_center_px"][ai]
    sigma_ant = np.hypot(c.antenna_label_sigma_px, extra)
    res_a = (pred_a - model.axy[ai]) / sigma_ant

    ti_idx = data["tx_idx"]
    vt = np.einsum("ni,nij->nj", tx[None, :] - cam[ti_idx, :3], R_all[ti_idx])
    if np.any(vt[:, 2] <= 0.1):
        return -np.inf, None
    qt = -vt[:, [1, 0]] / vt[:, 2:3]
    r2t = np.sum(qt * qt, axis=1)
    pred_t = qt * ((1.0 + data["cam_k0"][ti_idx] * r2t + data["cam_k1"][ti_idx] * r2t * r2t) * cam[ti_idx, 6])[:, None] + data["cam_center_px"][ti_idx]
    sigma_tx = np.hypot(c.antenna_label_sigma_px, txe)
    res_t = (pred_t - model.txy[ti_idx]) / sigma_tx

    gi = data["gps_idx"]
    res_gps = (cam[gi, :2] + bias[None, :] - model.gps[gi]) / model.gps_sigma[gi, None]
    ali = data["alt_idx"]
    res_alt = (cam[ali, 2] - model.alt[ali]) / c.altitude_sigma_m
    hi = data["hdg_idx"]
    delta_h = cam[hi, 4] - model.heading[hi]
    if np.any(np.abs(delta_h) >= np.pi):
        return -np.inf, None
    res_hdg = delta_h / c.heading_sigma_rad
    res_el = (cam[:, 3] - np.pi / 2) / c.elevation_sigma_rad
    res_ro = cam[:, 5] / c.roll_sigma_rad
    res_fo = (x_cam[:, 6] - data["log_focal0"]) / model.focal_sigma
    res_bi = bias / c.gps_common_sigma_m

    quad_terms = -0.5 * float(
        np.sum(res_a * res_a) + np.sum(res_t * res_t) + np.sum(res_gps * res_gps) +
        np.sum(res_alt * res_alt) + np.sum(res_hdg * res_hdg) +
        np.sum(res_el * res_el) + np.sum(res_ro * res_ro) + np.sum(res_fo * res_fo) +
        np.sum(res_bi * res_bi)
    )
    norm_terms = (
        -model.n_ant_label * 2 * np.log(sigma_ant / c.antenna_label_sigma_px)
        - 0.5 * (extra / c.antenna_extra_prior_px) ** 2
        - model.n_tx_label * 2 * np.log(sigma_tx / c.antenna_label_sigma_px)
        - 0.5 * (txe / c.antenna_extra_prior_px) ** 2
    )
    return lp_tie_terr + lp_hor + quad_terms + norm_terms, g_nl


def hit_and_run_slice_step(model, u, g_lin, pts, extra, txe, surr_lp, g_nl, cols, width, data, v_buf, rng):
    """Exact 1D Neal (2003) stepping-out + shrinkage slice sampler along a random unit direction in `cols`."""
    d_sub = rng.standard_normal(len(cols))
    d_sub /= np.linalg.norm(d_sub)
    dg_d = data["L_cal"][:, cols] @ d_sub
    dpts_d = (data["A_u"][:, cols] @ d_sub).reshape(-1, 3)

    log_y = surr_lp + np.log(rng.random())
    u_r = rng.random()
    L = -width * u_r
    R = L + width

    # Stepping out (capped at 6 expansions per side)
    for _ in range(6):
        lp_L, _ = eval_fast_surrogate(model, g_lin + L * dg_d, pts + L * dpts_d, extra, txe, data, v_buf)
        if lp_L <= log_y:
            break
        L -= width
    for _ in range(6):
        lp_R, _ = eval_fast_surrogate(model, g_lin + R * dg_d, pts + R * dpts_d, extra, txe, data, v_buf)
        if lp_R <= log_y:
            break
        R += width

    # Shrinkage
    while True:
        t = rng.uniform(L, R)
        g_lin_t = g_lin + t * dg_d
        pts_t = pts + t * dpts_d
        lp_t, g_nl_t = eval_fast_surrogate(model, g_lin_t, pts_t, extra, txe, data, v_buf)
        if lp_t > log_y:
            u_new = u.copy()
            u_new[cols] += t * d_sub
            return u_new, g_lin_t, pts_t, lp_t, g_nl_t
        if t < 0:
            L = t
        else:
            R = t


def block_mh_step(model, u, g_lin, pts, extra, txe, surr_lp, g_nl, cols, step_scale, data, v_buf, rng):
    """Reversible random-walk Metropolis-Hastings step on block `cols` of whitened eigenmodes."""
    du_sub = step_scale * rng.standard_normal(len(cols))
    g_lin_try = g_lin + data["L_cal"][:, cols] @ du_sub
    pts_try = pts + (data["A_u"][:, cols] @ du_sub).reshape(-1, 3)
    lp_try, g_nl_try = eval_fast_surrogate(model, g_lin_try, pts_try, extra, txe, data, v_buf)
    if g_nl_try is not None and np.log(rng.random()) < (lp_try - surr_lp):
        u_new = u.copy()
        u_new[cols] += du_sub
        return u_new, g_lin_try, pts_try, lp_try, g_nl_try, 1
    return u, g_lin, pts, surr_lp, g_nl, 0


def ess_step_calibrated(model, u, g_lin, pts, extra, txe, surr_lp, g_nl, cols, data, v_buf, rng):
    """Exact Elliptical Slice Sampling step on `cols` with tail-safe Gaussian reference N(0, diag(sig_ess[cols]^2))."""
    u_sub = u[cols]
    c_sub = data["c_ess"][cols]
    nu = data["sig_ess"][cols] * rng.standard_normal(len(cols))
    log_y = (surr_lp + float(np.sum(c_sub * u_sub * u_sub))) + np.log(rng.random())
    theta = rng.uniform(0.0, 2.0 * np.pi)
    theta_min = theta - 2.0 * np.pi
    theta_max = theta

    L_sub = data["L_cal"][:, cols]
    A_sub = data["A_u"][:, cols]
    g_base = g_lin - L_sub @ u_sub
    pts_base = pts - (A_sub @ u_sub).reshape(-1, 3)
    dg_u = L_sub @ u_sub
    dg_nu = L_sub @ nu
    dpts_u = (A_sub @ u_sub).reshape(-1, 3)
    dpts_nu = (A_sub @ nu).reshape(-1, 3)

    while True:
        ct, st = np.cos(theta), np.sin(theta)
        u_sub_try = ct * u_sub + st * nu
        g_lin_try = g_base + ct * dg_u + st * dg_nu
        pts_try = pts_base + ct * dpts_u + st * dpts_nu
        lp_try, g_nl_try = eval_fast_surrogate(model, g_lin_try, pts_try, extra, txe, data, v_buf)
        if g_nl_try is not None and (lp_try + float(np.sum(c_sub * u_sub_try * u_sub_try))) > log_y:
            u_new = u.copy()
            u_new[cols] = u_sub_try
            return u_new, g_lin_try, pts_try, lp_try, g_nl_try
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)


def update_extra_txe_exact(model, cam, ant, tx, ex_cur, txe_cur, surr_cur, ex_lp_cur, rng):
    s_ant_cur = model.scatter_logp(cam, ant, ex_cur)
    s_tx_cur = model.tx_scatter_logp(cam, tx, txe_cur)
    for _ in range(8):
        ex_cand = ex_cur + 0.45 * rng.standard_normal()
        if ex_cand > 0:
            s_ant_cand = model.scatter_logp(cam, ant, ex_cand)
            if np.isfinite(s_ant_cand) and np.log(rng.random()) < (s_ant_cand - s_ant_cur):
                diff = float(s_ant_cand - s_ant_cur)
                ex_cur, s_ant_cur = ex_cand, s_ant_cand
                surr_cur += diff
                ex_lp_cur += diff
        txe_cand = txe_cur + 0.35 * rng.standard_normal()
        if txe_cand > 0:
            s_tx_cand = model.tx_scatter_logp(cam, tx, txe_cand)
            if np.isfinite(s_tx_cand) and np.log(rng.random()) < (s_tx_cand - s_tx_cur):
                diff = float(s_tx_cand - s_tx_cur)
                txe_cur, s_tx_cur = txe_cand, s_tx_cand
                surr_cur += diff
                ex_lp_cur += diff
    return ex_cur, txe_cur, surr_cur, ex_lp_cur


def run_vol_chain(args):
    ch, u_start, ex_start, txe_start, data, tune, draws = args
    model = build_model()
    rng = np.random.default_rng(np.random.SeedSequence([20261017, ch]))
    v_buf = np.empty((len(model.oc), 3))

    u_cur = u_start.copy()
    ex_cur = float(ex_start)
    txe_cur = float(txe_start)
    g_lin_cur = data["g_center"] + data["L_cal"] @ u_cur
    pts_cur = data["pts_center"] + (data["A_u"] @ u_cur).reshape(-1, 3)
    surr_cur, g_nl_cur = eval_fast_surrogate(model, g_lin_cur, pts_cur, ex_cur, txe_cur, data, v_buf)
    while g_nl_cur is None:
        u_cur *= 0.8
        g_lin_cur = data["g_center"] + data["L_cal"] @ u_cur
        pts_cur = data["pts_center"] + (data["A_u"] @ u_cur).reshape(-1, 3)
        surr_cur, g_nl_cur = eval_fast_surrogate(model, g_lin_cur, pts_cur, ex_cur, txe_cur, data, v_buf)

    cam_cur, ant_cur, tx_cur, bias_cur, _, _, _ = unpack_211(
        model, g_nl_cur, ex_cur, txe_cur, data["g_bar"], data["pts_bar"], data["A_full"]
    )
    st_cur = model.pack(cam_cur, ant_cur, tx_cur, bias_cur, ex_cur, txe_cur, pts_cur)
    ex_lp_cur = float(model.logp(st_cur))

    # Build palindromic (forward-backward symmetric) schedule combining:
    # - Calibrated tail-safe Elliptical Slice Sampling (5 blocks covering 0..210)
    # - RW-MH mini-blocks on soft modes (0..35) and stiff modes (36..210)
    # - Hit-and-Run 1D Linear Slice steps on (0..18), (18..36), and (0..211)
    schedule = []
    for k in range(12):
        cols = np.arange(3 * k, 3 * k + 3)
        schedule.append(("rw", cols, 2.38 / np.sqrt(len(cols))))
    schedule.append(("slice", np.arange(0, 18), 2.5))
    schedule.append(("slice", np.arange(18, 36), 2.5))
    for cols in np.array_split(np.arange(0, 211), 5):
        schedule.append(("ess", cols, 0.0))
    for cols in np.array_split(np.arange(36, 211), 15):
        schedule.append(("rw", cols, 2.38 / np.sqrt(len(cols))))
    schedule.append(("slice", np.arange(0, 211), 2.0))

    palindromic_schedule = schedule + list(reversed(schedule))

    outer_acc = 0
    outer_n = 0
    records = []
    logp_trace = []

    for it in range(tune + draws):
        u_p = u_cur.copy()
        g_lin_p = g_lin_cur.copy()
        pts_p = pts_cur.copy()
        surr_p = surr_cur
        g_nl_p = g_nl_cur.copy()

        # Execute palindromic forward-backward sweep (strictly self-adjoint wrt surr_logp)
        for kind, cols, param in palindromic_schedule:
            if kind == "rw":
                u_p, g_lin_p, pts_p, surr_p, g_nl_p, _ = block_mh_step(
                    model, u_p, g_lin_p, pts_p, ex_cur, txe_cur, surr_p, g_nl_p, cols, param, data, v_buf, rng
                )
            elif kind == "ess":
                u_p, g_lin_p, pts_p, surr_p, g_nl_p = ess_step_calibrated(
                    model, u_p, g_lin_p, pts_p, ex_cur, txe_cur, surr_p, g_nl_p, cols, data, v_buf, rng
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

    return ch, np.array(records), np.array(logp_trace), outer_acc / max(outer_n, 1), st_cur


def main():
    t0 = time.time()
    model = build_model()
    with open("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/diffeo_vol_system.pkl", "rb") as f:
        data = pickle.load(f)
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

    tune, draws = 60, 300
    t_run = time.time()
    tasks = [(ch, starts[i][0], starts[i][1], starts[i][2], data, tune, draws) for i, ch in enumerate(chains)]
    with ProcessPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(run_vol_chain, tasks))
    print(f"5 Volume-Preserving Palindromic Chains ({tune}+{draws} outer) finished in {time.time()-t_run:.1f}s!", flush=True)

    for ch, rec, lpt, acc, _ in results:
        print(f"  Chain {ch}: outer exact MH acc = {acc:.2%}, final logp = {lpt[-1, 1]:.2f}, max logp = {lpt[:, 1].max():.2f}")

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

