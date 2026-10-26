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
CAMPAIGN = os.path.join(ROOT, "marjum-2026-07")
V4 = os.path.join(CAMPAIGN, "derived/geometry_posterior/v0004")
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src"))
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src/eigsep_terrain/marjum_geometry"))
sys.path.insert(0, os.path.join(ROOT, "data-analysis/scripts/marjum-2026-07/geometry"))

import marjum_mcmc_b21 as b
from marjum_camera import project
from marjum_mcmc import student_residual
import make_product as mp
from test_schur_mcmc import build_model, compute_schur_couplings


def prepare_schur_system(model, z_init):
    cam_bar, ant_bar, tx_bar, bias_bar, extra_bar, txe_bar, pts_bar = model.unpack(z_init)
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

    r0_hor = []
    J_hor = []
    for i in range(nc):
        x0 = x_bar[i]
        r0_h = model.horizon_residuals(i, cam_bar[i])
        J_h = np.zeros((len(r0_h), 7))
        for k_ax in range(7):
            up, dn = x0.copy(), x0.copy()
            up[k_ax] += eps_c[k_ax]; dn[k_ax] -= eps_c[k_ax]
            J_h[:, k_ax] = (model.horizon_residuals(i, b.camera_pose(up)) - model.horizon_residuals(i, b.camera_pose(dn))) / (2 * eps_c[k_ax])
        r0_hor.append(r0_h)
        J_hor.append(J_h)
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

    scale_211 = np.r_[np.tile(b.CAMERA_SCALE, nc), np.ones(8)]
    S_scaled = 0.5 * ((S * scale_211[:, None]) * scale_211[None, :] + ((S * scale_211[:, None]) * scale_211[None, :]).T)
    w, v = np.linalg.eigh(S_scaled)
    w_reg = np.maximum(w, 0.5)
    L_211 = scale_211[:, None] * (v / np.sqrt(w_reg))

    # Sparse landmark sensitivity matrix A_full: (npt, 3, 203) = -H_ll_inv_H_lc!
    A_full = -H_ll_inv_H_lc  # (npt, 3, 203)
    g_bar = np.r_[x_bar.ravel(), ant_bar, tx_bar, bias_bar]

    return g_bar, extra_bar, txe_bar, pts_bar, A_full, L_211, r0_hor, J_hor


def unpack_211(model, g, extra, txe, g_bar, pts_bar, A_full):
    dg_cam = g[:203] - g_bar[:203]
    x_cam = g[:203].reshape(model.nc, 7)
    cam = np.array([b.camera_pose(x) for x in x_cam])
    ant = g[203:206]
    tx = g[206:209]
    bias = g[209:211]
    pts = pts_bar + np.einsum("jac,c->ja", A_full, dg_cam)
    return cam, ant, tx, bias, float(extra), float(txe), pts


def surrogate_logp(model, g, extra, txe, g_bar, pts_bar, A_full, r0_hor, J_hor):
    if extra <= 0 or txe <= 0:
        return -np.inf
    cam, ant, tx, bias, extra, txe, pts = unpack_211(model, g, extra, txe, g_bar, pts_bar, A_full)
    t = model.terrain
    if not (t.e[0] + 2 < ant[0] < t.e[-1] - 2 and t.n[0] + 2 < ant[1] < t.n[-1] - 2 and float(t.height(ant[0], ant[1])) < ant[2] < 4000.0):
        return -np.inf
    if not (t.e[0] + 2 < tx[0] < t.e[-1] - 2 and t.n[0] + 2 < tx[1] < t.n[-1] - 2 and float(t.height(tx[0], tx[1])) - 5.0 < tx[2] < 4000.0):
        return -np.inf
    pt_lp = model.point_logp(cam=cam, points=pts)
    if not np.all(np.isfinite(pt_lp)):
        return -np.inf
    total = float(np.sum(pt_lp))
    tx_part = model.tx_scatter_logp(cam, tx, txe)
    if not np.isfinite(tx_part):
        return -np.inf
    total += tx_part
    c = model.config
    sigma = np.hypot(c.antenna_label_sigma_px, extra)
    dx_cam = (g[:203] - g_bar[:203]).reshape(model.nc, 7)
    for i in range(model.nc):
        if not model._camera_support(cam[i]):
            return -np.inf
        if model.has_ant_label[i]:
            ap, depth = project(cam[i], model.shapes[i], ant, model.distortion[i])
            if depth[0] <= 0.1:
                return -np.inf
            total -= 0.5 * float(np.sum(((ap[0] - model.axy[i]) / sigma) ** 2))
        r_h = r0_hor[i] + J_hor[i] @ dx_cam[i]
        total -= 0.5 * float(np.dot(r_h, r_h))
        if model.has_gps[i]:
            total -= 0.5 * float(np.sum(((cam[i, :2] + bias - model.gps[i]) / model.gps_sigma[i]) ** 2))
        if model.has_alt[i]:
            total -= 0.5 * float(((cam[i, 2] - model.alt[i]) / c.altitude_sigma_m) ** 2)
        if model.has_heading[i]:
            delta = cam[i, 4] - model.heading[i]
            if abs(delta) >= np.pi:
                return -np.inf
            total -= 0.5 * float((delta / c.heading_sigma_rad) ** 2)
        total -= 0.5 * float(((cam[i, 3] - np.pi / 2) / c.elevation_sigma_rad) ** 2 +
                             (cam[i, 5] / c.roll_sigma_rad) ** 2 +
                             (np.log(cam[i, 6] / model.focal[i]) / model.focal_sigma[i]) ** 2)
    total -= model.n_ant_label * 2 * np.log(sigma / c.antenna_label_sigma_px)
    total -= 0.5 * (extra / c.antenna_extra_prior_px) ** 2
    total -= 0.5 * float(np.sum((bias / c.gps_common_sigma_m) ** 2))
    return total


def run_surrogate_chain(args):
    ch, g_start, ex_start, txe_start, g_bar, pts_bar, A_full, L_211, r0_hor, J_hor, tune, draws, inner_sweeps = args
    model = build_model()
    rng = np.random.default_rng(np.random.SeedSequence([20261016, ch]))

    g_cur = g_start.copy()
    ex_cur = float(ex_start)
    txe_cur = float(txe_start)

    # Partition 211 eigenmodes into 21 blocks of ~10 modes
    blocks = np.array_split(np.arange(211), 21)
    block_scales = np.array([2.38 / np.sqrt(len(b_idx)) for b_idx in blocks])
    ex_scale = 0.5
    txe_scale = 0.5

    surr_cur = surrogate_logp(model, g_cur, ex_cur, txe_cur, g_bar, pts_bar, A_full, r0_hor, J_hor)
    state_cur = model.pack(*unpack_211(model, g_cur, ex_cur, txe_cur, g_bar, pts_bar, A_full))
    exact_cur = float(model.logp(state_cur))

    outer_acc = 0
    outer_n = 0
    records = []
    logp_trace = []

    for it in range(tune + draws):
        g_prop = g_cur.copy()
        ex_prop = ex_cur
        txe_prop = txe_cur
        surr_p = surr_cur

        b_acc = np.zeros(len(blocks))
        b_tot = np.zeros(len(blocks))

        for _ in range(inner_sweeps):
            for b_i in rng.permutation(len(blocks)):
                cols = blocks[b_i]
                du = block_scales[b_i] * rng.standard_normal(len(cols))
                g_cand = g_prop + L_211[:, cols] @ du
                s_cand = surrogate_logp(model, g_cand, ex_prop, txe_prop, g_bar, pts_bar, A_full, r0_hor, J_hor)
                b_tot[b_i] += 1
                if np.isfinite(s_cand) and np.log(rng.random()) < s_cand - surr_p:
                    g_prop, surr_p = g_cand, s_cand
                    b_acc[b_i] += 1
            # Update extra & tx_extra on surrogate
            ex_cand = ex_prop + ex_scale * rng.standard_normal()
            if ex_cand > 0:
                s_cand = surrogate_logp(model, g_prop, ex_cand, txe_prop, g_bar, pts_bar, A_full, r0_hor, J_hor)
                if np.isfinite(s_cand) and np.log(rng.random()) < s_cand - surr_p:
                    ex_prop, surr_p = ex_cand, s_cand
            txe_cand = txe_prop + txe_scale * rng.standard_normal()
            if txe_cand > 0:
                s_cand = surrogate_logp(model, g_prop, ex_prop, txe_cand, g_bar, pts_bar, A_full, r0_hor, J_hor)
                if np.isfinite(s_cand) and np.log(rng.random()) < s_cand - surr_p:
                    txe_prop, surr_p = txe_cand, s_cand

        # Exact Metropolis-Hastings outer correction test!
        state_prop = model.pack(*unpack_211(model, g_prop, ex_prop, txe_prop, g_bar, pts_bar, A_full))
        exact_prop = float(model.logp(state_prop))
        log_alpha = (exact_prop - surr_p) - (exact_cur - surr_cur)
        if it >= tune:
            outer_n += 1
        if np.isfinite(exact_prop) and np.log(rng.random()) < log_alpha:
            g_cur, ex_cur, txe_cur = g_prop, ex_prop, txe_prop
            surr_cur, exact_cur = surr_p, exact_prop
            state_cur = state_prop
            if it >= tune:
                outer_acc += 1

        if it < tune:
            rate = 0.5 / np.sqrt((it + 1) / 5.0)
            block_scales *= np.exp(rate * (b_acc / np.maximum(b_tot, 1) - 0.30))
        else:
            records.append(state_cur[:model.ng].copy())
            logp_trace.append((it - tune, exact_cur))

    return ch, np.array(records), np.array(logp_trace), outer_acc / max(outer_n, 1), state_cur


def main():
    t0 = time.time()
    model = build_model()
    z_init = np.load("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/state_v5_clean.npz")["state"]
    g_bar, extra_bar, txe_bar, pts_bar, A_full, L_211, r0_hor, J_hor = prepare_schur_system(model, z_init)

    # Polish g_bar on surrogate_logp first so g_bar is at the exact mode with extra_bar ~ 8.2px!
    from scipy.optimize import minimize
    u_opt = np.zeros(211)
    for _ in range(2):
        for b_idx in np.array_split(np.arange(211), 7):
            def obj(u_sub):
                u_try = u_opt.copy()
                u_try[b_idx] = u_sub
                g_try = g_bar + L_211 @ u_try
                val = surrogate_logp(model, g_try, extra_bar, txe_bar, g_bar, pts_bar, A_full, r0_hor, J_hor)
                return -val if np.isfinite(val) else 1e12
            res = minimize(obj, u_opt[b_idx], method="Powell", options=dict(xtol=1e-2, ftol=1e-3, maxfev=800))
            u_opt[b_idx] = res.x
    g_polished = g_bar + L_211 @ u_opt
    z_polished = model.pack(*unpack_211(model, g_polished, extra_bar, txe_bar, g_bar, pts_bar, A_full))
    print(f"Polished mode logp: {model.logp(z_polished):.2f} (was {model.logp(z_init):.2f}), Re-centering Schur system...", flush=True)
    g_bar, extra_bar, txe_bar, pts_bar, A_full, L_211, r0_hor, J_hor = prepare_schur_system(model, z_polished)
    print(f"Schur system ready in {time.time()-t0:.1f}s! Exact logp at g_bar: {model.logp(z_polished):.2f}", flush=True)

    # Create 5 overdispersed starting states using L_211
    chains = (0, 1, 5, 6, 7)
    starts = []
    for idx_c, ch in enumerate(chains):
        if idx_c == 0:
            starts.append((g_bar.copy(), extra_bar, txe_bar))
            continue
        rng = np.random.default_rng(20261016 + ch)
        for attempt in range(20):
            u_pert = (0.8 ** attempt) * 0.4 * rng.standard_normal(211)
            g_p = g_bar + L_211 @ u_pert
            ex_p = max(0.2, extra_bar + 0.3 * rng.standard_normal())
            txe_p = max(0.05, txe_bar + 0.1 * abs(rng.standard_normal()))
            st_p = model.pack(*unpack_211(model, g_p, ex_p, txe_p, g_bar, pts_bar, A_full))
            lp_p = model.logp(st_p)
            if np.isfinite(lp_p):
                print(f"Chain {ch} start logp: {lp_p:.2f}")
                starts.append((g_p, ex_p, txe_p))
                break

    tune, draws, inner_sweeps = 60, 240, 8
    t_run = time.time()
    tasks = [
        (ch, starts[i][0], starts[i][1], starts[i][2], g_bar, pts_bar, A_full, L_211, r0_hor, J_hor, tune, draws, inner_sweeps)
        for i, ch in enumerate(chains)
    ]
    with ProcessPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(run_surrogate_chain, tasks))
    print(f"5 Surrogate-MH chains ({tune}+{draws} outer sweeps x {inner_sweeps} inner) finished in {time.time()-t_run:.1f}s!", flush=True)

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
