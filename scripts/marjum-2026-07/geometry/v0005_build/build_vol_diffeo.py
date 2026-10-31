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
import numpy as np

ROOT = "/home/aparsons/projects/eigsep"
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src"))
sys.path.insert(0, os.path.join(ROOT, "eigsep_terrain/src/eigsep_terrain/marjum_geometry"))
sys.path.insert(0, os.path.join(ROOT, "data-analysis/scripts/marjum-2026-07/geometry"))
sys.path.insert(0, "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch")

from marjum_bundle import rotation
from marjum_camera import normalized
from marjum_mcmc import student_residual
from test_ess_mh_5chains import build_model, fast_terrain_height
from test_exact_gn_basis import surrogate_residuals
from test_surrogate_mh import unpack_211


def map_g_vol_preserving(g_lin, data):
    """Exact volume-preserving (det == 1, log_jac == 0) target-ray diffeomorphism for untied cameras."""
    g_bar = data["g_bar"]
    cam_bar = data["cam_bar"]
    ant = g_lin[203:206]
    tx = g_lin[206:209]
    ant_bar = g_bar[203:206]
    tx_bar = g_bar[206:209]
    g_out = g_lin.copy()
    for i in data["untied_idx"]:
        X = tx if data["is_tx_cam"][i] else ant
        X0 = tx_bar if data["is_tx_cam"][i] else ant_bar
        dX = X - X0
        dc_lin = g_lin[7*i:7*i+3] - g_bar[7*i:7*i+3]
        dang = g_lin[7*i+3:7*i+6] - g_bar[7*i+3:7*i+6]
        dlogf = g_lin[7*i+6] - g_bar[7*i+6]

        dv_lin = data["R_bar"][i].T @ (dX - dc_lin) + data["dv_dang"][i] @ dang
        dz = dv_lin[2]
        qb = data["q_bar"][i]
        zb = data["z_bar"][i]
        dq_df = data["dq_star_dlogf"][i]
        d2q_df2 = data["d2q_star_dlogf2"][i]
        dw0 = dv_lin[0] + qb[1] * dz + zb * dq_df[1] * dlogf
        dw1 = dv_lin[1] + qb[0] * dz + zb * dq_df[0] * dlogf

        z_new = zb + dz
        dq_star = dq_df * dlogf + 0.5 * d2q_df2 * (dlogf * dlogf)
        q_new = qb + dq_star
        v_new = np.array([-z_new * q_new[1] + dw0, -z_new * q_new[0] + dw1, z_new])
        th, ph, ti = g_lin[7*i+3], g_lin[7*i+4], g_lin[7*i+5]
        cth, sth = np.cos(th), np.sin(th)
        cph, sph = np.cos(ph), np.sin(ph)
        cti, sti = np.cos(ti), np.sin(ti)
        R_i = np.array([
            [cph * cth * cti - sph * sti, -cph * cth * sti - sph * cti, cph * sth],
            [sph * cth * cti + cph * sti, -sph * cth * sti + cph * cti, sph * sth],
            [-sth * cti, sth * sti, cth],
        ])
        g_out[7*i:7*i+3] = X - R_i @ v_new
    return g_out


def main():
    t0 = time.time()
    model = build_model()
    z_star = np.load("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/state_v5_joint_mode.npz")["state"]
    from test_ess_mh_5chains import prepare_calibrated_system
    from scipy.ndimage import gaussian_filter1d
    data, _, _ = prepare_calibrated_system(model, z_star)
    data["s0_smooth"] = gaussian_filter1d(data["s0_mat_all"], sigma=1.0, axis=1)
    data["A_flat"] = np.ascontiguousarray(data["A_full"].reshape(-1, 203))

    g_bar = data["g_bar"].copy()
    cam_bar = data["cam_bar"].copy()
    ant_bar = g_bar[203:206]
    tx_bar = g_bar[206:209]
    R_bar = np.array([rotation(cam_bar[i]) for i in range(model.nc)])
    v_bar = np.zeros((model.nc, 3))
    z_bar = np.zeros(model.nc)
    q_bar = np.zeros((model.nc, 2))
    q_star_bar = np.zeros((model.nc, 2))
    dq_star_dlogf = np.zeros((model.nc, 2))
    d2q_star_dlogf2 = np.zeros((model.nc, 2))
    dv_dang = np.zeros((model.nc, 3, 3))
    ray_xy = np.zeros((model.nc, 2))
    is_tx_cam = np.array([int(k) < 2209 for k in model.keys], dtype=bool)
    has_ties = np.array([len(idx) > 0 for idx in model.by_camera], dtype=bool)

    eps = 1e-4
    for i in range(model.nc):
        X0 = tx_bar if is_tx_cam[i] else ant_bar
        xy_t = model.txy[i] if is_tx_cam[i] else model.axy[i]
        ray_xy[i] = xy_t
        dX0 = X0 - cam_bar[i, :3]
        v_bar[i] = R_bar[i].T @ dX0
        z_bar[i] = v_bar[i, 2]
        q_bar[i] = -v_bar[i, [1, 0]] / z_bar[i]
        q0 = normalized(cam_bar[i], model.shapes[i], [xy_t], model.distortion[i])[0]
        q_star_bar[i] = q0

        cp, cm = cam_bar[i].copy(), cam_bar[i].copy()
        cp[6] *= np.exp(eps)
        cm[6] *= np.exp(-eps)
        qp = normalized(cp, model.shapes[i], [xy_t], model.distortion[i])[0]
        qm = normalized(cm, model.shapes[i], [xy_t], model.distortion[i])[0]
        dq_star_dlogf[i] = (qp - qm) / (2.0 * eps)
        d2q_star_dlogf2[i] = (qp - 2.0 * q0 + qm) / (eps * eps)

        for k_a in range(3):
            cp, cm = cam_bar[i].copy(), cam_bar[i].copy()
            cp[3 + k_a] += eps; cm[3 + k_a] -= eps
            dv_dang[i, :, k_a] = (rotation(cp).T @ dX0 - rotation(cm).T @ dX0) / (2.0 * eps)

    data.update(dict(
        R_bar=R_bar, v_bar=v_bar, z_bar=z_bar, q_bar=q_bar,
        q_star_bar=q_star_bar, dq_star_dlogf=dq_star_dlogf,
        d2q_star_dlogf2=d2q_star_dlogf2, dv_dang=dv_dang,
        ray_xy=ray_xy, is_tx_cam=is_tx_cam, has_ties=has_ties,
        untied_idx=np.flatnonzero(~has_ties),
        shapes=model.shapes, distortion=model.distortion
    ))

    # Compute exact Gauss-Newton Hessian H_gn under map_g_vol_preserving (where log_jac == 0!)
    ex, txe = data["extra_bar"], data["txe_bar"]
    scale_211 = np.r_[np.tile([1.0, 1.0, 1.0, 0.02, 0.02, 0.02, 0.02], model.nc), np.ones(8)]
    eps_fd = 1e-4 * scale_211
    r_star = surrogate_residuals(model, g_bar, ex, txe, data)
    J = np.empty((len(r_star), 211))
    for k in range(211):
        gp = g_bar.copy(); gp[k] += eps_fd[k]
        gm = g_bar.copy(); gm[k] -= eps_fd[k]
        J[:, k] = (surrogate_residuals(model, map_g_vol_preserving(gp, data), ex, txe, data) -
                   surrogate_residuals(model, map_g_vol_preserving(gm, data), ex, txe, data)) / (2 * eps_fd[k])
    H_gn = J.T @ J

    # Add cliff-normal precision for cameras with margin < 2.0m
    t = model.terrain
    for i in range(model.nc):
        e0, n0, z0 = cam_bar[i, :3]
        h0 = float(fast_terrain_height(t, e0, n0))
        margin = z0 - h0 - 0.1
        if margin < 2.0:
            dh_de = (float(fast_terrain_height(t, e0 - 0.5, n0)) - h0) / (-0.5)
            dh_dn = (float(fast_terrain_height(t, e0, n0 - 0.5)) - h0) / (-0.5)
            n_vec = np.array([-dh_de, -dh_dn, 1.0])
            H_gn[7*i:7*i+3, 7*i:7*i+3] += (4.0 / (margin ** 2)) * np.outer(n_vec, n_vec)

    H_scaled = 0.5 * ((H_gn * scale_211[:, None]) * scale_211[None, :] + ((H_gn * scale_211[:, None]) * scale_211[None, :]).T)
    w, v = np.linalg.eigh(H_scaled)
    print("Smallest 5 eigenvalues of H_scaled:", np.round(w[:5], 5))
    w_floored = np.maximum(w, 0.25)
    L_gn = scale_211[:, None] * (v / np.sqrt(w_floored))
    data["L_cal"] = L_gn.copy()
    data["g_center"] = g_bar.copy()

    # Only shrink modes where secant curvature at |u_k|=0.5 exceeds 1.05 (never shift g_center or inflate scales > 1.0!)
    c = model.config
    sigma_ant = np.hypot(c.antenna_label_sigma_px, ex)
    sigma_tx = np.hypot(c.antenna_label_sigma_px, txe)
    norm_terms = (
        -model.n_ant_label * 2 * np.log(sigma_ant / c.antenna_label_sigma_px)
        - 0.5 * (ex / c.antenna_extra_prior_px) ** 2
        - model.n_tx_label * 2 * np.log(sigma_tx / c.antenna_label_sigma_px)
        - 0.5 * (txe / c.antenna_extra_prior_px) ** 2
    )

    def eval_u_ref(u):
        g_nl = map_g_vol_preserving(data["g_center"] + data["L_cal"] @ u, data)
        r = surrogate_residuals(model, g_nl, ex, txe, data)
        return -0.5 * float(np.dot(r, r)) + norm_terms, g_nl

    lp0, g_c_nl = eval_u_ref(np.zeros(211))
    scales = np.ones(211)
    delta = 0.5
    for k in range(211):
        ek = np.zeros(211); ek[k] = delta
        lpp, _ = eval_u_ref(ek)
        lpm, _ = eval_u_ref(-ek)
        curv = -(lpp - 2.0 * lp0 + lpm) / (delta * delta)
        if curv > 1.05:
            scales[k] = np.clip(1.0 / np.sqrt(curv), 0.25, 1.0)
    data["L_cal"] = data["L_cal"] * scales[None, :]
    st_c = model.pack(*unpack_211(model, g_c_nl, ex, txe, data["g_bar"], data["pts_bar"], data["A_full"]))
    print(f"Calibrated at joint mode: surr lp0 = {lp0:.2f}, exact logp = {model.logp(st_c):.2f}, shrunk {np.sum(scales < 1.0)} modes")

    # Precompute fast contiguous BLAS arrays and Stage-1 Schur tables
    order = np.argsort(model.oc, kind="stable")
    oc_s = model.oc[order]
    op_s = model.op[order]
    xy_s = np.ascontiguousarray(model.xy[order])
    k0_s = np.ascontiguousarray(data["obs_k0"][order])
    k1_s = np.ascontiguousarray(data["obs_k1"][order])
    ctr_s = np.ascontiguousarray(data["obs_center"][order])
    counts = np.bincount(oc_s, minlength=model.nc)
    ends = np.cumsum(counts)
    starts = ends - counts
    cam_slices = [(i, slice(int(starts[i]), int(ends[i]))) for i in range(model.nc) if counts[i] > 0]

    data["oc_s"] = oc_s
    data["op_s"] = op_s
    data["xy_s"] = xy_s
    data["k0_s"] = k0_s
    data["k1_s"] = k1_s
    data["ctr_s"] = ctr_s
    data["cam_slices"] = cam_slices
    data["A_u"] = np.ascontiguousarray(data["A_flat"] @ data["L_cal"][:203, :])
    data["pts_center"] = data["pts_bar"].copy()
    data["ant_idx"] = np.flatnonzero(model.has_ant_label)
    data["tx_idx"] = np.flatnonzero(model.has_tx_label)
    data["gps_idx"] = np.flatnonzero(model.has_gps)
    data["alt_idx"] = np.flatnonzero(model.has_alt)
    data["hdg_idx"] = np.flatnonzero(model.has_heading)
    data["cam_center_px"] = np.array([[s[1] / 2.0, s[0] / 2.0] for s in model.shapes])
    data["cam_k0"] = np.array([float(d[0]) for d in model.distortion])
    data["cam_k1"] = np.array([float(d[1]) for d in model.distortion])
    data["log_focal0"] = np.log(model.focal)

    with open("/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/diffeo_vol_system.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Saved diffeo_vol_system.pkl in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
