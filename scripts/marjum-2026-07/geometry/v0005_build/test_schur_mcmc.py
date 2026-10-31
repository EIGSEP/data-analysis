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

v5_state_path = "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/fit_transmitter_v5_clean.npz"
map_state_path = "/home/aparsons/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch/state_v5_clean.npz"


def build_model():
    _, model_v4 = mp.load_model()
    cfg = b.Config(
        skyline_samples=4096,
        antenna_extra_prior_px=1.5,
        log_f_sigma_by_camera=model_v4.config.log_f_sigma_by_camera,
    )
    return b.Posterior(
        state_file=v5_state_path,
        dem_file=str(mp.DEM),
        meta_file=mp.INPUT / "meta.json",
        exif_file=mp.HERE / "inputs/exif_utm.npz",
        feature_dir=mp.INPUT / "cv_features",
        config=cfg,
    )


def compute_schur_couplings(model, cam_bar, pts_bar, ant_bar, tx_bar, bias_bar, extra_bar, txe_bar):
    """Compute per-observation landmark response A_{j, c} = -H_{ll}^{-1} H_{lc} and Schur camera factors."""
    c = model.config
    nc, npt = model.nc, model.npoint
    n_obs = len(model.xy)

    # 1. Derivative of tie residual r_obs (n_obs, 2) w.r.t. landmark (n_obs, 2, 3) and camera (n_obs, 2, 7)
    J_pt = np.zeros((n_obs, 2, 3))
    J_cam = np.zeros((n_obs, 2, 7))
    eps_p = 0.02
    eps_c = b.CAMERA_STEP * 0.2

    pred0 = np.empty_like(model.xy)
    for i, idx in enumerate(model.by_camera):
        pred0[idx], _ = project(cam_bar[i], model.shapes[i], pts_bar[model.op[idx]], model.distortion[i])
    tie0 = student_residual((pred0 - model.xy) / c.tie_sigma_px, c.student_df, 2)

    for ax in range(3):
        p_up = pts_bar.copy(); p_up[:, ax] += eps_p
        p_dn = pts_bar.copy(); p_dn[:, ax] -= eps_p
        pr_up, pr_dn = np.empty_like(model.xy), np.empty_like(model.xy)
        for i, idx in enumerate(model.by_camera):
            pr_up[idx], _ = project(cam_bar[i], model.shapes[i], p_up[model.op[idx]], model.distortion[i])
            pr_dn[idx], _ = project(cam_bar[i], model.shapes[i], p_dn[model.op[idx]], model.distortion[i])
        t_up = student_residual((pr_up - model.xy) / c.tie_sigma_px, c.student_df, 2)
        t_dn = student_residual((pr_dn - model.xy) / c.tie_sigma_px, c.student_df, 2)
        J_pt[:, :, ax] = (t_up - t_dn) / (2 * eps_p)

    for i, idx in enumerate(model.by_camera):
        if len(idx) == 0:
            continue
        x0 = b.camera_coordinates(cam_bar[i])
        pts_i = pts_bar[model.op[idx]]
        xy_i = model.xy[idx]
        for k_ax in range(7):
            up, dn = x0.copy(), x0.copy()
            up[k_ax] += eps_c[k_ax]; dn[k_ax] -= eps_c[k_ax]
            pr_u, _ = project(b.camera_pose(up), model.shapes[i], pts_i, model.distortion[i])
            pr_d, _ = project(b.camera_pose(dn), model.shapes[i], pts_i, model.distortion[i])
            tu = student_residual((pr_u - xy_i) / c.tie_sigma_px, c.student_df, 2)
            td = student_residual((pr_d - xy_i) / c.tie_sigma_px, c.student_df, 2)
            J_cam[idx, :, k_ax] = (tu - td) / (2 * eps_c[k_ax])

    # Terrain residual derivative (npt, 3)
    J_terr = np.zeros((npt, 3))
    for ax in range(3):
        p_up = pts_bar.copy(); p_up[:, ax] += eps_p
        p_dn = pts_bar.copy(); p_dn[:, ax] -= eps_p
        tu = student_residual((p_up[:, 2] - model.terrain.height(p_up[:, 0], p_up[:, 1])) / c.terrain_sigma_m, c.student_df)
        td = student_residual((p_dn[:, 2] - model.terrain.height(p_dn[:, 0], p_dn[:, 1])) / c.terrain_sigma_m, c.student_df)
        J_terr[:, ax] = (tu - td) / (2 * eps_p)

    # Assemble 3x3 H_ll for each landmark j
    H_ll = np.einsum("na,nb->nab", J_terr, J_terr) + np.eye(3)[None] * 1e-4
    jtj_obs = np.einsum("nra,nrb->nab", J_pt, J_pt)
    for a in range(3):
        for b_ax in range(3):
            H_ll[:, a, b_ax] += np.bincount(model.op, weights=jtj_obs[:, a, b_ax], minlength=npt)
    H_ll_inv = np.linalg.inv(H_ll)

    # For each observation n (linking cam c_n = model.oc[n] and landmark j_n = model.op[n]):
    # H_{l_j, c_n} = J_pt[n].T @ J_cam[n] (shape 3 x 7)
    # So A_obs[n] = -H_ll_inv[j_n] @ (J_pt[n].T @ J_cam[n]) (shape 3 x 7)!
    H_lc_obs = np.einsum("nra,nrb->nab", J_pt, J_cam)  # (n_obs, 3, 7)
    A_obs = -np.einsum("nab,nbc->nac", H_ll_inv[model.op], H_lc_obs)  # (n_obs, 3, 7)

    # Now compute Schur-complement camera proposal factors!
    # When camera i moves by dx_i and landmarks j in N(i) move by A_obs[n] @ dx_i:
    factors = []
    x_bar = np.array([b.camera_coordinates(cam_bar[i]) for i in range(nc)])
    for i in range(nc):
        idx_i = model.by_camera[i]
        pts_idx = np.unique(model.op[idx_i])
        # All observations of any landmark seen by camera i
        obs_mask = np.isin(model.op, pts_idx)
        obs_sub = np.flatnonzero(obs_mask)
        # Map from i's observation to landmark displacement
        A_i = np.zeros((npt, 3, 7))
        A_i[model.op[idx_i]] = A_obs[idx_i]

        def schur_res(dx_i):
            pose_i = b.camera_pose(x_bar[i] + dx_i)
            pts_mod = pts_bar.copy()
            pts_mod[pts_idx] += np.einsum("jab,b->ja", A_i[pts_idx], dx_i)
            # Camera i's non-tie + horizon residuals
            r_cam = model.camera_residuals(i, pose_i, ant_bar, pts_mod, bias_bar, extra_bar, tx=tx_bar, tx_extra=txe_bar)
            if r_cam is None:
                return None
            r_hor = model.horizon_residuals(i, pose_i)
            # Also include tie + terrain residuals on all other cameras sharing pts_idx!
            other_obs = obs_sub[model.oc[obs_sub] != i]
            if len(other_obs):
                pr_o = np.empty((len(other_obs), 2))
                for c_o in np.unique(model.oc[other_obs]):
                    m_o = model.oc[other_obs] == c_o
                    pr_o[m_o], _ = project(cam_bar[c_o], model.shapes[c_o], pts_mod[model.op[other_obs[m_o]]], model.distortion[c_o])
                r_other = student_residual((pr_o - model.xy[other_obs]) / c.tie_sigma_px, c.student_df, 2).ravel()
            else:
                r_other = np.empty(0)
            h_t = model.terrain.height(pts_mod[pts_idx, 0], pts_mod[pts_idx, 1])
            r_terr = student_residual((pts_mod[pts_idx, 2] - h_t) / c.terrain_sigma_m, c.student_df)
            return np.concatenate([r_cam, r_hor, r_other, r_terr])

        r0 = schur_res(np.zeros(7))
        J_s = np.zeros((len(r0), 7))
        for k_ax in range(7):
            up, dn = np.zeros(7), np.zeros(7)
            up[k_ax] = eps_c[k_ax]; dn[k_ax] = -eps_c[k_ax]
            ru, rd = schur_res(up), schur_res(dn)
            if ru is not None and rd is not None:
                J_s[:, k_ax] = (ru - rd) / (2 * eps_c[k_ax])
            elif ru is not None:
                J_s[:, k_ax] = (ru - r0) / eps_c[k_ax]
            elif rd is not None:
                J_s[:, k_ax] = (r0 - rd) / eps_c[k_ax]
        fac, _ = b._scaled_camera_factor(J_s)
        factors.append(fac)

    return x_bar, pts_bar.copy(), A_obs, np.asarray(factors)


class SchurChain:
    """Metropolis-within-Gibbs on the 213 global parameters with Schur-conditioned landmarks."""

    def __init__(self, model, z_start, x_bar, pts_bar, A_obs, cam_factors, rng):
        self.model = model
        self.rng = rng
        self.x_bar = x_bar
        self.pts_bar = pts_bar
        self.A_obs = A_obs
        self.cam_factor = cam_factors.copy()

        cam, ant, tx, bias, extra, txe, _ = model.unpack(z_start)
        self.cam = cam.copy()
        self.x_cam = np.array([b.camera_coordinates(c) for c in self.cam])
        self.ant = ant.copy()
        self.tx = tx.copy()
        self.bias = bias.copy()
        self.extra = float(extra)
        self.tx_extra = float(txe)
        self.points = self.compute_points(self.x_cam)
        self.horizon = np.array([model.horizon_logp(i, self.cam[i]) for i in range(model.nc)])

        self.antenna_factor = b.antenna_geometry(model, self.cam, self.ant, self.extra)
        self.transmitter_factor = b.transmitter_geometry(model, self.cam, self.tx, self.tx_extra)

        self.camera_scale = np.full(model.nc, 2.38 / np.sqrt(7))
        self.antenna_scale = 2.38 / np.sqrt(3)
        self.transmitter_scale = 2.38 / np.sqrt(3)
        self.extra_scale = 0.5
        self.tx_extra_scale = 0.5
        self.shift_scale = 0.3

        # Precompute per-camera affected landmark and observation subsets for O(1) local logp diffs!
        self.cam_pts_idx = []
        self.cam_A_local = []
        self.cam_other_obs_by_cam = []
        for i in range(model.nc):
            idx_i = model.by_camera[i]
            p_idx = np.unique(model.op[idx_i])
            self.cam_pts_idx.append(p_idx)
            A_loc = np.zeros((len(p_idx), 3, 7))
            # Map model.op[idx_i] to local index in p_idx
            loc_map = np.searchsorted(p_idx, model.op[idx_i])
            np.add.at(A_loc, loc_map, A_obs[idx_i])
            self.cam_A_local.append(A_loc)
            # Other cameras observing p_idx
            obs_mask = np.isin(model.op, p_idx) & (model.oc != i)
            obs_other = np.flatnonzero(obs_mask)
            by_c = []
            for c_o in np.unique(model.oc[obs_other]):
                sub = obs_other[model.oc[obs_other] == c_o]
                by_c.append((int(c_o), sub, np.searchsorted(p_idx, model.op[sub])))
            self.cam_other_obs_by_cam.append(by_c)

        self.counts = dict(cam=np.zeros(model.nc), cam_n=np.zeros(model.nc),
                           ant=0, ant_n=0, tx=0, tx_n=0, extra=0, extra_n=0,
                           txe=0, txe_n=0, shift=0, shift_n=0)

    def compute_points(self, x_cam):
        dx = x_cam - self.x_bar  # (nc, 7)
        d_obs = np.einsum("nab,nb->na", self.A_obs, dx[self.model.oc])
        pts = self.pts_bar.copy()
        for ax in range(3):
            pts[:, ax] += np.bincount(self.model.op, weights=d_obs[:, ax], minlength=self.model.npoint)
        return pts

    def local_cam_logp(self, i, pose_i, pts_local, horizon_i):
        m = self.model
        c = m.config
        if not m._camera_support(pose_i):
            return -np.inf
        p_idx = self.cam_pts_idx[i]
        t = m.terrain
        if len(p_idx):
            if np.any((pts_local[:, 0] <= t.e[0] + 2) | (pts_local[:, 0] >= t.e[-1] - 2) |
                      (pts_local[:, 1] <= t.n[0] + 2) | (pts_local[:, 1] >= t.n[-1] - 2)):
                return -np.inf
        # 1. Camera i's own terms (using pts_local for its tie points)
        idx_i = m.by_camera[i]
        if len(idx_i):
            loc_i = np.searchsorted(p_idx, m.op[idx_i])
            pred, depth = project(pose_i, m.shapes[i], pts_local[loc_i], m.distortion[i])
            if np.any(depth <= 0.1):
                return -np.inf
            tie = student_residual((pred - m.xy[idx_i]) / c.tie_sigma_px, c.student_df, 2)
            total = -0.5 * float(np.sum(tie * tie))
        else:
            total = 0.0

        sigma = np.hypot(c.antenna_label_sigma_px, self.extra)
        if m.has_ant_label[i]:
            ap, adepth = project(pose_i, m.shapes[i], self.ant, m.distortion[i])
            if adepth[0] <= 0.1:
                return -np.inf
            total -= 0.5 * float(np.sum(((ap[0] - m.axy[i]) / sigma) ** 2))
        if m.has_tx_label[i]:
            tp, tdepth = project(pose_i, m.shapes[i], self.tx, m.distortion[i])
            if tdepth[0] <= 0.1:
                return -np.inf
            total -= 0.5 * float(np.sum(((tp[0] - m.txy[i]) / np.hypot(c.antenna_label_sigma_px, self.tx_extra)) ** 2))
        if m.has_gps[i]:
            total -= 0.5 * float(np.sum(((pose_i[:2] + self.bias - m.gps[i]) / m.gps_sigma[i]) ** 2))
        if m.has_alt[i]:
            total -= 0.5 * float(((pose_i[2] - m.alt[i]) / c.altitude_sigma_m) ** 2)
        if m.has_heading[i]:
            delta = pose_i[4] - m.heading[i]
            if abs(delta) >= np.pi:
                return -np.inf
            total -= 0.5 * float((delta / c.heading_sigma_rad) ** 2)
        total -= 0.5 * float(((pose_i[3] - np.pi / 2) / c.elevation_sigma_rad) ** 2 +
                             (pose_i[5] / c.roll_sigma_rad) ** 2 +
                             (np.log(pose_i[6] / m.focal[i]) / m.focal_sigma[i]) ** 2)
        total += horizon_i

        # 2. Other cameras' tie residuals on p_idx and terrain residuals on p_idx
        if len(p_idx):
            for c_o, sub_obs, loc_o in self.cam_other_obs_by_cam[i]:
                pr_o, dp_o = project(self.cam[c_o], m.shapes[c_o], pts_local[loc_o], m.distortion[c_o])
                if np.any(dp_o <= 0.1):
                    return -np.inf
                t_o = student_residual((pr_o - m.xy[sub_obs]) / c.tie_sigma_px, c.student_df, 2)
                total -= 0.5 * float(np.sum(t_o * t_o))
            h_terr = t.height(pts_local[:, 0], pts_local[:, 1])
            terr = student_residual((pts_local[:, 2] - h_terr) / c.terrain_sigma_m, c.student_df)
            total -= 0.5 * float(np.sum(terr * terr))
        return total

    def update_cameras(self):
        m, rng = self.model, self.rng
        for i in rng.permutation(m.nc):
            p_idx = self.cam_pts_idx[i]
            pts_cur_loc = self.points[p_idx]
            cur_lp = self.local_cam_logp(i, self.cam[i], pts_cur_loc, self.horizon[i])
            dx_i = self.camera_scale[i] * (self.cam_factor[i] @ rng.standard_normal(7))
            x_prop = self.x_cam[i] + dx_i
            pose_prop = b.camera_pose(x_prop)
            self.counts["cam_n"][i] += 1
            if not m._camera_support(pose_prop):
                continue
            pts_prop_loc = pts_cur_loc + np.einsum("jab,b->ja", self.cam_A_local[i], dx_i)
            hor_prop = m.horizon_logp(i, pose_prop)
            cand_lp = self.local_cam_logp(i, pose_prop, pts_prop_loc, hor_prop)
            if np.log(rng.random()) < cand_lp - cur_lp:
                self.x_cam[i] = x_prop
                self.cam[i] = pose_prop
                self.points[p_idx] = pts_prop_loc
                self.horizon[i] = hor_prop
                self.counts["cam"][i] += 1

    def update_targets_and_nuisances(self):
        m, rng = self.model, self.rng
        # Antenna
        cur = m.antenna_logp(self.cam, self.ant, self.extra)
        for _ in range(8):
            prop = self.ant + self.antenna_scale * (self.antenna_factor @ rng.standard_normal(3))
            cand = m.antenna_logp(self.cam, prop, self.extra)
            self.counts["ant_n"] += 1
            if np.log(rng.random()) < cand - cur:
                self.ant, cur = prop, cand
                self.counts["ant"] += 1
        # Transmitter
        cur = m.transmitter_logp(self.cam, self.tx, self.tx_extra)
        for _ in range(8):
            prop = self.tx + self.transmitter_scale * (self.transmitter_factor @ rng.standard_normal(3))
            cand = m.transmitter_logp(self.cam, prop, self.tx_extra)
            self.counts["tx_n"] += 1
            if np.log(rng.random()) < cand - cur:
                self.tx, cur = prop, cand
                self.counts["tx"] += 1
        # Bias (exact Gibbs)
        use = m.has_gps
        w = 1.0 / m.gps_sigma[use] ** 2
        prec = w.sum() + 1.0 / m.config.gps_common_sigma_m ** 2
        mean = ((m.gps[use] - self.cam[use, :2]) * w[:, None]).sum(axis=0) / prec
        self.bias = mean + rng.standard_normal(2) / np.sqrt(prec)
        # Extra & tx_extra
        cur = m.scatter_logp(self.cam, self.ant, self.extra)
        for _ in range(4):
            prop = self.extra + self.extra_scale * rng.standard_normal()
            cand = m.scatter_logp(self.cam, self.ant, prop)
            self.counts["extra_n"] += 1
            if np.log(rng.random()) < cand - cur:
                self.extra, cur = prop, cand
                self.counts["extra"] += 1
        cur = m.tx_scatter_logp(self.cam, self.tx, self.tx_extra)
        for _ in range(4):
            prop = self.tx_extra + self.tx_extra_scale * rng.standard_normal()
            cand = m.tx_scatter_logp(self.cam, self.tx, prop)
            self.counts["txe_n"] += 1
            if np.log(rng.random()) < cand - cur:
                self.tx_extra, cur = prop, cand
                self.counts["txe"] += 1

    def update_shift(self):
        m, rng = self.model, self.rng
        cur = m.logp(self.state())
        delta = self.shift_scale * rng.standard_normal(3)
        x_cam_p = self.x_cam.copy()
        x_cam_p[:, :3] += delta
        cam_p = self.cam.copy()
        cam_p[:, :3] += delta
        ant_p = self.ant + delta
        tx_p = self.tx + delta
        pts_p = self.compute_points(x_cam_p)
        cand = m.logp(m.pack(cam_p, ant_p, tx_p, self.bias, self.extra, self.tx_extra, pts_p))
        self.counts["shift_n"] += 1
        if np.log(rng.random()) < cand - cur:
            self.x_cam, self.cam, self.ant, self.tx, self.points = x_cam_p, cam_p, ant_p, tx_p, pts_p
            self.horizon = np.array([m.horizon_logp(i, self.cam[i]) for i in range(m.nc)])
            self.counts["shift"] += 1

    def adapt(self, rate):
        def tune(s, acc, tot, target):
            tot = np.maximum(tot, 1)
            return s * np.exp(rate * (acc / tot - target))
        self.camera_scale = tune(self.camera_scale, self.counts["cam"], self.counts["cam_n"], 0.25)
        self.antenna_scale = float(tune(self.antenna_scale, self.counts["ant"], self.counts["ant_n"], 0.30))
        self.transmitter_scale = float(tune(self.transmitter_scale, self.counts["tx"], self.counts["tx_n"], 0.30))
        self.extra_scale = float(tune(self.extra_scale, self.counts["extra"], self.counts["extra_n"], 0.44))
        self.tx_extra_scale = float(tune(self.tx_extra_scale, self.counts["txe"], self.counts["txe_n"], 0.44))
        self.shift_scale = float(tune(self.shift_scale, self.counts["shift"], self.counts["shift_n"], 0.30))
        for k in self.counts:
            self.counts[k] = np.zeros(self.model.nc) if k.startswith("cam") else 0

    def state(self):
        return self.model.pack(self.cam, self.ant, self.tx, self.bias, self.extra, self.tx_extra, self.points)


def run_worker(args):
    ch, z_start, x_bar, pts_bar, A_obs, cam_factors, tune, draws = args
    model = build_model()
    rng = np.random.default_rng(np.random.SeedSequence([20261016, ch]))
    chain = SchurChain(model, z_start, x_bar, pts_bar, A_obs, cam_factors, rng)
    records = []
    for it in range(tune + draws):
        chain.update_cameras()
        chain.update_targets_and_nuisances()
        if it % 5 == 0:
            chain.update_shift()
        if it < tune:
            if (it + 1) % 20 == 0:
                chain.adapt(min(1.5, 4.0 / np.sqrt((it + 1) / 20)))
        else:
            records.append(chain.state()[:model.ng].copy())
    return ch, np.array(records), chain.state(), float(model.logp(chain.state()))


def main():
    t0 = time.time()
    model = build_model()
    z_map = np.load(map_state_path)["state"]
    cam_bar, ant_bar, tx_bar, bias_bar, extra_bar, txe_bar, pts_bar = model.unpack(z_map)
    x_bar, pts_bar, A_obs, cam_factors = compute_schur_couplings(
        model, cam_bar, pts_bar, ant_bar, tx_bar, bias_bar, extra_bar, txe_bar
    )
    print(f"Computed Schur couplings in {time.time()-t0:.1f}s!", flush=True)
    for i in range(model.nc):
        sd = np.sqrt(np.sum(cam_factors[i] ** 2, axis=1)) * (2.38 / np.sqrt(7))
        if i in (0, 3, 14, 17, 21, 23, 25, 26):
            print(f"  {model.keys[i]} Schur prop SD: pos={np.round(sd[:3], 3)}m, ang={np.round(sd[3:6]*1000, 2)}mrad, logf={sd[6]:.4f}")

    # Create 5 overdispersed starts on the Schur manifold
    chains = (0, 1, 5, 6, 7)
    starts = []
    ant_fac = b.antenna_geometry(model, cam_bar, ant_bar, extra_bar)
    tx_fac = b.transmitter_geometry(model, cam_bar, tx_bar, txe_bar)
    for idx_c, ch in enumerate(chains):
        if idx_c == 0:
            starts.append(z_map.copy())
            continue
        rng = np.random.default_rng(20261016 + ch)
        x_c = x_bar.copy()
        for i in range(model.nc):
            x_c[i] += 0.5 * (cam_factors[i] @ rng.standard_normal(7))
        c_p = np.array([b.camera_pose(x) for x in x_c])
        # Compute Schur-coupled landmarks at x_c!
        dx = x_c - x_bar
        d_obs = np.einsum("nab,nb->na", A_obs, dx[model.oc])
        p_p = pts_bar.copy()
        for ax in range(3):
            p_p[:, ax] += np.bincount(model.op, weights=d_obs[:, ax], minlength=model.npoint)
        a_p = ant_bar + 0.8 * (ant_fac @ rng.standard_normal(3))
        t_p = tx_bar + 0.8 * (tx_fac @ rng.standard_normal(3))
        b_p = bias_bar + 0.3 * rng.standard_normal(2)
        ex_p = max(0.3, extra_bar + 0.3 * rng.standard_normal())
        txe_p = max(0.3, txe_bar + 0.3 * rng.standard_normal())
        st_p = model.pack(c_p, a_p, t_p, b_p, ex_p, txe_p, p_p)
        print(f"Chain {ch} start logp: {model.logp(st_p):.2f} (MAP: {model.logp(z_map):.2f})", flush=True)
        starts.append(st_p)

    tune, draws = 150, 400
    t_run = time.time()
    tasks = [(ch, starts[idx_c], x_bar, pts_bar, A_obs, cam_factors, tune, draws) for idx_c, ch in enumerate(chains)]
    with ProcessPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(run_worker, tasks))
    print(f"5 Schur chains ({tune}+{draws} sweeps) finished in {time.time()-t_run:.1f}s!", flush=True)

    traces = np.stack([r[1] for r in res])
    for r in res:
        print(f"  Chain {r[0]} final logp: {r[3]:.2f}")
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
