import numpy as np
import numba


@numba.njit
def tie_and_terrain_numba(
    cam, R_all, pts, op_s, cam_ids, cam_starts, cam_ends, k0_s, k1_s, ctr_s, xy_s,
    denom_tie, factor_tie,
    dem_data, e0, n0, inv_res, denom_terr, factor_terr
):
    n_slices = len(cam_ids)
    sum_tie = 0.0
    for s_idx in range(n_slices):
        i = cam_ids[s_idx]
        st = cam_starts[s_idx]
        en = cam_ends[s_idx]
        ce = cam[i, 0]
        cn = cam[i, 1]
        cu = cam[i, 2]
        f_i = cam[i, 6]
        r00, r01, r02 = R_all[i, 0, 0], R_all[i, 0, 1], R_all[i, 0, 2]
        r10, r11, r12 = R_all[i, 1, 0], R_all[i, 1, 1], R_all[i, 1, 2]
        r20, r21, r22 = R_all[i, 2, 0], R_all[i, 2, 1], R_all[i, 2, 2]
        for idx in range(st, en):
            p_idx = op_s[idx]
            de = pts[p_idx, 0] - ce
            dn = pts[p_idx, 1] - cn
            du = pts[p_idx, 2] - cu
            vz = de * r02 + dn * r12 + du * r22
            if vz <= 0.1:
                return -np.inf
            vx = de * r00 + dn * r10 + du * r20
            vy = de * r01 + dn * r11 + du * r21
            inv_vz = 1.0 / vz
            q0 = -vy * inv_vz
            q1 = -vx * inv_vz
            r2 = q0 * q0 + q1 * q1
            df = (1.0 + k0_s[idx] * r2 + k1_s[idx] * r2 * r2) * f_i
            dx = q0 * df + ctr_s[idx, 0] - xy_s[idx, 0]
            dy = q1 * df + ctr_s[idx, 1] - xy_s[idx, 1]
            sum_tie += np.log1p((dx * dx + dy * dy) / denom_tie)

    H, W = dem_data.shape
    n_pts = pts.shape[0]
    sum_terr = 0.0
    for j in range(n_pts):
        r = (pts[j, 1] - n0) * inv_res
        c = (pts[j, 0] - e0) * inv_res
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
        dr = r - r0
        dc = c - c0
        if r0 < 0:
            r0 = 0
        elif r0 > H - 2:
            r0 = H - 2
        if c0 < 0:
            c0 = 0
        elif c0 > W - 2:
            c0 = W - 2
        z00 = float(dem_data[r0, c0])
        z01 = float(dem_data[r0, c0 + 1])
        z10 = float(dem_data[r0 + 1, c0])
        z11 = float(dem_data[r0 + 1, c0 + 1])
        h = (1.0 - dr) * ((1.0 - dc) * z00 + dc * z01) + dr * ((1.0 - dc) * z10 + dc * z11)
        dz = pts[j, 2] - h
        sum_terr += np.log1p((dz * dz) / denom_terr)

    return factor_tie * sum_tie + factor_terr * sum_terr


@numba.njit
def tie_and_terrain_and_grad_numba(
    cam, R_all, pts, op_s, cam_ids, cam_starts, cam_ends, k0_s, k1_s, ctr_s, xy_s,
    denom_tie, df_tie,
    dem_data, e0, n0, inv_res, denom_terr, df_terr,
    grad_cam, grad_R, grad_pts
):
    n_slices = len(cam_ids)
    sum_tie = 0.0
    c_tie = -(df_tie + 2.0) / denom_tie
    for s_idx in range(n_slices):
        i = cam_ids[s_idx]
        st = cam_starts[s_idx]
        en = cam_ends[s_idx]
        ce = cam[i, 0]
        cn = cam[i, 1]
        cu = cam[i, 2]
        f_i = cam[i, 6]
        r00, r01, r02 = R_all[i, 0, 0], R_all[i, 0, 1], R_all[i, 0, 2]
        r10, r11, r12 = R_all[i, 1, 0], R_all[i, 1, 1], R_all[i, 1, 2]
        r20, r21, r22 = R_all[i, 2, 0], R_all[i, 2, 1], R_all[i, 2, 2]
        gc0, gc1, gc2, gc6 = 0.0, 0.0, 0.0, 0.0
        gr00, gr01, gr02 = 0.0, 0.0, 0.0
        gr10, gr11, gr12 = 0.0, 0.0, 0.0
        gr20, gr21, gr22 = 0.0, 0.0, 0.0
        for idx in range(st, en):
            p_idx = op_s[idx]
            de = pts[p_idx, 0] - ce
            dn = pts[p_idx, 1] - cn
            du = pts[p_idx, 2] - cu
            vz = de * r02 + dn * r12 + du * r22
            if vz <= 0.1:
                return -np.inf, False
            vx = de * r00 + dn * r10 + du * r20
            vy = de * r01 + dn * r11 + du * r21
            inv_vz = 1.0 / vz
            q0 = -vy * inv_vz
            q1 = -vx * inv_vz
            r2 = q0 * q0 + q1 * q1
            k0 = k0_s[idx]
            k1 = k1_s[idx]
            poly = 1.0 + k0 * r2 + k1 * r2 * r2
            df = poly * f_i
            dx = q0 * df + ctr_s[idx, 0] - xy_s[idx, 0]
            dy = q1 * df + ctr_s[idx, 1] - xy_s[idx, 1]
            u_t = (dx * dx + dy * dy) / denom_tie
            sum_tie += np.log1p(u_t)

            w_t = c_tie / (1.0 + u_t)
            gdx = w_t * dx
            gdy = w_t * dy
            gq_dot = gdx * q0 + gdy * q1
            gc6 += gq_dot * poly
            g_r2 = gq_dot * f_i * (k0 + 2.0 * k1 * r2)
            gq0 = gdx * df + 2.0 * q0 * g_r2
            gq1 = gdy * df + 2.0 * q1 * g_r2
            gv0 = -gq1 * inv_vz
            gv1 = -gq0 * inv_vz
            gv2 = -(gq0 * q0 + gq1 * q1) * inv_vz

            gp0 = gv0 * r00 + gv1 * r01 + gv2 * r02
            gp1 = gv0 * r10 + gv1 * r11 + gv2 * r12
            gp2 = gv0 * r20 + gv1 * r21 + gv2 * r22
            grad_pts[p_idx, 0] += gp0
            grad_pts[p_idx, 1] += gp1
            grad_pts[p_idx, 2] += gp2
            gc0 -= gp0
            gc1 -= gp1
            gc2 -= gp2
            gr00 += de * gv0; gr01 += de * gv1; gr02 += de * gv2
            gr10 += dn * gv0; gr11 += dn * gv1; gr12 += dn * gv2
            gr20 += du * gv0; gr21 += du * gv1; gr22 += du * gv2

        grad_cam[i, 0] += gc0
        grad_cam[i, 1] += gc1
        grad_cam[i, 2] += gc2
        grad_cam[i, 6] += gc6
        grad_R[i, 0, 0] += gr00; grad_R[i, 0, 1] += gr01; grad_R[i, 0, 2] += gr02
        grad_R[i, 1, 0] += gr10; grad_R[i, 1, 1] += gr11; grad_R[i, 1, 2] += gr12
        grad_R[i, 2, 0] += gr20; grad_R[i, 2, 1] += gr21; grad_R[i, 2, 2] += gr22

    H, W = dem_data.shape
    n_pts = pts.shape[0]
    sum_terr = 0.0
    c_terr = -(df_terr + 1.0)
    for j in range(n_pts):
        r = (pts[j, 1] - n0) * inv_res
        c = (pts[j, 0] - e0) * inv_res
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
        dr = r - r0
        dc = c - c0
        if r0 < 0:
            r0 = 0
        elif r0 > H - 2:
            r0 = H - 2
        if c0 < 0:
            c0 = 0
        elif c0 > W - 2:
            c0 = W - 2
        z00 = float(dem_data[r0, c0])
        z01 = float(dem_data[r0, c0 + 1])
        z10 = float(dem_data[r0 + 1, c0])
        z11 = float(dem_data[r0 + 1, c0 + 1])
        h = (1.0 - dr) * ((1.0 - dc) * z00 + dc * z01) + dr * ((1.0 - dc) * z10 + dc * z11)
        dh_de = ((1.0 - dr) * (z01 - z00) + dr * (z11 - z10)) * inv_res
        dh_dn = ((1.0 - dc) * (z10 - z00) + dc * (z11 - z01)) * inv_res
        dz = pts[j, 2] - h
        dz2 = dz * dz
        sum_terr += np.log1p(dz2 / denom_terr)
        w_tr = c_terr * dz / (denom_terr + dz2)
        grad_pts[j, 0] -= w_tr * dh_de
        grad_pts[j, 1] -= w_tr * dh_dn
        grad_pts[j, 2] += w_tr

    lp = -0.5 * (df_tie + 2.0) * sum_tie - 0.5 * (df_terr + 1.0) * sum_terr
    return lp, True


@numba.njit
def eval_crest_skyline_numba(dem_data, e0, e_last, n0, n_last, res, cam_pos, az, frac_sub, log_end0, dist0):
    M, K = frac_sub.shape
    H, W = dem_data.shape
    out = np.empty(M, dtype=np.float64)
    inv_res = 1.0 / res
    slopes = np.empty(K, dtype=np.float64)
    inv_tau = 2000.0
    for m in range(M):
        ce = np.cos(az[m])
        sn = np.sin(az[m])
        c_e = cam_pos[m, 0]
        c_n = cam_pos[m, 1]
        c_u = cam_pos[m, 2]
        safe_ce = ce if abs(ce) > 1e-12 else (1e-12 if ce >= 0 else -1e-12)
        safe_sn = sn if abs(sn) > 1e-12 else (1e-12 if sn >= 0 else -1e-12)
        de = (e_last - c_e) / safe_ce if ce >= 0 else (e0 - c_e) / safe_ce
        dn = (n_last - c_n) / safe_sn if sn >= 0 else (n0 - c_n) / safe_sn
        end = de if de < dn else dn
        if end - res > 2.0:
            end = end - res
        else:
            end = 2.0
        dlog = np.log(end) - log_end0[m]
        max_slope = -1e30
        for k in range(K):
            df = dlog * frac_sub[m, k]
            scale = 1.0 + df * (1.0 + 0.5 * df * (1.0 + 0.3333333333333333 * df))
            dist = dist0[m, k] * scale
            e_pt = c_e + ce * dist
            n_pt = c_n + sn * dist
            r = (n_pt - n0) * inv_res
            c = (e_pt - e0) * inv_res
            r0 = int(np.floor(r))
            c0 = int(np.floor(c))
            dr = r - r0
            dc = c - c0
            if r0 < 0:
                r0 = 0
            elif r0 > H - 2:
                r0 = H - 2
            if c0 < 0:
                c0 = 0
            elif c0 > W - 2:
                c0 = W - 2
            z00 = float(dem_data[r0, c0])
            z01 = float(dem_data[r0, c0 + 1])
            z10 = float(dem_data[r0 + 1, c0])
            z11 = float(dem_data[r0 + 1, c0 + 1])
            z = (1.0 - dr) * ((1.0 - dc) * z00 + dc * z01) + dr * ((1.0 - dc) * z10 + dc * z11)
            slope = (z - c_u) / dist
            slopes[k] = slope
            if slope > max_slope:
                max_slope = slope
        w_sum = 0.0
        for k in range(K):
            diff_s = (slopes[k] - max_slope) * inv_tau
            if diff_s > -15.0:
                w_sum += np.exp(diff_s)
        s_smooth = max_slope + np.log(w_sum) / inv_tau
        out[m] = np.arctan(s_smooth)
    return out


@numba.njit
def eval_crest_skyline_and_grad_numba(dem_data, e0, e_last, n0, n_last, res, cam_pos, az, frac_sub, log_end0, dist0):
    M, K = frac_sub.shape
    H, W = dem_data.shape
    out = np.empty(M, dtype=np.float64)
    ds_dpos = np.empty((M, 3), dtype=np.float64)
    ds_daz = np.empty(M, dtype=np.float64)
    inv_res = 1.0 / res
    slopes = np.empty(K, dtype=np.float64)
    inv_dists = np.empty(K, dtype=np.float64)
    dz_des = np.empty(K, dtype=np.float64)
    dz_dns = np.empty(K, dtype=np.float64)
    dscales = np.empty(K, dtype=np.float64)
    inv_tau = 2000.0
    for m in range(M):
        ce = np.cos(az[m])
        sn = np.sin(az[m])
        c_e = cam_pos[m, 0]
        c_n = cam_pos[m, 1]
        c_u = cam_pos[m, 2]
        safe_ce = ce if abs(ce) > 1e-12 else (1e-12 if ce >= 0 else -1e-12)
        safe_sn = sn if abs(sn) > 1e-12 else (1e-12 if sn >= 0 else -1e-12)
        de = (e_last - c_e) / safe_ce if ce >= 0 else (e0 - c_e) / safe_ce
        dn = (n_last - c_n) / safe_sn if sn >= 0 else (n0 - c_n) / safe_sn
        e_active = de < dn
        raw_end = de if e_active else dn
        if raw_end - res > 2.0:
            end = raw_end - res
            inv_end = 1.0 / end
            if e_active:
                dlog_dce = (-1.0 / safe_ce) * inv_end
                dlog_dcn = 0.0
                dlog_daz = (de * sn / safe_ce) * inv_end
            else:
                dlog_dce = 0.0
                dlog_dcn = (-1.0 / safe_sn) * inv_end
                dlog_daz = (-dn * ce / safe_sn) * inv_end
        else:
            end = 2.0
            dlog_dce = 0.0
            dlog_dcn = 0.0
            dlog_daz = 0.0
        dlog = np.log(end) - log_end0[m]
        max_slope = -1e30
        for k in range(K):
            f_k = frac_sub[m, k]
            df = dlog * f_k
            scale = 1.0 + df * (1.0 + 0.5 * df * (1.0 + 0.3333333333333333 * df))
            dscale_ddf = 1.0 + df * (1.0 + 0.5 * df)
            dist = dist0[m, k] * scale
            e_pt = c_e + ce * dist
            n_pt = c_n + sn * dist
            r = (n_pt - n0) * inv_res
            c = (e_pt - e0) * inv_res
            r0 = int(np.floor(r))
            c0 = int(np.floor(c))
            dr = r - r0
            dc = c - c0
            if r0 < 0:
                r0 = 0
            elif r0 > H - 2:
                r0 = H - 2
            if c0 < 0:
                c0 = 0
            elif c0 > W - 2:
                c0 = W - 2
            z00 = float(dem_data[r0, c0])
            z01 = float(dem_data[r0, c0 + 1])
            z10 = float(dem_data[r0 + 1, c0])
            z11 = float(dem_data[r0 + 1, c0 + 1])
            z = (1.0 - dr) * ((1.0 - dc) * z00 + dc * z01) + dr * ((1.0 - dc) * z10 + dc * z11)
            inv_d = 1.0 / dist
            slope = (z - c_u) * inv_d
            slopes[k] = slope
            inv_dists[k] = inv_d
            dz_des[k] = ((1.0 - dr) * (z01 - z00) + dr * (z11 - z10)) * inv_res
            dz_dns[k] = ((1.0 - dc) * (z10 - z00) + dc * (z11 - z01)) * inv_res
            dscales[k] = f_k * (dscale_ddf / scale)
            if slope > max_slope:
                max_slope = slope
        w_sum = 0.0
        g_e, g_n, g_u, g_az = 0.0, 0.0, 0.0, 0.0
        for k in range(K):
            diff_s = (slopes[k] - max_slope) * inv_tau
            if diff_s > -15.0:
                w = np.exp(diff_s)
                w_sum += w
                id_k = inv_dists[k]
                dze = dz_des[k]
                dzn = dz_dns[k]
                ds_ddlog = dscales[k] * (ce * dze + sn * dzn - slopes[k])
                g_e += w * (dze * id_k + ds_ddlog * dlog_dce)
                g_n += w * (dzn * id_k + ds_ddlog * dlog_dcn)
                g_u -= w * id_k
                g_az += w * ((-sn * dze + ce * dzn) + ds_ddlog * dlog_daz)
        s_smooth = max_slope + np.log(w_sum) / inv_tau
        out[m] = np.arctan(s_smooth)
        inv_w = 1.0 / (w_sum * (1.0 + s_smooth * s_smooth))
        ds_dpos[m, 0] = g_e * inv_w
        ds_dpos[m, 1] = g_n * inv_w
        ds_dpos[m, 2] = g_u * inv_w
        ds_daz[m] = g_az * inv_w
    return out, ds_dpos, ds_daz


def map_single_cam_vol_pos(cam7_lin, X, X0, g_bar_i, R_bar_i, dv_dang_i, qb_i, zb_i, dq_df_i, d2q_df2_i):
    dX = X - X0
    dc_lin = cam7_lin[:3] - g_bar_i[:3]
    dang = cam7_lin[3:6] - g_bar_i[3:6]
    dlogf = cam7_lin[6] - g_bar_i[6]
    dv_lin = R_bar_i.T @ (dX - dc_lin) + dv_dang_i @ dang
    dz = dv_lin[2]
    dw0 = dv_lin[0] + qb_i[1] * dz + zb_i * dq_df_i[1] * dlogf
    dw1 = dv_lin[1] + qb_i[0] * dz + zb_i * dq_df_i[0] * dlogf
    z_new = zb_i + dz
    dq_star = dq_df_i * dlogf + 0.5 * d2q_df2_i * (dlogf * dlogf)
    q_new = qb_i + dq_star
    v_new = np.array([-z_new * q_new[1] + dw0, -z_new * q_new[0] + dw1, z_new])
    th, ph, ti = cam7_lin[3], cam7_lin[4], cam7_lin[5]
    cth, sth = np.cos(th), np.sin(th)
    cph, sph = np.cos(ph), np.sin(ph)
    cti, sti = np.cos(ti), np.sin(ti)
    R_i = np.array([
        [cph * cth * cti - sph * sti, -cph * cth * sti - sph * cti, cph * sth],
        [sph * cth * cti + cph * sti, -sph * cth * sti + cph * cti, sph * sth],
        [-sth * cti, sth * sti, cth],
    ])
    return X - R_i @ v_new


def pullback_vol_diffeo_grad(g_lin, grad_g_nl, data):
    grad_g_lin = grad_g_nl.copy()
    eps_m = 1e-6
    g_bar = data["g_bar"]
    ant = g_lin[203:206]
    tx = g_lin[206:209]
    ant_bar = g_bar[203:206]
    tx_bar = g_bar[206:209]
    for i in data["untied_idx"]:
        is_tx = data["is_tx_cam"][i]
        X = tx if is_tx else ant
        X0 = tx_bar if is_tx else ant_bar
        c7 = g_lin[7*i:7*i+7]
        gb_i = g_bar[7*i:7*i+7]
        Rb_i = data["R_bar"][i]
        dvd_i = data["dv_dang"][i]
        qb_i = data["q_bar"][i]
        zb_i = data["z_bar"][i]
        dq_i = data["dq_star_dlogf"][i]
        d2q_i = data["d2q_star_dlogf2"][i]
        g_pos = grad_g_nl[7*i:7*i+3]
        for k_loc in range(7):
            cp, cm = c7.copy(), c7.copy()
            cp[k_loc] += eps_m; cm[k_loc] -= eps_m
            dp = (map_single_cam_vol_pos(cp, X, X0, gb_i, Rb_i, dvd_i, qb_i, zb_i, dq_i, d2q_i) -
                  map_single_cam_vol_pos(cm, X, X0, gb_i, Rb_i, dvd_i, qb_i, zb_i, dq_i, d2q_i)) / (2.0 * eps_m)
            if k_loc < 3:
                dp[k_loc] -= 1.0
            grad_g_lin[7*i + k_loc] += float(np.dot(g_pos, dp))
        t_offset = 206 if is_tx else 203
        for k_t in range(3):
            Xp, Xm = X.copy(), X.copy()
            Xp[k_t] += eps_m; Xm[k_t] -= eps_m
            dp = (map_single_cam_vol_pos(c7, Xp, X0, gb_i, Rb_i, dvd_i, qb_i, zb_i, dq_i, d2q_i) -
                  map_single_cam_vol_pos(c7, Xm, X0, gb_i, Rb_i, dvd_i, qb_i, zb_i, dq_i, d2q_i)) / (2.0 * eps_m)
            grad_g_lin[t_offset + k_t] += float(np.dot(g_pos, dp))
    return grad_g_lin
