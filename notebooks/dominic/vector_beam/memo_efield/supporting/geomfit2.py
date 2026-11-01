"""Step 2, pixel-free: predict each sample from OTHER-dwell samples in a Gaussian
angular neighbourhood (smooth in geometry), and re-fit the block phases from zero
at every geometry (no inherited v0013 phase)."""
import sys, numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree
sys.path.insert(0, "../..")
import geometry_v0013 as G
from eigsep_base.rotations import enu_to_body, body_to_enu
from eigsep_data.beam_mapping import vector_beam as vb
from eigsep_data.rfi import build_frequency_trust_mask, flag_unbracketed_fills

MODEL = G.model(); P0 = np.asarray(G.PARAMS, float)
SIGMA = np.radians(1.0); RMAX = 3 * SIGMA; DWELL_S = 30.0; BLOCK = 20


def geometry(params, p_body, az, el, amp=True):
    R = MODEL.rotations(az, el, params)
    tx = np.asarray(MODEL.transmitter_enu, float) + np.array([params[3], params[4], 0.0])
    A = np.asarray(MODEL.antenna_enu, float)
    h = tx - (A + body_to_enu(np.asarray(p_body, float), R))
    dist = np.linalg.norm(h, axis=-1, keepdims=True)
    rhat = enu_to_body(h / dist, R)
    tg = MODEL.transmitter(params); E = []
    for a in (0, 1):
        e = enu_to_body(tg.field_enu(a), R); e = e - np.sum(e * rhat, -1, keepdims=True) * rhat
        E.append(e * (np.linalg.norm(tx - A) / dist) if amp else e)
    return rhat, E[0], E[1]


def load(inputs):
    z = np.load(inputs)
    nc = z["Ex_all"].shape[1]; fm = 171.875 + (np.arange(nc) - 60) * 1.953125
    trust = build_frequency_trust_mask(fm, min_freq=87.0, fm_low=87.0, fm_high=108.0, max_freq=230.0)
    Exa, Eya = z["Ex_all"].copy(), z["Ey_all"].copy()
    Exa[flag_unbracketed_fills(z["Ex_meas"], trust)] = np.nan; Eya[flag_unbracketed_fills(z["Ey_meas"], trust)] = np.nan
    ch = []
    for d in z["d_idx"][:-1]:
        d = int(d); Ex, Ey = Exa[:, d], Eya[:, d]
        v = z["sel"] & ~z["bad_time"] & np.isfinite(Ex) & np.isfinite(Ey)
        meas = (z["Ex_meas"] if d % 2 == 0 else z["Ey_meas"])[:, d]
        ch.append(dict(d=d, arm=d % 2, Ex=np.where(v, Ex, 0), Ey=np.where(v, Ey, 0), v=v, meas=meas & v))
    return z, ch


def neighbours(rhat, times, anyvalid):
    """Sparse W[i, j] = exp(-theta^2 / 2 sigma^2) for other-dwell j within RMAX of i."""
    idx = np.flatnonzero(anyvalid); tree = cKDTree(rhat[idx])
    M = tree.sparse_distance_matrix(tree, 2 * np.sin(RMAX / 2), output_type="coo_matrix")
    i, j, c = idx[M.row], idx[M.col], M.data
    keep = np.abs(times[i] - times[j]) > DWELL_S
    i, j, c = i[keep], j[keep], c[keep]
    th = 2 * np.arcsin(np.clip(c / 2, 0, 1))
    n = len(rhat)
    return sp.csr_matrix((np.exp(-th ** 2 / (2 * SIGMA ** 2)), (i, j)), shape=(n, n))


def predict(W, rhat, E1v, E2v, Ex, Ey, v, g, cond_max=900.0):
    """Kernel-weighted per-sample solve (normal equations, one prior row per neighbour)."""
    Wc = W.multiply(v[None, :].astype(float)).tocsr()
    outer = (np.einsum("ni,nj->nij", E1v, E1v) + np.einsum("ni,nj->nij", E2v, E2v)
             + np.einsum("ni,nj->nij", rhat, rhat)).reshape(-1, 9)
    Mm = (Wc @ outer).reshape(-1, 3, 3)
    rhs = Wc @ (E1v * Ex[:, None]) + Wc @ (E2v * (Ey / g)[:, None])
    wsum = np.asarray(Wc.sum(1)).ravel()
    ok = wsum > 0
    ev = np.linalg.eigvalsh(Mm[ok]); good = np.zeros(len(ok), bool)
    good[ok] = (ev[:, 0] > 0) & (ev[:, -1] / np.maximum(ev[:, 0], 1e-300) < cond_max)
    u = np.full((len(ok), 3), np.nan + 0j)
    u[good] = np.linalg.solve(Mm[good], rhs[good][..., None])[..., 0]
    return np.einsum("ni,ni->n", E1v, u), g * np.einsum("ni,ni->n", E2v, u)


def evaluate(params, p_body, z, ch, gs, use=None, iters=15, return_phase=False):
    """Phase re-fitted from zero at this geometry; returns pooled other-dwell error over `use` channels."""
    use = range(len(ch)) if use is None else use
    rhat, E1v, E2v = geometry(params, p_body, z["az"], z["el"])
    anyv = np.any([ch[k]["v"] for k in use], 0)
    W = neighbours(rhat, z["times"], anyv)
    N = len(rhat); phase = np.zeros(N); known = np.ones(N, bool)
    for it in range(iters):
        rot = np.exp(-1j * np.where(known, phase, 0)); data, pred = [], []
        for k in use:
            c = ch[k]; Ex, Ey = c["Ex"] * rot, c["Ey"] * rot; v = c["v"] & known
            ex, ey = predict(W, rhat, E1v, E2v, Ex, Ey, v, gs[k])
            d, p = (Ex, ex) if c["arm"] == 0 else (Ey, ey)
            m = c["meas"] & known; data.append(np.where(m, d, np.nan)); pred.append(np.where(m, p, np.nan))
        dph = vb.fit_block_phase(data, pred, BLOCK, 3)
        ok = np.isfinite(dph); known = ok if it == 0 else known & ok
        phase[ok] += dph[ok]
    num = den = 0.0
    for d_, p_ in zip(data, pred):
        r = np.exp(-1j * dph); m = np.isfinite(d_) & np.isfinite(p_) & np.isfinite(r)
        num += np.sum(np.abs(d_[m] * r[m] - p_[m]) ** 2); den += np.sum(np.abs(d_[m]) ** 2)
    err = float(np.sqrt(num / den))
    return (err, phase, known) if return_phase else err
