"""Mount tilts: R = Rz(psi) Ry(tau1) Rx(el) Ry(tau2) Rz(az).
tau1: whole mount tilted about the horizontal axis perpendicular to the axle (rigid; ~ tx direction shift).
tau2: roll axis not perpendicular to the elevation axle (non-rigid; new information)."""
import sys, json, numpy as np, healpy as hp
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import geomfit2 as gf
from eigsep_base.rotations import enu_to_body, body_to_enu, rx, rz, vector_to_spherical
from eigsep_data.beam_mapping.tx_fit import stripe_split


def ry(deg):
    a = np.radians(np.asarray(deg, float)); c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def rotations(params, az, el, tau1=0.0, tau2=0.0):
    M = gf.MODEL
    return rz(M.psi_deg) @ ry(tau1) @ rx(np.asarray(el) + params[2]) @ ry(tau2) @ rz(np.asarray(az) + M.az_offset_deg + params[1])


def geometry(params, az, el, tau1=0.0, tau2=0.0):
    R = rotations(params, az, el, tau1, tau2)
    tx = np.asarray(gf.MODEL.transmitter_enu, float) + np.array([params[3], params[4], 0.0])
    h = tx - np.asarray(gf.MODEL.antenna_enu, float); rhat = enu_to_body(h / np.linalg.norm(h), R)
    tg = gf.MODEL.transmitter(params); E = []
    for a in (0, 1):
        e = enu_to_body(tg.field_enu(a), R); E.append(e - np.sum(e * rhat, -1, keepdims=True) * rhat)
    return rhat, E[0], E[1]


def hfss_rms(z, ch, split, params, tau1=0.0, tau2=0.0):
    rhat, E1v, E2v = geometry(params, z["az"], z["el"], tau1, tau2)
    th, ph = vector_to_spherical(rhat); pxs = hp.ang2pix(32, th, ph); out = []
    for k, c in enumerate(ch):
        D = np.abs(c["Ex"] if c["arm"] == 0 else c["Ey"]) ** 2; m = c["meas"]
        uh = z["beam"][z["h_idx"][k]][:, pxs].T; e = E1v if c["arm"] == 0 else E2v
        cH = np.abs(np.einsum("ij,ij->i", e, np.conj(uh))) ** 2; tr, te = m & (split == 0), m & (split == 2)
        gain = D[tr] @ cH[tr] / (cH[tr] @ cH[tr]); out.append(np.linalg.norm(D[te] - gain * cH[te]) / np.linalg.norm(D[te]))
    return float(np.median(out))


def consistency(z, ch, gs, params, tau1=0.0, tau2=0.0, use=None, iters=30):
    """geomfit2.evaluate with the tilted geometry (monkeypatch its geometry function)."""
    orig = gf.geometry
    gf.geometry = lambda p, pb, az, el, amp=True: geometry(p, az, el, tau1, tau2)
    try:
        return gf.evaluate(params, [0, 0, 0], z, ch, gs, use=use, iters=iters)
    finally:
        gf.geometry = orig
