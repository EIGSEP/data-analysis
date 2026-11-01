"""Axle tilt tau1 under right- vs left-handed azimuth, at v0005 selected state vs posterior-mean positions.
HFSS-free: other-dwell consistency (even/odd channel halves), closure g refit at every geometry.
HFSS: held-out stripe rms, az offset + alpha found by a global scan per handedness then refined per tau1.
el zero fixed at v0013's +0.93 deg; tx and antenna fixed at the named positions."""
from common import *
import healpy as hp
from eigsep_base.rotations import vector_to_spherical
from eigsep_data.beam_mapping import vector_beam as vb
ph = np.load("../../selfcal_phase.npz")["phase"]
rot = np.exp(-1j * np.where(np.isfinite(ph), ph, 0)); okph = np.isfinite(ph)


def g_at(pos, hand, off, al, tau1):
    z, p = setup(pos, hand, off, al)
    rhat, E1v, E2v = tilt.geometry(p, z["az"], z["el"], tau1)
    th, phi = vector_to_spherical(rhat); pxs = hp.ang2pix(32, th, phi)
    return [vb.fit_closure_g(pxs, rhat, E1v, E2v, c["Ex"] * rot, c["Ey"] * rot, c["v"] & okph) for c in ch]


def global_scan(pos, hand):
    best = (np.inf,)
    for off in np.arange(-180, 180, 5.0):
        for al in np.arange(0, 180, 5.0):
            r = hfss(pos, hand, off, al, 0.0)
            if r < best[0]: best = (r, off, al)
    return best


def refine(pos, hand, off, al, tau1):
    best = (np.inf,)
    for o in off + np.arange(-4, 4.01, 1.0):
        for a in al + np.arange(-6, 6.01, 2.0):
            r = hfss(pos, hand, o, a, tau1)
            if r < best[0]: best = (r, o, a)
    return best


taus = np.arange(-10, 10.01, 1.0); out = {}
for hand in ("R", "L"):
    g0 = global_scan("mean", hand); print(f"{hand} global HFSS scan at tau1=0: rms {g0[0]:.4f} off {g0[1]:+.0f} alpha {g0[2]:.0f}", flush=True)
    for pos in ("sel", "mean"):
        key = f"{hand}_{pos}"; rows = []; off, al = g0[1], g0[2]
        for t in taus:
            gg = g_at(pos, hand, off, al, t)
            ce, co = cons(pos, hand, off, al, t, EV) if False else (None, None)
            z, p = setup(pos, hand, off, al)
            ce = tilt.consistency(z, ch, gg, p, tau1=t, use=EV); co = tilt.consistency(z, ch, gg, p, tau1=t, use=OD)
            h, off, al = refine(pos, hand, off, al, t)
            rows.append(dict(tau1=float(t), cons_even=ce, cons_odd=co, hfss=h, az_off=float(off), alpha=float(al) % 180))
            print(f"  {key} tau1 {t:+5.1f}: cons {ce:.4f}/{co:.4f}  hfss {h:.4f} (off {off:+.0f}, alpha {al % 180:.0f})", flush=True)
        out[key] = rows; json.dump(out, open("compare.json", "w"), indent=1)
