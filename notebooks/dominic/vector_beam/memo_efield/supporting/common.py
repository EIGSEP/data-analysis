"""Shared setup for the 2026-10-27 handedness x positions tilt comparison.
Left-handed az is implemented as the right-handed model fed -az (offset refit)."""
import sys, json, dataclasses, numpy as np
S = "."; sys.path.insert(0, S); sys.argv = [sys.argv[0], S]
import tilt, geomfit2 as gf
from eigsep_data.beam_mapping.tx_fit import stripe_split
z0, ch = gf.load(S + "/inputs.npz"); split = stripe_split(z0["az"])
g = json.load(open(S + "/geomscan2.json"))["g"]; gs = [a * np.exp(1j * p) for a, p in g]
EV = [k for k, c in enumerate(ch) if c["d"] % 2 == 0]; OD = [k for k, c in enumerate(ch) if c["d"] % 2 == 1]
EL0 = float(gf.P0[2])  # v0013 el zero correction, 0.93 deg
POS = {  # memo 002: v0005 selected state (used 10-21) vs adopted posterior means (10-23)
    "sel": (np.array([1655.37, 2030.96, 1777.88]), np.array([1651.04, 2026.72, 1684.66])),
    "mean": (np.array([1655.13, 2030.96, 1777.85]), np.array([1650.61, 2027.05, 1684.62])),
}
BASE = gf.MODEL


def setup(pos, hand, off, alpha):
    ant, tx = POS[pos]
    gf.MODEL = dataclasses.replace(BASE, antenna_enu=ant, transmitter_enu=tx, az_offset_deg=float(off), alpha0_deg=float(alpha) % 180)
    z = dict(z0); z["az"] = z0["az"] if hand == "R" else -np.asarray(z0["az"])
    return z, np.array([0.0, 0.0, EL0, 0.0, 0.0])


def hfss(pos, hand, off, alpha, tau1):
    z, p = setup(pos, hand, off, alpha); return tilt.hfss_rms(z, ch, split, p, tau1=tau1)


def cons(pos, hand, off, alpha, tau1, use):
    z, p = setup(pos, hand, off, alpha); return tilt.consistency(z, ch, gs, p, tau1=tau1, use=use)
