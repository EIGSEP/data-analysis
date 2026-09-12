"""Two checks: is the uniform-rate angle axis self-consistent, and is the comb radiated?"""
import numpy as np
from common import load, db, OUT, COMB_RESIDUE, COMB_SPILL, ROT, CTRL

D = load()
fr, ch, t = D["fr"], D["ch"], D["tsec"]
tones = np.load(OUT / "tones.npy")
band = (fr > 150) & (fr < 195)
tone_m = (ch % 16 == COMB_RESIDUE) & band
cont_m = ~np.isin(ch % 16, COMB_SPILL) & band
c = db(np.nanmedian(D["d"][ROT][:, tone_m], 1)) - db(np.nanmedian(D["d"][ROT][:, cont_m], 1))

print("=== is a uniform-rate angle axis self-consistent? ===")
print("under uniform rotation the +-180 deg points are the SAME orientation,")
print("so the response there must match.\n")
for nm, (a, b, e) in {"turn at t=940.6 s": (869.7, 940.6, 1012.5),
                      "turn at t=1316.4 s": (1246.6, 1316.4, 1389.4)}.items():
    lo, hi = a + 0.5 * (b - a), b + 0.5 * (e - b)
    w = 3.0
    m0 = D["sky"] & (np.abs(t - lo) < w)
    m1 = D["sky"] & (np.abs(t - hi) < w)
    print(f"{nm}: R(-180) = {np.nanmedian(c[m0]):5.1f} dB, R(+180) = {np.nanmedian(c[m1]):5.1f} dB "
          f"-> mismatch {np.nanmedian(c[m0]) - np.nanmedian(c[m1]):+5.1f} dB")
print("\nboth turns mismatch by the same sign and size, so it is reproducible structure,")
print("not noise: the motion is not a uniform 360 deg rotation, and a degree axis built")
print("on that assumption would be misleading.  The figure uses time instead.\n")

print("=== is the comb radiated (picked up by the antenna) or conducted? ===")
states, counts = np.unique(D["sw"], return_counts=True)
for s, n in zip(states, counts):
    m = (D["sw"] == s)
    if n < 5:
        continue
    ctr_ = np.nanmedian(db(np.nanmedian(D["d"][ROT][m][:, tone_m], 1))
                        - db(np.nanmedian(D["d"][ROT][m][:, cont_m], 1)))
    lvl = np.nanmedian(db(np.nanmedian(D["d"][ROT][m][:, cont_m], 1)))
    print(f"   sw_state {s:3d} ({n:4d} ints): continuum {lvl:5.1f} dB, comb contrast {ctr_:5.1f} dB")
print("\nthe comb is present only while the receiver looks at the antenna, and it modulates")
print("with platform orientation, so it reaches the receiver through the antenna.")
