"""Is the second-half structure orientation-locked (beam/multipath) or kinematic?

Overlay every turn in the episode.  Structure fixed in the antenna frame repeats turn to
turn; structure that moves is the platform's rate varying.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from common import load, db, flanking, OUT, COMB_RESIDUE, COMB_SPILL, ROT

D = load()
fr, ch, t = D["fr"], D["ch"], D["tsec"]
nchan = fr.size
tones = np.load(OUT / "tones.npy")
band = (fr > 150) & (fr < 195)
c = np.where(D["sky"],
             db(np.nanmedian(D["d"][ROT][:, (ch % 16 == COMB_RESIDUE) & band], 1))
             - db(np.nanmedian(D["d"][ROT][:, ~np.isin(ch % 16, COMB_SPILL) & band], 1)), np.nan)

k = (t > 640) & (t < 1530) & np.isfinite(c)
x, ys = t[k], uniform_filter1d(c[k], 5)
pk = [i for i in range(3, len(ys) - 3) if ys[i] == max(ys[i - 3:i + 4]) and ys[i] > 18.0]
tall = []
for i in pk:
    if not tall or x[i] - x[tall[-1]] > 20:
        tall.append(i)
    elif ys[i] > ys[tall[-1]]:
        tall[-1] = i
tm = x[tall]

print(f"{'centre':>8} {'halves [s]':>13} {'1st null':>9} {'2nd min':>8} {'depth1':>7} {'depth2':>7}")
rows = []
for n in range(1, len(tm) - 1):
    a, b, e = tm[n - 1], tm[n], tm[n + 1]
    if not (60 < b - a < 95 and 60 < e - b < 95):
        continue
    m = D["sky"] & (np.abs(t - b) <= 0.25 * (b - a) + 0.25 * (e - b)) & np.isfinite(c)
    tau, v = t[m] - b, uniform_filter1d(c[m], 3)
    neg, pos = tau < -4, tau > 4
    t1, t2 = tau[neg][np.argmin(v[neg])], tau[pos][np.argmin(v[pos])]
    d1, d2 = v[np.abs(tau) < 2].max() - v[neg].min(), v[np.abs(tau) < 2].max() - v[pos].min()
    rows.append((b, tau, v, t1, t2))
    print(f"{b:8.1f} {b-a:6.1f}/{e-b:5.1f} {t1:9.1f} {t2:8.1f} {d1:7.1f} {d2:7.1f}")

t1s = np.array([r[3] for r in rows]); t2s = np.array([r[4] for r in rows])
print(f"\n1st null at {t1s.mean():.1f} +- {t1s.std():.1f} s   "
      f"2nd minimum at {t2s.mean():+.1f} +- {t2s.std():.1f} s")
print(f"|t1| vs t2: {np.abs(t1s).mean():.1f} vs {t2s.mean():.1f} s "
      f"-> the two minima are NOT symmetric about the maximum")

# per-tone: where is the second minimum, as a function of frequency?
A, B, E = np.load(OUT / "turn.npy")
HALF = 0.25 * (B - A) + 0.25 * (E - B)
sel = D["sky"] & (np.abs(t - B) <= HALF)
tau = (t - B)[sel]
print(f"\n{'MHz':>7} {'1st null t':>11} {'2nd min t':>10}  (continuum-subtracted)")
for cc in tones[::4]:
    tot = D["d"][ROT][sel, cc]
    ex = tot - np.nanmedian(D["d"][ROT][sel][:, flanking(cc, nchan)], 1)
    v = uniform_filter1d(db(np.where(ex > 0, ex, np.nan)), 3)
    neg, pos = tau < -4, tau > 4
    print(f"{fr[cc]:7.1f} {tau[neg][np.nanargmin(v[neg])]:11.1f} {tau[pos][np.nanargmin(v[pos])]:10.1f}")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for b, tau_, v, *_ in rows:
    axs[0].plot(tau_, v, lw=0.9, label=f"t={b:.0f} s")
axs[0].set_xlabel("time from maximum [s]"); axs[0].set_ylabel("comb contrast [dB]")
axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3); axs[0].set_title("every usable turn overlaid", fontsize=10)
allt = np.concatenate([r[1] for r in rows]); allv = np.concatenate([r[2] for r in rows])
axs[1].plot(allt, allv, ".", ms=3, alpha=0.6)
axs[1].set_xlabel("time from maximum [s]"); axs[1].set_ylabel("comb contrast [dB]")
axs[1].grid(alpha=0.3); axs[1].set_title("pooled", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "12_asymmetry.png", dpi=110)
