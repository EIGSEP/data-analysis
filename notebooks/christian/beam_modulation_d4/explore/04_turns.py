"""Are the turns uniform?  Locate the nulls and maxima directly instead of folding."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from common import load, db, OUT, COMB_RESIDUE, COMB_SPILL, ROT

D = load()
fr, ch = D["fr"], D["ch"]
tone = ch % 16 == COMB_RESIDUE
cont = ~np.isin(ch % 16, COMB_SPILL)
band = (fr > 150) & (fr < 195)
d = D["d"][ROT]
c = db(np.nanmedian(d[:, tone & band], 1)) - db(np.nanmedian(d[:, cont & band], 1))
c = np.where(D["sky"], c, np.nan)
t = D["tsec"]

T0, T1 = 640.0, 1530.0
k = (t > T0) & (t < T1) & np.isfinite(c)
x, y = t[k], c[k]
ys = uniform_filter1d(y, 5)

def extrema(v, sign):
    return [i for i in range(3, len(v) - 3)
            if sign * v[i] == max(sign * v[i - 3:i + 4]) and sign * v[i] > sign * np.median(v)]

mins = [i for i in range(3, len(ys) - 3) if ys[i] == min(ys[i - 3:i + 4]) and ys[i] < 10]
maxs = [i for i in range(3, len(ys) - 3) if ys[i] == max(ys[i - 3:i + 4]) and ys[i] > 14]
# thin out near-duplicates
def thin(idx, gap=10.0):
    out = []
    for i in idx:
        if not out or x[i] - x[out[-1]] > gap:
            out.append(i)
        elif (ys[i] < ys[out[-1]]) == (idx is mins):
            out[-1] = i
    return out
mins, maxs = thin(mins), thin(maxs)
print(f"{len(mins)} nulls, {len(maxs)} maxima")
dn = np.diff(x[mins]); dm = np.diff(x[maxs])
print(f"null-to-null   spacing: {np.median(dn):.1f} s  (spread {dn.min():.1f}-{dn.max():.1f})")
print(f"peak-to-peak   spacing: {np.median(dm):.1f} s  (spread {dm.min():.1f}-{dm.max():.1f})")
print("maxima heights:", " ".join(f"{ys[i]:.1f}" for i in maxs))
print("null depths   :", " ".join(f"{ys[i]:.1f}" for i in mins))

fig, ax = plt.subplots(figsize=(16, 4.5))
ax.plot(x, y, lw=0.6, color="0.6")
ax.plot(x, ys, lw=1.0)
ax.plot(x[mins], ys[mins], "vC3", ms=6, label="nulls")
ax.plot(x[maxs], ys[maxs], "^C2", ms=6, label="maxima")
ax.set_xlabel("seconds since 2025-07-20 01:01:18 UTC")
ax.set_ylabel("comb contrast, 150-195 MHz [dB]")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "04_turns.png", dpi=110)

print()
print("maxima  t[s]  height   gap to previous")
prev = None
for i in maxs:
    print(f"   {x[i]:7.1f}  {ys[i]:5.1f}" + (f"   {x[i]-prev:6.1f}" if prev else ""))
    prev = x[i]
tall = [i for i in maxs if ys[i] > 18.0]
print("\ntall maxima (>18 dB) spacing:", np.round(np.diff(x[tall]), 1))
