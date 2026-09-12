"""Score every candidate turn and pick the one the figure will show.

A turn is centred on a tall response maximum and runs to the half-turn points either side.
Tall maxima are taken to be one full turn apart, not a half: the two maxima per turn differ
by ~6 dB, which only happens if they are 180 deg apart on a beam with different fore and
aft response.  Phase is linear in time within each half, so a varying rate between halves
does not distort the shape.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from common import load, db, flanking, OUT, COMB_RESIDUE, COMB_SPILL, ROT

D = load()
fr, ch, t = D["fr"], D["ch"], D["tsec"]
nchan = fr.size
tones = np.load(OUT / "tones.npy")
d = D["d"][ROT]
band = (fr > 150) & (fr < 195)
c = np.where(D["sky"],
             db(np.nanmedian(d[:, (ch % 16 == COMB_RESIDUE) & band], 1))
             - db(np.nanmedian(d[:, ~np.isin(ch % 16, COMB_SPILL) & band], 1)), np.nan)

k = (t > 640) & (t < 1530) & np.isfinite(c)
x, ys = t[k], uniform_filter1d(c[k], 5)
maxs = [i for i in range(3, len(ys) - 3) if ys[i] == max(ys[i - 3:i + 4]) and ys[i] > 18.0]
thin = []
for i in maxs:
    if not thin or x[i] - x[thin[-1]] > 20:
        thin.append(i)
    elif ys[i] > ys[thin[-1]]:
        thin[-1] = i
tm = x[thin]
print("tall maxima at", np.round(tm, 1))

BIN = 10.0   # ~67 integrations per turn at 1.07 s cadence, so 5 deg bins would be empty
edges = np.arange(-180.0, 180.0 + BIN, BIN)
ctr = 0.5 * (edges[:-1] + edges[1:])
ZERO = int(np.argmin(np.abs(ctr)))


def phase_of(a, b, e):
    """Phase in deg for a turn centred on b, half-turns bounded by a and e."""
    ph = np.where(t < b, 360.0 * (t - b) / (b - a), 360.0 * (t - b) / (e - b))
    return ph, (t >= a + 0.5 * (b - a)) & (t <= b + 0.5 * (e - b)) & D["sky"]


def profile(series, ph, sel):
    j = np.digitize(ph, edges) - 1
    out = np.full(len(ctr), np.nan)
    for i in range(len(ctr)):
        v = series[sel & (j == i)]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = np.median(v)
    return out - np.nanmedian(out[ZERO - 1:ZERO + 2])


print(f"\n{'turn':>4} {'centre t':>9} {'half [s]':>14} {'cover':>6} {'depth med':>10} {'rough':>7}")
best, rows = None, []
for n in range(1, len(tm) - 1):
    a, b, e = tm[n - 1], tm[n], tm[n + 1]
    if not (60 < b - a < 95 and 60 < e - b < 95):
        continue
    ph, sel = phase_of(a, b, e)
    prof = [profile(db(d[:, cc]), ph, sel) for cc in tones]
    cover = np.mean([np.isfinite(p).mean() for p in prof])
    if cover < 0.9:
        continue
    depth = np.array([np.nanmax(p) - np.nanmin(p) for p in prof])
    rough = np.median([np.nanmedian(np.abs(np.diff(p, 2))) for p in prof])
    rows.append((n, b, a, e, np.median(depth), rough, sel.sum()))
    print(f"{n:4d} {b:9.1f} {b-a:6.1f}/{e-b:5.1f} {cover:6.2f} {np.median(depth):10.1f} {rough:7.2f}")

# prefer deep and smooth
score = [r[4] / (1.0 + r[5]) for r in rows]
pick = rows[int(np.argmax(score))]
print(f"\npicked turn centred at t = {pick[1]:.1f} s, halves {pick[1]-pick[2]:.1f}/{pick[3]-pick[1]:.1f} s, "
      f"{pick[6]} integrations, median depth {pick[4]:.1f} dB")
np.save(OUT / "turn.npy", np.array([pick[2], pick[1], pick[3]]))

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(x, ys, lw=1.0)
for r in rows:
    ax.axvspan(r[2] + 0.5 * (r[1] - r[2]), r[1] + 0.5 * (r[3] - r[1]), alpha=0.12, color="C2")
ax.axvspan(pick[2] + 0.5 * (pick[1] - pick[2]), pick[1] + 0.5 * (pick[3] - pick[1]), alpha=0.3, color="C3")
ax.set_xlabel("seconds since 2025-07-20 01:01:18 UTC")
ax.set_ylabel("comb contrast 150-195 MHz [dB]"); ax.grid(alpha=0.3)
ax.set_title("candidate turns (green), the one the figure shows (red)")
fig.tight_layout(); fig.savefig(OUT / "06_pick_turn.png", dpi=110)
