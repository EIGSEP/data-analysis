"""Measure the turn period by folding, since deployment 4 has no encoder.

Grid-search the period, fold the rotating receiver's tone contrast on each trial, and keep
the one that gives the crispest fold -- the ratio of the variance of the binned profile to
the residual scatter within bins.  The full turn carries two unequal maxima, so the true
period wins over its half if the two maxima really do differ.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, db, OUT, COMB_RESIDUE, COMB_SPILL, ROT

D = load()
fr, ch = D["fr"], D["ch"]
tone = ch % 16 == COMB_RESIDUE
cont = ~np.isin(ch % 16, COMB_SPILL)
band = (fr > 150) & (fr < 195)

T0, T1 = 660.0, 1520.0          # the turning interval, tightened from 02_overview
sel = D["sky"] & (D["tsec"] > T0) & (D["tsec"] < T1)
d = D["d"][ROT]
y = (db(np.nanmedian(d[:, tone & band], 1)) - db(np.nanmedian(d[:, cont & band], 1)))[sel]
x = D["tsec"][sel]
m = np.isfinite(y); x, y = x[m], y[m]
print(f"{len(x)} integrations over {x[-1] - x[0]:.0f} s")


def sharpness(P, nb=48):
    ph = np.mod(x, P) / P * nb
    j = ph.astype(int) % nb
    prof = np.array([np.median(y[j == b]) if (j == b).sum() > 2 else np.nan for b in range(nb)])
    if np.isnan(prof).any():
        return -np.inf, prof
    resid = y - prof[j]
    return np.nanvar(prof) / np.nanvar(resid), prof


grid = np.arange(40.0, 130.0, 0.02)
score = np.array([sharpness(P)[0] for P in grid])
best = grid[np.argmax(score)]
print(f"best period {best:.2f} s   (score {score.max():.2f})")
for P in (best / 2, best, best * 2):
    print(f"   P = {P:7.2f} s -> sharpness {sharpness(P)[0]:6.2f}")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(grid, score, lw=0.8)
axs[0].axvline(best, color="r", lw=0.8)
axs[0].set_xlabel("trial period [s]"); axs[0].set_ylabel("fold sharpness")
for P, ax in ((best / 2, axs[1]), (best, axs[2])):
    ph = np.mod(x, P) / P * 360.0
    ax.plot(ph, y, ".", ms=2, alpha=0.5)
    _, prof = sharpness(P)
    ax.plot(np.arange(48) / 48 * 360 + 360 / 96, prof, "r-", lw=1.5)
    ax.set_title(f"folded on {P:.2f} s", fontsize=10)
    ax.set_xlabel("phase [deg]"); ax.set_ylabel("contrast [dB]")
fig.tight_layout(); fig.savefig(OUT / "03_period.png", dpi=110)

# turn-to-turn repeatability: how much does each turn differ from the mean profile?
nturn = int((x[-1] - x[0]) // best)
print(f"{nturn} complete turns in the interval")
_, prof = sharpness(best)
for i in range(nturn):
    a, b = x[0] + i * best, x[0] + (i + 1) * best
    k = (x >= a) & (x < b)
    j = (np.mod(x[k], best) / best * 48).astype(int) % 48
    print(f"   turn {i:2d}: rms about the mean profile {np.sqrt(np.mean((y[k] - prof[j])**2)):5.2f} dB, "
          f"depth {np.nanmax(y[k]) - np.nanmin(y[k]):5.1f} dB")
