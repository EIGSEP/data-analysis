"""Sanity-check the phase mapping, and compare the two usable turns."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from common import load, db, OUT, COMB_RESIDUE, COMB_SPILL, ROT

D = load()
fr, ch, t = D["fr"], D["ch"], D["tsec"]
tones = np.load(OUT / "tones.npy")
band = (fr > 150) & (fr < 195)
c = db(np.nanmedian(D["d"][ROT][:, (ch % 16 == COMB_RESIDUE) & band], 1)) \
    - db(np.nanmedian(D["d"][ROT][:, ~np.isin(ch % 16, COMB_SPILL) & band], 1))

BIN = 10.0
edges = np.arange(-180.0, 180.0 + BIN, BIN)
ctr = 0.5 * (edges[:-1] + edges[1:])
ZERO = int(np.argmin(np.abs(ctr)))
TURNS = {"picked (t=940.6 s)": (869.7, 940.6, 1012.5),
         "other  (t=1316.4 s)": (1246.6, 1316.4, 1389.4)}

fig, axs = plt.subplots(2, 2, figsize=(13, 7))
norm = Normalize(fr[tones].min(), fr[tones].max())
smap = ScalarMappable(norm, plt.cm.plasma)
for col, (nm, (a, b, e)) in enumerate(TURNS.items()):
    ph = np.where(t < b, 360.0 * (t - b) / (b - a), 360.0 * (t - b) / (e - b))
    sel = (t >= a + 0.5 * (b - a)) & (t <= b + 0.5 * (e - b)) & D["sky"]
    j = np.digitize(ph, edges) - 1

    ax = axs[0, col]
    ax.plot(t[sel], c[sel], ".-", ms=3, lw=0.7)
    ax.axvline(b, color="r", lw=0.8, label="turn centre (0 deg)")
    for h in (a + 0.5 * (b - a), b + 0.5 * (e - b)):
        ax.axvline(h, color="0.5", lw=0.8, ls="--")
    ax2 = ax.twiny(); ax2.set_xlim(ph[sel][0], ph[sel][-1]); ax2.set_xlabel("phase [deg]", fontsize=8)
    ax.set_title(nm, fontsize=10); ax.set_xlabel("seconds"); ax.set_ylabel("contrast [dB]")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axs[1, col]
    for cc in tones:
        s = np.where(sel, db(D["d"][ROT][:, cc]), np.nan)
        p = np.array([np.nanmedian(s[sel & (j == i)]) if (sel & (j == i)).any() else np.nan
                      for i in range(len(ctr))])
        p = p - np.nanmedian(p[ZERO - 1:ZERO + 2])
        ax.plot(ctr, p, color=smap.to_rgba(fr[cc]), lw=0.9)
    ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.3)
    ax.set_xlabel("rotation phase [deg]"); ax.set_ylabel("dB rel. 0 deg")
fig.colorbar(smap, ax=axs[1, :], label="Frequency [MHz]", pad=0.02)
fig.savefig(OUT / "08_check.png", dpi=110, bbox_inches="tight")

# where do the nulls land, per band?
for nm, (a, b, e) in TURNS.items():
    ph = np.where(t < b, 360.0 * (t - b) / (b - a), 360.0 * (t - b) / (e - b))
    sel = (t >= a + 0.5 * (b - a)) & (t <= b + 0.5 * (e - b)) & D["sky"]
    j = np.digitize(ph, edges) - 1
    for lo, hi in [(55, 85), (115, 145), (150, 195)]:
        m = np.isin(ch, tones) & (fr > lo) & (fr < hi)
        s = np.where(sel, db(np.nanmedian(D["d"][ROT][:, m], 1)), np.nan)
        p = np.array([np.nanmedian(s[sel & (j == i)]) if (sel & (j == i)).any() else np.nan
                      for i in range(len(ctr))])
        p = p - np.nanmedian(p[ZERO - 1:ZERO + 2])
        neg, pos = ctr < 0, ctr > 0
        print(f"{nm}  {lo:3d}-{hi:3d} MHz: nulls at {ctr[neg][np.nanargmin(p[neg])]:+5.0f} and "
              f"{ctr[pos][np.nanargmin(p[pos])]:+5.0f} deg, depth {np.nanmax(p) - np.nanmin(p):5.1f} dB")
