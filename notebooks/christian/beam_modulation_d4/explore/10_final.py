"""The figure, with time through one turn on the x axis."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter
from common import load, db, flanking, OUT, ROT, CTRL

D = load()
fr, ch, t = D["fr"], D["ch"], D["tsec"]
nchan = fr.size
tones = np.load(OUT / "tones.npy")
A, B, E = np.load(OUT / "turn.npy")
HALF = 0.25 * (B - A) + 0.25 * (E - B)      # half a turn, in seconds
sel = D["sky"] & (np.abs(t - B) <= HALF)
tau = t - B
FLAG_DB = 3.0
print(f"one turn = {2 * HALF:.1f} s, {sel.sum()} integrations")


def curve(series):
    s = np.where(sel, series, np.nan)
    fill = np.nanmedian(s[sel])
    smooth = median_filter(np.nan_to_num(s, nan=fill), size=5, mode="nearest")
    flagged = np.abs(s - smooth) > FLAG_DB
    s = np.where(flagged, np.nan, s)
    ref = np.nanmedian(s[sel & (np.abs(tau) < 2.0)])
    return s[sel] - ref, int(flagged[sel].sum())


prof, nflag = {}, 0
for c in tones:
    prof[c], f_ = curve(db(D["d"][ROT][:, c]))
    nflag += f_
depth = lambda p: np.nanmax(p) - np.nanmin(p)
td = np.array([depth(prof[c]) for c in tones])
sky = np.array([depth(curve(db(np.nanmedian(D["d"][ROT][:, flanking(c, nchan)], 1)))[0]) for c in tones])

print(f"tones                       : {len(tones)}, {fr[tones].min():.1f}-{fr[tones].max():.1f} MHz, "
      f"spacing {16 * (fr[1] - fr[0]):.3f} MHz")
print(f"rotating receiver, tones    : {td.min():.1f}-{td.max():.1f} dB (median {np.median(td):.1f})")
print(f"rotating receiver, non-comb : median {np.median(sky):.2f} dB "
      f"-> tones {np.median(td) / np.median(sky):.0f}x deeper")
for k in CTRL:
    print(f"stationary key {k}, same tones: {depth(curve(db(np.nanmedian(D['d'][k][:, tones], 1)))[0]):.2f} dB")
for nm, lo, hi in [("before", 60.0, 600.0), ("after", 1560.0, 2000.0)]:
    m = D["sky"] & (t > lo) & (t < hi)
    s = db(np.nanmedian(D["d"][ROT][:, tones], 1))[m]
    n = min(len(s), sel.sum())
    print(f"rotating receiver parked {nm:6s}: {np.nanmax(s[:n]) - np.nanmin(s[:n]):.2f} dB over {n} integrations")
print(f"samples flagged             : {nflag}")

norm = Normalize(fr[tones].min(), fr[tones].max())
smap = ScalarMappable(norm, plt.cm.plasma)
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for c in tones:
    ax.plot(tau[sel], prof[c], color=smap.to_rgba(fr[c]), lw=0.9, alpha=0.95)
ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
ax.set_xlim(-HALF, HALF)
ax.set_xticks([-30, -15, 0, 15, 30])
ax.grid(alpha=0.25, lw=0.5)
ax.set_xlabel("Time through one platform turn [s]", fontsize=9)
ax.set_ylabel("Received power relative to $t=0$ [dB]", fontsize=9)
ax.tick_params(labelsize=8)
cb = fig.colorbar(smap, ax=ax, pad=0.02)
cb.set_label("Frequency [MHz]", fontsize=9); cb.ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "10_final.png", dpi=150, bbox_inches="tight")
