"""First full draft of the figure, in the deployment-5 style, plus the controls."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter
from common import load, db, flanking, OUT, COMB_RESIDUE, COMB_SPILL, ROT, CTRL

D = load()
fr, ch, t = D["fr"], D["ch"], D["tsec"]
nchan = fr.size
tones = np.load(OUT / "tones.npy")
a, b, e = np.load(OUT / "turn.npy")

BIN = 10.0
FLAG_DB = 3.0
edges = np.arange(-180.0, 180.0 + BIN, BIN)
ctr = 0.5 * (edges[:-1] + edges[1:])
ZERO = int(np.argmin(np.abs(ctr)))
ph = np.where(t < b, 360.0 * (t - b) / (b - a), 360.0 * (t - b) / (e - b))
sel = (t >= a + 0.5 * (b - a)) & (t <= b + 0.5 * (e - b)) & D["sky"]
bin_idx = np.digitize(ph, edges) - 1
print(f"turn: {sel.sum()} integrations, {(b - a + e - b) / 2:.1f} s")


def profile(series, flag=True):
    s = np.where(sel, series, np.nan)
    if flag:
        fill = np.nanmedian(s[sel])
        smooth = median_filter(np.nan_to_num(s, nan=fill), size=5, mode="nearest")
        s = np.where(np.abs(s - smooth) > FLAG_DB, np.nan, s)
    out = np.full(len(ctr), np.nan)
    for i in range(len(ctr)):
        v = s[sel & (bin_idx == i)]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = np.median(v)
    return out - np.nanmedian(out[ZERO - 1:ZERO + 2])


tone_fr = fr[tones]
depth = lambda p: np.nanmax(p) - np.nanmin(p)

prof = {c: profile(db(D["d"][ROT][:, c])) for c in tones}
td = np.array([depth(prof[c]) for c in tones])
sky = np.array([depth(profile(db(np.nanmedian(D["d"][ROT][:, flanking(c, nchan)], 1)))) for c in tones])
print(f"rotating receiver, tones    : {td.min():.1f}-{td.max():.1f} dB (median {np.median(td):.1f})")
print(f"rotating receiver, non-comb : median {np.median(sky):.2f} dB "
      f"-> tones {np.median(td) / np.median(sky):.0f}x deeper")
for k in CTRL:
    g = profile(db(np.nanmedian(D["d"][k][:, tones], 1)))
    print(f"stationary key {k}, same tones: {depth(g):.2f} dB")

# the same receiver while it is parked, immediately before and after the turn
for nm, lo, hi in [("before", 60.0, 600.0), ("after", 1560.0, 2000.0)]:
    m = D["sky"] & (t > lo) & (t < hi)
    s = db(np.nanmedian(D["d"][ROT][:, tones], 1))[m]
    n = min(len(s), sel.sum())
    print(f"rotating receiver parked {nm}: {np.nanmax(s[:n]) - np.nanmin(s[:n]):.2f} dB over {n} integrations")

norm = Normalize(tone_fr.min(), tone_fr.max())
smap = ScalarMappable(norm, plt.cm.plasma)
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for c in tones:
    ax.plot(ctr, prof[c], color=smap.to_rgba(fr[c]), lw=0.9, alpha=0.95)
ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180])
ax.grid(alpha=0.25, lw=0.5)
ax.set_xlabel("Platform rotation phase [deg]", fontsize=9)
ax.set_ylabel("Received power relative to $0^\\circ$ [dB]", fontsize=9)
ax.tick_params(labelsize=8)
cb = fig.colorbar(smap, ax=ax, pad=0.02)
cb.set_label("Frequency [MHz]", fontsize=9); cb.ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "07_draft.png", dpi=150, bbox_inches="tight")
