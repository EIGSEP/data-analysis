import numpy as np, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
z = np.load(sys.argv[1]); out = sys.argv[2]
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
chans = [70, 82, 87, 89]
fig, axs = plt.subplots(len(chans), 2, figsize=(13, 2.6*len(chans)), sharex=True)
for r, d in enumerate(chans):
    k = list(z["d_idx"]).index(d); v = z["valid"][k] & z["sel"]; t = np.flatnonzero(v)
    for c, (arm, meas) in enumerate((("Ex", "Ex_meas"), ("Ey", "Ey_meas"))):
        ax = axs[r, c]; p = np.abs(z[arm+"_all"][v, d])**2; m = z[meas][v, d]; med = np.median(p)
        ax.semilogy(t[m], p[m]/med, ".", ms=2, color=BLUE, label="measured")
        ax.semilogy(t[~m], p[~m]/med, ".", ms=2, color=ORANGE, label="DPSS-filled")
        bad = p/med > 100
        ax.semilogy(t[bad], p[bad]/med, "o", ms=7, mfc="none", mec=INK, mew=1, label=">100× median")
        ax.axhline(100, color=MUTED, lw=0.8, ls="--")
        ax.set_title(f"d{d} ({z['freq'][k]:.1f} MHz)  |{arm}|²  — {m.mean():.0%} measured, {bad.sum()} cells >100×",
                     fontsize=10, loc="left", color=INK)
        if c == 0: ax.set_ylabel("power / median")
        ax.grid(axis="y", color="#ecebe7", lw=0.6)
axs[0, 0].legend(loc="upper right", fontsize=8, frameon=False, markerscale=2)
for ax in axs[-1]: ax.set_xlabel("integration index")
fig.suptitle("Absurd cells are all DPSS fills of the unmeasured arm (log scale)", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig(out, dpi=110, facecolor="#fcfcfb")
