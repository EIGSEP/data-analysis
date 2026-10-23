import numpy as np, json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = sys.argv[1]; R = json.load(open(f"{S}/variants.json")); L = np.load(f"{S}/variants_loo.npz")
INK, MUTED, BLUE, ORANGE, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#ecebe7"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
# fig2: metrics per channel
fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for ax, (key, lab) in zip(axs.flat, [("loo_power", "LOO held-out power corr (HFSS-free)"), ("power", "power corr vs HFSS"),
                                     ("y", "y-component corr vs HFSS"), ("z", "z-component corr vs HFSS")]):
    for var, col, mk in (("baseline", ORANGE, "o"), ("bracketed", BLUE, "s")):
        f = [r["freq"] for r in R[var]]; v = [r.get(key, np.nan) for r in R[var]]
        ax.plot(f, v, "-", marker=mk, ms=5, lw=2, color=col, label={"baseline": "before (Phase 2)", "bracketed": "after: unbracketed fills → NaN"}[var])
    ax.set_title(lab, loc="left", fontsize=10, color=INK); ax.grid(axis="y", color=GRID); ax.set_ylim(-0.1 if key == "loo_power" else 0.6, 1.02)
axs[0, 0].annotate("d89: no valid samples\n(fill always extrapolated)", xy=(228.5, 0.0), xytext=(212, 0.1), fontsize=8, color=MUTED,
                   arrowprops=dict(arrowstyle="->", color=MUTED))
axs[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
for ax in axs[1]: ax.set_xlabel("frequency (MHz)")
fig.suptitle("Effect of dropping extrapolated DPSS fills, per channel", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(f"{S}/fig2_metrics.png", dpi=110, facecolor="#fcfcfb")
# fig3: LOO scatter
fig, axs = plt.subplots(2, 4, figsize=(15, 7.2))
for c, d in enumerate((77, 82, 85, 87)):
    for r, var in enumerate(("baseline", "bracketed")):
        ax = axs[r, c]; p, t = L[f"{var}_d{d}_pred"], L[f"{var}_d{d}_true"]; m = np.median(t)
        ax.loglog(t / m, p / m, ".", ms=2, alpha=0.4, color=ORANGE if r == 0 else BLUE)
        lim = [1e-3, max(1e2, (t / m).max(), (p / m).max())]; ax.plot(lim, lim, color=MUTED, lw=0.8)
        ax.set_title(f"d{d} {'before' if r == 0 else 'after'}:  r = {np.corrcoef(p, t)[0, 1]:.3f}", loc="left", fontsize=10)
        ax.grid(color=GRID, lw=0.5)
        if c == 0: ax.set_ylabel("predicted / median")
        if r == 1: ax.set_xlabel("held-out measured / median")
fig.suptitle("Leave-one-out: predicting each held-out visit from the others (log–log, 1:1 line)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(f"{S}/fig3_loo_scatter.png", dpi=110, facecolor="#fcfcfb")
