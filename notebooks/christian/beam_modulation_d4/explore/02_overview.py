"""Where in this 34-min window is the platform actually turning?"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, db, OUT, COMB_RESIDUE, COMB_SPILL, ROT, CTRL, KEYS

D = load()
fr, ch = D["fr"], D["ch"]
tone = (ch % COMB_RESIDUE.__class__(16) == COMB_RESIDUE) if False else (ch % 16 == COMB_RESIDUE)
cont = ~np.isin(ch % 16, COMB_SPILL)
band = (fr > 150) & (fr < 195)

print("sw_state census:", dict(zip(*[a.tolist() for a in np.unique(D["sw"], return_counts=True)])))
print(f"sky integrations: {D['sky'].sum()} / {len(D['sky'])}")

fig, axs = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
for k in KEYS:
    d = D["d"][k]
    c = db(np.nanmedian(d[:, tone & band], 1)) - db(np.nanmedian(d[:, cont & band], 1))
    axs[0].plot(D["tsec"][D["sky"]] / 60, c[D["sky"]], lw=0.7,
                label=f"key {k}" + (" (rotating)" if k == ROT else " (control)"))
    axs[1].plot(D["tsec"][D["sky"]] / 60, db(np.nanmedian(d[:, cont & band], 1))[D["sky"]],
                lw=0.7, label=f"key {k}")
axs[0].set_ylabel("comb contrast, 150-195 MHz [dB]")
axs[1].set_ylabel("non-comb continuum [dB]")
axs[1].set_xlabel("minutes since 2025-07-20 01:01:18 UTC")
for ax in axs:
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
fig.suptitle("deployment 4, local 07-19 18:15 window")
fig.tight_layout(); fig.savefig(OUT / "02_overview.png", dpi=110)

# bound the turning interval: rolling peak-to-peak of the rotating receiver's contrast
d = D["d"][ROT]
c = db(np.nanmedian(d[:, tone & band], 1)) - db(np.nanmedian(d[:, cont & band], 1))
c = np.where(D["sky"], c, np.nan)
W = 70
pp = np.full(len(c), np.nan)
for i in range(len(c) - W):
    v = c[i:i + W]
    if np.isfinite(v).sum() > 0.8 * W:
        pp[i] = np.nanmax(v) - np.nanmin(v)
turning = pp > 8
idx = np.where(turning)[0]
print(f"turning from t = {D['tsec'][idx[0]]:.0f} s to {D['tsec'][idx[-1] + W]:.0f} s "
      f"({(D['tsec'][idx[-1] + W] - D['tsec'][idx[0]]) / 60:.1f} min)")
