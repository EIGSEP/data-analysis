"""Which comb channels are usable tones, and how far above the continuum do they sit?"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, db, flanking, OUT, COMB_RESIDUE, COMB_SPILL, FM_BAND, ROT, CTRL

D = load()
fr, ch = D["fr"], D["ch"]
nchan = fr.size
d = D["d"][ROT]

# the platform is stationary for the first 10 min and the last 9 min of the window;
# use the stationary lead-in for the tone census so rotation cannot bias it
lead = D["sky"] & (D["tsec"] < 600)
print(f"stationary lead-in: {lead.sum()} integrations")

spec = db(np.nanmedian(d[lead], 0))
resid = np.array([np.nanmedian(spec[(ch % 16 == r) & (fr > 50) & (fr < 200)]) for r in range(16)])
print("median level by channel residue (50-200 MHz):")
for r in range(16):
    print(f"   {r:2d}: {resid[r] - np.median(resid):+6.2f} dB" + ("   <- comb" if r == COMB_RESIDUE else ""))

cands = [c for c in ch[ch % 16 == COMB_RESIDUE] if 50.0 < fr[c] < 200.0]
rows = []
for c in cands:
    nb = flanking(c, nchan)
    snr = spec[c] - np.nanmedian(spec[nb])
    rough = np.nanmedian(np.abs(np.diff(db(np.nanmedian(d[lead][:, nb], 1)), 2)))
    fm = FM_BAND[0] < fr[c] < FM_BAND[1]
    rows.append((c, fr[c], snr, rough, fm))

print(f"\n{'ch':>5} {'MHz':>8} {'tone-continuum':>15} {'flank roughness':>16}  verdict")
for c, f_, snr, rough, fm in rows:
    why = "FM band" if fm else ("RFI" if rough > 0.05 else ("weak" if snr < 3 else "keep"))
    print(f"{c:5d} {f_:8.2f} {snr:15.1f} {rough:16.4f}  {why}")

keep = [r for r in rows if not r[4] and r[3] <= 0.05 and r[2] >= 3.0]
print(f"\n{len(keep)} tones kept, {min(r[1] for r in keep):.1f}-{max(r[1] for r in keep):.1f} MHz")

fig, axs = plt.subplots(2, 1, figsize=(14, 7))
axs[0].plot(fr, spec, lw=0.5, color="0.6", label="all channels")
axs[0].plot(fr[[r[0] for r in rows]], [spec[r[0]] for r in rows], "o", ms=3, label="comb residue 0")
axs[0].plot([r[1] for r in keep], [spec[r[0]] for r in keep], "o", ms=4, label="kept")
axs[0].axvspan(*FM_BAND, color="0.85", zorder=0)
axs[0].set_xlim(40, 210); axs[0].set_ylabel("stationary spectrum [dB]"); axs[0].legend(fontsize=8)
axs[1].plot([r[1] for r in rows], [r[2] for r in rows], "o-", ms=3)
axs[1].axhline(3.0, color="r", lw=0.8, ls=":")
axs[1].axvspan(*FM_BAND, color="0.85", zorder=0)
axs[1].set_xlim(40, 210); axs[1].set_xlabel("frequency [MHz]")
axs[1].set_ylabel("tone above local continuum [dB]")
for ax in axs: ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "05_tones.png", dpi=110)
np.save(OUT / "tones.npy", np.array([r[0] for r in keep]))
