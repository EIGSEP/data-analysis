"""Are the shallow low-frequency shapes real, or just the additive continuum capping them?

Raw channel power is tone + sky + receiver noise.  When the tone is nulled the channel
only falls to the continuum, so the deepest drop a channel can show is capped by its
tone-to-continuum ratio at the peak.  Subtract the continuum (in linear power, as the
paper's 2026 figure does) and the cap goes away.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from common import load, db, flanking, OUT, ROT

D = load()
fr, ch, t = D["fr"], D["ch"], D["tsec"]
nchan = fr.size
tones = np.load(OUT / "tones.npy")
A, B, E = np.load(OUT / "turn.npy")
HALF = 0.25 * (B - A) + 0.25 * (E - B)
sel = D["sky"] & (np.abs(t - B) <= HALF)
tau = (t - B)[sel]
d = D["d"][ROT]

raw, sub, snr, noise = {}, {}, {}, {}
for c in tones:
    nb = flanking(c, nchan)
    tot = d[sel, c]
    cont = np.nanmedian(d[sel][:, nb], 1)
    ref = np.abs(tau) < 2.0
    raw[c] = db(tot) - db(np.nanmedian(tot[ref]))
    ex = tot - cont                                  # linear-power continuum subtraction
    sub[c] = db(np.where(ex > 0, ex, np.nan)) - db(np.nanmedian(ex[ref]))
    snr[c] = db(np.nanmedian(tot[ref])) - db(np.nanmedian(cont[ref]))
    # noise on the continuum estimate sets how deep a subtracted tone can be believed
    noise[c] = db(np.nanmedian(ex[ref])) - db(np.nanstd(cont))

dep = lambda p: np.nanmax(p) - np.nanmin(p)
print(f"{'MHz':>7} {'tone-cont':>10} {'raw depth':>10} {'sub depth':>10} {'headroom':>9}")
for c in tones:
    print(f"{fr[c]:7.1f} {snr[c]:10.1f} {dep(raw[c]):10.1f} {dep(sub[c]):10.1f} {noise[c]:9.1f}")

rawd = np.array([dep(raw[c]) for c in tones])
subd = np.array([dep(sub[c]) for c in tones])
s = np.array([snr[c] for c in tones])
print(f"\nraw depth vs tone-to-continuum: median |difference| {np.median(np.abs(rawd - s)):.1f} dB, "
      f"correlation {np.corrcoef(rawd, s)[0,1]:.3f}")
lo, hi = fr[tones] < 160, fr[tones] >= 160
print(f"below 160 MHz: raw {np.median(rawd[lo]):5.1f} dB -> subtracted {np.median(subd[lo]):5.1f} dB "
      f"(tone sits {np.median(s[lo]):.1f} dB over continuum)")
print(f"above 160 MHz: raw {np.median(rawd[hi]):5.1f} dB -> subtracted {np.median(subd[hi]):5.1f} dB "
      f"(tone sits {np.median(s[hi]):.1f} dB over continuum)")

norm = Normalize(fr[tones].min(), fr[tones].max())
smap = ScalarMappable(norm, plt.cm.plasma)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for c in tones:
    axs[0].plot(tau, raw[c], color=smap.to_rgba(fr[c]), lw=0.8)
    axs[1].plot(tau, sub[c], color=smap.to_rgba(fr[c]), lw=0.8)
axs[0].set_title("raw channel power", fontsize=10)
axs[1].set_title("continuum subtracted (as the 2026 paper figure)", fontsize=10)
for ax in axs[:2]:
    ax.set_xlim(-HALF, HALF); ax.grid(alpha=0.3)
    ax.set_xlabel("time through turn [s]"); ax.set_ylabel("dB rel. $t=0$")
axs[1].set_ylim(-45, 15)
axs[2].plot(s, rawd, "o", ms=4, label="raw")
axs[2].plot(s, subd, "s", ms=4, label="subtracted")
axs[2].plot([0, 28], [0, 28], "k:", lw=0.8, label="depth = tone-to-continuum")
axs[2].set_xlabel("tone above continuum at peak [dB]"); axs[2].set_ylabel("measured depth [dB]")
axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)
fig.colorbar(smap, ax=axs[1], label="MHz", pad=0.02)
fig.tight_layout(); fig.savefig(OUT / "11_continuum.png", dpi=110)
