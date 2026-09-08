"""Three ways to render the curves where the tone stops being detectable."""
import os, runpy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

os.environ.pop("BEAM_PNG", None)
g = runpy.run_path("/home/christian/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti/docs/render_beam_modulation.py")
binned, subtract, noise_sigma = g["binned"], g["subtract"], g["noise_sigma"]
tones, freq, A, B, ZERO, NSIG = g["tones"], g["freq"], g["A"], g["B"], g["ZERO"], g["NSIG"]
centres, to_db = g["centres"], g["to_db"]

data = {}
for c in tones:
    mu = binned(subtract(c), A, B)
    ref = np.nanmedian(mu[ZERO - 1:ZERO + 2])
    sig = noise_sigma(c)
    pos = mu > 0
    data[c] = (np.where(pos, to_db(np.where(pos, mu, np.nan) / ref), np.nan),
               mu > NSIG * sig, to_db(NSIG * sig / ref))

tf = freq[np.array(tones)]
norm = Normalize(tf.min(), tf.max())
smap = ScalarMappable(norm, plt.cm.plasma)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)

for ax, style in zip(axes, ("hard stop (current)", "faint continuation", "floor marked")):
    for c in tones:
        full, ok, floor = data[c]
        col = smap.to_rgba(freq[c])
        if style == "hard stop (current)":
            ax.plot(centres, np.where(ok, full, np.nan), color=col, lw=0.7)
        elif style == "faint continuation":
            ax.plot(centres, full, color=col, lw=0.7, alpha=0.22)
            ax.plot(centres, np.where(ok, full, np.nan), color=col, lw=0.7)
        else:
            ax.plot(centres, np.where(ok, full, np.nan), color=col, lw=0.7)
            last = np.where(ok)[0]
            for j in (last.min(), last.max()) if last.size else ():
                ax.plot(centres[j], full[j], "v", color=col, ms=2.5)
            ax.plot([-178, -172], [floor, floor], color=col, lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
    ax.set_xlim(-180, 180); ax.set_ylim(-38, 4)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(style, fontsize=9); ax.set_xlabel("Platform rotation angle [deg]", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Injected tone power rel. $0^\\circ$ [dB]", fontsize=8)
fig.tight_layout()
out = Path(__file__).parent / "52_null_styles.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
nb = sum(1 for c in tones if not data[c][1].all())
print(f"{len(tones)} tones; {len(tones)-nb} unbroken (null resolved), {nb} floor-limited")
