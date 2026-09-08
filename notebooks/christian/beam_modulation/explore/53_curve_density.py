"""Do we need the solid/faint distinction, or just fewer curves?"""
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
tones, freq, centres, profs = g["tones"], g["freq"], g["centres"], g["profs"]

tf = freq[np.array(tones)]
smap = ScalarMappable(Normalize(tf.min(), tf.max()), plt.cm.plasma)
few = tones[::4]

panels = [("all 27, no threshold", tones, False, False),
          ("all 27, stop at threshold", tones, True, False),
          (f"{len(few)} tones, no threshold", few, False, False),
          (f"{len(few)} tones, solid + faint", few, True, True)]

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1), sharey=True)
for ax, (title, sel, trunc, faint) in zip(axes, panels):
    for c in sel:
        full, ok = profs[c]
        col = smap.to_rgba(freq[c])
        if faint:
            ax.plot(centres, full, color=col, lw=0.8, alpha=0.25)
        ax.plot(centres, np.where(ok, full, np.nan) if trunc else full,
                color=col, lw=0.8, alpha=0.95)
    ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
    ax.set_xlim(-180, 180); ax.set_ylim(-35, 4)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Platform rotation angle [deg]", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Injected tone power rel. $0^\\circ$ [dB]", fontsize=8)
fig.tight_layout()
out = Path(__file__).parent / "53_curve_density.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}; subset = {', '.join(f'{freq[c]:.0f}' for c in few)} MHz")
