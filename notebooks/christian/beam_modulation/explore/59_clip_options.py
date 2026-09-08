"""The figure with the y-axis left to autoscale, against the -33 dB clip."""
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
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))
for ax, clip in zip(axes, (None, (-40, 4), (-37, 4), (-33, 4))):
    for c in tones:
        ax.plot(centres, profs[c][0], color=smap.to_rgba(freq[c]), lw=0.7, alpha=0.95)
    ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
    ax.set_xlim(-180, 180)
    if clip:
        ax.set_ylim(*clip)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlabel("Platform rotation angle [deg]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("autoscale" if clip is None else f"clipped at ${clip[0]}$ dB", fontsize=9)
axes[0].set_ylabel("Injected tone power rel. $0^\\circ$ [dB]", fontsize=8)
fig.tight_layout()
out = Path(__file__).parent / "59_clip_options.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
lo = min(np.nanmin(profs[c][0]) for c in tones)
deepest_sig = min(np.nanmin(np.where(profs[c][1], profs[c][0], np.nan)) for c in tones)
print(f"autoscale reaches {lo:.1f} dB; deepest measured point is {deepest_sig:.1f} dB")
print(f"-> autoscale spends {abs(lo - deepest_sig):.1f} dB of the axis on noise")
print(f"wrote {out}")
