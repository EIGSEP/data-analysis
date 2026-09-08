"""Does the secondary dip at the top of the band evolve smoothly with frequency?

A smoothness cut on each tone's own profile was briefly used to reject 224.6 MHz,
on the assumption that its split null was a channel glitch. Plotting the whole
top of the band together shows the same secondary dip near -45 deg in every tone
from 201 to 240 MHz, deepening progressively with frequency, so it is structure
in the response and the cut was discarding it. The cut was removed.
"""
import os
import runpy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.pop("BEAM_PNG", None)
g = runpy.run_path("/home/christian/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti/docs/render_beam_modulation.py")
chan, freq, centres = g["chan"], g["freq"], g["centres"]
excess_profile, finite, is_fm = g["excess_profile"], g["finite"], g["is_fm"]
kept = set(g["tones"])

top = [c for c in chan[(chan % 16 == 8) & (freq > 200) & (freq < 245)]
       if finite[c] and not is_fm[c]]
fig, ax = plt.subplots(figsize=(7, 3.6))
for c in top:
    prof, _ = excess_profile(c)
    ax.plot(centres, prof, "-" if c in kept else "--", lw=1.1,
            label=f"{freq[c]:.0f} MHz{'' if c in kept else '  (rejected)'}")
ax.set_xlim(-180, 180)
ax.set_ylim(-30, 5)
ax.grid(alpha=0.3)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xlabel("Platform rotation angle [deg]")
ax.set_ylabel("rel. $0^\\circ$ [dB]")
ax.set_title("Top of the band: does the split null evolve smoothly?", fontsize=10)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
out = Path(__file__).parent / "57_top_band.png"
fig.savefig(out, dpi=200, bbox_inches="tight")

core = np.abs(centres) < 60
for c in top:
    prof, _ = excess_profile(c)
    print(f"  {freq[c]:6.1f} MHz  {'kept' if c in kept else 'rejected':>8}  "
          f"max kink {np.nanmax(np.abs(np.diff(prof[core], 2))):.1f} dB")
print(f"wrote {out}")
