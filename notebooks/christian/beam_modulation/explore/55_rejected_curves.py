"""Every comb tone that the figure throws away, and why."""
import os, runpy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.pop("BEAM_PNG", None)
g = runpy.run_path("/home/christian/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti/docs/render_beam_modulation.py")
d4, chan, freq, centres = g["d4"], g["chan"], g["freq"], g["centres"]
binned, flanking, excess_profile = g["binned"], g["flanking"], g["excess_profile"]
to_db, is_fm, finite = g["to_db"], g["is_fm"], g["finite"]
A, B, ZERO, NSIG = g["A"], g["B"], g["ZERO"], g["NSIG"]
MAX_ROUGHNESS, COMB_RESIDUE = g["MAX_ROUGHNESS"], g["COMB_RESIDUE"]
kept = set(g["tones"])

groups = {"no data": [], "flanking channels RFI-rough": [],
          "tone not significant": [], "FM band, no clean flanks": []}
core = np.abs(centres) < 60
for c in chan[(chan % 16 == COMB_RESIDUE) & (freq > 50) & (freq < 250)]:
    if c in kept:
        continue
    if is_fm[c]:
        # inside the FM band every neighbour is excluded too, so there is no
        # continuum reference and no excess can be formed -- nothing to draw
        groups["FM band, no clean flanks"].append((c, None))
        continue
    if not finite[c] or not flanking(c):
        groups["no data / no clean flanks"].append((c, None)); continue
    prof, ok = excess_profile(c)
    nb = flanking(c)
    rough = np.nanmedian(np.abs(np.diff(binned(to_db(np.nanmedian(d4[:, nb], 1)), A, B), 2)))
    if rough >= MAX_ROUGHNESS:
        groups["flanking channels RFI-rough"].append((c, prof)); continue
    groups["tone not significant"].append((c, prof))

plot = {k: v for k, v in groups.items() if v}
fig, axes = plt.subplots(1, len(plot), figsize=(3.6 * len(plot), 3.3), sharey=True)
axes = np.atleast_1d(axes)
for ax, (name, items) in zip(axes, plot.items()):
    drawn = 0
    for c, prof in items:
        if prof is None or not np.any(np.isfinite(prof)):
            continue
        ax.plot(centres, prof, lw=0.9, label=f"{freq[c]:.0f}")
        drawn += 1
    if drawn == 0:
        ax.text(0.5, 0.5, "no continuum reference,\nnothing to plot:\n"
                + ", ".join(f"{freq[c]:.0f}" for c, _ in items) + " MHz",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
    ax.set_xlim(-180, 180); ax.set_ylim(-35, 8)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(f"{name}\n({len(items)} tones)", fontsize=9)
    ax.set_xlabel("Platform rotation angle [deg]", fontsize=8)
    ax.tick_params(labelsize=7)
    if drawn:
        ax.legend(fontsize=6, ncol=2, title="MHz", title_fontsize=6, loc="lower center")
axes[0].set_ylabel("Injected tone power rel. $0^\\circ$ [dB]", fontsize=8)
nodata = sum(1 for v in groups.values() for _, p in v
             if p is None or not np.any(np.isfinite(p)))
fig.suptitle(f"{len(kept)} of {len(kept) + sum(len(v) for v in groups.values())} in-band comb "
             f"channels kept; {nodata} rejected channels have no continuum reference and so no curve", fontsize=9, y=1.04)
fig.tight_layout()
out = Path(__file__).parent / "55_rejected_curves.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"kept {len(kept)} tones; rejected:")
for k, v in groups.items():
    fr = ", ".join(f"{freq[c]:.0f}" for c, _ in v)
    print(f"  {k:<28} {len(v):2d}   {fr}")
print(f"wrote {out}")
