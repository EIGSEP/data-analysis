"""Both transmitter polarizations through the same rotation, same reduction.

Residue 8 and residue 0 are the two polarizations of the 2026 transmitter, on
alternating channels of the comb. At the azimuth of the paper figure ($-90$ deg)
residue 8 is the one aligned with the single-polarization bowtie; residue 0 is
the orthogonal one and couples ~8 dB more weakly (see 54_pol_alignment).
"""
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
d4, chan, freq, centres, el = g["d4"], g["chan"], g["freq"], g["centres"], g["el"]
binned, flanking, excess_profile = g["binned"], g["flanking"], g["excess_profile"]
to_db, is_fm, finite = g["to_db"], g["is_fm"], g["finite"]
A, B, ZERO = g["A"], g["B"], g["ZERO"]
MAX_R, MAX_KINK, NSIG = g["MAX_ROUGHNESS"], g["MAX_KINK"], g["NSIG"]
core = np.abs(centres) < 60
pk = np.zeros(len(el), bool); pk[A:B] = True; pk &= np.abs(el) < 30


def select(res):
    keep = {}
    for c in chan[(chan % 16 == res) & (freq > 50) & (freq < 250) & finite & ~is_fm]:
        nb = flanking(c)
        if not nb:
            continue
        rough = np.nanmedian(np.abs(np.diff(binned(to_db(np.nanmedian(d4[:, nb], 1)), A, B), 2)))
        if rough >= MAX_R:
            continue
        prof, ok = excess_profile(c)
        if not ok[ZERO] or ok.sum() < 20:
            continue
        if np.nanmax(np.abs(np.diff(prof[core], 2))) > MAX_KINK:
            continue
        keep[c] = (prof, ok)
    return keep


sets = {8: select(8), 0: select(0)}
allf = np.concatenate([freq[list(v)] for v in sets.values()])
smap = ScalarMappable(Normalize(allf.min(), allf.max()), plt.cm.plasma)

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
titles = {8: "residue 8 — aligned polarization", 0: "residue 0 — orthogonal polarization"}
for ax, res in zip(axes, (8, 0)):
    for c, (prof, _) in sets[res].items():
        ax.plot(centres, prof, color=smap.to_rgba(freq[c]), lw=0.7, alpha=0.95)
    snr = np.median([to_db(np.nanmedian(d4[pk][:, c])) - to_db(np.nanmedian(d4[pk][:, flanking(c)]))
                     for c in sets[res]])
    ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
    ax.set_xlim(-180, 180); ax.set_ylim(-30, 4)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(f"{titles[res]}\n{len(sets[res])} tones, median tone/continuum {snr:.1f} dB",
                 fontsize=9)
    ax.set_xlabel("Platform rotation angle [deg]", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Injected tone power rel. $0^\\circ$ [dB]", fontsize=8)
cb = fig.colorbar(smap, ax=axes, pad=0.02)
cb.set_label("Frequency [MHz]", fontsize=8); cb.ax.tick_params(labelsize=7)
out = Path(__file__).parent / "56_both_combs.png"
fig.savefig(out, dpi=200, bbox_inches="tight")

for res in (8, 0):
    d = np.array([-np.nanmin(np.where(ok, p, np.nan))
                  for p, ok in sets[res].values()])  # measured part only
    fr = freq[list(sets[res])]
    print(f"residue {res}: {len(d):2d} tones, {fr.min():.1f}-{fr.max():.1f} MHz, "
          f"depth {d.min():.1f}-{d.max():.1f} dB (median {np.median(d):.1f})")
print(f"wrote {out}")
