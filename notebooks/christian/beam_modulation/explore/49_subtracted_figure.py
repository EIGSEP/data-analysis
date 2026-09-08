"""Draft of the continuum-subtracted figure."""
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "48_subtracted_depths.py").read().split("tones, rows = []")[0])

tones, profs = [], []
for c in chan[(chan % 16 == COMB_RESIDUE) & (freq > 50) & (freq < 200) & finite & ~is_fm]:
    nb = flanking(c)
    rough = np.nanmedian(np.abs(np.diff(binned(to_db(np.nanmedian(d4[:, nb], 1)), A, B)[0], 2)))
    if rough >= MAX_ROUGHNESS:
        continue
    cont = np.nanmedian(d4[:, nb], axis=1)
    ex_mu, ex_sd = binned(d4[:, c] - cont, A, B, flag=False)
    ok = ex_mu > NSIG * ex_sd
    if not ok[ZERO] or ok.sum() < 20:
        continue
    tones.append(c)
    profs.append(np.where(ok, to_db(np.where(ok, ex_mu, np.nan) / ex_mu[ZERO]), np.nan))

tone_freq = freq[np.array(tones)]
norm = Normalize(tone_freq.min(), tone_freq.max())
smap = ScalarMappable(norm, plt.cm.plasma)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for c, p in zip(tones, profs):
    ax.plot(centres, p, color=smap.to_rgba(freq[c]), lw=0.7, alpha=0.95)
ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
ax.set_xlim(-180, 180)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.grid(alpha=0.25, lw=0.4)
ax.set_xlabel("Platform rotation angle [deg]", fontsize=8)
ax.set_ylabel("Injected tone power rel. $0^\\circ$ [dB]", fontsize=8)
ax.tick_params(labelsize=7)
cb = fig.colorbar(smap, ax=ax, pad=0.02)
cb.set_label("Frequency [MHz]", fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).parent / "49_subtracted.png", dpi=200, bbox_inches="tight")
d = np.array([-np.nanmin(p) for p in profs])
print(f"{len(tones)} tones {tone_freq.min():.1f}-{tone_freq.max():.1f} MHz; "
      f"depth {d.min():.1f}-{d.max():.1f} dB, median {np.median(d):.1f}")
