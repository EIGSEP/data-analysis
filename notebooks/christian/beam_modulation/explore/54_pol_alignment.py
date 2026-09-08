"""Which comb is aligned with the bowtie at the azimuth of the figure?

The transmitter is dual-polarization and its two polarizations sit on alternating
comb channels: residue 8 and residue 0. The suspended bowtie is single
polarization, so as the platform azimuth steps between rotations the coupling to
the two transmitter polarizations should trade off. If instead one comb is
uniformly weaker at every azimuth, the difference is transmit power, not
alignment.
"""
import os, runpy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.pop("BEAM_PNG", None)
g = runpy.run_path("/home/christian/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti/docs/render_beam_modulation.py")
d4, chan, freq, el, az = g["d4"], g["chan"], g["freq"], g["el"], g["az"]
to_db, is_fm, finite = g["to_db"], g["is_fm"], g["finite"]
rotations, rot_az, ROT = g["rotations"], g["rot_az"], g["ROT"]
COMB_SPILL = g["COMB_SPILL"]


def flank(c):
    return [c + o for o in range(-6, 7) if 3 <= abs(o) <= 6 and 0 <= c + o < len(chan)
            and finite[c + o] and (c + o) % 16 not in COMB_SPILL and not is_fm[c + o]]


sets = {}
for res in (8, 0):
    sets[res] = [c for c in chan[(chan % 16 == res) & (freq > 56) & (freq < 221)]
                 if finite[c] and not is_fm[c] and flank(c)]
    print(f"residue {res}: {len(sets[res])} candidate tones")

rows = []
for r, (a, b) in enumerate(rotations):
    pk = np.zeros(len(el), bool)
    pk[a:b] = True
    pk &= np.abs(el) < 30.0
    if pk.sum() < 5:
        continue
    vals = {}
    for res, cs in sets.items():
        vals[res] = np.median([to_db(np.nanmedian(d4[pk][:, c]))
                               - to_db(np.nanmedian(d4[pk][:, flank(c)])) for c in cs])
    rows.append((r, rot_az[r], vals[8], vals[0]))

rows = np.array(rows)
good = rows[rows[:, 0] <= 29]          # well-behaved first half of the raster
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(good[:, 1], good[:, 2], "o-", ms=3, lw=1, label="residue 8")
ax.plot(good[:, 1], good[:, 3], "s-", ms=3, lw=1, label="residue 0")
ax.axvline(-90, color="0.5", ls="--", lw=1, label="azimuth of the figure")
ax.set_xlabel("Platform azimuth [deg]")
ax.set_ylabel("Median tone / continuum at peak [dB]")
ax.set_title("Coupling of the two transmitter polarizations vs azimuth", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
out = Path(__file__).parent / "54_pol_alignment.png"
fig.savefig(out, dpi=200, bbox_inches="tight")

i = int(np.argmin(np.abs(good[:, 1] - (-90.0))))
print(f"\nat the figure's azimuth ({good[i,1]:+.0f} deg): "
      f"residue 8 = {good[i,2]:.1f} dB, residue 0 = {good[i,3]:.1f} dB "
      f"-> residue {'8' if good[i,2] > good[i,3] else '0'} is stronger by "
      f"{abs(good[i,2]-good[i,3]):.1f} dB")
print(f"across azimuth: residue 8 spans {good[:,2].min():.1f}-{good[:,2].max():.1f} dB, "
      f"residue 0 spans {good[:,3].min():.1f}-{good[:,3].max():.1f} dB")
print(f"difference (8 - 0) spans {np.min(good[:,2]-good[:,3]):+.1f} to "
      f"{np.max(good[:,2]-good[:,3]):+.1f} dB, "
      f"correlation with azimuth {np.corrcoef(good[:,1], good[:,2]-good[:,3])[0,1]:+.2f}")
print(f"wrote {out}")
