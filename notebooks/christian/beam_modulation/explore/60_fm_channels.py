"""Can the FM-band comb channels be recovered, or is the tone simply buried?

Two questions. (i) Is the injected tone detectable at all in the FM band? The
transmitter was off before ~18:14 UTC, so comparing the comb channel against its
local neighbours with the transmitter off and again during the raster separates
"tone present but continuum hard to estimate" from "tone buried under FM".
(ii) What do the raw rotation profiles look like there?
"""
import os, runpy
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.pop("BEAM_PNG", None)
g = runpy.run_path("/home/christian/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti/docs/render_beam_modulation.py")
d4, chan, freq, centres = g["d4"], g["chan"], g["freq"], g["centres"]
binned, to_db, is_fm, A, B, ZERO = g["binned"], g["to_db"], g["is_fm"], g["A"], g["B"], g["ZERO"]
el, COMB_SPILL = g["el"], g["COMB_SPILL"]

DATA = Path("/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered")
OFF = ["corr_20260717_174008Z.h5", "corr_20260717_174217Z.h5", "corr_20260717_174426Z.h5",
       "corr_20260717_174635Z.h5", "corr_20260717_174844Z.h5"]
off = np.concatenate([h5py.File(DATA / n, "r")["data/4"][:].astype(np.float64) for n in OFF])
off[off <= 0] = np.nan

fm_tones = chan[(chan % 16 == 8) & is_fm]
ref_tones = [c for c in g["tones"] if 110 < freq[c] < 145][:4]

def local(c, arr, allow_fm):
    nb = [c + o for o in range(-6, 7) if 3 <= abs(o) <= 6 and 0 <= c + o < len(chan)
          and (c + o) % 16 not in COMB_SPILL and (allow_fm or not is_fm[c + o])]
    return np.nanmedian(np.nanmedian(arr[:, nb], axis=1))

pk = np.zeros(len(el), bool); pk[A:B] = True; pk &= np.abs(el) < 30
print("  freq      tone/local, OFF     tone/local, ON      rise when transmitter on")
for c in list(fm_tones) + ref_tones:
    a = to_db(np.nanmedian(off[:, c])) - to_db(local(c, off, True))
    b = to_db(np.nanmedian(d4[pk][:, c])) - to_db(local(c, d4[pk], True))
    tag = "  FM" if is_fm[c] else "  ref"
    print(f" {freq[c]:6.1f}  {a:14.2f} dB {b:16.2f} dB {b-a:19.2f} dB{tag}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
for c in fm_tones:
    a1.plot(centres, binned(to_db(d4[:, c]), A, B) - np.nanmedian(binned(to_db(d4[:, c]), A, B)[ZERO-1:ZERO+2]),
            lw=1.0, label=f"{freq[c]:.0f}")
a1.set_title("FM-band comb channels, raw", fontsize=9); a1.legend(fontsize=6, ncol=2)
for c in ref_tones:
    a2.plot(centres, binned(to_db(d4[:, c]), A, B) - np.nanmedian(binned(to_db(d4[:, c]), A, B)[ZERO-1:ZERO+2]),
            lw=1.0, label=f"{freq[c]:.0f}")
a2.set_title("kept tones just above the FM band, raw", fontsize=9); a2.legend(fontsize=6)
for ax in (a1, a2):
    ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180]); ax.grid(alpha=0.3)
    ax.set_xlabel("Platform rotation angle [deg]", fontsize=8); ax.tick_params(labelsize=7)
a1.set_ylabel("Raw power rel. $0^\\circ$ [dB]", fontsize=8)
fig.tight_layout()
out = Path(__file__).parent / "60_fm_channels.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nwrote {out}")
