"""Figure for the method choice: why DPSS loses to the local median here.

Transmitter-off data (17:40-17:48 UTC), where the comb channels carry continuum
only, so the prediction can be checked against truth.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/christian/Documents/research/hera_filters")
from hera_filters import dspec  # noqa: E402

DATA = Path("/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered")
OFF = ["corr_20260717_174008Z.h5", "corr_20260717_174217Z.h5", "corr_20260717_174426Z.h5",
       "corr_20260717_174635Z.h5", "corr_20260717_174844Z.h5"]
COMB_RESIDUE, COMB_SPILL, FM_BAND = 8, (7, 8, 9, 15, 0, 1), (86.0, 110.0)

d = np.concatenate([h5py.File(DATA / n, "r")["data/4"][:].astype(np.float64) for n in OFF])
d[d <= 0] = np.nan
chan = np.arange(d.shape[1])
freq = chan * 0.244140625
RADIOM = 1.0 / np.sqrt(0.244140625e6 * 0.537)
finite = np.isfinite(d).all(0)
is_fm = (freq > FM_BAND[0]) & (freq < FM_BAND[1])
band = (freq > 48.0) & (freq < 235.0) & finite
clean = band & ~np.isin(chan % 16, COMB_SPILL) & ~is_fm
tones = chan[band & (chan % 16 == COMB_RESIDUE) & ~is_fm]
spec = np.nanmedian(d, axis=0)


def dpss_model(logfit, tau_ns=100.0):
    xb = freq[band] * 1e6
    y = np.log10(d[:, band]) if logfit else d[:, band]
    basis, _ = dspec.dpss_operator(xb, [0.0], [tau_ns * 1e-9], eigenval_cutoff=[1e-9])
    basis = np.real(basis)
    m = clean[band]
    mod = (basis @ (np.linalg.pinv(basis[m]) @ y[:, m].T)).T
    return 10 ** mod if logfit else mod


def flank(c):
    return [c + o for o in range(-6, 7) if 3 <= abs(o) <= 6 and clean[c + o]]


ti = np.searchsorted(chan[band], tones)
truth = d[:, tones]
curves = {
    "DPSS, linear power": np.nanmedian(np.abs(dpss_model(False)[:, ti] / truth - 1), 0),
    "DPSS, log power": np.nanmedian(np.abs(dpss_model(True)[:, ti] / truth - 1), 0),
    "flanking median": np.nanmedian(np.abs(
        np.array([np.nanmedian(d[:, flank(c)], 1) for c in tones]).T / truth - 1), 0),
}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
for lab, y in curves.items():
    a1.semilogy(freq[tones], 100 * y, "o-", ms=3, lw=1, label=lab)
a1.axhline(100 * RADIOM, color="0.4", ls="--", lw=1, label="radiometer noise")
a1.set_xlabel("Frequency [MHz]"); a1.set_ylabel("Error at comb channel [%]")
a1.set_title("Predicting a masked comb channel\n(transmitter off, truth known)", fontsize=9)
a1.legend(fontsize=7); a1.grid(alpha=0.3)

mod_log = np.nanmedian(dpss_model(True), 0)
mi = np.searchsorted(chan[band], chan[band])
res = spec[band] / mod_log - 1.0
cb = clean[band]
a2.plot(freq[band][cb], 100 * res[cb], ".", ms=2.5, color="tab:orange",
        label="DPSS log model, clean channels")
a2.plot(freq[tones], 100 * (spec[tones] / mod_log[np.searchsorted(chan[band], tones)] - 1),
        "o", ms=4, color="tab:red", label="DPSS log model, comb channels")
fl_res = np.array([spec[c] / np.nanmedian(spec[flank(c)]) - 1 for c in tones])
a2.plot(freq[tones], 100 * fl_res, "s", ms=4, mfc="none", color="tab:green",
        label="flanking median, comb channels")
a2.axhline(0, color="0.6", lw=0.6)
a2.set_ylim(-25, 25)
a2.set_xlabel("Frequency [MHz]"); a2.set_ylabel("Residual, data/model - 1 [%]")
a2.set_title("Where the band-spanning fit goes wrong", fontsize=9)
a2.legend(fontsize=7); a2.grid(alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "51_dpss_vs_local.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
for lab, y in curves.items():
    print(f"  {lab:<20} band-median error {100*np.median(y):.2f} %")
