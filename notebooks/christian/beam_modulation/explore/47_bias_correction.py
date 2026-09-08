"""Both continuum estimators are systematics-limited, not noise-limited.

45/46 measured 1.6% (flanking median) and 2.5% (DPSS in log power) error at the
comb channels, against a 0.28% radiometer noise. If that error is a static
property of the bandpass, the transmitter-off window measures it and it can be
divided out. Split-half within the off window tests whether it is stable enough
to be worth applying.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

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
n = d.shape[0] // 2
print(f"{d.shape[0]} off integrations, split {n}/{d.shape[0]-n}, {len(tones)} comb channels\n")


def flanking_est(data):
    return np.array([np.nanmedian(data[:, [c + o for o in range(-6, 7) if 3 <= abs(o) <= 6
                                           and clean[c + o]]], 1) for c in tones]).T


def dpss_est(data, tau_ns=100.0):
    xb = freq[band] * 1e6
    y = np.log10(data[:, band])
    basis, _ = dspec.dpss_operator(xb, [0.0], [tau_ns * 1e-9], eigenval_cutoff=[1e-9])
    basis = np.real(basis)
    m = clean[band]
    model = 10 ** (basis @ (np.linalg.pinv(basis[m]) @ y[:, m].T)).T
    return model[:, np.searchsorted(chan[band], tones)]


truth = d[:, tones]
print(" estimator          raw_err   bias-corrected   vs_radiometer   bias spread")
for name, fn in (("flanking median", flanking_est), ("DPSS log 100 ns", dpss_est)):
    est = fn(d)
    raw = np.nanmedian(np.abs(est / truth - 1.0))
    bias = np.nanmedian(est[:n] / truth[:n], axis=0)          # trained on first half
    corr = np.nanmedian(np.abs(est[n:] / bias / truth[n:] - 1.0))   # tested on second
    print(f" {name:<17} {100*raw:6.3f} %   {100*corr:10.3f} %   {corr/RADIOM:10.2f}x   "
          f"{10*np.log10(bias.min()):+.2f} to {10*np.log10(bias.max()):+.2f} dB")

# is the residual after correction random or still structured?
est = flanking_est(d)
bias = np.nanmedian(est[:n] / truth[:n], axis=0)
r = est[n:] / bias / truth[n:] - 1.0
print(f"\n flanking, corrected: per-integration scatter {100*np.nanstd(r):.3f} %, "
      f"time-median residual per channel {100*np.nanmedian(np.abs(np.nanmedian(r, 0))):.3f} %")
print(f" radiometer noise on a 13-channel median estimate ~ {100*RADIOM*np.sqrt(1+1/13.):.3f} %")
