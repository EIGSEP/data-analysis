"""How accurately can the continuum under a comb channel be predicted?

The comb transmitter was off until ~18:14 UTC on 2026-07-17, so in the earlier
files the comb channels contain continuum only and their true value is known.
That is the exact prediction task the subtraction needs, with the real mask
geometry (3 masked channels, nearest usable channel two away).

Compares a global DPSS model (hera_filters) against the flanking median used by
the current figure.
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
COMB_RESIDUE, COMB_SPILL = 8, (7, 8, 9, 15, 0, 1)
FM_BAND, FIT_BAND = (86.0, 110.0), (48.0, 235.0)

blocks = []
for n in OFF:
    with h5py.File(DATA / n, "r") as f:
        d = f["data/4"][:].astype(np.float64)
        blocks.append(d)
d = np.concatenate(blocks)
d[d <= 0] = np.nan
nchan = d.shape[1]
chan = np.arange(nchan)
freq = chan * 0.244140625
print(f"transmitter-off data: {d.shape[0]} integrations x {nchan} channels")

in_band = (freq > FIT_BAND[0]) & (freq < FIT_BAND[1])
is_fm = (freq > FM_BAND[0]) & (freq < FM_BAND[1])
clean = in_band & ~np.isin(chan % 16, COMB_SPILL) & ~is_fm & np.isfinite(d).all(0)
tones = chan[(chan % 16 == COMB_RESIDUE) & in_band & ~is_fm & np.isfinite(d).all(0)]

RADIOM = 1.0 / np.sqrt(0.244140625e6 * 0.537)
xb, db, cleanb = freq[in_band] * 1e6, d[:, in_band], clean[in_band]
tb = np.searchsorted(chan[in_band], tones)

# confirm the comb really is off
snr = np.nanmedian(d[:, tones], 0) / np.array(
    [np.nanmedian(np.nanmedian(d[:, [c + o for o in range(-6, 7)
                                     if 3 <= abs(o) <= 6 and clean[c + o]]], 1)) for c in tones])
print(f"tone/continuum with transmitter off: median {10*np.log10(np.median(snr)):+.3f} dB "
      f"(range {10*np.log10(snr.min()):+.2f} to {10*np.log10(snr.max()):+.2f})\n")

print(" estimator                err_at_tone_chan   vs_radiometer")
for tau_ns in (50.0, 100.0, 200.0):
    basis, _ = dspec.dpss_operator(xb, [0.0], [tau_ns * 1e-9], eigenval_cutoff=[1e-9])
    basis = np.real(basis)
    model = (basis @ (np.linalg.pinv(basis[cleanb]) @ db[:, cleanb].T)).T
    err = np.nanmedian(np.abs(model[:, tb] / db[:, tb] - 1.0))
    print(f" DPSS {tau_ns:5.0f} ns ({basis.shape[1]:3d} modes)  {100*err:10.3f} %   {err/RADIOM:10.2f}x")

fl = np.array([np.nanmedian(d[:, [c + o for o in range(-6, 7)
                                  if 3 <= abs(o) <= 6 and clean[c + o]]], 1) for c in tones]).T
err_fl = np.nanmedian(np.abs(fl / d[:, tones] - 1.0))
print(f" flanking median           {100*err_fl:10.3f} %   {err_fl/RADIOM:10.2f}x")
np.savez_compressed(Path(__file__).parent / "gap_validation.npz",
                    tone_freq=freq[tones], err_flanking=err_fl)
