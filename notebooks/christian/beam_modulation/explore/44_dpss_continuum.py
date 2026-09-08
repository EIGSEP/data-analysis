"""DPSS continuum model for the injected-tone channels.

The published figure plots raw channel power, which is tone + sky continuum, so
each curve flattens onto the continuum pedestal instead of following the beam
down. This models the continuum with a DPSS basis (hera_filters) fitted to the
non-comb channels and evaluated at the tone channels, so the tone excess can be
isolated.

Validation here: (i) which delay half-width to use, (ii) hold-out error on clean
channels compared with the radiometer noise, (iii) resulting modulation depth
against the raw and flanking-median reductions.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/home/christian/Documents/research/hera_filters")
from hera_filters import dspec  # noqa: E402

SIDECAR = Path("/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment5/motor_scan_20260717_key4.h5")

COMB_RESIDUE, COMB_SPILL = 8, (7, 8, 9, 15, 0, 1)
FM_BAND, FIT_BAND = (86.0, 110.0), (48.0, 235.0)

with h5py.File(SIDECAR, "r") as f:
    d4 = f["data4"][:].astype(np.float64)
    freq = f["freqs"][:]
    el = f["el_counts"][:] * (1.0 / f.attrs["counts_per_deg"])
    az = f["az_counts"][:] * (1.0 / f.attrs["counts_per_deg"])

d4[d4 <= 0] = np.nan
nchan = d4.shape[1]
chan = np.arange(nchan)

in_band = (freq > FIT_BAND[0]) & (freq < FIT_BAND[1])
is_comb = np.isin(chan % 16, COMB_SPILL)
is_fm = (freq > FM_BAND[0]) & (freq < FM_BAND[1])
clean0 = in_band & ~is_comb & ~is_fm & np.isfinite(d4).all(0)

print(f"channels: {nchan} total, {in_band.sum()} in fit band, {clean0.sum()} clean for fitting")
print(f"channel width {1e3*(freq[1]-freq[0]):.2f} kHz; comb period {16*(freq[1]-freq[0]):.3f} MHz "
      f"-> mask alias delay {1e3/(16*(freq[1]-freq[0])):.0f} ns")

x = freq * 1e6  # Hz

# --- convention check: dpss_operator half-widths in ns or s? mode count tells us.
for unit, val in (("as-seconds", 100e-9), ("as-nanosec", 100.0)):
    try:
        B, n = dspec.dpss_operator(x[in_band], [0.0], [val], eigenval_cutoff=[1e-9])
        print(f"  half_width passed {unit:<11} -> {B.shape[1]:5d} modes")
    except Exception as e:  # noqa: BLE001
        print(f"  half_width passed {unit:<11} -> failed: {type(e).__name__}")

# half-widths are in seconds when x is in Hz
DUMP_S, CHAN_HZ = 0.537, (freq[1] - freq[0]) * 1e6
RADIOM = 1.0 / np.sqrt(CHAN_HZ * DUMP_S)
print(f"\nradiometer fractional noise, one dump one channel: {100*RADIOM:.3f} %")

xb, db = x[in_band], d4[:, in_band]
cleanb = clean0[in_band]
rng = np.random.default_rng(0)


def fit_continuum(tau_s, fit_mask, data=db):
    """DPSS continuum model for every integration, fitted on fit_mask channels."""
    basis, _ = dspec.dpss_operator(xb, [0.0], [tau_s], eigenval_cutoff=[1e-9])
    basis = np.real(basis)
    pinv = np.linalg.pinv(basis[fit_mask])
    coeffs = pinv @ data[:, fit_mask].T          # (nmode, ntime)
    return basis @ coeffs, basis.shape[1]        # (nchan_band, ntime)


print("\n tau_ns  modes   holdout_err   vs_radiometer   resid_on_fit")
best = None
for tau_ns in (30.0, 50.0, 100.0, 150.0, 200.0):
    idx = np.where(cleanb)[0]
    held = rng.choice(idx, size=idx.size // 5, replace=False)
    fit_mask = cleanb.copy()
    fit_mask[held] = False
    model, nm = fit_continuum(tau_ns * 1e-9, fit_mask)
    err = np.nanmedian(np.abs(model[held].T / db[:, held] - 1.0))
    res = np.nanmedian(np.abs(model[fit_mask].T / db[:, fit_mask] - 1.0))
    print(f" {tau_ns:6.0f}  {nm:5d}   {100*err:9.3f} %   {err/RADIOM:11.2f}x   {100*res:8.3f} %")
    if best is None or err < best[1]:
        best = (tau_ns, err)

TAU_NS = 100.0
print(f"\nusing tau = {TAU_NS:.0f} ns (below the {1e3/(16*(freq[1]-freq[0])):.0f} ns mask-alias delay)")
