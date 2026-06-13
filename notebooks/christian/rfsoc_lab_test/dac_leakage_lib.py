"""Analysis helpers for the RFSoC DAC spectral-leakage measurement.

The RFSoC emits a comb of tones (every 16 channels). Combined with
injected noise it is recorded as an auto-correlation in ``data/3``.
These helpers locate the comb, characterize the noise floor / dynamic
range, and stack the tone wings to measure spillover into neighbouring
channels.
"""

import h5py
import numpy as np

N_CHAN = 1024
COMB_START = 128
COMB_STEP = 16
COMB_STOP = 944  # inclusive
BAND = (120, 950)  # in-band channels used for floor statistics
DF_MHZ = 250.0 / N_CHAN  # channel width, ~0.2441 MHz


def comb_channels(start=COMB_START, step=COMB_STEP, stop=COMB_STOP):
    """Channel indices of the comb tones (``stop`` inclusive)."""
    return np.arange(start, stop + 1, step)


def load_auto(path, key="3"):
    """Time-averaged auto-correlation from one HDF5 file.

    Returns ``(spec, freqs, n_acc)`` with ``spec`` shape ``(1024,)``.
    """
    with h5py.File(path, "r") as f:
        data = f[f"data/{key}"][:]  # (n_acc, 1024)
        freqs = f["header/freqs"][:]
    return data.mean(axis=0), freqs, data.shape[0]


def in_band_mask(n_chan=N_CHAN, band=BAND):
    """Boolean mask, True for channels inside ``band``."""
    lo, hi = band
    mask = np.zeros(n_chan, dtype=bool)
    mask[lo:hi] = True
    return mask


def detect_comb(spec, rel_threshold=100.0, band=BAND):
    """Channels exceeding ``rel_threshold`` x the in-band median.

    Verified to return exactly ``comb_channels()`` on all four files.
    """
    band_mask = in_band_mask(spec.size, band)
    med = np.median(spec[band_mask])
    return np.where((spec > rel_threshold * med) & band_mask)[0]
