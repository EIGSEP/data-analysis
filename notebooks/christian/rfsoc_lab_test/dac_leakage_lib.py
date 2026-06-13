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
