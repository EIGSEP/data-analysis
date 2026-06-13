import os

import numpy as np
import pytest

import dac_leakage_lib as lib

HERE = os.path.dirname(os.path.abspath(__file__))


def _synthetic(floor=0.0, amp=1e8, leak=0.0, hw_leak=1):
    """Spectrum with a comb of amplitude ``amp`` on a flat ``floor``,
    each tone leaking ``leak * amp`` into +/- ``hw_leak`` neighbours."""
    spec = np.full(lib.N_CHAN, float(floor))
    for t in lib.comb_channels():
        spec[t] = amp
        for d in range(1, hw_leak + 1):
            spec[t - d] += leak * amp
            spec[t + d] += leak * amp
    return spec


def test_comb_channels():
    tones = lib.comb_channels()
    assert tones[0] == 128
    assert tones[-1] == 944
    assert len(tones) == 52
    assert np.all(np.diff(tones) == 16)
