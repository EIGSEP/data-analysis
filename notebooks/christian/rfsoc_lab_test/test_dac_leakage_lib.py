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


def test_detect_comb_recovers_comb():
    spec = _synthetic(floor=1.0, amp=1e6, leak=0.0)
    assert np.array_equal(lib.detect_comb(spec), lib.comb_channels())


def test_load_auto_real_file():
    path = os.path.join(HERE, "noise_floor_15db.h5")
    spec, freqs, n_acc = lib.load_auto(path)
    assert spec.shape == (lib.N_CHAN,)
    assert freqs.shape == (lib.N_CHAN,)
    assert n_acc == 10
    assert np.array_equal(lib.detect_comb(spec), lib.comb_channels())
