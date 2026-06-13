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


def test_floor_amplitude_dynamic_range():
    spec = _synthetic(floor=500.0, amp=1e8, leak=0.0)
    tones = lib.comb_channels()
    assert lib.noise_floor(spec, tones) == pytest.approx(500.0)
    assert lib.tone_amplitude(spec, tones) == pytest.approx(1e8)
    assert lib.dynamic_range(spec, tones) == pytest.approx(2e5)


def test_stacked_profile_recovers_leakage():
    leak = 3e-4
    spec = _synthetic(floor=0.0, amp=1e8, leak=leak, hw_leak=1)
    tones = lib.comb_channels()
    offsets, mean, err = lib.stacked_profile(spec, tones, half_width=8)
    c = int(np.where(offsets == 0)[0][0])
    assert mean[c] == pytest.approx(1.0)
    assert mean[c - 1] == pytest.approx(leak)
    assert mean[c + 1] == pytest.approx(leak)
    assert mean[c + 2] == pytest.approx(0.0)
    assert mean[c - 2] == pytest.approx(0.0)
    assert offsets[0] == -8 and offsets[-1] == 8
    assert err.shape == mean.shape
