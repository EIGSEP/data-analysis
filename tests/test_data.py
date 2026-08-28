"""Tests for eigsep_data.data."""

import warnings

import numpy as np

from eigsep_data import data


def _raster(n_plat=6, plat_len=20, ramp_len=3, step_deg=10.0):
    """Synthetic az raster: stable plateaus joined by short transitions."""
    seg, az_pot = [], []
    for k in range(n_plat):
        seg += [2 * k] * plat_len
        az_pot += list(
            np.full(plat_len, step_deg * k)
            + np.random.default_rng(k).normal(0, 0.01, plat_len)
        )
        if k < n_plat - 1:
            seg += [2 * k + 1] * ramp_len
            az_pot += list(
                np.linspace(step_deg * k, step_deg * (k + 1), ramp_len + 2)[
                    1:-1
                ]
            )
    return np.array(az_pot, dtype=float), np.array(seg, dtype=float)


class TestExtractCleanPotDataV2:
    """A dropped pot sample must not take its neighbours with it."""

    def test_single_nan_does_not_destroy_plateau(self):
        # np.median propagates NaN, and the median is broadcast over the
        # whole plateau and then into the ramps on both sides -- one bad
        # sample used to cost ~26.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[30] = np.nan
        out = data.extract_clean_pot_data_v2(y, az_step)
        assert not np.isnan(out).any()

    def test_partial_dropout_run_recovers(self):
        # Real Jul-17 dropout runs span many samples but rarely a whole
        # plateau; those must still yield a usable median.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[23:40] = np.nan  # 17 of the 20 samples in one plateau
        out = data.extract_clean_pot_data_v2(y, az_step)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out[23:43], 10.0, atol=0.1)

    def test_nan_at_edges_does_not_flood_array(self):
        # The first/last plateau median fills everything before/after it.
        az_pot, az_step = _raster()
        for idx in (5, len(az_pot) - 5):
            y = az_pot.copy()
            y[idx] = np.nan
            out = data.extract_clean_pot_data_v2(y, az_step)
            assert not np.isnan(out).any()

    def test_fully_missing_plateau_stays_nan_but_is_contained(self):
        # No measurement exists here, so it must stay flagged missing
        # rather than being interpolated across -- but it must not
        # spread beyond its own samples.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[23:43] = np.nan
        out = data.extract_clean_pot_data_v2(y, az_step)
        assert np.isnan(out[23:43]).all()
        assert np.isfinite(out[:23]).all()
        assert np.isfinite(out[43:]).all()

    def test_fully_missing_plateau_emits_no_warning(self):
        # np.nanmedian over an all-NaN slice warns; the plateau must be
        # skipped before that happens.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[23:43] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data.extract_clean_pot_data_v2(y, az_step)

    def test_clean_data_unchanged(self):
        az_pot, az_step = _raster()
        base = data.extract_clean_pot_data_v2(az_pot, az_step)
        assert not np.isnan(base).any()
        np.testing.assert_allclose(base[:20], 0.0, atol=0.1)
        np.testing.assert_allclose(base[23:43], 10.0, atol=0.1)

    def test_untouched_plateaus_bit_identical(self):
        # A NaN in one plateau must not perturb any other.
        az_pot, az_step = _raster()
        base = data.extract_clean_pot_data_v2(az_pot, az_step)
        y = az_pot.copy()
        y[30] = np.nan
        out = data.extract_clean_pot_data_v2(y, az_step)
        np.testing.assert_array_equal(out[:20], base[:20])
        np.testing.assert_array_equal(out[46:], base[46:])
