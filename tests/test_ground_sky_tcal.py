"""Binned load-temperature windows through the campaign consumer."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from eigsep_data.products.tcal import load_temperature


@pytest.fixture
def consumer(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "marjum-2026-07"
        / "ground_sky"
        / "common.py"
    )
    spec = importlib.util.spec_from_file_location("ground_sky_common", path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    calls = []

    class Calibration:
        def _solutions(self, campaign, version):
            return {"keys": ["4"], "freqs": np.arange(12.0)}

        def fetch(self, *args, **kwargs):
            t = np.asarray(args[-1])
            window = kwargs.get("t_load_window")
            calls.append((t.copy(), kwargs))
            # Actual approved API: inclusive finite means and nearest fallback.
            load, _, _ = load_temperature(
                np.arange(20.0),
                300.0 + np.arange(20.0) ** 2,
                t,
                tol_s=1.0,
                window=window,
            )
            return {
                "scale": np.full((t.size, 12), 2.0),
                "offset": np.repeat(load[:, None], 12, axis=1),
            }

    monkeypatch.setattr(
        "eigsep_data.products.base.get", lambda _: Calibration()
    )
    return common, calls


def test_default_keeps_nearest_rule_and_omits_window(consumer, tmp_path):
    common, calls = consumer
    s, o, f = common.tcal_coefficients([2.0, 4.0, 6.0], "v0004", tmp_path, 2)
    np.testing.assert_array_equal(o[:, 0], [304.0, 316.0, 336.0])
    assert all(kw == {} for _, kw in calls)
    assert s.shape == (3, 12) and f.shape == (12,)


def test_windows_keep_alignment_across_chunks(consumer, tmp_path):
    common, calls = consumer
    t = np.arange(2.0, 12.0, 2.0)
    edges = (t - 1.0, t + 1.0)
    _, o, _ = common.tcal_coefficients(
        t, "v0004", tmp_path, 2, t_load_window=edges
    )
    # Includes both edge readings; excludes the readings just outside them.
    np.testing.assert_allclose(o[:, 0], 300.0 + t**2 + 2.0 / 3.0)
    for edge, expected in zip(
        zip(*(kw["t_load_window"] for _, kw in calls)), edges
    ):
        np.testing.assert_array_equal(np.concatenate(edge), expected)
    assert [len(times) for times, _ in calls] == [2, 2, 1]


def test_scalar_edges_broadcast_before_chunking(consumer, tmp_path):
    common, _ = consumer
    _, o, _ = common.tcal_coefficients(
        [2.0, 4.0, 6.0], "v0004", tmp_path, 2, t_load_window=(1.0, 3.0)
    )
    np.testing.assert_allclose(o[:, 0], 300.0 + 14.0 / 3.0)


def test_bad_edge_shape_fails_before_fetch(consumer, tmp_path):
    common, calls = consumer
    with pytest.raises(ValueError):
        common.tcal_coefficients(
            [2.0, 4.0, 6.0],
            "v0004",
            tmp_path,
            2,
            t_load_window=([1.0, 2.0], [3.0, 4.0]),
        )
    assert calls == []


def test_empty_and_nan_windows_keep_api_fallback(consumer, tmp_path):
    common, _ = consumer
    _, o, _ = common.tcal_coefficients(
        [2.0, 4.0],
        "v0004",
        tmp_path,
        1,
        t_load_window=([2.1, np.nan], [2.2, np.nan]),
    )
    np.testing.assert_array_equal(o[:, 0], [304.0, 316.0])


def test_calibrate_passes_bin_edges_and_keeps_scale_sigma(consumer, tmp_path):
    common, calls = consumer
    z = {
        "t": np.array([2.0, 4.0]),
        "t0": np.array([1.0, 3.0]),
        "t1": np.array([3.0, 5.0]),
        "chans": np.array([5]),
        "yc": np.full((2, 1, 6), 10.0),
        "nc": np.ones((2, 1, 6)),
        "sem": np.array([[0.1], [0.2]]),
    }
    out = common.calibrate(z, "v0004", tmp_path)
    np.testing.assert_allclose(
        out["T"][:, 0], [324.0 + 2.0 / 3.0, 336.0 + 2.0 / 3.0]
    )
    np.testing.assert_array_equal(out["scale"], [[2.0], [2.0]])
    np.testing.assert_array_equal(out["sigma"], [[0.2], [0.4]])
    for actual, expected in zip(
        calls[-1][1]["t_load_window"], (z["t0"], z["t1"])
    ):
        np.testing.assert_array_equal(actual, expected)
    for name in ("t0", "t1"):
        with pytest.raises(KeyError, match=name):
            common.calibrate(
                {k: v for k, v in z.items() if k != name}, "v0004", tmp_path
            )
