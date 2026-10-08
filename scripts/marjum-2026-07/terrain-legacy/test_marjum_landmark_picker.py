import json

import numpy as np

from marjum_landmark_picker import LandmarkPicker


def make_picker(tmp_path, distortion=True):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "IMG_0001.HEIC").touch()
    values = dict(
        keys=np.array(["0001"]),
        cameras=np.array([[0, 0, 0, 0, 0, 0, 1000.0]]),
        shapes=np.array([[1000, 1200]]),
        antenna=np.array([0.0, 0.0, 100.0]),
    )
    if distortion:
        values["distortion"] = np.array([[0.03, -0.01]])
    fit = tmp_path / "fit.npz"
    np.savez(fit, **values)
    landmarks = {
        "antenna": {
            "pixel_field": "ant_px", "fit_file": fit,
            "position_field": "antenna",
        },
        "transmitter": {
            "pixel_field": "transmitter_px", "fit_file": None,
            "position_field": "transmitter",
        },
    }
    return LandmarkPicker(image_dir, fit, landmarks,
                          tmp_path / "meta.json", radius_m=3)


def test_physical_crop_and_missing_landmark_fallback(tmp_path):
    picker = make_picker(tmp_path)
    bounds, center, reason = picker.expected_crop(
        "0001", "antenna", (1000, 1200))
    assert np.allclose(center, [600, 500])
    assert bounds[0] < center[0] < bounds[1]
    assert bounds[2] < center[1] < bounds[3]
    assert "±3 m" in reason
    bounds, center, reason = picker.expected_crop(
        "0001", "transmitter", (1000, 1200))
    assert bounds is None and center is None
    assert "no current physical solution" in reason


def test_absent_distortion_defaults_to_zero(tmp_path):
    picker = make_picker(tmp_path, distortion=False)
    assert np.array_equal(picker.distortion["0001"], [0, 0])
    assert picker.expected_crop("0001", "antenna", (1000, 1200))[0] is not None


def test_landmarks_are_saved_independently(tmp_path):
    picker = make_picker(tmp_path)
    picker.set_pixel("0001", "antenna", (10, 20))
    picker.set_pixel("0001", "transmitter", (30, 40))
    saved = json.loads((tmp_path / "meta.json").read_text())
    assert saved["0001"]["ant_px"] == [10.0, 20.0]
    assert saved["0001"]["transmitter_px"] == [30.0, 40.0]
