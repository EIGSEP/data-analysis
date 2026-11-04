"""Tests for the v6 pointing-table columns: tilt_deg and el_deg from accel.

Synthetic box sweeps are made with a known sensor misalignment and known
axle tilts, calibrated with picohost's own fit_el_calibration (the routine
that made the Marjum calibration), recorded as correlator files, and run
through build_table.build. Run with::

    python -m pytest notebooks/arp/marjum-2026-07/pointing/tests
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os

import h5py
import numpy as np
import pytest
from picohost import imu_geometry as ig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fuse = _load("fuse", "fuse.py")
extract = _load("extract", "extract.py")
build_table = _load("build_table", "build_table.py")

G = 9.81
CADENCE = extract.KNOWN_CADENCES_S[1]
# The sensor is turned in the box by Rz(beta) Ry(alpha): 3.82 deg in all.
ALPHA, BETA = -2.55, 2.88
# The BNO08x's axis senses relative to the box (notebook 035's primary
# hypothesis: specific force, (sx, sy, sz) = (-1, +1, -1)).
D = np.diag([-1.0, 1.0, -1.0])
# An in-plane accelerometer bias (m/s^2). A bias along the axle is not
# observable from a single-axis sweep (fit_accel_sphere_coplanar pins it
# to zero), so it would enter the tilt directly; it is left out here.
BIAS_INPLANE = np.array([0.0, 0.035, -0.027])
TAUS = (0.0, 2.0, -2.0, 6.0, -6.0)


def _rot_y(a):
    a = np.radians(a)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                     [-np.sin(a), 0, np.cos(a)]])


def _rot_z(a):
    a = np.radians(a)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]])


RMIS = _rot_z(BETA) @ _rot_y(ALPHA)
# Sensor axes in the box frame are the columns of RMIS D; a box vector v
# reads (RMIS D)^T v = D RMIS^T v.
TO_SENSOR = D @ RMIS.T


def up_in_box(el_deg, tau_deg):
    """R_box^T z for R_box = Ry(tau) Rx(el) (TxGeometryModel without az)."""
    el = np.radians(np.atleast_1d(np.asarray(el_deg, float)))
    tau = np.radians(np.broadcast_to(np.asarray(tau_deg, float), el.shape))
    return np.column_stack([-np.sin(tau), np.sin(el) * np.cos(tau),
                            np.cos(el) * np.cos(tau)])


def bias_sensor():
    """BIAS_INPLANE with its along-axle component removed (sensor frame)."""
    axle = TO_SENSOR @ np.array([1.0, 0.0, 0.0])
    return BIAS_INPLANE - (BIAS_INPLANE @ axle) * axle


def reading(el_deg, tau_deg, scale=1.0):
    """Raw accelerometer readings (m/s^2), (n, 3)."""
    return scale * G * up_in_box(el_deg, tau_deg) @ TO_SENSOR.T + bias_sensor()


@pytest.fixture(scope="module")
def calibration():
    """picohost's fit on a 12-stop sweep at a 1.3 deg tilt, motor home 0."""
    motor = np.linspace(-150, 150, 12)
    sec, _ = ig.fit_el_calibration(reading(motor, 1.3), None, motor)
    cal = sec["imu_el"]
    assert 3.0 < cal["mount_misalign_deg"] < 4.5
    return cal


def firmware_el(raw, cal):
    """What the firmware publishes per sample (picohost PicoIMU)."""
    return np.array([ig.el_from_imu(ig.precondition(a, cal["accel_bias"]),
                                    cal["M"]) for a in raw])


# -- functions -------------------------------------------------------------

def test_el_from_unit_is_firmware_el(calibration):
    el = np.linspace(-179.5, 179.5, 721)
    raw = reading(el, 2.0)
    u = fuse.imu_unit(raw, calibration["accel_bias"])
    assert np.allclose(fuse.el_from_unit(u, calibration["M"]),
                       firmware_el(raw, calibration), atol=1e-9)


def test_tilt_recovers_tau(calibration):
    raster = reading(np.linspace(-170, 170, 400), 5.9)
    n = fuse.axle_normal(fuse.imu_unit(raster, calibration["accel_bias"]))
    for tau in TAUS:
        u = fuse.imu_unit(reading(np.linspace(-80, 80, 81), tau),
                          calibration["accel_bias"])
        assert np.abs(fuse.axle_tilt(u, n) - tau).max() < 1e-6


def test_tilt_sign_follows_tilt_sign():
    u = fuse.imu_unit(reading([10.0, 20.0], 3.0), bias_sensor())
    n = TO_SENSOR @ np.array([-1.0, 0.0, 0.0])
    assert n[0] > 0
    assert np.allclose(fuse.axle_tilt(u, n, sign=+1), 3.0)
    assert np.allclose(fuse.axle_tilt(u, n, sign=-1), -3.0)


def test_imu_unit_masks_missing_and_zero():
    a = np.array([[np.nan, 0, 9.8], [0, 0, 0], [0, 0, -9.8]], float)
    u = fuse.imu_unit(a, np.zeros(3))
    assert np.isnan(u[:2]).all()
    assert np.allclose(u[2], [0, 0, -1])


def test_linear_mean_wrap_is_fixed(calibration):
    """Three samples near +-180 deg: the linear el mean breaks, accel not."""
    centre = np.array([179.0, 179.85, -179.95, -179.0])
    el = centre[:, None] + np.array([-0.3, 0.0, 0.3])[None, :]
    raw = reading(el.ravel(), 2.0).reshape(4, 3, 3)
    fw = firmware_el(raw.reshape(-1, 3), calibration).reshape(4, 3)
    linear = fw.mean(axis=1)
    u = fuse.imu_unit(raw.mean(axis=1), calibration["accel_bias"])
    v6 = fuse.el_from_unit(u, calibration["M"])
    truth = firmware_el(reading(centre, 2.0), calibration)
    assert np.abs(fuse.wrap180(v6 - truth)).max() < 1e-3
    straddle = np.abs(fuse.wrap180(linear - truth)) > 1.0
    assert straddle.tolist() == [False, True, True, False]


def test_quasi_static():
    a = np.array([[0, 0, 9.80665], [0, 0, 1.03 * 9.80665],
                  [np.nan, 0, 9.8]])
    assert fuse.quasi_static(a).tolist() == [True, False, False]


def test_calibration_must_be_unique(tmp_path, calibration):
    paths = []
    for k, cal in enumerate(({"imu_el": calibration}, {},
                             {"imu_el": dict(calibration, accel_scale=1.0)})):
        p = tmp_path / f"f{k}.h5"
        with h5py.File(p, "w") as f:
            f["header/imu_calibration"] = json.dumps(cal)
        paths.append(str(p))
    sec, info = extract.imu_el_calibration(paths[:2])
    assert sec == calibration
    assert (info["n_files_with"], info["n_files_without"]) == (1, 1)
    with pytest.raises(ValueError):
        extract.imu_el_calibration(paths)


# -- the table -------------------------------------------------------------

def _ts(text):
    return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc).timestamp()


def _write_file(data_dir, t0, raw, el_fw, cal):
    """One correlator file holding imu_el records only."""
    n = len(raw)
    times = t0 + CADENCE * np.arange(n)
    close = dt.datetime.fromtimestamp(times[-1] + 1.0, dt.timezone.utc)
    name = close.strftime("corr_%Y%m%d_%H%M%SZ.h5")
    recs = []
    for a, e in zip(raw, el_fw):
        if not np.isfinite(a).all():
            recs.append({"status": "error", "accel_x": None,
                         "accel_y": None, "accel_z": None, "el_deg": None})
            continue
        recs.append({"status": "update", "yaw": 0.0, "pitch": 0.0,
                     "roll": 0.0, "accel_x": a[0], "accel_y": a[1],
                     "accel_z": a[2], "el_deg": float(e)})
    with h5py.File(os.path.join(data_dir, name), "w") as f:
        f["header/times"] = times
        f["header/imu_calibration"] = json.dumps(cal)
        f["metadata/imu_el"] = json.dumps(recs)
    return name, times


@pytest.fixture(scope="module")
def table(tmp_path_factory, calibration):
    """A small campaign: a 07-17 raster at 5.9 deg, then sweeps and parks."""
    root = tmp_path_factory.mktemp("campaign")
    data_dir = root / "data"
    data_dir.mkdir()
    (root / "curation").mkdir()
    cal_hdr = {"imu_el": calibration, "metadata": {"n_stops": 12}}
    blocks, truth = [], {}

    def add(era, t0, el, tau, header=cal_hdr, scale=None, drop=()):
        raw = reading(el, tau)
        if scale is not None:
            i, k = scale
            raw[i] = reading(el[i], tau, scale=k)[0]
        fw = firmware_el(raw, calibration)
        for i in drop:
            raw[i] = np.nan
        name, times = _write_file(str(data_dir), _ts(t0), raw, fw, header)
        blocks.append((era, name, times))
        truth[name] = dict(tau=tau, el=np.asarray(el, float), fw=fw)
        return name

    # the raster: three full-circle sweeps inside fuse.RASTER_UTC
    for k, t0 in enumerate(("2026-07-17 20:30:00", "2026-07-17 20:40:00",
                            "2026-07-17 20:50:00")):
        add("~91m", t0, np.linspace(-170, 170, 400) + k, 5.9)
    # elevation sweeps at the test tilts, some files without the header
    # calibration (as 1514 Marjum files are)
    for k, tau in enumerate(TAUS):
        add("~87.5m", f"2026-07-17 0{3 + k}:00:00",
            np.linspace(-80, 80, 300), tau,
            header={} if k % 2 else cal_hdr, drop=(5,) if k == 0 else ())
    # a park at el 30 deg (static rows, one not quasi-static)
    add("~87.5m", "2026-07-17 09:00:00", np.full(200, 30.0), 0.4,
        scale=(10, 1.05))
    # a park through +-180 deg: rows are means of three firmware samples
    centre = 179.2 + 0.008 * np.arange(200)
    el3 = centre[:, None] + np.array([-0.3, 0.0, 0.3])[None, :]
    raw3 = reading(el3.ravel(), 2.1).reshape(-1, 3, 3)
    fw3 = firmware_el(raw3.reshape(-1, 3), calibration).reshape(-1, 3)
    name, times = _write_file(str(data_dir), _ts("2026-07-18 01:30:00"),
                              raw3.mean(axis=1), fw3.mean(axis=1), cal_hdr)
    blocks.append(("~91m-reraise", name, times))
    truth[name] = dict(tau=2.1, el=fuse.wrap180(centre),
                       fw=fw3.mean(axis=1),
                       fw_centre=firmware_el(reading(centre, 2.1),
                                             calibration))

    with open(root / "curation" / "mode_table.jsonl", "w") as fh:
        for era, name, times in blocks:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            fh.write(json.dumps({
                "t_start_utc": dt.datetime.fromtimestamp(
                    times[0], dt.timezone.utc).strftime(fmt),
                "t_end_utc": dt.datetime.fromtimestamp(
                    times[-1], dt.timezone.utc).strftime(fmt),
                "height_era": era, "phase": "C", "file_first": name,
                "file_last": name, "n_files": 1}) + "\n")
    eras = {k: {"offset_source": "anchor",
                "agl_m_vs_ground_under_antenna": 90.0}
            for k in ("30m", "87.5m", "91m", "91m-reraise")}
    with open(root / "curation" / (build_table.HORIZON_PROFILES + ".json"),
              "w") as fh:
        json.dump({"eras": eras}, fh)

    tab, _, _, info = build_table.build(str(data_dir), 0.0, 2e9, str(root))
    return tab, info, truth


def _rows(tab, name):
    return tab["file"] == name


def test_table_tilt_recovers_tau(table):
    tab, info, truth = table
    assert info["axle"]["n_rows"] == 1200
    for name, tr in truth.items():
        m = _rows(tab, name) & np.isfinite(tab["tilt_deg"])
        err = np.abs(tab["tilt_deg"][m] - tr["tau"]).max()
        if "fw_centre" in tr:
            assert err < 1e-4          # rows averaged over 0.6 deg of el
        else:
            assert err < 1e-6, (name, tr["tau"], err)


def test_table_tilt_missing_and_sigma(table):
    tab, info, truth = table
    name = [n for n, t in truth.items() if t["tau"] == 0.0][0]
    m = _rows(tab, name)
    assert np.isnan(tab["tilt_deg"][m][5])
    assert np.isnan(tab["tilt_sigma_deg"][m][5])
    assert info["calibration"]["n_files_without"] == 2
    ok = np.isfinite(tab["tilt_deg"])
    assert np.allclose(tab["tilt_sigma_deg"][ok],
                       np.hypot(info["tilt_noise_deg"],
                                fuse.IMU_TILT_AXIS_SIGMA))


def test_table_el_unchanged_without_wrap(table):
    tab, _, truth = table
    for name, tr in truth.items():
        if "fw_centre" in tr:
            continue
        m = _rows(tab, name) & np.isfinite(tab["imu_el_deg"])
        assert np.abs(fuse.wrap180(tab["el_deg"][m]
                                   - tab["imu_el_deg"][m])).max() < 1e-9
        assert not np.any(tab["flags"][m] & fuse.FLAG_EL_WRAP_CORRECTED)


def test_table_wrap_fixed_and_flagged(table):
    tab, _, truth = table
    name = [n for n, t in truth.items() if "fw_centre" in t][0]
    tr = truth[name]
    m = _rows(tab, name)
    assert np.abs(fuse.wrap180(tab["el_deg"][m] - tr["fw_centre"])).max() \
        < 1e-3
    broken = np.abs(fuse.wrap180(tr["fw"] - tr["fw_centre"])) > 1.0
    assert broken.sum() > 0
    flagged = (tab["flags"][m] & fuse.FLAG_EL_WRAP_CORRECTED) != 0
    assert np.array_equal(flagged, broken)
    # the raw firmware mean is kept
    assert np.allclose(tab["imu_el_deg"][m], tr["fw"])


def test_table_quasi_static_flag(table):
    tab, _, truth = table
    name = [n for n, t in truth.items() if t["tau"] == 0.4][0]
    bit = (tab["flags"][_rows(tab, name)]
           & fuse.FLAG_IMU_NOT_QUASI_STATIC) != 0
    assert np.flatnonzero(bit).tolist() == [10]
    assert not np.any(tab["flags"][~_rows(tab, name)]
                      & fuse.FLAG_IMU_NOT_QUASI_STATIC)
