"""Tests for v6's flag and quality rules on synthetic rows.

- ``EL_SOLUTION_GLITCH`` does not make an ``EL_WRAP_CORRECTED`` row suspect;
- ``IMU_NOT_QUASI_STATIC`` and ``EL_UNDEFINED_TILT`` do;
- ``UNCOMMANDED_MOTION`` runs on the accelerometer el, so neither the
  firmware mean's wrap glitches nor the motor fallback's 0 deg fill set it.

Run with::

    python -m pytest notebooks/arp/marjum-2026-07/pointing/tests
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fuse = _load("fuse", "fuse.py")
build_table = _load("build_table", "build_table.py")

OK = build_table.QUALITY_OK
SUSPECT = build_table.QUALITY_SUSPECT
GAP = build_table.QUALITY_GAP
GLITCH = fuse.FLAG_EL_SOLUTION_GLITCH
WRAP = fuse.FLAG_EL_WRAP_CORRECTED
# Set on every row of the table since v4; never decides quality.
BASE = fuse.FLAG_HEIGHT_ASSUMED


def _quality(*flags, hdr_bad=None):
    f = np.array([BASE | x for x in flags], dtype=np.int32)
    if hdr_bad is None:
        hdr_bad = np.zeros(f.size, bool)
    return build_table.quality_from_flags(f, hdr_bad).tolist()


# -- quality ---------------------------------------------------------------

def test_quality_v5_criteria_kept():
    assert _quality(0, fuse.FLAG_EL_NO_IMU, fuse.FLAG_AZ_SLEW,
                    fuse.FLAG_UNCOMMANDED, GLITCH,
                    fuse.FLAG_EL_POST_FAILURE | fuse.FLAG_EL_STUCK,
                    fuse.FLAG_NO_ESTIMATE | fuse.FLAG_AZ_SLEW) == [
        OK, SUSPECT, SUSPECT, SUSPECT, SUSPECT, OK, GAP]
    assert _quality(0, 0, hdr_bad=np.array([True, False])) == [SUSPECT, OK]
    fallback = fuse.FLAG_AZ_FUSED_FALLBACK
    assert _quality(fallback, fallback | fuse.FLAG_AZ_NO_POT) == [OK, SUSPECT]


def test_wrap_corrected_glitch_is_ok():
    """A wrap row with only bit 10 is 'ok'; the bit itself is unchanged."""
    assert _quality(WRAP, WRAP | GLITCH) == [OK, OK]


def test_wrap_corrected_other_bits_decide():
    assert _quality(WRAP | GLITCH | fuse.FLAG_UNCOMMANDED,
                    WRAP | GLITCH | fuse.FLAG_AZ_SLEW,
                    WRAP | GLITCH | fuse.FLAG_IMU_NOT_QUASI_STATIC,
                    GLITCH) == [SUSPECT] * 4


def test_not_quasi_static_is_suspect():
    assert _quality(fuse.FLAG_IMU_NOT_QUASI_STATIC,
                    fuse.FLAG_IMU_NOT_QUASI_STATIC | WRAP) == [SUSPECT] * 2


def test_undefined_tilt_is_suspect():
    assert _quality(fuse.FLAG_EL_UNDEFINED_TILT) == [SUSPECT]


def test_flag_bits_distinct():
    bits = list(fuse.FLAG_NAMES)
    assert len(set(bits)) == len(bits)
    assert fuse.FLAG_NAMES[fuse.FLAG_EL_UNDEFINED_TILT] == "EL_UNDEFINED_TILT"
    assert fuse.FLAG_EL_UNDEFINED_TILT == 1 << 16


# -- UNCOMMANDED_MOTION on the accelerometer el -----------------------------

N = 400


def _park_near_180(seed=0):
    """A park at el ~ +/-180 deg with the motor still (0 steps)."""
    rng = np.random.default_rng(seed)
    el = fuse.wrap180(180.0 + 0.1 * rng.standard_normal(N))
    motor = np.zeros(N)
    return el, motor


def _bit5(flags):
    return (flags & fuse.FLAG_UNCOMMANDED) != 0


def test_firmware_wrap_glitches_fire_v5_not_v6():
    el_acc, motor = _park_near_180()
    fw = el_acc.copy()
    fw[::25] = 0.0                       # linear-mean glitches
    fw[12::50] = 60.0
    _, _, flags_v5 = fuse.fuse_elevation(fw, motor)
    assert _bit5(flags_v5).any()
    _, _, flags_v6 = fuse.fuse_elevation(fw, motor,
                                         uncommanded_el_deg=el_acc)
    assert not _bit5(flags_v6).any()


def test_motor_fallback_zeros_do_not_fire():
    el_acc, motor = _park_near_180(1)
    fw = el_acc.copy()
    fw[::20] = np.nan                    # no imu_el: el_deg takes the motor
    el_imu6 = np.where(np.isfinite(fw), el_acc, np.nan)
    el, _, flags = fuse.fuse_elevation(fw, motor,
                                       uncommanded_el_deg=el_imu6)
    assert np.all(el[::20] == 0.0)       # the fallback fill
    assert not _bit5(flags).any()
    # el_deg itself would fire: the fill is a 180 deg jump at every gap
    assert fuse.detect_uncommanded(el, motor).any()


def test_real_uncommanded_motion_still_fires():
    el_acc = np.linspace(30.0, 80.0, N)  # moving, motor still
    motor = np.zeros(N)
    _, _, flags = fuse.fuse_elevation(el_acc, motor,
                                      uncommanded_el_deg=el_acc)
    assert _bit5(flags).any()
