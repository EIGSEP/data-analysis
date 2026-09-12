"""Rotation/transmitter beam mapping utilities.

The conventions here match Dominic's ``eigsep_data.beam_sim`` notebook:
``R = R_el @ R_az`` maps the receiver frame into the top frame, and a
transmitter heading is therefore transformed with ``R.T``.  The module is
NumPy/SciPy based so it can be used from notebooks without requiring JAX.

The two useful layers are:

* :func:`fuse_pointing` turns motor counts, potentiometer azimuth, and IMU
  acceleration into a common pointing table.
* :class:`PolarizationBeamMapper` fits an off-axis transmitter heading and
  polarization angle, then solves non-negative theta/phi beam gains at the
  sampled directions from alternating transmitter arms.

Beam maps may be supplied as HEALPix arrays or as a callable returning
``(g_theta, g_phi)`` for ``(theta, phi)``.
"""

from dataclasses import dataclass

import numpy as np
from eigsep_sim.coord import rot_m as _eigsep_rot_m
from scipy.optimize import least_squares, nnls


MOTOR_DEG_PER_STEP = 180.0 / 1.13e4


def _rot_axis(angle, axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, z*z*C+c],
    ])


def rotation_matrix(az_deg, el_deg, az_axis=(0, 0, 1), el_axis=(1, 0, 0)):
    """Return the receiver rotation matrix used by the beam simulations.

    az_axis is directed: an encoder reporting a right-handed angle about
    a downward shaft uses (0, 0, -1) in the topocentric frame.
    """
    return _eigsep_rot_m(np.deg2rad(el_deg), np.asarray(el_axis, float)) @ _eigsep_rot_m(
        np.deg2rad(az_deg), np.asarray(az_axis, float)
    )


def imu_elevation_deg(accel_xyz):
    """Convert the v007 notebook's IMU acceleration convention to elevation."""
    a = np.asarray(accel_xyz, dtype=float)
    return np.rad2deg(np.unwrap(
        np.arctan2(np.sign(a[..., 1]) * np.linalg.norm(a[..., :2], axis=-1), -a[..., 2]),
        period=2 * np.pi,
    ))


def _circular_mean_deg(values, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(values.shape[0])
    weights = np.asarray(weights, dtype=float).reshape((-1,) + (1,) * (values.ndim - 1))
    good = np.isfinite(values)
    z = np.sum(weights * np.where(good, np.exp(1j * np.deg2rad(values)), 0), axis=0)
    norm = np.sum(weights * good, axis=0)
    return np.rad2deg(np.angle(z / np.maximum(norm, 1e-30)))


@dataclass
class PointingStreams:
    """Synthetic or measured pointing streams before fusion."""
    true_az_deg: np.ndarray
    true_el_deg: np.ndarray
    motor_az_steps: np.ndarray
    motor_el_steps: np.ndarray
    pot_az_deg: np.ndarray
    imu_accel: np.ndarray
    motor_slip_az_steps: np.ndarray
    motor_slip_el_steps: np.ndarray


def simulate_pointing_streams(true_az_deg, true_el_deg, pot_sigma_deg=0.5,
                              motor_sigma_steps=2.0, slip_az_steps=0.0,
                              slip_el_steps=0.0, imu_sigma=0.01, seed=0):
    """Generate noisy pot/encoder/IMU streams with persistent gear slip."""
    rng = np.random.default_rng(seed)
    az = np.asarray(true_az_deg, float); el = np.asarray(true_el_deg, float)
    slip_az = np.broadcast_to(slip_az_steps, az.shape).astype(float)
    slip_el = np.broadcast_to(slip_el_steps, el.shape).astype(float)
    motor_az = az / MOTOR_DEG_PER_STEP + slip_az + rng.normal(0, motor_sigma_steps, az.shape)
    motor_el = el / MOTOR_DEG_PER_STEP + slip_el + rng.normal(0, motor_sigma_steps, el.shape)
    pot = az + rng.normal(0, pot_sigma_deg, az.shape)
    # Invert the v007 IMU elevation convention: ay=sin(el), -az=cos(el).
    accel = np.stack([np.zeros_like(el), np.sin(np.deg2rad(el)),
                      -np.cos(np.deg2rad(el))], axis=-1)
    accel += rng.normal(0, imu_sigma, accel.shape)
    return PointingStreams(az, el, motor_az, motor_el, pot, accel, slip_az, slip_el)


def estimate_motor_slip_steps(motor_steps, reference_deg):
    """Robustly estimate a constant encoder offset against a reference angle."""
    residual = MOTOR_DEG_PER_STEP * np.asarray(motor_steps) - np.asarray(reference_deg)
    return float(np.nanmedian(residual) / MOTOR_DEG_PER_STEP)


@dataclass
class PointingTable:
    az_deg: np.ndarray
    el_deg: np.ndarray
    az_motor_deg: np.ndarray
    az_pot_deg: np.ndarray
    el_imu_deg: np.ndarray
    el_motor_deg: np.ndarray


def fuse_pointing(motor_az_steps, motor_el_steps, pot_az_deg=None,
                  imu_accel=None, az_weight=0.5, el_weight=0.5):
    """Fuse available motor, potentiometer, and IMU pointing streams.

    Motor counts use the calibration from ``EIGSEP_data_explore_v007``.
    Missing values may be NaN.  Azimuth is combined on the circle; elevation
    is combined linearly.  The returned table retains each input stream for
    diagnostics and fitting.
    """
    azm = MOTOR_DEG_PER_STEP * np.asarray(motor_az_steps, float)
    elm = MOTOR_DEG_PER_STEP * np.asarray(motor_el_steps, float)
    azp = np.full_like(azm, np.nan) if pot_az_deg is None else np.asarray(pot_az_deg, float)
    eli = np.full_like(elm, np.nan) if imu_accel is None else imu_elevation_deg(imu_accel)
    az = azm.copy()
    both = np.isfinite(azm) & np.isfinite(azp)
    az[both] = _circular_mean_deg(np.stack([azm[both], azp[both]]), [az_weight, 1 - az_weight])
    az[~np.isfinite(azm) & np.isfinite(azp)] = azp[~np.isfinite(azm) & np.isfinite(azp)]
    el = elm.copy()
    both = np.isfinite(elm) & np.isfinite(eli)
    el[both] = (el_weight * elm[both] + (1 - el_weight) * eli[both])
    el[~np.isfinite(elm) & np.isfinite(eli)] = eli[~np.isfinite(elm) & np.isfinite(eli)]
    return PointingTable(az, el, azm, azp, eli, elm)


def _sph_basis(theta, phi):
    s, c = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    r = np.stack([cp*s, sp*s, c], axis=-1)
    th = np.stack([cp*c, sp*c, -s], axis=-1)
    ph = np.stack([-sp, cp, np.zeros_like(theta)], axis=-1)
    return r, th, ph


def vector_to_spherical(v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.arccos(np.clip(v[..., 2], -1, 1)), np.mod(np.arctan2(v[..., 1], v[..., 0]), 2*np.pi)


@dataclass
class TransmitterGeometry:
    heading_top: np.ndarray
    alpha_deg: float = 0.0

    def __post_init__(self):
        self.heading_top = np.asarray(self.heading_top, float)
        self.heading_top /= np.linalg.norm(self.heading_top)

    def field_top(self, arm=0):
        """Return the driven E2 dipole arm in the top frame.

        Dominic's HFSS workflow defines arm 0 as ``E1=0, E2=1`` with
        dipole axes rotated by ``alpha``; arm 1 repeats that E2 drive after
        rotating the dipole pair by 90 degrees.  This is intentionally not
        the same as driving rotated ``ax1`` on arm 0.
        """
        alpha = np.deg2rad(self.alpha_deg + (90.0 if arm else 0.0))
        return np.array([-np.sin(alpha), np.cos(alpha), 0.0])


class PolarizationBeamMapper:
    """Forward model and fitting tools for alternating transmitter spikes."""

    def __init__(self, beam_theta, beam_phi, sampler=None):
        self.beam_theta = beam_theta
        self.beam_phi = beam_phi
        self.sampler = sampler

    def gains(self, theta, phi):
        if self.sampler is not None:
            return tuple(np.asarray(x, float) for x in self.sampler(theta, phi))
        try:
            import healpy as hp
        except ImportError as exc:
            raise ImportError("healpy is required for HEALPix beam arrays") from exc
        px = hp.ang2pix(hp.npix2nside(np.asarray(self.beam_theta).size), theta, phi)
        return np.asarray(self.beam_theta)[px], np.asarray(self.beam_phi)[px]

    def design(self, az_deg, el_deg, heading_top, alpha_deg, arms):
        az_deg, el_deg = np.broadcast_arrays(az_deg, el_deg)
        arms = np.asarray(arms, int)
        coeff = np.zeros((az_deg.size, 2))
        theta = np.zeros(az_deg.size); phi = np.zeros(az_deg.size)
        for i, (az, el, arm) in enumerate(zip(az_deg.flat, el_deg.flat, arms.flat)):
            R = rotation_matrix(az, el)
            h = R.T @ heading_top
            theta[i], phi[i] = vector_to_spherical(h)
            _, thhat, phhat = _sph_basis(theta[i], phi[i])
            e = R.T @ TransmitterGeometry(heading_top, alpha_deg).field_top(arm)
            coeff[i] = [(e @ thhat)**2, (e @ phhat)**2]
        gth, gph = self.gains(theta, phi)
        return coeff, theta, phi, np.asarray(gth), np.asarray(gph)

    def predict(self, az_deg, el_deg, geometry, arms, scale=1.0, offset=0.0):
        coeff, theta, phi, gth, gph = self.design(
            az_deg, el_deg, geometry.heading_top, geometry.alpha_deg, arms
        )
        return offset + scale * (gth * coeff[:, 0] + gph * coeff[:, 1]), (theta, phi)

    def fit_heading(self, az_deg, el_deg, arms, observed, initial,
                    fit_alpha=True, fit_scale=True, fit_offset=True):
        """Fit transmitter heading (az/el), polarization angle and amplitudes."""
        y = np.asarray(observed, float)
        arms = np.asarray(arms, int)
        initial = np.asarray(initial, float)
        n = 2 + int(fit_alpha) + int(fit_scale) + int(fit_offset)
        x0 = np.zeros(n); x0[:2] = initial[:2]; j = 2
        if fit_alpha: x0[j] = initial[2] if initial.size > 2 else 0; j += 1
        if fit_scale: x0[j] = 1.0; j += 1
        if fit_offset: x0[j] = 0.0
        def unpack(x):
            heading = np.array([np.cos(np.deg2rad(x[0]))*np.cos(np.deg2rad(x[1])),
                                np.sin(np.deg2rad(x[0]))*np.cos(np.deg2rad(x[1])),
                                np.sin(np.deg2rad(x[1]))])
            j = 2; alpha = x[j] if fit_alpha else (initial[2] if initial.size > 2 else 0); j += int(fit_alpha)
            scale = x[j] if fit_scale else 1.; j += int(fit_scale)
            offset = x[j] if fit_offset else 0.
            return TransmitterGeometry(heading, alpha), scale, offset
        def residual(x):
            geom, scale, offset = unpack(x)
            pred, _ = self.predict(az_deg, el_deg, geom, arms, scale, offset)
            return (pred - y) / max(np.nanstd(y), 1e-12)
        result = least_squares(residual, x0)
        geom, scale, offset = unpack(result.x)
        result.geometry, result.scale, result.offset = geom, scale, offset
        return result

    def fit_sampled_gains(self, az_deg, el_deg, geometry, arms, observed):
        """Recover non-negative theta/phi gains independently at each pixel."""
        coeff, theta, phi, _, _ = self.design(az_deg, el_deg, geometry.heading_top, geometry.alpha_deg, arms)
        if self.sampler is None:
            import healpy as hp
            pixels = hp.ang2pix(hp.npix2nside(np.asarray(self.beam_theta).size), theta, phi)
        else:
            # Callable samplers cannot define a stable pixel identity.
            pixels = np.arange(theta.size)
        y = np.asarray(observed, float)
        out = {}
        for px in np.unique(pixels):
            m = pixels == px
            out[int(px)] = nnls(coeff[m], y[m])[0]
        return out, theta, phi
