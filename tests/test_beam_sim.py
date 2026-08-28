"""Tests for eigsep_data.beam_sim."""

import healpy
import jax.numpy as jnp
import numpy as np

from eigsep_data.beam_sim import (
    RotatingAntennaCartesian,
    TransmitterAntenna,
    power_sim,
)

NSIDE = 8
NPIX = healpy.nside2npix(NSIDE)


def _power(beam_cart, conjugate_beam, E2):
    """Simulated power over a small az/el grid."""
    rx = RotatingAntennaCartesian(
        beam_cart=jnp.asarray(beam_cart), conjugate_beam=conjugate_beam
    )
    tx = TransmitterAntenna(
        E1=1.0, E2=E2, heading_top=jnp.array([0, 0, -1]), alpha=60
    )
    az = jnp.array(np.deg2rad(np.linspace(0, 350, 36)))
    el = jnp.array(np.deg2rad(np.linspace(-40, 40, 36)))
    return np.asarray(power_sim(rx, tx, az, el, normalize=False)[0])


class TestConjugateBeam:
    def test_flag_changes_elliptical_response(self):
        # conj(W).E and W.E differ whenever either field is elliptically
        # polarized; the flag exists to correct a flipped HFSS phase
        # convention, so it must actually reach the PLF.
        beam = np.zeros((3, NPIX), dtype=complex)
        beam[0] = 1.0
        beam[1] = 0.6j
        p_true = _power(beam, True, 1.0j)
        p_false = _power(beam, False, 1.0j)
        assert not np.allclose(p_true, p_false)

    def test_flag_is_noop_for_linear_polarization(self):
        # A real (up to global phase) beam and a linear TX are invariant
        # under conjugation -- guards against the branch flipping a sign
        # it should not.
        beam = np.zeros((3, NPIX), dtype=complex)
        beam[0] = 1.0
        np.testing.assert_allclose(
            _power(beam, True, 0.0), _power(beam, False, 0.0)
        )

    def test_default_is_conjugated(self):
        rng = np.random.default_rng(0)
        beam = rng.normal(size=(3, NPIX)) + 1j * rng.normal(size=(3, NPIX))
        rx = RotatingAntennaCartesian(beam_cart=jnp.asarray(beam))
        assert rx.conjugate_beam is True
        np.testing.assert_allclose(
            _power(beam, True, 1.0j), _power(beam, rx.conjugate_beam, 1.0j)
        )

    def test_flag_survives_pytree_roundtrip(self):
        # power_sim is jitted, so the flag rides in the aux data; a lost
        # round-trip would silently restore the hardcoded behaviour.
        beam = np.zeros((3, NPIX), dtype=complex)
        beam[0] = 1.0
        rx = RotatingAntennaCartesian(
            beam_cart=jnp.asarray(beam), conjugate_beam=False
        )
        leaves, aux = rx.tree_flatten()
        assert RotatingAntennaCartesian.tree_unflatten(
            aux, leaves
        ).conjugate_beam is False
