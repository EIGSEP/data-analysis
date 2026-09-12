"""End-to-end HFSS transmitter simulation and sampled-beam recovery.

Run with a checkout of the HFSS asset, for example::

    python demo_hfss_tx_recovery.py /path/to/eigsep_data/hfss_beam_maps/bowtie_beam.npz
"""

import sys
import numpy as np

from rotation_beam import PolarizationBeamMapper, TransmitterGeometry
from tx_beam_sim import HFSSBeamSet, ground_heading, simulate_hfss


def simulate_gain_model(beam, az, el, geometry, arms):
    """Generate alternating-arm powers from the HFSS spherical gain maps."""
    from rotation_beam import PolarizationBeamMapper
    rows = []
    pxs = []
    for fi in range(beam.gain_th.shape[0]):
        mapper = PolarizationBeamMapper(beam.gain_th[fi], beam.gain_ph[fi])
        p, _ = mapper.predict(az, el, geometry, arms)
        rows.append(p)
        _, theta, phi, _, _ = mapper.design(az, el, geometry.heading_top,
                                             geometry.alpha_deg, arms)
        import healpy as hp
        pxs.append(hp.ang2pix(beam.nside, theta, phi))
    return np.asarray(rows), pxs


def main(path):
    beam = HFSSBeamSet.from_npz(path)
    # Each pointing is observed with both adjacent transmitter polarizations.
    az = np.repeat(np.linspace(-170, 170, 90), 2)
    el = np.repeat(18 * np.sin(np.linspace(0, 2 * np.pi, 90)), 2)
    arms = np.tile([0, 1], 90)
    for east, north in [(0, 0), (25, 0), (0, 25), (-30, 20)]:
        geom = TransmitterGeometry(ground_heading(east, north), alpha_deg=17)
        simulated, _ = simulate_gain_model(beam, az, el, geom, arms)
        # Recover the first HFSS frequency's two polarization gains.
        mapper = PolarizationBeamMapper(beam.gain_th[0], beam.gain_ph[0])
        recovered, _, _ = mapper.fit_sampled_gains(az, el, geom, arms, simulated[0])
        expected = {}
        _, theta, phi, _, _ = mapper.design(az, el, geom.heading_top,
                                             geom.alpha_deg, arms)
        import healpy as hp
        pixels = hp.ang2pix(beam.nside, theta, phi)
        for px in np.unique(pixels):
            expected[int(px)] = np.array([beam.gain_th[0, px], beam.gain_ph[0, px]])
        err = [np.linalg.norm(recovered[k] - expected[k]) /
               max(np.linalg.norm(expected[k]), 1e-12) for k in recovered]
        print(f"offset E/N={east:+g}/{north:+g} m: "
              f"{len(recovered)} pixels, median relative beam error "
              f"{np.median(err):.3e}, max {np.max(err):.3e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: demo_hfss_tx_recovery.py BEAM_NPZ")
    main(sys.argv[1])
