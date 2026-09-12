"""Synthetic truth-recovery demonstration for ``eigsep_data.beam_mapping``."""

import numpy as np

from eigsep_data.beam_mapping import PolarizationBeamMapper, TransmitterGeometry


def analytic_beam(theta, phi):
    # Two deliberately different polarization patterns, both positive.
    return (
        0.25 + np.cos(theta) ** 2 * (1 + 0.35 * np.cos(2 * phi)),
        0.15 + np.sin(theta) ** 2 * (1 - 0.25 * np.cos(2 * phi)),
    )


def main():
    rng = np.random.default_rng(4)
    az = np.linspace(-170, 170, 240)
    el = 18 * np.sin(np.linspace(0, 4 * np.pi, az.size))
    arms = np.arange(az.size) % 2
    mapper = PolarizationBeamMapper(None, None, sampler=analytic_beam)
    truth = TransmitterGeometry(
        [np.cos(np.deg2rad(32)) * np.cos(np.deg2rad(-18)),
         np.sin(np.deg2rad(32)) * np.cos(np.deg2rad(-18)),
         np.sin(np.deg2rad(-18))],
        alpha_deg=23,
    )
    observed, _ = mapper.predict(az, el, truth, arms, scale=2.4, offset=0.1)
    observed += rng.normal(0, 0.002 * observed.std(), observed.size)
    fit = mapper.fit_heading(az, el, arms, observed, [5, 5, 5])
    print(f"truth heading az/el = 32/-18 deg, alpha = 23 deg")
    print("fit   heading az/el = %.3f/%.3f deg, alpha = %.3f deg" % tuple(fit.x[:3]))
    print("fit scale/offset    = %.4f/%.4f" % (fit.scale, fit.offset))
    print("normalized RMS       = %.4g" % np.sqrt(np.mean(fit.fun ** 2)))
    return fit


if __name__ == "__main__":
    main()
