"""
The production beam fit's geometry: geometry release v0004 positions with
the corrections fitted by derived/beam/empirical_raster_v0013 (memo 005).

The campaign repository holding imgs/fits/v0004_marjum_geometry/shared.json
and empirical_raster_v0013/provenance.json is not public, so the values
are transcribed from public records, each cited below. Where two records
give the same quantity they agree.

- Positions (UTM raster grid of DEM v0002, metres): printed by
  notebooks/arp/marjum-2026-07/single_ch_beam_fit.html (cell output,
  "antenna [1656.22 2031.83 1778.6 ], transmitter [1652.18 2025.33
  1683.48]", geometry release v0004). Memo 001 Table 6 gives the same to
  0.1 m. The transmitter is then 4.6 deg from nadir, as memo 005 states.
- psi: memos 003 and 005 (142.164 deg, the highline direction).
- az offset and el correction at full precision:
  scripts/marjum-2026-07/ground_sky/summary.json, "pointing" block
  (az_offset_deg -16.035010698302443 = -16 + az_zero_delta,
  el_offset_deg 0.9319537656380783), beam product v0013.
- alpha0 = 96 deg and the five corrections to three decimals:
  single_ch_beam_fit.html ("geometry: psi 142.164, az offset -16.0,
  alpha0 96.0; corrections {...}"); memo 005 Table 1 to two.
- Transmitter offsets sit on the bound 1.5/sqrt(2) m
  (beam_mapping.tx_fit.DEFAULT_BOUNDS), +east, -north (memo 005).

Geometry release v0005 (memo 002, 2026-10-17) supersedes v0004's
positions, but no beam fit on v0005 exists yet, and v0013's corrections
are only meaningful with the positions they were fitted on.
"""

import numpy as np

from eigsep_data.beam_mapping.tx_fit import DEFAULT_BOUNDS, TxGeometryModel

ANTENNA_ENU = np.array([1656.22, 2031.83, 1778.60])
TRANSMITTER_ENU = np.array([1652.18, 2025.33, 1683.48])
PSI_DEG = 142.164
AZ_OFFSET_DEG = -16.0
ALPHA0_DEG = 96.0

_AZ_TOTAL = -16.035010698302443
_EL_TOTAL = 0.9319537656380783
PARAMS = np.array(
    [
        -0.149,  # polarization_delta_deg
        _AZ_TOTAL - AZ_OFFSET_DEG,  # az_zero_delta_deg
        _EL_TOTAL,  # el_zero_delta_deg
        DEFAULT_BOUNDS[3],  # tx_east_delta_m, at its bound
        -DEFAULT_BOUNDS[4],  # tx_north_delta_m, at its bound
    ]
)


def model():
    return TxGeometryModel(
        psi_deg=PSI_DEG,
        antenna_enu=ANTENNA_ENU,
        transmitter_enu=TRANSMITTER_ENU,
        alpha0_deg=ALPHA0_DEG,
        az_offset_deg=AZ_OFFSET_DEG,
    )
