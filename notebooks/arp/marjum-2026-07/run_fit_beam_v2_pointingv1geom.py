"""Run fit_beam_v2 against the pointing_table@v1 refit geometry.

The review checkpoint compares two geometries:
  beam_fits_v2_report.json               <- v007_multichannel_consensus.json
  beam_fits_v2_pointingv1geom_report.json <- ..._pointingv1.json

Both must be regenerated with the same corrected masking or the comparison is
apples-to-oranges. This driver produces the second one.

It deliberately does NOT re-fit the geometry. That refit is a known false
minimum -- it lands 86.3 deg from the surveyed transmitter direction while
*improving* RMS, and substituting the surveyed heading changes median
normalized RMS by 0.0005. Re-running it under a new mask would just produce a
different arbitrary heading and would be new analysis, not a correction.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_beam_v2 as v2

v2.CONSENSUS_JSON = str(HERE / "v007_multichannel_consensus_pointingv1.json")
v2.OUT_PREFIX = str(HERE / "beam_fits_v2_pointingv1geom")
v2.main()
