"""
HFSS-free self-calibration of the common cross-correlation phase on the
Phase 2 inputs (vb.self_calibrate_phase), written to selfcal_phase.npz.

The cross phase at a fixed pointing repeats within a dwell but jumps
between dwells, identically in every channel; visits to one pixel then
disagree. The phase is fitted per block of BLOCK integrations from
leave-one-dwell-out predictions, so single-dwell pixels and a sample's
own dwell never enter it. NaN marks blocks the data leave unconstrained.

    POINTING_TABLE=... VECTOR_BEAM_CACHE=... python selfcal_phase.py
"""

import time
from pathlib import Path

import numpy as np

from eigsep_data.beam_mapping import vector_beam as vb

import pipeline
import run_phase2

HERE = Path(__file__).resolve().parent
BLOCK = 20  # integrations, ~11 s


def main():
    t0 = time.time()
    inp = pipeline.load()
    (pxs, r_hat, E1v, E2v), _, sel_b, _, _ = run_phase2.phase2_geometry(inp)
    order = np.argsort(inp["times"])
    assert np.array_equal(
        order, np.arange(len(order))
    ), "rows not in time order"
    channels = [
        dict(ch, valid=ch["valid"] & sel_b)
        for ch in inp["channels"]
        if (ch["valid"] & sel_b).sum() > 50
    ]
    phase, info = vb.self_calibrate_phase(
        pxs, r_hat, E1v, E2v, channels, block=BLOCK, verbose=True
    )
    np.savez(
        HERE / "selfcal_phase.npz",
        phase=phase,
        times=inp["times"],
        block=BLOCK,
        d_idx=np.array([ch["d_idx"] for ch in channels]),
        updates_deg=np.array(info["updates_deg"]),
    )
    print(
        f"phase on {np.isfinite(phase).sum()} of {len(phase)} integrations, "
        f"{info['n_iter']} iterations, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
