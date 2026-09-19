"""Receiver gain g_rx solutions for the beam-scan window, from B7.

rf-calibrator's B7 (`abscal/trx_phaseC.npz`, `MEMO_B7_first_calibration.md`)
delivers per-cycle absolute g_rx(nu, t) and T_rx(nu, t) over Phase C: 66
switching cycles, 2026-07-17 04:11 -> 2026-07-18 02:57 UTC. The beam-scan
window sits entirely inside that span.

Why this matters for the beam fit. `measured_tx` is a *channel difference*,
auto[c] - 0.5*(auto[c-1] + auto[c+1]). With auto = g_rx * (T_sky + T_rx), and
g_rx smooth over three adjacent 244 kHz channels, the difference carries g_rx
as a **linear multiplicative factor**. So a drift in g_rx multiplies the data
by a time-varying factor that a single per-channel fit amplitude cannot absorb.
Across the beam-scan window at 173.83 MHz, g_rx runs 473 -> 293, a factor 1.61.
Dividing it out removes that systematic. This is the standing
`receiver_regime_caveat` in beam_fits_v2, now correctable.

The frequency grids match exactly (max |df| = 0 over the 101 beam channels), so
channels are indexed directly with no frequency interpolation.

*** The honest caveat, and it is a big one. ***
The cycles are not spread evenly over the beam scan. Their spacing across the
window is, in minutes:

    12.9, 10.7, 342.4, 66.6, 12.9, 12.9

That 342-minute hole -- 19:42:14 -> 01:24:40 UTC -- contains **44.3% of the
used samples**, and it is exactly the interval that rf-calibrator's B7 brackets
the receiver regime change to. So within the gap we know g_rx was 473 before
and 318 after, and nothing about the shape in between. A linear interpolation
imposes a smooth ramp on what may well have been a step. `explorer_model.py`
therefore also exposes `CAL_ANCHOR_MIN`, the distance in minutes from each
sample to its nearest real cycle, so consumers can see where the correction is
anchored and where it is a guess.

Run: LD_LIBRARY_PATH=$MAMBA/envs/arp/lib $MAMBA/envs/arp/bin/python3 build_cal_gain.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TRX = "/mnt/data02/eigsep/abscal/trx_phaseC.npz"
OUT = HERE / "cal_gain.npz"

cache = np.load(HERE / "beam_explorer_cache.npz")
side = np.load(HERE / "beam_explorer_sidecar.npz", allow_pickle=True)
chans = cache["channels"]
times = side["times"].astype(float)

b7 = np.load(TRX, allow_pickle=True)
order = np.argsort(b7["sol_times"])
sol_times = b7["sol_times"][order].astype(float)
gain = b7["gain"][order]          # (ncyc, 1024)
t_rx = b7["t_rx"][order]
freqs = b7["freqs"]

# frequency grids must agree exactly; assert rather than silently interpolate
if np.max(np.abs(freqs[chans] - cache["freqs_mhz"])) > 1e-9:
    raise SystemExit("abscal and cache frequency grids disagree")

g_ch = gain[:, chans]             # (ncyc, nchan)

# Store the CYCLE solutions, not a per-sample expansion. 101 x 66 floats is
# 27 kB; the per-sample expansion is 22 MB and would more than double the
# explorer's payload for no information. explorer_model.py does the linear
# interpolation onto the sample times at load (it is one np.interp per
# channel, milliseconds).
nchan = len(chans)
print(f"channels {nchan}  cycles {sol_times.size}")
finite_per_chan = (np.isfinite(g_ch) & (g_ch > 0)).sum(axis=0)
print(f"cycles usable per channel: min {finite_per_chan.min()}, "
      f"max {finite_per_chan.max()}")
i712 = int(np.argmin(np.abs(chans - 712)))
# window bounds from the data, not hardcoded
tv = times[times > 0]
W0, W1 = float(tv.min()), float(tv.max())
inw = (sol_times >= W0) & (sol_times <= W1)
print(f"ch712 g_rx across the {int(inw.sum())} in-window cycles: "
      f"{g_ch[inw, i712].min():.1f} .. {g_ch[inw, i712].max():.1f} "
      f"(ratio {g_ch[inw, i712].max()/g_ch[inw, i712].min():.2f})")

np.savez_compressed(
    OUT,
    sol_times=sol_times,
    gain_cycles=g_ch.astype(np.float32),
    trx_cycles=t_rx[:, chans].astype(np.float32),
    channels=chans,
    source=np.array(TRX),
    gap_start_utc=np.array("2026-07-17T19:42:14Z"),
    gap_end_utc=np.array("2026-07-18T01:24:40Z"),
    gap_minutes=np.array(342.4),
)
print(f"wrote {OUT}  ({OUT.stat().st_size/1e3:.0f} kB)")
