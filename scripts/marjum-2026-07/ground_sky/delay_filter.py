"""Delay low-pass of per-integration box-air spectra, applied before binning.

Box-air's power fluctuates on seconds-to-minutes timescales by several times
radiometer noise, and averaging into bins leaves most of it (memo 007,
section "In flight"). At 87.5 m the fluctuation's power is at delays of
300-900 ns, near the ground-bounce delay 2h/c; the sky and ground signal the
global fit models is spectrally smooth. This module removes the long-delay
part of each integration before the bins are averaged:

1. ``P_slow``: a centred running mean of the flag-applied power over
   ``ref_s`` seconds, per channel; ``delta = P / P_slow - 1``.
2. Within each sub-band, a weighted least-squares fit of a DPSS basis
   spanning |delay| < ``tau_ns`` to ``delta``, one fit per integration.
3. The filtered power is ``P_slow * (1 + fit)``.

Masks: the fit uses one mask for every integration of a ``block_s`` block,
the union of their flags (memo 007; spectral-analysis skill: a mask that
varies from one integration to the next scatters power). A block whose union
masks more than ``max_flag`` of a sub-band is dropped (NaN). Channels masked
in the union are NaN in the output: the fit's value there is never used as
data. Channels outside every sub-band pass through unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

#: Sub-bands fitted separately (MHz), either side of the FM band.
BANDS = ((50.0, 85.0), (110.0, 250.0))


def dpss_basis(nch, dch_mhz, tau_ns):
    """DPSS vectors (nch, K) concentrated in |delay| < tau_ns."""
    nw = tau_ns * 1e-9 * nch * dch_mhz * 1e6
    k = max(1, int(2 * nw) + 1)
    return dpss(nch, nw, k).T if nw > 0.5 else np.ones((nch, 1)) / np.sqrt(nch)


def delay_lowpass(t, freqs_mhz, D, tau_ns, block_s=5.0, ref_s=600.0, bands=BANDS,
                  max_flag=0.5, ridge=1e-6):
    """Filter D (rows = integrations at times t, NaN = flagged). Returns (Df, stats).

    ``tau_ns=None`` applies only the block-union mask and block dropping, with
    no filter: the control that differs from the filter in that alone.
    """
    t = np.asarray(t, float)
    f = np.asarray(freqs_mhz, float)
    Df = D.copy()
    ps = pd.DataFrame(D, index=pd.to_datetime(t, unit="s")).rolling(
        f"{int(ref_s)}s", center=True, min_periods=10).mean().to_numpy()
    blk = np.floor((t - t[0]) / block_s).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1]])
    ends = np.r_[starts[1:], len(t)]
    stats = {"cells": 0, "flag_in": 0, "flag_union": 0, "dropped_blocks": 0, "blocks": 0}
    dch = float(np.median(np.diff(f)))
    for lo, hi in bands:
        cols = np.flatnonzero((f >= lo) & (f < hi))
        if cols.size < 8:
            continue
        A = dpss_basis(cols.size, dch, tau_ns) if tau_ns is not None else None
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = D[:, cols] / ps[:, cols] - 1
        bad = ~np.isfinite(delta)
        out = np.full((len(t), cols.size), np.nan)
        for a, b in zip(starts, ends):
            stats["blocks"] += 1
            U = bad[a:b].any(0)
            stats["cells"] += (b - a) * cols.size
            stats["flag_in"] += int(bad[a:b].sum())
            stats["flag_union"] += (b - a) * int(U.sum())
            if U.mean() > max_flag:
                stats["dropped_blocks"] += 1
                continue
            if A is None:
                out[a:b] = np.where(U, np.nan, D[a:b][:, cols])
                continue
            Ak = A[~U]
            G = Ak.T @ Ak
            G += ridge * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
            c = np.linalg.solve(G, Ak.T @ delta[a:b][:, ~U].T)       # (K, nrow)
            fit = (A @ c).T
            fit[:, U] = np.nan
            out[a:b] = ps[a:b][:, cols] * (1 + fit)
        Df[:, cols] = out
    return Df, stats
