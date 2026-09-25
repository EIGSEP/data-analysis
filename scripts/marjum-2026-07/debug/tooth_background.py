"""Local DPSS background under the transmitter comb teeth, marjum-2026-07 beam raster.

Exploratory, single-study code, shared by build_tooth_selection.py and
fit_beam.py so both use one definition of the background.

Chosen in data-analysis/notebooks/arp/marjum-2026-07/debug/:
tooth_background_debug.ipynb (Aaron approved replacing gap differencing,
2026-09-24) and tooth_background_basis_debug.ipynb (basis choice).

For every row (one spectrum), channels in a 48 MHz window are fitted with DPSS
modes of 150 ns delay half-width. Each tooth and its +/-1 neighbours are masked
(filterbank leakage), the FM band is masked, and outliers on either side are
clipped symmetrically at 4 sigma. The fit is evaluated at the tooth. Symmetric
clipping matters: the transmitter's broadband amplifier noise is part of the
background under a tooth, and rejecting positive excess (as the v3-beta RFI
flagger does) biases the model low by 3-6%.

Units: counts in, counts out. Channel c is at c * 250/1024 MHz. Teeth are
channels with c % 8 == 0. NaN where no window covers a tooth or a fit fails.
"""
import numpy as np
from scipy.signal.windows import dpss

DF = 250.0 / 1024            # MHz per channel
NCHAN = 1024
SMOOTH_NS = 150.0            # DPSS delay half-width
WINDOW_MHZ = 48.0            # wide enough to separate ~550 ns structure from the 512 ns comb alias
STEP_MHZ = 24.0
LIMITS_MHZ = (40.0, 249.5)   # fit windows stay inside these
FM_MHZ = (87.0, 108.5)
CLIP = 4.0

CH = np.arange(NCHAN)
FREQ = CH * DF
NEAR_TOOTH = np.abs((CH + 4) % 8 - 4) <= 1
FM = (FREQ > FM_MHZ[0]) & (FREQ < FM_MHZ[1])

DESCRIPTION = dict(
    method='local DPSS background per spectrum, evaluated at each tooth',
    smooth_halfwidth_ns=SMOOTH_NS, window_mhz=WINDOW_MHZ, step_mhz=STEP_MHZ,
    window_limits_mhz=list(LIMITS_MHZ), masked='tooth and +/-1 channels; FM band %g-%g MHz' % FM_MHZ,
    clipping='symmetric, %g sigma (1.4826 MAD), iterated to convergence' % CLIP,
    chosen_in=['debug/tooth_background_debug.ipynb', 'debug/tooth_background_basis_debug.ipynb'])


def dpss_modes(n, halfwidth_ns):
    """DPSS modes over n contiguous channels with delay half-width halfwidth_ns."""
    nw = halfwidth_ns * n * DF * 1e-3          # ns x GHz
    k = int(np.floor(2 * nw)) + 1
    return dpss(n, nw, Kmax=k).T if k > 1 else dpss(n, nw)[:, None]


def robust_fit(y, A, base, clip=CLIP, iters=6):
    """Least squares of y on A over `base`, dropping |residual| > clip*sigma on
    either side until the mask stops changing. Returns the prediction on all
    channels and the final fit mask."""
    use = base.copy()
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[use], y[use], rcond=None)
        r = y - A @ coef
        sigma = 1.4826 * np.median(np.abs(r[use]))
        new = base & (np.abs(r) <= clip * sigma)
        if np.array_equal(new, use):
            break
        use = new
    return A @ coef, use


def windows(teeth):
    """(window channels, teeth evaluated there). Window centres are every STEP_MHZ;
    each tooth is evaluated in the window whose centre is nearest."""
    teeth = np.asarray(teeth)
    lo, hi = LIMITS_MHZ
    centres = np.arange(lo + WINDOW_MHZ / 2, hi - WINDOW_MHZ / 2 + STEP_MHZ, STEP_MHZ)
    centres = np.clip(centres, lo + WINDOW_MHZ / 2, hi - WINDOW_MHZ / 2)
    nearest = np.argmin(np.abs(FREQ[teeth][:, None] - centres[None, :]), axis=1)
    out = []
    for k, c in enumerate(centres):
        ev = teeth[nearest == k]
        if len(ev):
            idx = np.flatnonzero((FREQ >= c - WINDOW_MHZ / 2) & (FREQ < c + WINDOW_MHZ / 2))
            out.append((idx, ev))
    return out


def tooth_background(spectra, teeth, return_residual_rms=False):
    """Background at each tooth for each spectrum.

    spectra: (rows, 1024) counts. teeth: channel indices (each % 8 == 0).
    Returns (rows, len(teeth)) counts. With return_residual_rms, also returns
    (rows, len(teeth)): the robust rms of the fractional fit residual on the
    off-tooth channels within +/-4 of each tooth (offsets 2-4), a per-tooth
    measure of how well the background fits locally.
    """
    spectra = np.asarray(spectra, float)
    teeth = np.asarray(teeth)
    assert np.all(teeth % 8 == 0)
    col = {c: i for i, c in enumerate(teeth)}
    out = np.full((len(spectra), len(teeth)), np.nan)
    local = np.full_like(out, np.nan)
    for idx, ev in windows(teeth):
        A = dpss_modes(len(idx), SMOOTH_NS)
        base0 = ~NEAR_TOOTH[idx] & ~FM[idx]
        pos = [c - idx[0] for c in ev]
        near = [np.r_[p - 4:p - 1, p + 2:p + 5] for p in pos]
        for r, row in enumerate(spectra[:, idx]):
            base = base0 & np.isfinite(row) & (row > 0)
            if base.sum() < A.shape[1] + 5:
                continue
            scale = np.median(row[base])
            pred, _ = robust_fit(row / scale, A, base)
            for c, p, nb in zip(ev, pos, near):
                out[r, col[c]] = pred[p] * scale
                if return_residual_rms:
                    nb = nb[(nb >= 0) & (nb < len(idx))]
                    x = row[nb] / scale / pred[nb] - 1
                    local[r, col[c]] = np.sqrt(np.mean(x ** 2))
    return (out, local) if return_residual_rms else out
