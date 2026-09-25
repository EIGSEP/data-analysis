"""B16: DPSS smooth-band model + residual PCA, per antenna.

Rev 4 (2026-09-16, Aaron's correction, relayed via experimental-strategist):
**the fit, residual, and PCA are now in LINEAR amplitude (raw counts),
not log10.** Rev 2's log-amplitude choice for these three was wrong --
Aaron's direct correction: linear is the correct unit for fitting and
amplitudes; log is the usual choice for plots/display only. DPSS
smooth-band fitting on log power spectra was an incorrect operation.
Log10 is retained ONLY where the notebook builder uses it for a display
colormap/axis, never for anything that feeds the fit, residual array,
or `masked_pca()`. See `derived/smooth_model/v0/manifest.json`
revision_notes for what changed numerically as a result.

Rev 2 (2026-09-15, Aaron's review): four corrections applied.

1. ~~**Log-amplitude, not linear.** The fit, residual, and PCA all now
   operate on `logp = log10(max(raw, 1))` -- the same convention v0's
   own `detectors.py` uses everywhere (`transient_track`,
   `persistent_track`, comb detection), not reinvented, and the
   installed default for any amplitude-domain work going forward.~~
   **Superseded by rev 4 above -- this was wrong.**
2. **Verified, not assumed, that v0's mask was applied before the fit**
   (it was -- weights, not just a post-hoc PCA exclusion). But two
   narrow lines (~135.7 MHz box-gnd, ~125.5 MHz box-air) that dominated
   rev 1's residual PCA turn out to be CLEAN (unflagged) by v0's own
   bitfield 86-92% of the time at that exact channel, over the pilot
   window -- checked directly against the per-pixel category array, not
   inferred. v0 does not reliably catch these. Per the review:
   `iterative_refine()` below fits once, finds new outliers directly
   from the fit residual (things v0's bitfield misses), adds them to
   the mask, and refits.
3. **Masked/weighted PCA**, not naive SVD on a residual matrix with
   flagged pixels zeroed. `masked_pca()` computes a weighted channel
   covariance (`sum_t w_ti w_tj x_ti x_tj / sum_t w_ti w_tj`, weights
   divided back out -- a real weighted covariance, not zero-fill-and-
   pretend-N-is-unchanged) so flagged pixels stop dominating the basis.
4. Waterfall plotting helpers for the (data | residual | residual x
   mask) three-panel deliverable live in the notebook builder, not
   here, since they're pure presentation of these arrays.

Flags: v0's own per-(time,channel) category bitfield (regenerated,
unmodified) OR the campaign-wide self-comb/transmitter channel mask
already built and validated for B15's PCA exclusion
(`b15_event_survey.self_comb_channel_mask`) -- reused here as
instructed, not re-derived. Plus, after `iterative_refine()`, any
residual-identified outliers on top of that.

Model: hera_filters.dspec.fourier_filter, mode='dpss_leastsq', a single
delay window centered at zero delay, half-width 40 ns (starting point
per the ticket). x is frequency in Hz; **filter_half_widths must be
passed in SECONDS despite the installed hera_filters' own docstring
saying "nanosec"** -- passing 40 (as literal nanoseconds) makes the
internal DPSS time-bandwidth product balloon and raises
`ValueError: NW must be less than M/2`; passing 40e-9 works. This is a
real installed-version behavior, not a guess -- confirmed by triggering
the error and reading the traceback back to dpss_operator's NW
computation before fixing it.

Restricted to the analysis band (45-235 MHz), same convention as v0's
own `detectors.BAND_ANALYSIS` -- not reinvented here.

hera_filters availability: confirmed present in the arp env
(`hera_filters.dspec.fourier_filter`, `dpss_operator`,
`DPSS_DEFAULTS_1D`) -- not silently substituted for anything else.
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

from eigsep_data.flagging import build_masks as B
from eigsep_data.flagging import detectors as D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # sibling study modules in this directory
import b15_event_survey as EV  # noqa: E402
import hera_filters.dspec as dspec  # noqa: E402

DATA_ROOT = EV.DATA_ROOT  # portable default + MARJUM_DATA_ROOT override, see b15_event_survey.py
# box-gnd, box-air, box-gnd-on-a-different-SNAP-input. "3" is added
# 2026-09-17 per Aaron's direct confirmation: input 3 is the SAME
# physical box-gnd as input 0, just wired to a different SNAP input
# during campaign Phase A(late)/B (curation/select_files.py's
# PHASE_INPUTS: "3 (+4)" for A, "3","4" for B) -- not a different
# antenna, so the same fit/mask logic applies unchanged, just reading
# a different raw key. "0" and "3" never co-occur as live raw groups
# in the same file (checked directly), so there is no ambiguity about
# which one to prefer. Deliberately NOT extended to "1"/"5": per that
# same table, those are literal MUX COPIES of "0"/"4" (duplicate
# signal, not new sky coverage) -- fitting them would be redundant,
# not additional RFI-environment information.
INPUTS = ("0", "3", "4")
FILTER_HALF_WIDTH_NS = 40.0
FILTER_CENTER_NS = 0.0
REFINE_NSIG = 6.0  # matches v0's persistent_track clip_sigma convention

modes = B.load_mode_table(os.path.join(DATA_ROOT, "curation", "mode_table.jsonl"))


def _dpss_fit(freqs_hz, amp, wgts):
    mdl, res, info = dspec.fourier_filter(
        freqs_hz, amp, wgts,
        filter_centers=[FILTER_CENTER_NS * 1e-9],
        filter_half_widths=[FILTER_HALF_WIDTH_NS * 1e-9],
        mode="dpss_leastsq", filter_dims=1,
        **dspec.DPSS_DEFAULTS_1D,
    )
    return np.real(mdl), np.real(res), info


def fit_file(path, fname, inp, self_mask, band, refine=True):
    """Returns a dict with freqs_hz, amp (data, LINEAR amplitude --
    raw counts, rev 4 correction), model, residual, flagged (mask
    actually used in the fit), and n_refined (count of pixels added by
    iterative_refine), for one file/antenna. None on error.
    """
    m = B.mode_for(modes, fname)
    tx_on = bool(m and m.get("tx_comb") == "on")
    _fname, per_input, freqs_mhz, err = B.process_file((path, tx_on))
    if err or inp not in per_input:
        return None
    cat = per_input[inp]["cat"]

    with h5py.File(path, "r") as h:
        raw = h["data/" + inp][:].astype(np.float64)
    # Rev 4: fit in LINEAR amplitude (raw counts), not log10. Aaron's
    # correction (2026-09-16): linear is the correct unit for fitting
    # and amplitudes; log stays fine for plots/display only. No clamp
    # needed here (unlike the log10 version) -- overflow (raw < 0) is
    # already caught by v0's OVERFLOW bit -> zero-weighted below, so a
    # stray negative sample just carries zero weight into the fit.
    amp_full = raw

    flagged = (cat != D.CLEAN) | self_mask[None, :]
    freqs_hz = freqs_mhz[band] * 1e6
    amp = amp_full[:, band]
    wgts = np.where(flagged[:, band], 0.0, 1.0)

    model, residual, info = _dpss_fit(freqs_hz, amp, wgts)
    n_refined = 0
    fair_std_before = fair_std_after = None

    if refine:
        # Find outliers the fit residual itself reveals that v0's
        # bitfield + self-comb mask did NOT already flag -- this is
        # the loop the review asked for, run once (not to convergence,
        # stated explicitly below in the notebook).
        #
        # Scale is computed PER CHANNEL (across time, within this
        # file), not one global number for the whole file: a single
        # file-wide MAD mixes channels with genuinely different
        # residual scatter (band-edge rolloff, channels adjacent to
        # persistent structure, ...), which over-thresholds the
        # quietest channels and under-thresholds the noisiest ones. A
        # first version of this used one global scale and flagged an
        # implausible ~2% of all pixels at nsig=6; per-channel scale is
        # the statistically correct comparison and is checked against
        # that finding, not assumed to fix it.
        still_good = wgts > 0
        new_outliers = np.zeros_like(still_good)
        for j in range(residual.shape[1]):
            col_good = still_good[:, j]
            if col_good.sum() < 20:
                continue
            col = residual[col_good, j]
            scale = 1.4826 * np.median(np.abs(col - np.median(col)))
            if scale > 1e-9:
                new_outliers[:, j] = col_good & (np.abs(residual[:, j]) > REFINE_NSIG * scale)
        n_refined = int(new_outliers.sum())
        if n_refined:
            # Fair, apples-to-apples check of Aaron's claim: refitting
            # with the enlarged mask can only improve (or leave
            # unchanged) the model's agreement with the data that
            # remains unflagged in BOTH the before and after fit -- so
            # compare both models against exactly that final-good set,
            # not against "all pixels including the ones just zeroed
            # out" (which is what a prior version of this diagnostic
            # did, and is not a fair comparison).
            final_good = still_good & ~new_outliers
            model_before = model
            wgts = np.where(new_outliers, 0.0, wgts)
            flagged_full = flagged.copy()
            flagged_full[:, band] = flagged_full[:, band] | new_outliers
            flagged = flagged_full
            model, residual, info = _dpss_fit(freqs_hz, amp, wgts)
            resid_before_final = (amp - model_before)[final_good]
            resid_after_final = (amp - model)[final_good]
            fair_std_before = float(np.std(resid_before_final))
            fair_std_after = float(np.std(resid_after_final))

    n_terms = info.get("filter_params", {}).get("axis_1", {}).get(
        "basis_options", {})
    return {
        "freqs_hz": freqs_hz,
        "amp": amp,             # linear amplitude data, counts (what was fit)
        "data": amp,            # alias for readability in the notebook
        "model": model,
        "residual": residual,
        "flagged": flagged[:, band],
        "n_refined": n_refined,
        "n_terms_info": n_terms,
        "fair_std_before": fair_std_before,
        "fair_std_after": fair_std_after,
    }


def masked_pca(X, W, n_modes=10):
    """Weighted/masked PCA: a real weighted covariance, not naive SVD
    on a residual matrix with flagged pixels zeroed to 0 and treated as
    measured. `X`: (n_time, n_chan) residual. `W`: (n_time, n_chan)
    weights (0/1 here, but works for soft weights).

    C_jk = sum_t W_tj W_tk X_tj X_tk / sum_t W_tj W_tk

    -- the weights are divided back out (`/ sum_t W_tj W_tk`), so a
    pair of channels that are jointly unflagged less often still gets a
    correctly *normalized average* covariance instead of an
    artificially shrunken one from a zero-filled matrix over the full
    (unweighted) sample count. Returns (var_frac, eigenvectors) sorted
    by descending eigenvalue, eigenvectors are rows (n_modes, n_chan).
    """
    Xw = X * W
    # Weighted mean per channel, using only that channel's own weight
    # (not the pairwise weight -- this matches how a weighted mean is
    # ordinarily defined, one axis at a time).
    w_sum = W.sum(axis=0)
    w_sum_safe = np.where(w_sum > 0, w_sum, 1.0)
    mean_j = Xw.sum(axis=0) / w_sum_safe
    Xc = np.where(W > 0, X - mean_j[None, :], 0.0) * W

    pair_w = W.T @ W  # (n_chan, n_chan), sum_t W_tj W_tk
    cov_num = Xc.T @ Xc  # sum_t W_tj X_tj W_tk X_tk (mean already removed & weighted)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(pair_w > 0, cov_num / np.where(pair_w > 0, pair_w, 1.0), 0.0)
    # Symmetrize defensively (floating point).
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order].T  # rows are eigenvectors now
    var_frac = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
    return var_frac[:n_modes], eigvecs[:n_modes]


DERIVED_DIR = os.path.join(DATA_ROOT, "derived", "smooth_model", "v0")


def write_companion(path, fname, results_by_input, band, freqs_mhz):
    """Write the smooth model to `derived/smooth_model/v0/<original
    filename>` -- **not** into `data/` and **not** as a same-directory
    sibling with a suffix.

    2026-09-15 incident, fixed same day: an earlier version of this
    function wrote `<path in data/>.smooth_model.h5` directly beside the
    source file. That matched the bare `data/*.h5` glob every other
    pipeline in this campaign uses (`load_v007_data`,
    `data_space_rfi_mask`, ...), which crashed at least one reader and
    silently shifted beam-analyst's negative-index file-count-dependent
    window slicing -- caught by beam-analyst, stopped by
    experimental-strategist. `data/` must stay a closed, raw, immutable
    set; per-file derived products now have a documented home
    (`INDEX.md`, "2026-09-15 convention"): `derived/<kind>/v0/`, same
    base filename, no double `.h5.<suffix>` suffix (the directory itself
    disambiguates), with a `manifest.json` beside them.
    """
    os.makedirs(DERIVED_DIR, exist_ok=True)
    out_path = os.path.join(DERIVED_DIR, fname)
    with h5py.File(out_path, "w") as h:
        h.attrs["product"] = "smooth_band_model"
        h.attrs["campaign"] = "marjum-2026-07"
        h.attrs["source_file"] = fname
        h.attrs["method"] = "hera_filters.dspec.fourier_filter, dpss_leastsq"
        h.attrs["amplitude_domain"] = "linear (raw counts) -- rev 4, Aaron's 2026-09-16 correction"
        h.attrs["filter_center_ns"] = FILTER_CENTER_NS
        h.attrs["filter_half_width_ns"] = FILTER_HALF_WIDTH_NS
        h.attrs["flag_basis"] = ("flags/v0 category bitfield (regenerated) "
                                  "OR campaign-wide self-comb channel mask, "
                                  "OR iterative residual-based refinement "
                                  f"({REFINE_NSIG} sigma, one pass)")
        h.create_dataset("freqs_mhz", data=freqs_mhz[band])
        for inp, res in results_by_input.items():
            g = h.create_group(f"input_{inp}")
            g.create_dataset("model", data=res["model"].astype(np.float32),
                              compression="gzip", compression_opts=4)
            g.create_dataset("residual", data=res["residual"].astype(np.float32),
                              compression="gzip", compression_opts=4)
            g.create_dataset("flagged", data=res["flagged"].astype(np.uint8),
                              compression="gzip", compression_opts=4)
    return out_path


if __name__ == "__main__":
    from datetime import datetime, timezone

    self_freqs, self_mask = EV.self_comb_channel_mask()

    t0 = datetime(2026, 7, 17, 20, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 7, 17, 21, 0, 0, tzinfo=timezone.utc)
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "data", "corr_20260717_??????Z.h5")))
    win = [f for f in files if t0 <= D.file_close_time(os.path.basename(f)) <= t1]
    print(f"pilot window: {len(win)} files")

    band = (self_freqs >= D.BAND_ANALYSIS[0]) & (self_freqs <= D.BAND_ANALYSIS[1])
    n_written = 0
    all_residuals = {k: [] for k in INPUTS}
    n_refined_total = {k: 0 for k in INPUTS}
    for path in win:
        fname = os.path.basename(path)
        results_by_input = {}
        for inp in INPUTS:
            r = fit_file(path, fname, inp, self_mask, band)
            if r is None:
                continue
            results_by_input[inp] = r
            all_residuals[inp].append(r["residual"])
            n_refined_total[inp] += r["n_refined"]
        if results_by_input:
            out = write_companion(path, fname, results_by_input, band, self_freqs)
            n_written += 1
    print(f"wrote {n_written} companion files")
    for inp in INPUTS:
        if all_residuals[inp]:
            stacked = np.concatenate(all_residuals[inp], axis=0)
            print(f"input {inp}: residual matrix {stacked.shape}, "
                  f"std={np.std(stacked):.3g}, n_refined={n_refined_total[inp]}")
