"""Prototype RFI flagger for marjum-2026-07 -- cross-correlation based.

STATUS: prototype for inspection. Not a product; writes nothing.

I/O goes through `eigsep_data`, not h5py. Nothing here opens a raw file,
resolves an input key, or parses a filename for a timestamp: a
`Selection` comes in, `load_bundle` does the reading, and a `Bundle`
with its `flags` filled in comes out. That is deliberate -- an earlier
version of this module reimplemented the loader badly enough to have its
own per-phase key table (`PHASE_PAIR`), its own filename-time parser and
its own header reader, and each was a place to be wrong independently of
the four other implementations of the same join that `bundle.py` exists
to replace.

**Existing flags are never read.** Every `load_bundle` call here passes
`products=[]`. This detector's job is to produce a mask, and seeding it
from `flags/v0` or `flags/v2` -- whose DPSS bit is itself an artefact --
would make it impossible to say what it found on its own.

What comes back is a real `eigsep_data.bundle.Bundle`, so
`b.data`, `b.freqs_mhz`, `b.t`, `b.meta` and `b.provenance` mean what
they mean everywhere else, and `b.flags` returns this module's bitfield
because it is filled into `b.products["flags"]["mask"]` under the same
key `products/flags.py` uses. Code written against a bundle loaded from
disk works against one of these unchanged, which is the point: when the
algorithm settles, writing these masks out in the `flags/<version>/`
schema and loading them back should change nothing for a consumer.

Why the method is shaped the way it is
--------------------------------------
Two detectors that see different things, plus a second pass that can
change its mind:

    raw_transient_flags   per channel along time, raw autos, no model
    coherence + z_coh     |V_ab|/sqrt(Pa Pb) against a Rayleigh floor
    line_statistic        per-channel floor detrended along frequency
    broadband_times       whole integrations when the band lights up
    (then) DPSS fit on what survives -> auto_residual_z -> re-decide

The thresholds are calibrated rather than tuned. For a cross,
|r| is Rayleigh with sigma = 1/sqrt(2N); for an auto, the scatter about
the continuum is model/sqrt(N); N = dnu*tau comes from the index's own
`integration_time` column. `nsig` defaults to 6 on all three z-scores.

See `rfi_flag_prototype.ipynb` for the measurements behind each choice.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, binary_dilation

# ---------------------------------------------------------------- constants

N_CHAN = 1024
CHAN_WIDTH_MHZ = 250.0 / 1024.0

# Same analysis band as flagging/detectors.py BAND_ANALYSIS.
BAND_ANALYSIS = (45.0, 235.0)

# Known emitter bands, from flagging/detectors.py. Excluded from every
# fitted scale and from the continuum fit, as v003 cell 8 does before it
# computes anything. Still reported as flagged -- excluded from the
# statistics, not from the product.
A_PRIORI_BANDS = {
    "FM": (88.0, 108.0),
    "ORBCOMM": (136.5, 138.5),
}

# Digital clock harmonics of the 250 MHz sample clock: ch 512 = 125.000 MHz
# (fs/2) and ch 768 = 187.500 MHz (3fs/4). Both show up at high coherence
# and are instrumental.
CLOCK_HARMONIC_CHANS = (512, 768)

# Bands that move TOGETHER under micrometeor scattering.
METEOR_BANDS = {
    "FM": (88.0, 108.0),
    "DTV_LO": (54.0, 88.0),
    "DTV_HI": (174.0, 216.0),
}

# The 2026 digital self-comb: 1.953125 MHz = exactly 8 channels.
SELF_COMB_SPACING_CHAN = 8

# Band for the gross-power monitor: above the analysis band the bandpass
# has rolled off ~3 decades, but a broadband transient still lands in it.
BAND_QUIET = (236.0, 249.0)

# ------------------------------------------------------------ the bitfield
#
# Same shape as flags/<version>/flag_bits.json, so this can be written out
# in the existing schema without reinterpretation. uint16, and deliberately
# NOT overlapping v0's bit meanings -- these are different categories
# produced by a different detector, and reusing v0's numbering would invite
# a consumer to read one as the other.

CLEAN = 0
BIT_OFF_SKY = 1 << 0        # receiver on a load, not the antenna
BIT_OVERFLOW = 1 << 1       # int32 accumulator wrap
BIT_A_PRIORI = 1 << 2       # known emitter band, masked before any statistic
BIT_TRANSIENT = 1 << 3      # per-channel time outlier on the raw autos
BIT_BROADBAND = 1 << 4      # whole integration: large fraction of band lit
BIT_GROSS = 1 << 5          # whole integration: quiet-band power excursion
BIT_METEOR = 1 << 6         # whole integration: FM and DTV rose together
BIT_COHERENT = 1 << 7       # per-pixel cross-correlation outlier
BIT_LINE = 1 << 8           # persistent narrow line (whole channel)
BIT_OCCUPANCY = 1 << 9      # channel flagged too often to be worth keeping
BIT_RETRACTED = 1 << 10     # flagged in pass 1, withdrawn after the DPSS fit
BIT_RESIDUAL = 1 << 11      # post-DPSS auto-residual outlier (second pass)

FLAG_BITS = {
    "encoding": "uint16 bitfield per (time, channel)",
    "axes": ["time", "channel"],
    "clean_value": 0,
    "bits": [
        {"bit": 0, "value": int(BIT_OFF_SKY), "name": "off_sky", "rfi": False,
         "meaning": "rfswitch != RFANT; receiver on a load, not sky"},
        {"bit": 1, "value": int(BIT_OVERFLOW), "name": "overflow", "rfi": False,
         "meaning": "int32 accumulator wrap (negative auto)"},
        {"bit": 2, "value": int(BIT_A_PRIORI), "name": "a_priori", "rfi": True,
         "meaning": "known emitter band / clock harmonic / comb tooth, "
                    "masked before any statistic was computed"},
        {"bit": 3, "value": int(BIT_TRANSIENT), "name": "transient", "rfi": True,
         "meaning": "per-channel time outlier on the raw autos, model-free"},
        {"bit": 4, "value": int(BIT_BROADBAND), "name": "broadband", "rfi": True,
         "meaning": "whole integration: >frac of the band tripped at once"},
        {"bit": 5, "value": int(BIT_GROSS), "name": "gross_power", "rfi": True,
         "meaning": "whole integration: quiet-band power excursion"},
        {"bit": 6, "value": int(BIT_METEOR), "name": "meteor", "rfi": True,
         "meaning": "whole integration: FM and DTV rose together"},
        {"bit": 7, "value": int(BIT_COHERENT), "name": "coherent", "rfi": True,
         "meaning": "per-pixel outlier in |V_ab|/sqrt(Pa Pb) vs Rayleigh floor"},
        {"bit": 8, "value": int(BIT_LINE), "name": "line", "rfi": True,
         "meaning": "persistent narrow line: channel noise floor stands "
                    "above its neighbours'"},
        {"bit": 9, "value": int(BIT_OCCUPANCY), "name": "occupancy", "rfi": True,
         "meaning": "channel flagged in more than frac of its valid samples"},
        {"bit": 11, "value": int(BIT_RESIDUAL), "name": "residual", "rfi": True,
         "meaning": "stands above the radiometric noise of the auto after the "
                    "DPSS continuum is removed (second pass). Distinct from "
                    "'coherent': it needs no cross, so it is the only "
                    "per-pixel evidence available where the cross is "
                    "quantisation-limited."},
        {"bit": 10, "value": int(BIT_RETRACTED), "name": "retracted",
         "rfi": False,
         "meaning": "raised in pass 1 and withdrawn after the DPSS fit; "
                    "informational, not an RFI claim"},
    ],
}

#: Bits that mean "interference was detected here".
RFI_BITS = (BIT_A_PRIORI | BIT_TRANSIENT | BIT_BROADBAND | BIT_GROSS
            | BIT_METEOR | BIT_COHERENT | BIT_LINE | BIT_OCCUPANCY
            | BIT_RESIDUAL)
#: Bits that mean "this sample is not usable", for reasons that are not RFI.
UNUSABLE_BITS = BIT_OFF_SKY | BIT_OVERFLOW


# --------------------------------------------------------------- stage 0-1

def band_masks(freqs):
    """(in_band, a_priori_excluded, quiet_band) channel masks."""
    in_band = (freqs >= BAND_ANALYSIS[0]) & (freqs <= BAND_ANALYSIS[1])
    excl = np.zeros_like(in_band)
    for lo, hi in A_PRIORI_BANDS.values():
        excl |= (freqs >= lo) & (freqs <= hi)
    excl[list(CLOCK_HARMONIC_CHANS)] = True
    quiet = (freqs >= BAND_QUIET[0]) & (freqs <= BAND_QUIET[1])
    return in_band, excl, quiet


def self_comb_chans(n_chan=N_CHAN, spacing=SELF_COMB_SPACING_CHAN,
                    halfwidth=1):
    """Channels on the digital self-comb grid, PLUS `halfwidth` either side.

    The margin is not padding-for-safety, it is measured. On
    corr_20260717_201113Z.h5 the median coherence is 0.858 on the teeth
    (ch%8==0) and ~0.057 everywhere else -- and of the 152 channels the
    detector then lights up outside the a priori mask, 73 sit at ch%8==1 and
    74 at ch%8==7. Every one is an immediate neighbour of a tooth: the
    channelizer leaks, and the leak is far above the detection threshold.

    A comb mask that marks only the exact teeth therefore leaves 2 of every
    8 channels looking like fresh detections -- 20.8% of the band on that
    window, which is the same order as the overflagging this prototype
    exists to fix, arrived at by a completely different route. With
    halfwidth=1 the same window detects 2.1%.
    """
    m = np.zeros(n_chan, dtype=bool)
    idx = np.arange(0, n_chan, spacing)
    for d in range(-halfwidth, halfwidth + 1):
        j = idx + d
        m[j[(j >= 0) & (j < n_chan)]] = True
    return m


def on_sky(meta, unknown="assume_sky"):
    """Rows where the receiver is looking at the ANTENNA.

    `rfswitch` is an index column, so this is a column comparison rather than
    a metadata parse. Not optional: the switch schedule cycles
    RFANT / RFNOFF / RFNON / RFAMB / RFSP1_* every hour, and whole files are
    the receiver staring at a load -- corr_20260714_144344Z.h5 is 240
    integrations of RFAMB end to end.

    **`MISSING` is not `off sky`.** The switch stream is absent from the early
    campaign files: corr_20260712_220026Z.h5 and corr_20260713_023613Z.h5 are
    240 rows of `MISSING` each. Testing `== "RFANT"` marks those rows off-sky,
    which disables every detector downstream -- the first two campaign days
    came back with nothing flagged but the a priori mask, and read as a clean
    sky rather than as an absent measurement.

    So `unknown` decides explicitly, and defaults to the conservative
    direction for a FLAGGER: assume the data is on sky and let the detectors
    run. Excluding a little calibration data from a mask is a far smaller
    error than producing no mask at all for two days. Pass
    `unknown="assume_load"` to get the strict reading.
    """
    state = meta["rfswitch"].astype(str).str.upper()
    known = ~state.isin(["MISSING", "NAN", "NONE", ""])
    is_ant = state == "RFANT"
    if unknown == "assume_sky":
        return np.asarray(is_ant | ~known)
    if unknown == "assume_load":
        return np.asarray(is_ant)
    raise ValueError("unknown must be 'assume_sky' or 'assume_load'")


def overflow(p):
    """int32 accumulator wrap. Autos are non-negative by construction, so a
    negative sample is unambiguous -- no threshold. Same test as
    flagging/detectors.overflow_mask()."""
    return p < 0


def raw_transient_flags(p, n_samples, nsig=6.0, n_false=None, med_width=9,
                        empirical_floor=True, min_counts=16.0):
    """Plain statistical outlier test on the RAW autos, per channel along time.

    No model, no cross, no DPSS. This should have been the first stage of the
    pipeline from the start and was not: the prototype went straight to
    coherence, which needs a working cross correlation and says nothing
    outside the channels it is computed on. The cost of leaving it out was
    concrete -- a single-channel event 855x above its own time median sat
    unflagged because the only pre-DPSS detector was the coherence one, and
    the entire 236-249 MHz range had no per-channel test at all.

    The scale is the radiometric one again, so this needs no tuning: an auto
    averaged over N samples has fractional noise 1/sqrt(N), so the expected
    scatter about a channel's own time trend is `trend / sqrt(N)`. At
    N = 131072 that is 0.28%, and a 855x excursion is z ~ 3e5. These events
    are not marginal detections; nothing subtle was required to find them.

    A running median along time (`med_width`) rather than a flat median, so
    slow gain drift is not mistaken for a transient -- same construction as
    `flagging/detectors.transient_track()`, which v0 has always had and which
    this prototype dropped when it replaced the auto path with coherence.

    One-sided: interference adds power.

    Runs over ALL channels, including outside BAND_ANALYSIS. The analysis band
    is where the science is, but the rest of the spectrum is where the cleanest
    diagnostics live, and a detector that cannot see 238 MHz cannot tell you
    that your 236-249 MHz quiet-band monitor has a transmitter in it.
    """
    trend = median_filter(p, size=(med_width, 1), mode="nearest")
    resid_ = p - trend
    sig_rad = np.maximum(trend, 0.0) / np.sqrt(max(n_samples, 1.0))
    sig = sig_rad
    if empirical_floor:
        # The radiometric scale is only the right one where the channel
        # actually carries power. Measured on corr_20260717_034832Z.h5: in
        # mid-band the empirical MAD sits within 2-30% of the radiometric
        # prediction (ratio 1.02-1.29), but at 238 MHz it is 13x it and at
        # 14 MHz 537x -- those channels are dominated by quantisation and the
        # ADC noise floor, not by the averaged-power statistics the radiometer
        # equation describes. Taking the larger of the two keeps the test
        # calibrated where the physics applies and honest where it does not.
        sig_emp = 1.4826 * np.median(
            np.abs(resid_ - np.median(resid_, axis=0)), axis=0)
        sig = np.maximum(sig_rad, sig_emp[None, :])

    # Channels with essentially no power cannot be tested at all: below ~25 MHz
    # the median is 0 or 0.5 counts, so every scale estimate collapses to zero
    # and any single count becomes an arbitrarily large z. A first version of
    # this flagged channels 1-8 at z ~ 4e4, which is a division by zero wearing
    # a detection's clothes. `min_counts` is the smallest median power a
    # channel must carry to be worth testing.
    testable = np.median(p, axis=0) >= min_counts
    if not testable.any():
        # Nothing in this file carries enough power to test. Report nothing
        # rather than everything.
        zero = np.zeros(p.shape, dtype=bool)
        return zero, np.zeros(p.shape), np.inf
    sig = np.maximum(sig, 1.0)          # integer counts: never claim sub-LSB
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid_ / np.where(sig > 0, sig, np.inf)
    z = np.where(np.isfinite(z), z, 0.0)
    z = np.where(testable[None, :], z, 0.0)

    # Threshold: a plain n-sigma cut by default, because that is the knob
    # people actually want to reason about. `nsig=6` on a 240 x ~930 testable
    # file corresponds to about 2e-4 expected false positives -- roughly four
    # orders of magnitude stricter than the `n_false=1` setting it replaces
    # (z = 4.44). Pass `nsig=None, n_false=<k>` to go back to specifying a
    # tolerated false-positive count instead; the two are alternatives, not
    # both applied.
    if nsig is not None:
        thr = float(nsig)
    else:
        thr = gaussian_threshold(
            max(int(testable.sum()) * p.shape[0], 1),
            1.0 if n_false is None else n_false)
    return z > thr, z, thr


def broadband_times(transient, in_band, frac=0.15):
    """Integrations where a large fraction of the band tripped the transient
    test at once -- flag the WHOLE integration, not just the channels that
    individually cleared threshold.

    The per-channel test alone is not enough for a broadband event, and
    integration 89 of corr_20260717_034832Z.h5 is the clean demonstration.
    It carries a +4.1% excess across 25-100 MHz on box-air only, falling
    monotonically with frequency (5.1% at 40-50 MHz, 0.6% at 120-130 MHz),
    for one integration. 24% of in-band channels clear 6 sigma -- and the
    other 76% are contaminated at the same few-percent level while sitting
    individually below threshold. Leaving them in means the DPSS model for
    that row is fitted to 498 contaminated channels, which is precisely the
    "scattering off unflagged RFI" the residual panel shows.

    None of the other time tests see it: the quiet-band monitor looks at
    236-249 MHz and the event stops around 109 MHz, while the FM/DTV test
    normalises by a reference continuum that the event lifts too, so it
    cancels itself out. This is the case they do not cover.

    `flagging/detectors.broadband_times()` is the same test with the same
    default, and it is the morphological discriminator v0 has always used to
    separate broadband transients (airplane reflections, lightning) from
    narrowband emitters. This prototype dropped it along with
    `transient_track` and had to rediscover the need.

    On that file the threshold is not delicate: the in-band transient fraction
    has median 0.004 and p99 0.17, and 0.15 selects 4 integrations of 240
    (89, 128, 164, 222) while 0.25 selects none.
    """
    if in_band.sum() == 0:
        return np.zeros(transient.shape[0], dtype=bool)
    return transient[:, in_band].mean(axis=1) > frac


def gross_power_time_flags(pa, pb, quiet, nsig=5.0):
    """Per-integration flags from binned power in the quiet band.

    v003 cell 8's `pwr_oob`: mean over a band with no sky in it, median
    removed, scaled by its own MAD. One number per integration, so it costs
    nothing and is completely independent of any spectral model. Two-sided
    here on purpose -- a dropout (power vanishing) is as much a data-quality
    event as a transient, and v004's whole subject is dropouts.
    """
    flags = np.zeros(pa.shape[0], dtype=bool)
    if quiet.sum() < 4:
        return flags
    for p in (pa, pb):
        # MEDIAN, not mean, across the quiet band. The band is not as quiet as
        # its name suggests -- 238.037 MHz carries a narrow emitter that hits
        # 1806x its own median on corr_20260717_034832Z.h5 -- and a mean over
        # 53 channels is dragged bodily by one such channel. That made this
        # statistic fire on narrow events it was never meant to detect, which
        # then suppressed the per-pixel flags on those integrations. A median
        # responds to a genuine broadband lift and ignores a single tone,
        # which is the division of labour intended: narrow lines belong to
        # `raw_transient_flags`, broadband excursions belong here.
        pwr = np.median(p[:, quiet], axis=1)
        pwr = pwr - np.median(pwr)
        mad = 1.4826 * np.median(np.abs(pwr))
        if mad <= 0:
            continue
        # The quiet band is quantisation-limited, not radiometric: its
        # per-integration median scatters by 6.6% where the radiometer
        # equation predicts 0.04%, and the distribution is heavy-tailed and
        # integer-valued. An n-sigma cut assuming Gaussianity fired on 21 of
        # 240 integrations here. Take the threshold from the band's own
        # empirical distribution instead, with nsig as a floor.
        hi = np.percentile(np.abs(pwr), 99.0)
        flags |= np.abs(pwr) > max(nsig * mad, hi * 1.5)
    return flags


# ----------------------------------------------------------------- stage 2

def quantisation_ok(cross, in_band, max_zero_frac=0.02):
    """Is the cross big enough, in counts, for the coherence to mean anything?

    Found the hard way in phase A. `data/02` there has a median |V| of ~500
    counts against autos of ~1.6e5, and 21-38% of in-band samples have |V|
    EXACTLY zero -- the int32 cross has rounded them away. A quantile-fitted
    Rayleigh sigma on that is 0, the z-score is meaningless, and the detector
    silently reports a clean sky.

    So this is a precondition, not a quality score: below it the
    cross-correlation detector does not apply and the file needs the
    auto-only path instead.
    """
    z = np.mean(np.abs(cross[:, in_band]) == 0)
    return bool(z <= max_zero_frac), float(z)


def coherence(pa, pb, cross, floor=1.0):
    """|V_ab| / sqrt(P_a P_b) -- the fraction of the two autos' amplitude
    that is actually correlated between the antennas.

    This is the statistic the auto-only pipeline does not have. Receiver
    noise is independent between the two receivers and averages down in the
    cross; a common interferer does not. It is also naturally normalised:
    the bandpass, the ~648 ns reflection ripple, and the accumulator-length
    doubling all divide out, so a single threshold means the same thing at
    50 MHz and at 200 MHz, and on both sides of the 07-15 acc_len split.
    That is what lets step 4 use one percentile for the whole band.
    """
    denom = np.sqrt(np.maximum(pa, floor) * np.maximum(pb, floor))
    return np.abs(cross) / denom


def n_independent_samples(meta):
    """N = delta_nu * tau per row, from the index's own columns.

    `integration_time` and `nchan` are already in `Selection.meta`, so this
    opens nothing. On a phase-C row that is 0.244140625e6 * 0.536870912 =
    131072, which equals `corr_acc_len / nchan` -- and it halves before the
    07-15 15:55 accumulator-length doubling, which is exactly why it is read
    per row and never hardcoded.
    """
    nchan = np.asarray(meta["nchan"], dtype=float)
    tau = np.asarray(meta["integration_time"], dtype=float)
    dnu = 250e6 / np.where(nchan > 0, nchan, np.nan)
    return dnu * tau


def rayleigh_sigma_theory(n_samples):
    """Radiometric noise floor of the coherence.

    For two inputs whose noise is independent, the normalised cross
    correlation r = V_ab / sqrt(P_a P_b) has real and imaginary parts that are
    each zero-mean Gaussian with variance 1/(2N). So |r| is RAYLEIGH with
    sigma = 1/sqrt(2N), and everything follows without a single tuned
    parameter:

        median|r| = sigma * sqrt(2 ln 2)
        P(|r| > z * sigma) = exp(-z^2 / 2)

    That last line is the point. A threshold can be set from a false-alarm
    rate instead of from a percentile or an n-sigma guess, and it means the
    same thing in every channel, every file and every campaign phase.
    """
    return 1.0 / np.sqrt(2.0 * n_samples)


def rayleigh_sigma_empirical(coh, keep, q=0.25):
    """Per-channel Rayleigh sigma, fitted from a LOW quantile.

    Inverting the Rayleigh CDF: sigma = quantile_q / sqrt(-2 ln(1-q)).

    The quantile has to be a low one. The upper half of the distribution is
    where the RFI lives -- measured on corr_20260717_034832Z.h5, the 10th and
    25th percentiles track Rayleigh to within 1% and 2%, while the 90th sits
    50% high and the 99th is 3x high. Fitting the scale on the median or above
    would be fitting it partly to the signal being searched for.

    Fitted per channel rather than once per file because the true floor is not
    only radiometric: the two antennas share some sky and some cross-talk, and
    that common-mode contribution is frequency-dependent. Note what this costs:
    a channel carrying a tone at ALL times has an inflated sigma and hides from
    this test. That is what `line_statistic()` is for.
    """
    denom = np.sqrt(-2.0 * np.log(1.0 - q))
    n_chan = coh.shape[1]
    sig = np.full(n_chan, np.nan)
    for j in range(n_chan):
        col = coh[keep[:, j], j] if keep.ndim == 2 else coh[:, j]
        if col.size >= 20:
            sig[j] = np.quantile(col, q) / denom
    return sig


def false_alarm_threshold(n_pixels, n_expected_false=1.0):
    """Rayleigh z at which `n_expected_false` pixels survive by chance.

    P(z) = exp(-z^2/2), so z = sqrt(-2 ln(p)). For a 240 x 778 file and one
    tolerated false positive that is z = 5.0. This replaces the n-sigma guess
    and the percentile cap in one move: it is a statement about how often the
    detector is allowed to be wrong, which is a thing we can actually have an
    opinion about.
    """
    p = max(float(n_expected_false) / max(float(n_pixels), 1.0), 1e-300)
    return float(np.sqrt(-2.0 * np.log(p)))


def gaussian_threshold(n_pixels, n_expected_false=1.0):
    """One-sided Gaussian z for `n_expected_false` survivors, the auto-domain
    counterpart of `false_alarm_threshold()`.

    The two noise models are different and must not share a threshold. |r| in
    the cross is Rayleigh; a power residual in an auto is Gaussian about the
    model. Same false-alarm statement, different quantile function.
    """
    from scipy.special import erfcinv
    # p must stay strictly inside (0, 1): p = 1 gives erfcinv(2) = -inf, and a
    # -inf threshold flags every sample. That is reachable -- three of the
    # campaign's files are entirely zero, so no channel is testable, n_pixels
    # collapses to 1 and n_expected_false=1 makes p exactly 1. The scan that
    # found it reported "240 integrations flagged, max z = 0", which is the
    # shape of a division-by-nothing rather than a detection.
    p = float(n_expected_false) / max(float(n_pixels), 1.0)
    p = min(max(p, 1e-300), 1.0 - 1e-12)
    return float(np.sqrt(2.0) * erfcinv(2.0 * p))


def auto_residual_z(resid, model, n_samples, good=None):
    """Post-DPSS residual in units of the auto's noise -- radiometric where the
    continuum model is adequate, empirical where it is not.

    The radiometric part is computable: for a spectrum averaged over N
    independent samples the scatter about the continuum is `model/sqrt(N)`.
    On files where the 40 ns model tracks the data that prediction is right to
    a few percent -- measured empirical/predicted per-channel scatter of 1.06
    and 1.01 on corr_20260717_034832Z.h5 and corr_20260713_194333Z.h5. The
    autos reach the radiometric limit; the cross coherence sits a factor ~2
    above it, which is itself a clue about that excess since a genuine
    sensitivity shortfall would show in both.

    But that only holds where the model FITS. On the self-comb file and in
    phase B the 40 ns continuum cannot follow the data, the residual is full
    of model error rather than noise, and a threshold set at the radiometric
    prediction fires on 30-48% of pixels -- over-flagging of exactly the kind
    this prototype exists to remove, reintroduced through the back door.

    So the scale used is the LARGER of the radiometric prediction and a robust
    per-channel MAD of the residual itself. Where the model is good the two
    agree and the threshold stays calibrated; where it is bad the empirical
    scale takes over and the detector reports what stands above the *actual*
    scatter rather than above an idealisation of it.

    The ratio between them is returned, because it is a direct and useful
    measure of how badly the continuum model is failing on a given file.
    """
    sig_rad = model / np.sqrt(max(n_samples, 1.0))
    sig_emp = np.full(resid.shape[1], np.nan)
    for j in range(resid.shape[1]):
        col = resid[good[:, j], j] if good is not None else resid[:, j]
        if col.size >= 20:
            sig_emp[j] = 1.4826 * np.median(np.abs(col - np.median(col)))
    rad_chan = np.nanmedian(np.where(sig_rad > 0, sig_rad, np.nan), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        inflation = np.where(rad_chan > 0, sig_emp / rad_chan, np.nan)
    sig = np.maximum(sig_rad, np.nan_to_num(sig_emp, nan=0.0)[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / np.where(sig > 0, sig, np.inf)
    return np.where(np.isfinite(z), z, 0.0), inflation


def combine_decision(z_coh, z_res, n_pixels, n_false=1.0,
                     coh_nsig=6.0, res_nsig=6.0):
    """Final flag from BOTH domains, with provisional flags genuinely revisable.

    A flag raised before the DPSS fit is provisional. Once the continuum is
    modelled we can ask a second, independent question -- does this sample
    still stand above the RADIOMETRIC noise of the auto? -- and the two
    answers together beat either alone.

    The rule has to be built so that the answer can come out "no", or the
    second pass is decoration. So `z_coh` alone does not carry a flag through:

      * `z_res > t_hi`                     flag  (auto evidence, calibrated)
      * `z_coh > t_hi` and `z_res > t_lo`  flag  (cross evidence, corroborated)
      * `z_coh > t_lo` and `z_res > t_lo`  flag  (weak, but in both domains)
      * `z_coh > t_hi` and `z_res <= t_lo` RETRACT

    That last line is the point of the second pass. A pixel can be strongly
    coherent between the two antennas and yet carry no excess power over the
    continuum model -- common-mode gain, a shared systematic, a correlated
    feature of the bandpass. The cross says "correlated", which is not the
    same claim as "interference", and only the auto can settle it.

    The corroboration threshold is set so the JOINT false-alarm probability
    matches the single-domain one: for independent statistics p_lo^2 = p_hi,
    so t_lo is the threshold at sqrt(p_hi). Independence is an assumption and
    only partly true -- both statistics are built from the same autos -- which
    biases the corroborated branch towards over-flagging. It is the branch to
    drop first if false positives ever matter more than sensitivity.

    Returns (final, retracted, promoted, thresholds).
    """
    # Straight n-sigma when given, false-alarm-derived otherwise. The "low"
    # corroboration thresholds are always derived, since their whole point is
    # that the JOINT false-alarm probability matches the single-domain one:
    # p_lo^2 = p_hi, so t_lo is the threshold at sqrt(p_hi).
    if coh_nsig is not None:
        t_hi_coh = float(coh_nsig)
        p_hi_coh = float(np.exp(-0.5 * t_hi_coh ** 2))          # Rayleigh tail
    else:
        t_hi_coh = false_alarm_threshold(n_pixels, n_false)
        p_hi_coh = max(float(n_false) / max(float(n_pixels), 1.0), 1e-300)
    if res_nsig is not None:
        t_hi_res = float(res_nsig)
        from scipy.special import erfc
        p_hi_res = float(0.5 * erfc(t_hi_res / np.sqrt(2.0)))    # Gaussian tail
    else:
        t_hi_res = gaussian_threshold(n_pixels, n_false)
        p_hi_res = max(float(n_false) / max(float(n_pixels), 1.0), 1e-300)

    t_lo_coh = float(np.sqrt(-2.0 * np.log(min(np.sqrt(p_hi_coh), 1 - 1e-12))))
    t_lo_res = gaussian_threshold(1.0 / max(np.sqrt(p_hi_res), 1e-300), 1.0)

    auto_strong = z_res > t_hi_res
    corroborated = (z_coh > t_lo_coh) & (z_res > t_lo_res)
    final = auto_strong | corroborated

    provisional = z_coh > t_hi_coh
    retracted = provisional & ~final
    promoted = final & ~provisional
    return final, retracted, promoted, dict(
        t_hi_coh=t_hi_coh, t_hi_res=t_hi_res,
        t_lo_coh=t_lo_coh, t_lo_res=t_lo_res)


def line_statistic(sigma_chan, keep_chan, width=41):
    """Persistent-line detector: per-channel noise floor, detrended along
    frequency.

    A channel carrying a steady tone has an inflated Rayleigh sigma, and
    `rayleigh_sigma_empirical` cannot see that -- the tone is in every sample,
    so it is in the quantile too. Along frequency it is obvious: the channel's
    floor stands above its neighbours'.

    So the two detectors are cleanly split by what they can see, rather than
    one test being asked to do both jobs:

        z_pixel  (time axis)      transient and intermittent interference
        z_line   (frequency axis) persistent tones, however steady

    This is the piece B16 has no equivalent of at all.
    """
    s = sigma_chan.copy()
    bad = ~np.isfinite(s)
    if bad.all():
        return np.zeros_like(s), np.zeros_like(s)
    s[bad] = np.nanmedian(s)
    smooth = median_filter(s, size=width, mode="nearest")
    resid = s - smooth
    ref = resid[keep_chan & np.isfinite(sigma_chan)]
    if ref.size < 32:
        return resid, np.zeros_like(resid)
    scale = 1.4826 * np.median(np.abs(ref - np.median(ref)))
    if scale <= 0:
        return resid, np.zeros_like(resid)
    return resid, resid / scale


def meteor_scatter_flags(pa, freqs, nsig=6.0, min_bands=2):
    """Times where the FM and DTV bands rise TOGETHER, relative to the
    continuum between them.

    The naive version -- each band's mean power, median-removed, MAD-scaled --
    does not work, and the way it fails is instructive. Measured on a quiet
    phase-A file, the three band excesses correlate at r = 1.000 with each
    other. That is not meteor scatter; it is the receiver's overall gain
    drifting and carrying every band with it. Correlation between the bands
    is therefore worthless as a signature: it is ~1 whether or not anything
    is happening.

    So each band is normalised by a REFERENCE continuum taken from the parts
    of the analysis band with no broadcast service in them (108-136, 155-174
    and 216-235 MHz). A common gain change divides out exactly; a propagation
    event that lights up the transmitter bands and nothing else survives.

    An event is a time where at least `min_bands` of the three exceed `nsig`.
    Requiring several bands together is what makes this a propagation detector
    rather than a power detector: a local broadband transient lifts the
    reference continuum too and cancels itself out here, while one band alone
    is a station fading up, which is not an event.

    Returns (flags, per-band excess) so the joint behaviour can be shown
    rather than asserted.
    """
    ref = np.zeros(freqs.shape, dtype=bool)
    for lo, hi in ((108.0, 136.0), (155.0, 174.0), (216.0, 235.0)):
        ref |= (freqs >= lo) & (freqs < hi)
    if ref.sum() < 8:
        return np.zeros(pa.shape[0], dtype=bool), {}
    ref_p = np.mean(pa[:, ref], axis=1)
    ref_p = np.where(ref_p > 0, ref_p, np.nan)

    exc = {}
    for name, (lo, hi) in METEOR_BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        if sel.sum() < 4:
            continue
        ratio = np.mean(pa[:, sel], axis=1) / ref_p
        med = np.nanmedian(ratio)
        mad = 1.4826 * np.nanmedian(np.abs(ratio - med))
        exc[name] = np.nan_to_num((ratio - med) / mad) if mad > 0 \
            else np.zeros_like(ratio)
    if not exc:
        return np.zeros(pa.shape[0], dtype=bool), {}
    hot = np.sum([e > nsig for e in exc.values()], axis=0)
    return hot >= min_bands, exc


# ----------------------------------------------------------------- stage 3-4

# ----------------------------------------------------------------- stage 5-6

def occupancy_kill(flags, keep, frac=0.30):
    """Channels flagged in more than `frac` of their valid samples are killed
    outright for the whole file.

    v003 cell 8: `bad_chans = np.where(np.sum(1-mask, axis=0) > 10)[0]`, then
    the channel is zeroed everywhere. A tone that is on for most of a file
    gets a ragged, partial mask from any per-sample test -- some samples over
    threshold, some just under -- and the residue is worse than useless,
    because downstream averaging then sees a biased subset. Promoting it to a
    whole-channel kill is the honest thing to do and is how persistent lines
    get caught without a time-variance requirement.
    """
    n_valid = keep.sum(axis=0)
    n_flag = (flags & keep).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        occ = np.where(n_valid > 0, n_flag / np.maximum(n_valid, 1), 0.0)
    return occ > frac, occ


def dilate(flags, n_time=1, n_chan=1):
    """Grow flags by one sample in each axis.

    v003 cell 15 does the time-axis version of this
    (`valid[1:-1] &= valid[:-2] & valid[2:]`). The shoulders of a real event
    sit below any threshold that the core of it clears, and leaving them is
    how a partially-flagged transient biases whatever averages over it.
    """
    if n_time == 0 and n_chan == 0:
        return flags
    struct = np.ones((2 * n_time + 1, 2 * n_chan + 1), dtype=bool)
    return binary_dilation(flags, structure=struct)


# ------------------------------------------------------- DPSS and the fit

def delay_diagnostic(freqs_mhz, median_residual, med_over_mad=None, nsig=6.0):
    """Is a residual's delay structure reflections, or narrow RFI?

    A question worth asking before anyone widens a continuum model to "absorb
    the ripple", because the two look superficially alike in a delay plot and
    the right response to them is opposite.

    A discrete reflection is a localised peak at its own delay. A narrow spike
    in FREQUENCY is a delta function, so it is FLAT in delay -- it fills the
    whole range out to the Nyquist delay 1/(2 dnu), and interference between
    two such spikes puts closely spaced peaks all the way across. Reading
    those peaks as reflection delays and notching them into the model is
    fitting the RFI into the continuum, which is the exact opposite of what
    the model is for.

    This returns the delay spectrum before and after excising the outlier
    channels, plus the concentration of the residual power. If excising a
    handful of channels collapses the delay structure, it was never
    reflections.

    Measured on corr_20260717_034832Z.h5/input_0 after B16's own fit: ONE
    channel (94.48 MHz, the strongest FM carrier) holds 81% of the
    median-residual power and the top three hold 94%; the delay spectrum is
    flat out to the 2048 ns Nyquist delay; excising the spikes drops the
    absolute power above 300 ns by 5.9x and the median-residual rms by 4x.
    It was the FM line and the fit's ringing response to it, throughout.
    """
    med = np.nan_to_num(median_residual)
    dch = (freqs_mhz[1] - freqs_mhz[0]) * 1e6

    def _dsp(x):
        w = np.hanning(x.size)
        return np.abs(np.fft.rfft(x * w))

    dly = np.fft.rfftfreq(med.size, d=dch) * 1e9
    dsp_all = _dsp(med)

    if med_over_mad is None:
        spike = np.zeros(med.size, dtype=bool)
    else:
        spike = np.abs(np.nan_to_num(med_over_mad)) > nsig
    cleaned = med.copy()
    if spike.any() and (~spike).sum() > 4:
        idx = np.arange(med.size)
        cleaned[spike] = np.interp(idx[spike], idx[~spike], med[~spike])
    dsp_clean = _dsp(cleaned)

    p = med ** 2
    order = np.argsort(p)[::-1]
    conc = np.cumsum(p[order]) / max(p.sum(), 1e-300)
    return {
        "delay_ns": dly,
        "nyquist_delay_ns": 1.0 / (2.0 * dch) * 1e9,
        "dsp": dsp_all,
        "dsp_excised": dsp_clean,
        "spike_chans": spike,
        "frac_power_top1": float(conc[0]) if conc.size else np.nan,
        "frac_power_top3": float(conc[min(2, conc.size - 1)]) if conc.size else np.nan,
        "rms_before": float(np.std(med)),
        "rms_after": float(np.std(cleaned)),
    }


def fit_weights(good, freqs_mhz, a_priori=True, clock_harmonics=True):
    """Fit weights that keep the KNOWN strong emitters out of the continuum fit.

    This is the one-line change that matters most to B16, and it is the same
    move v003 cell 8 makes before computing anything (`fmask[chs_fm] = 0`).
    B16 weights by v0's bitfield and the self-comb mask, neither of which
    reliably zeroes the FM band -- so the brightest carrier in the data sits
    inside the least-squares fit, drags the model, and rings.

    Measured on corr_20260717_034832Z.h5/input_0, changing nothing but the
    weights:

        fit weights                      chans |med|>6 MAD   rms(median resid)
        v0 bits + self-comb (B16 today)              0.236            4.74e+05
        + FM and ORBCOMM zero-weighted               0.036            5.68e+03

    An 83x reduction in the residual's persistent structure, from masking two
    bands that were already known about. It also takes B16's own uncentred
    refine criterion from flagging 27.1% of good pixels to 7.6%, before the
    centring fix is applied at all.
    """
    w = np.where(good, 1.0, 0.0)
    if a_priori:
        excl = np.zeros(freqs_mhz.shape, dtype=bool)
        for lo, hi in A_PRIORI_BANDS.values():
            excl |= (freqs_mhz >= lo) & (freqs_mhz <= hi)
        if clock_harmonics:
            for ch in CLOCK_HARMONIC_CHANS:
                near = np.argmin(np.abs(freqs_mhz - ch * 250.0 / N_CHAN))
                excl[near] = True
        w[:, excl] = 0.0
    return w


# ------------------------------------------------------------------ pipeline

DEFAULTS = dict(
    line_width=41,
    n_false=1.0,
    sigma_quantile=0.25,
    occ_frac=0.30,
    transient_width=9,
    broadband_frac=0.15,
    gross_nsig=5.0,
    meteor_nsig=6.0,
    comb_halfwidth=1,
    switch_unknown="assume_sky",
    dilate_time=0,
    dilate_chan=0,
    # Straight n-sigma thresholds. Set any to None to fall back on the
    # false-alarm-rate derivation. NOTE the auto-domain ones are Gaussian and
    # the coherence one is RAYLEIGH, so the same number is a different
    # false-alarm probability: at 6, P = 1e-9 Gaussian vs 1.5e-8 Rayleigh.
    transient_nsig=6.0,
    coh_nsig=6.0,
    res_nsig=6.0,
)


def _cross_key(meta, key_a, key_b):
    """The raw key holding the cross of two inputs, checked against the file.

    The cross is named by concatenating the two input keys in ascending
    order -- '04' in phase C, '35' in B, '02' in A. Rather than carry a
    per-phase table (an earlier version of this module did, and it was one
    more thing to keep in step with `curation/select_files.py`), the key is
    built from whatever `load_bundle` resolved for the two antennas and then
    checked against the `data_keys` the index recorded for each file.

    Raises if the selection spans files that disagree, instead of silently
    using the first file's answer for all of them.
    """
    lo, hi = sorted((str(key_a), str(key_b)), key=lambda k: int(k))
    want = f"{lo}{hi}"
    present = []
    for keys in meta["data_keys"].astype(str).unique():
        present.append(want in str(keys).split(","))
    if not all(present):
        raise ValueError(
            f"cross key {want!r} (from inputs {key_a},{key_b}) is not present "
            f"in every selected file. The selection probably spans a wiring "
            f"phase; select within one phase, or pass cross_key= explicitly."
        )
    return want


def load_inputs(selection, antenna="box-gnd", partner="box-air",
                band_mhz=None, cross_key=None, key=None, partner_key=None):
    """Read the two autos and their cross for *selection*.

    Three `load_bundle` calls, all with `products=[]` -- **existing flags are
    never loaded**. The antenna is resolved from each file's own header by
    `load_bundle`, so no input key is hardcoded here; box-gnd is input 0 on
    2026-07-17 and input 2 on 07-13 and this does not have to know that.

    `band_mhz` defaults to None -- the FULL axis, not the analysis band. The
    analysis band is applied downstream as a mask, never as a slice, because
    two detectors need channels outside it: the gross-power monitor lives at
    236-249 MHz, and the clock-comb tones that fire the transient test include
    238.037 MHz. Slicing to 45-235 first silently disables both, which is how
    an earlier version of this reported zero gross-power events.

    Returns `(bundle_a, bundle_b, cross)` where the first is the bundle the
    flags will be attached to.
    """
    # `key`/`partner_key` bypass the antenna-name lookup. Needed, not a
    # convenience: in phase B the files' own `input_to_ant` header declares
    # {0: box-air, 2: box-gnd, 4: viv-N, 5: viv-E} while the live data keys
    # are ['3','35','4','5'] -- neither named antenna is present, and
    # `load_bundle` correctly refuses. The header is stale for that wiring,
    # so the caller has to say which inputs it means.
    if key is not None:
        b_a = selection.load_bundle(key=key, products=[], band_mhz=band_mhz)
    else:
        b_a = selection.load_bundle(antenna=antenna, products=[],
                                    band_mhz=band_mhz)
    if partner_key is not None:
        b_b = selection.load_bundle(key=partner_key, products=[],
                                    band_mhz=band_mhz)
    else:
        b_b = selection.load_bundle(antenna=partner, products=[],
                                    band_mhz=band_mhz)
    key_a = key or b_a.provenance["keys"][0]
    key_b = partner_key or b_b.provenance["keys"][0]
    ckey = cross_key or _cross_key(b_a.meta, key_a, key_b)
    b_x = selection.load_bundle(key=ckey, products=[], band_mhz=band_mhz)
    if not (b_a.data.shape == b_b.data.shape == b_x.data.shape):
        raise ValueError(
            f"shape mismatch between inputs: {b_a.data.shape}, "
            f"{b_b.data.shape}, {b_x.data.shape}")
    return b_a, b_b, b_x


def flag_bundle(selection, antenna="box-gnd", partner="box-air",
                band_mhz=None, self_comb=False, cross_key=None,
                key=None, partner_key=None, second_pass=True, **kw):
    """Flag *selection* and hand back a `Bundle` with its mask filled in.

    The returned object is the bundle `load_bundle` built for *antenna*, with
    `products["flags"] = {"mask": <uint16 bitfield>}` -- the same key
    `eigsep_data.products.flags` populates -- so `b.flags` works and anything
    written against a bundle read from disk works against this unchanged.
    `b.provenance["products"]["flags"]` records the parameters rather than
    a version string, because this is not a released version.

    Bits are `FLAG_BITS`; `RFI_BITS` and `UNUSABLE_BITS` are the two masks
    worth combining.
    """
    p = dict(DEFAULTS)
    p.update(kw)

    b_a, b_b, b_x = load_inputs(selection, antenna=antenna, partner=partner,
                                band_mhz=band_mhz, cross_key=cross_key,
                                key=key, partner_key=partner_key)
    freqs = np.asarray(b_a.freqs_mhz, dtype=float)
    pa = np.asarray(b_a.data, dtype=np.float64)
    pb = np.asarray(b_b.data, dtype=np.float64)
    cross = np.asarray(b_x.data)
    meta = b_a.meta

    n_row = pa.shape[0]
    n_samp = float(np.nanmedian(n_independent_samples(meta)))

    # The analysis band is a MASK on the full axis, not a slice of it.
    in_band = (freqs >= BAND_ANALYSIS[0]) & (freqs <= BAND_ANALYSIS[1])
    excl = np.zeros(freqs.shape, dtype=bool)
    for lo, hi in A_PRIORI_BANDS.values():
        excl |= (freqs >= lo) & (freqs <= hi)
    for ch in CLOCK_HARMONIC_CHANS:
        j = int(np.argmin(np.abs(freqs - ch * CHAN_WIDTH_MHZ)))
        if abs(freqs[j] - ch * CHAN_WIDTH_MHZ) < CHAN_WIDTH_MHZ:
            excl[j] = True
    if self_comb:
        # Teeth PLUS `comb_halfwidth` either side: the channelizer leaks, and
        # measured on corr_20260717_201113Z.h5 the teeth sit at coherence 0.86
        # against 0.06 elsewhere while their immediate neighbours account for
        # essentially every remaining detection.
        ch_idx = np.round(freqs / CHAN_WIDTH_MHZ).astype(int)
        tooth = (ch_idx % SELF_COMB_SPACING_CHAN) == 0
        for d in range(1, int(p["comb_halfwidth"]) + 1):
            tooth |= np.roll(tooth, d) | np.roll(tooth, -d)
        excl |= tooth
    quiet = (freqs >= BAND_QUIET[0]) & (freqs <= BAND_QUIET[1])

    mask = np.zeros(pa.shape, dtype=np.uint16)

    # --- stage 0: samples that are not usable at all -----------------------
    sky = on_sky(meta, unknown=p["switch_unknown"])
    n_unknown = int((meta["rfswitch"].astype(str).str.upper() == "MISSING").sum())
    mask[~sky, :] |= BIT_OFF_SKY
    ovf = overflow(pa) | overflow(pb)
    mask[ovf] |= BIT_OVERFLOW
    instrument_bad = ovf | (~sky)[:, None]

    mask[:, excl] |= BIT_A_PRIORI

    # --- stage 1: model-free per-channel transient test on the raw autos ---
    tr_a, z_tr_a, tr_thr = raw_transient_flags(
        pa, n_samp, nsig=p["transient_nsig"], n_false=p["n_false"],
        med_width=p["transient_width"])
    tr_b, z_tr_b, _ = raw_transient_flags(
        pb, n_samp, nsig=p["transient_nsig"], n_false=p["n_false"],
        med_width=p["transient_width"])
    transient = (tr_a | tr_b) & ~instrument_bad
    mask[transient] |= BIT_TRANSIENT

    # --- stage 2: whole-integration tests ----------------------------------
    t_broad = broadband_times(transient, in_band, frac=p["broadband_frac"]) & sky
    # The quiet-band monitor needs channels the analysis band excludes, which
    # is why `load_inputs` does not slice.
    if quiet.sum() >= 4:
        t_gross = gross_power_time_flags(pa, pb, quiet, nsig=p["gross_nsig"]) & sky
    else:
        # Only reachable if the caller sliced the axis; say so rather than
        # returning a silent zero.
        t_gross = np.zeros(n_row, dtype=bool)
    t_meteor, meteor_exc = meteor_scatter_flags(pa, freqs, nsig=p["meteor_nsig"])
    t_meteor &= sky
    mask[t_broad, :] |= BIT_BROADBAND
    mask[t_gross, :] |= BIT_GROSS
    mask[t_meteor, :] |= BIT_METEOR
    t_bad = t_broad | t_gross | t_meteor

    # --- stage 3: coherence ------------------------------------------------
    quant_ok, zero_frac = quantisation_ok(cross, in_band)
    coh = coherence(pa, pb, cross)
    estimation_ok = ~(instrument_bad | t_bad[:, None])
    report_ok = ~instrument_bad          # deliberately NOT gated on t_bad:
    # "excluded from a fitted scale" and "not reported as flagged" are
    # different statements, and conflating them let a transient that TRIPPED a
    # time flag escape the per-pixel mask entirely.
    sig_hat = rayleigh_sigma_empirical(coh, estimation_ok, q=p["sigma_quantile"])
    sig_theory = rayleigh_sigma_theory(n_samp)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_coh = coh / sig_hat[None, :]
    z_coh = np.where(np.isfinite(z_coh), z_coh, 0.0)
    n_elig = int(estimation_ok.sum())
    z_thr = (float(p["coh_nsig"]) if p["coh_nsig"] is not None
             else false_alarm_threshold(max(n_elig, 1), p["n_false"]))
    if quant_ok:
        mask[(z_coh > z_thr) & report_ok] |= BIT_COHERENT

    # --- stage 4: persistent lines ----------------------------------------
    keep_chan = in_band & ~excl
    line_resid, z_line = line_statistic(sig_hat, keep_chan, width=p["line_width"])
    line_thr = (float(p["coh_nsig"]) if p["coh_nsig"] is not None
                else false_alarm_threshold(max(int(keep_chan.sum()), 1),
                                           p["n_false"]))
    line_chans = (z_line > line_thr) & in_band if quant_ok \
        else np.zeros(freqs.shape, dtype=bool)
    mask[:, line_chans] |= BIT_LINE

    # --- stage 5: occupancy ------------------------------------------------
    detected = (mask & (BIT_TRANSIENT | BIT_COHERENT | BIT_LINE)) > 0
    killed, occ = occupancy_kill(detected, report_ok, frac=p["occ_frac"])
    mask[:, killed] |= BIT_OCCUPANCY

    prov = dict(b_a.provenance)
    prov["products"] = dict(prov.get("products", {}))
    prov["products"]["flags"] = {
        "version": None,
        "generator": "rfi_proto.flag_bundle (PROTOTYPE, not a released "
                     "version -- see rfi_flag_prototype.ipynb)",
        "bits": FLAG_BITS,
        "params": {k: p[k] for k in sorted(p)},
        "partner_antenna": partner,
        "cross_key": b_x.provenance.get("key"),
        "n_independent_samples": n_samp,
        "sigma_hat_over_theory": float(
            np.nanmedian(sig_hat[keep_chan]) / sig_theory)
        if np.isfinite(sig_theory) and sig_theory > 0 else None,
        "rfswitch_missing_rows": n_unknown,
        "cross_quantisation_ok": quant_ok,
        "cross_zero_fraction": zero_frac,
        "thresholds": {"transient": float(tr_thr), "coherence": float(z_thr),
                       "line": float(line_thr)},
    }
    b_a.provenance = prov
    b_a.products = dict(b_a.products)
    b_a.products["flags"] = {
        "mask": mask,
        # Kept alongside the bitfield because the evidence is per antenna and
        # the mask is a union across them; losing that distinction made four
        # whole-integration flags invisible in the diagnostics.
        "z_coherence": z_coh,
        "z_transient_a": z_tr_a,
        "z_transient_b": z_tr_b,
        "coherence": coh,
        "sigma_hat": sig_hat,
        "meteor_excess": meteor_exc,
    }

    if second_pass:
        b_a = _second_pass(b_a, pa, freqs, in_band, n_samp, z_coh, p,
                           use_cross=quant_ok)
    return b_a


def _second_pass(bundle, amp, freqs, in_band, n_samp, z_coh, p,
                 use_cross=True):
    """Fit a DPSS continuum on what pass 1 left standing, then re-decide.

    A flag raised before the continuum is modelled is provisional. With a
    model in hand the auto can be asked an independent question -- does this
    sample still stand above the RADIOMETRIC noise? -- and the answer is
    allowed to be no, which is the whole point.
    """
    import hera_filters.dspec as dspec

    mask = bundle.products["flags"]["mask"]
    usable = (mask & UNUSABLE_BITS) == 0
    flagged = (mask & RFI_BITS) > 0
    keep = (usable & ~flagged)[:, in_band]
    amp = amp[:, in_band]
    w = np.where(keep, 1.0, 0.0)

    mdl, _, _ = dspec.fourier_filter(
        freqs[in_band] * 1e6, amp, w, filter_centers=[0.0],
        filter_half_widths=[40e-9], mode="dpss_leastsq", filter_dims=1,
        **dspec.DPSS_DEFAULTS_1D)
    mdl = np.real(mdl)

    # hera_filters returns an ALL-ZERO model rather than raising when a wide
    # contiguous block is zero-weighted at a band edge. On the labelled event
    # file that produced a residual equal to the raw autos and a mask over 93%
    # of the band: it looked like a detection and was a failed fit. The second
    # pass is a thing that can decline to run.
    dpss_ok = bool(np.all(np.isfinite(mdl)) and np.mean(mdl > 0) > 0.5)
    info = bundle.provenance["products"]["flags"]
    info["dpss_ok"] = dpss_ok
    if not dpss_ok:
        info["second_pass"] = "declined: degenerate DPSS fit"
        return bundle

    resid = amp - mdl
    z_res_band, inflation_band = auto_residual_z(resid, mdl, n_samp, good=keep)
    # Lift back onto the full axis so every array in products/ shares it.
    z_res = np.zeros(mask.shape)
    z_res[:, in_band] = z_res_band
    inflation = np.full(mask.shape[1], np.nan)
    inflation[in_band] = inflation_band
    model_full = np.full(mask.shape, np.nan)
    model_full[:, in_band] = mdl
    n_pix = int(keep.size)
    if not use_cross:
        # The cross is quantisation-limited, so z_coh carries no information
        # and the corroborated branch would be corroborating with noise. Fall
        # back to auto evidence alone -- which is the whole reason this bit
        # exists separately from `coherent`.
        z_for_combine = np.zeros_like(z_res)
    else:
        z_for_combine = z_coh
    final, retracted, promoted, thr = combine_decision(
        z_for_combine, z_res, n_pix, n_false=p["n_false"],
        coh_nsig=p["coh_nsig"], res_nsig=p["res_nsig"])

    # Structural flags are not up for revision by a pixel test.
    final &= in_band[None, :]
    structural = (mask & (BIT_OFF_SKY | BIT_OVERFLOW | BIT_A_PRIORI
                          | BIT_BROADBAND | BIT_GROSS | BIT_METEOR
                          | BIT_LINE | BIT_OCCUPANCY)) > 0
    provisional = (mask & (BIT_TRANSIENT | BIT_COHERENT)) > 0
    retracted = provisional & ~final & ~structural
    # ~x on a Python int is negative; mask off by XOR-ing within the dtype.
    clear = np.uint16(0xFFFF ^ (BIT_TRANSIENT | BIT_COHERENT))
    mask[retracted] &= clear
    mask[retracted] |= BIT_RETRACTED
    # The second pass's own detections get their own bit. Folding them into
    # `coherent` misattributed auto-only evidence to the cross -- visibly so
    # in phase A, where the cross is unusable and `coherent` was still firing.
    mask[final & ~structural] |= BIT_RESIDUAL

    bundle.products["flags"]["mask"] = mask
    bundle.products["flags"]["smooth_model"] = model_full
    bundle.products["flags"]["z_residual"] = z_res
    bundle.products["flags"]["model_inflation"] = inflation
    info["second_pass"] = "applied"
    info["thresholds"].update({f"combine_{k}": float(v)
                               for k, v in thr.items()})
    info["n_retracted"] = int(retracted.sum())
    info["n_promoted"] = int(promoted.sum())
    info["model_inflation_median"] = float(np.nanmedian(inflation))
    return bundle


def measure_margin(bundle, max_off=8):
    """How far from a flagged channel does the excess coherence actually reach?

    Answers the complaint that flags land on clean samples several channels
    and integrations either side of real interference, by measuring the
    profile instead of picking a dilation width and hoping. Measured on
    corr_20260717_034832Z.h5 it is flat at z ~ 1.0 from offset 1 on BOTH axes
    -- the immediate neighbours are already at the noise floor -- which is why
    `dilate_time` and `dilate_chan` default to 0.
    """
    prod = bundle.products["flags"]
    mask = prod["mask"]
    z = prod["z_coherence"]
    freqs = np.asarray(bundle.freqs_mhz, dtype=float)
    ib = (freqs >= BAND_ANALYSIS[0]) & (freqs <= BAND_ANALYSIS[1])
    fl = (mask & RFI_BITS) > 0
    occ = fl.mean(axis=0)
    cores = np.where(ib & (occ > 0.8))[0]
    prof_f = {}
    for off in range(0, max_off + 1):
        vals = []
        for c in cores:
            for s_ in ((c + off,) if off == 0 else (c - off, c + off)):
                if 0 <= s_ < z.shape[1] and ib[s_] and occ[s_] < 0.2:
                    vals.append(np.median(z[:, s_]))
        prof_f[off] = float(np.median(vals)) if vals else np.nan

    # Along time, per-sample detections only: whole-channel categories are set
    # at every integration and would make every row "flagged".
    per_sample = (mask & (BIT_TRANSIENT | BIT_COHERENT)) > 0
    quiet_chan = ib & (mask[0] & (BIT_LINE | BIT_OCCUPANCY | BIT_A_PRIORI)) == 0
    t_any = per_sample[:, ib].any(axis=1)
    prof_t = {}
    for off in range(0, max_off + 1):
        vals = []
        for t in np.where(t_any)[0]:
            for s_ in ((t + off,) if off == 0 else (t - off, t + off)):
                if 0 <= s_ < z.shape[0] and not t_any[s_] and quiet_chan.any():
                    vals.append(np.median(z[s_, quiet_chan]))
        prof_t[off] = float(np.median(vals)) if vals else np.nan
    return prof_f, prof_t


def refine_b16_style(resid_auto, good, nsig=6.0, centred=True, one_sided=True):
    """B16's `iterative_refine` criterion, with the two corrections toggleable.

    Kept here so the notebook can show the size of each correction on real
    data rather than asserting it. `centred=False, one_sided=False`
    reproduces what flagging/b16_dpss_model.py does today.
    """
    out = np.zeros_like(good)
    for j in range(resid_auto.shape[1]):
        col_good = good[:, j]
        if col_good.sum() < 20:
            continue
        col = resid_auto[col_good, j]
        med = np.median(col)
        scale = 1.4826 * np.median(np.abs(col - med))
        if scale <= 1e-9:
            continue
        dev = resid_auto[:, j] - (med if centred else 0.0)
        out[:, j] = col_good & ((dev > nsig * scale) if one_sided
                                else (np.abs(dev) > nsig * scale))
    return out
