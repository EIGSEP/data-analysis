"""Noise-source temperature from the 2026-10-08 interim bench run.

Wiring and sensors are as recorded in
notebooks/charlie/bench_runs/2026-10-08_interim/README.md:

- hot load on the "Amb/Hot Load" port, switch state RFAMB, thermistor
  tempctrl_load (pull-up 9.98 kOhm);
- ambient 50 Ohm reference on the Feed port, switch state RFANT, thermistor
  tempctrl_lna2 (GP27, pull-up 9.97 kOhm);
- noise diode on/off, RFNON/RFNOFF; its board thermistor is
  rfswitch_therm.temp_therm2 (GP28) on the bench board;
- receiver: antenna "box-air" (correlator input 0), acc_len 2**28 (1.074 s).

Timestamps are UTC: the run started 2026-10-08 ~18:03 PDT, 2026-10-09 01:03 UTC.

What comes from the packages, and what is here:

- eigsep_data: the metadata index, per-state selections, contiguous visits
  (here one per dwell), the spectra (Selection.load_bundle), and the RFI
  detectors (flagging.detectors: overflow_mask, transient_track,
  persistent_track);
- eigsep_cal: the switched ratio Q (dicke.calc_Tant_star), the S11
  calibration to plane P (S11, with the packaged switch-path and internal-OSL
  characterizations) and the mismatch factor (dicke.mismatch);
- numpy.polyfit: the straight-line fits, one per channel.

This module adds only what those don't do: the thermistor conversion with
the measured pull-ups, the weighted mean of each dwell with wrapped samples
masked, the grouping of dwells into switch cycles, the split of cycles into
steady, heating and cooling, the S11 sweeps interpolated onto the correlator
channels and the cycle times, the radiometer noise of Q, and the channel mask.

Model (memo 004 section 3, the minimal tier of eigsep_cal memo M003 section
7.1): P_s = g (M_s T_s + T_r), with M_s = mismatch(Gamma_s, Gamma_rec) at P and
T_r the same in every switch state (no receiver noise waves). Substituting
into Q gives Q = (M_hot T_hot - M_ref T_ref) / T_NS, with
T_NS = (P_NON - P_ref) / g: the noise source's excess over the reference as
this receiver takes it. That definition absorbs the noise source's own
mismatch, so it depends on Gamma_rec. With the lab measurement of the noise
source's reflection at P (GAMMA_NS_FILE), its mismatch M_ns can be removed:
tnon_terms gives T_NON, the noise-source path's temperature at P when on,
which does not depend on the receiver. Loss in the switch paths (their
available gain) is not yet included.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from eigsep_cal import S11
from eigsep_cal.dicke import calc_Tant_star, mismatch
from eigsep_data import MetadataIndex
from eigsep_data.flagging.detectors import overflow_mask, persistent_track, transient_track

DATA = Path("/mnt/data02/eigsep/scratch/switch_bench_tests")
ANTENNA = "box-air"
HOT, REF, NON, NOFF = "RFAMB", "RFANT", "RFNON", "RFNOFF"
STATES = (REF, HOT, NON, NOFF)
ENBW_HZ = 250e6 / 1024           # channel width; the PFB's ENBW is not yet measured
#: Rows of one state closer than this belong to one dwell (integrations are
#: 1.07 s apart within a dwell; a state recurs ~50 s later).
DWELL_GAP_S = 5.0
#: A cycle is steady if the hot load's thermistor changes by less than this
#: (K/min). 1 K/min is what the current heater and thermistor allow; the
#: heating/cooling fits test whether that is tight enough.
STEADY_K_PER_MIN = 1.0
#: The bench S11 files (ants11_*, recs11_*), and the S11 DUT names of the loads.
S11_PATTERN = "*s11_*.h5"
S11_DUT = {HOT: "amb", REF: "ant"}
#: Lab measurement of the noise-source path's reflection at P (VNA on the field
#: board's LNA port, switch on RFNON/RFNOFF), 2026-10-15. Board temperature not
#: varied yet.
GAMMA_NS_FILE = Path("/mnt/data02/eigsep/scratch/rfn_sparameters_10_15.npz")
#: Channels outside this band are masked (MHz).
BAND_MHZ = (50.0, 245.0)

#: Thermistors: YSI 44909, Steinhart-Hart fit to YSI's 30 kOhm table
#: (pico-firmware tempctrl.h LOAD_THERM_SH_*); pull-ups measured 2-wire.
SH = (9.362744897e-4, 2.209138866e-4, 1.280636863e-7)
PULLUP = {"tempctrl_load": 9980.0, "tempctrl_lna2": 9970.0, "tempctrl_lna1": 9910.0}


def thermistor_K(resistance_logged, pullup):
    """Temperature (K) from a tempctrl channel's logged resistance.

    The firmware computes `resistance` assuming a 10 kOhm pull-up; scale it to
    the measured pull-up, then apply Steinhart-Hart."""
    r = np.asarray(resistance_logged, float) * pullup / 10000.0
    lr = np.log(r)
    a, b, c = SH
    return 1.0 / (a + b * lr + c * lr ** 3)


def index():
    """The eigsep_data metadata index over the bench files."""
    return MetadataIndex(str(DATA))


def add_temperatures(meta):
    """The index's metadata with physical temperatures (K) added."""
    meta = meta.copy()
    meta["T_hot"] = thermistor_K(meta["tempctrl_load_resistance"], PULLUP["tempctrl_load"])
    meta["T_ref"] = thermistor_K(meta["tempctrl_lna2_resistance"], PULLUP["tempctrl_lna2"])
    meta["T_diode_board"] = meta["rfswitch_therm_temp_therm2"] + 273.15
    meta["T_lna3"] = thermistor_K(meta["tempctrl_lna1_resistance"], PULLUP["tempctrl_lna1"])
    return meta


def load_state(idx, state):
    """One switch state: its metadata (with temperatures, dwell ids and each
    row's position in its dwell) and its spectra, row for row.

    Dwells are eigsep_data visits with a short gap; UNKNOWN rows (the switch
    moving during the integration) are a different state and never included."""
    sel = idx.select(rfswitch=state)
    b = sel.load_bundle(antenna=ANTENNA, products=[])
    meta = add_temperatures(sel.meta.reset_index(drop=True))
    if not np.array_equal(meta["time_best"].to_numpy(), np.asarray(b.t)):
        raise ValueError(f"{state}: bundle rows are not in the metadata's order")
    meta["dwell"] = sel.visits(gap_s=DWELL_GAP_S)
    meta["pos_in_dwell"] = meta.groupby("dwell").cumcount()
    return meta, np.asarray(b.freqs_mhz), np.asarray(b.data, float), b


def rfi_flags(spec, freqs):
    """Per-sample RFI flags for one state's integrations (time-ordered), from
    eigsep_data's detectors: int32 wraps (overflow_mask), departures from a
    smooth time trend per channel (transient_track), and steady narrowband
    channels (persistent_track, which checks 45-235 MHz only).

    Wrapped samples are replaced by their channel's median log power before the
    two tracks run, so they don't drag the baselines (they stay flagged).
    Returns per-sample flags (wraps | transients) and per-channel persistent
    flags."""
    wrap = overflow_mask(spec)
    valid = ~np.all(spec == 0, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log10(np.where(spec > 0, spec, np.nan))
    fill = np.nanmedian(logp[valid], axis=0)
    logp = np.where(np.isfinite(logp), logp, fill)
    trans, _, _ = transient_track(logp, valid)
    persist, _ = persistent_track(logp, valid, freqs)
    return wrap | trans, persist


def dwell_means(meta, spec, flags=None, skip_first=0):
    """Weighted mean spectrum of each dwell of one state.

    Weight = integration time; flagged samples (`flags`, per sample; by
    default only int32 wraps) and all-zero rows get zero weight; the first
    `skip_first` integrations of each dwell are dropped. Returns a table (one
    row per dwell) and the (ndwell, nchan) mean powers."""
    if flags is None:
        flags = overflow_mask(spec)
    keep = (meta["pos_in_dwell"].to_numpy() >= skip_first) & (meta["dwell"].to_numpy() >= 0)
    good = ~flags & ~np.all(spec == 0, axis=1)[:, None] & keep[:, None]
    w = np.where(good, meta["integration_time"].to_numpy()[:, None], 0.0)
    rows, P = [], []
    for d, g in meta[keep].groupby("dwell"):
        i = g.index.to_numpy()
        ws = w[i].sum(axis=0)
        P.append(np.where(ws > 0, (w[i] * np.where(good[i], spec[i], 0)).sum(0) / np.where(ws > 0, ws, 1), np.nan))
        tw = g["integration_time"].to_numpy()
        rows.append(dict(t=np.average(g["time_best"], weights=tw), n=len(i),
                         T_hot=np.average(g["T_hot"], weights=tw),
                         T_ref=np.average(g["T_ref"], weights=tw),
                         T_diode_board=np.average(g["T_diode_board"], weights=tw),
                         heater_on=bool(g["tempctrl_load_enabled"].fillna(False).astype(bool).any())))
    return pd.DataFrame(rows), np.array(P)


def cycles(dwells, max_span_s=90.0):
    """Group dwells into switch cycles REF -> HOT -> NON -> NOFF.

    `dwells` maps each state to its (table, powers) from dwell_means. A cycle
    starts at each REF dwell and takes the first HOT, NON and NOFF dwells after
    it, if all of them start before the next REF dwell and within max_span_s.
    Returns one table row per complete cycle and the four (ncycle, nchan)
    power arrays. Label the cycles with classify()."""
    t_ref = dwells[REF][0]["t"].to_numpy()
    rows, out = [], {s: [] for s in STATES}
    for k, t0 in enumerate(t_ref):
        t_next = t_ref[k + 1] if k + 1 < len(t_ref) else np.inf
        pick = {REF: k}
        for s in (HOT, NON, NOFF):
            t = dwells[s][0]["t"].to_numpy()
            j = np.flatnonzero((t > t0) & (t < t_next) & (t - t0 < max_span_s))
            if j.size:
                pick[s] = j[0]
        if len(pick) < 4:
            continue
        hot = dwells[HOT][0].iloc[pick[HOT]]
        rows.append(dict(t=hot["t"], T_hot=hot["T_hot"], T_ref=dwells[REF][0]["T_ref"].iloc[k],
                         T_diode_board=dwells[NON][0]["T_diode_board"].iloc[pick[NON]],
                         heater_on=hot["heater_on"],
                         **{f"n_{s}": dwells[s][0]["n"].iloc[pick[s]] for s in STATES}))
        for s in STATES:
            out[s].append(dwells[s][1][pick[s]])
    return pd.DataFrame(rows), {s: np.array(v) for s, v in out.items()}


def classify(cyc, steady_k_per_min=None):
    """Label each cycle steady, heating or cooling by the hot load's rate of
    change between neighbouring cycles (K/min). The threshold defaults to the
    module's STEADY_K_PER_MIN as it is when called."""
    if steady_k_per_min is None:
        steady_k_per_min = STEADY_K_PER_MIN
    cyc = cyc.copy()
    cyc["rate_K_per_min"] = np.gradient(cyc["T_hot"].to_numpy(), cyc["t"].to_numpy()) * 60
    steady = cyc["rate_K_per_min"].abs() < steady_k_per_min
    cyc["kind"] = np.where(steady, "steady", np.where(cyc["rate_K_per_min"] > 0, "heating", "cooling"))
    return cyc


def q_hot(cp):
    """Q = (P_hot - P_ref) / (P_NON - P_ref), per cycle and channel: the switched
    ratio, i.e. eigsep_cal's T* with T_NS = 1 and a 0 K reference."""
    return calc_Tant_star(cp[HOT], cp[NON], cp[REF], T_ns=1.0, T_load=0.0)


# ------------------------------------------------------------ mismatch

def load_s11():
    """The bench S11 sweeps, calibrated with eigsep_cal's packaged 2026
    characterizations (planes raw, vna, dut and lna = P)."""
    s = S11(str(DATA), year=2026, pattern=S11_PATTERN)
    s.calibrate()
    return s


def s11_at(s, dut, plane, freqs, t):
    """Gamma of `dut` at `plane`, on the correlator channels `freqs` (MHz) and
    at times `t` (unix): linear in frequency (NaN outside the sweep), then
    linear in time between sweeps (held before the first and after the last).
    Returns (len(t), len(freqs)) complex."""
    ts, g = s.get_all_s11s(dut, plane)
    fv = np.asarray(s.freqs_mhz)
    gc = np.array([np.interp(freqs, fv, x.real, left=np.nan, right=np.nan)
                   + 1j * np.interp(freqs, fv, x.imag, left=np.nan, right=np.nan) for x in g])
    u = np.interp(t, ts, np.arange(len(ts)))
    lo = np.floor(u).astype(int)
    hi = np.minimum(lo + 1, len(ts) - 1)
    w = (u - lo)[:, None]
    return (1 - w) * gc[lo] + w * gc[hi]


def mismatch_factors(s, freqs, t):
    """Mismatch factors M at P of the hot load and the reference, and the
    three Gammas used, at cycle times `t`."""
    g_rec = s11_at(s, "rec", "default", freqs, t)
    g = {st: s11_at(s, S11_DUT[st], "lna", freqs, t) for st in (HOT, REF)}
    M = {st: mismatch(g[st], g_rec) for st in (HOT, REF)}
    return M, g, g_rec


def fit_line(x, Q, use):
    """Straight-line fit y = a + b x, per channel, with numpy.polyfit, over the
    cycles in `use` (`Q` is the ordinate y). `x` is (ncycle,) for one abscissa
    or (ncycle, nchan) when it differs by channel. Non-finite
    points are left out per channel; a channel with fewer than 4 points gets
    NaN. Returns intercept a, slope b, the slope's standard error (from
    polyfit's covariance, i.e. from the residuals), the number of cycles
    fitted, and the residuals (NaN where not fitted)."""
    nchan = Q.shape[1]
    x = np.broadcast_to(x[:, None] if x.ndim == 1 else x, Q.shape)
    a, b, sb = (np.full(nchan, np.nan) for _ in range(3))
    n = np.zeros(nchan, int)
    res = np.full(Q.shape, np.nan)
    for k in range(nchan):
        ok = use & np.isfinite(x[:, k]) & np.isfinite(Q[:, k])
        n[k] = ok.sum()
        if n[k] < 4:
            continue
        (b[k], a[k]), cov = np.polyfit(x[ok, k], Q[ok, k], 1, cov=True)
        sb[k] = np.sqrt(cov[0, 0])
        res[ok, k] = Q[ok, k] - a[k] - b[k] * x[ok, k]
    return dict(a=a, b=b, sb=sb, n=n, res=res)


# ------------------------------------------------------------ channel mask

def sigma_q(cyc, cp, tau_s):
    """Radiometer noise of Q per cycle and channel, from each dwell's mean
    power and number of integrations: sigma_P = P / sqrt(dnu tau n)."""
    sig = {st: cp[st] / np.sqrt(ENBW_HZ * tau_s * cyc[f"n_{st}"].to_numpy()[:, None]) for st in STATES}
    D = cp[NON] - cp[REF]
    Q = (cp[HOT] - cp[REF]) / D
    return np.sqrt(sig[HOT] ** 2 + ((Q - 1) * sig[REF]) ** 2 + (Q * sig[NON]) ** 2) / np.abs(D)


def channel_mask(freqs, fit, sq, use, persistent, band=BAND_MHZ, chi2_factor=3.0,
                 max_rel_err=0.05):
    """Channels to leave out of T_NS, by reason (True = masked):

    - band: outside `band`;
    - persistent: a steady narrowband emitter in any state (rfi_flags);
    - chi2: the fit residuals, in units of radiometer noise, have a mean square
      more than `chi2_factor` times the in-band median (drift or interference);
    - rel_err: the slope's standard error exceeds `max_rel_err` of the slope,
      or the slope is not positive (a weak or absent noise-source step).

    The last two are about the fit, not RFI. Medians are used here only to
    find bad channels. Returns the per-reason masks, their union, and the
    per-channel chi2."""
    inband = (freqs >= band[0]) & (freqs <= band[1])
    z = fit["res"] / sq
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chi2 = np.nanmean(np.where(use[:, None], z ** 2, np.nan), axis=0)
    m = {"band": ~inband, "persistent": inband & persistent}
    m["chi2"] = inband & ~(chi2 <= chi2_factor * np.nanmedian(chi2[inband & ~persistent]))
    m["rel_err"] = inband & ~((fit["b"] > 0) & (fit["sb"] / fit["b"] <= max_rel_err))
    m["any"] = m["band"] | m["persistent"] | m["chi2"] | m["rel_err"]
    return m, chi2


def gamma_ns(freqs):
    """The noise-source path's reflection at P (RFNON) on the correlator
    channels, linear in frequency (NaN outside the measured range), and the
    largest |Gamma_RFNON - Gamma_RFNOFF| (they should agree, the diode being
    behind the 30 dB pad)."""
    d = np.load(GAMMA_NS_FILE)
    f = d["freqs"] / 1e6
    on = np.interp(freqs, f, d["RFNON"].real, left=np.nan, right=np.nan) \
        + 1j * np.interp(freqs, f, d["RFNON"].imag, left=np.nan, right=np.nan)
    return on, float(np.abs(d["RFNON"] - d["RFNOFF"]).max())


def tnon_terms(Q, M, M_ns, T_hot, T_ref):
    """Regressor and ordinate for the noise source's temperature at P.

    With the noise source's mismatch M_ns known, the model gives
    M_hot T_hot - M_ref T_ref = Q (M_ns T_NON - M_ref T_ref), i.e.
    y = M_hot T_hot - (1 - Q) M_ref T_ref = T_NON * x with x = Q M_ns, where
    T_NON is the noise-source path's source temperature at P when on: a
    property of the source, not of the receiver. All temperatures in K.
    Returns (x, y), each (ncycle, nchan)."""
    T_hot = np.asarray(T_hot)[:, None]
    T_ref = np.asarray(T_ref)[:, None]
    y = M[HOT] * T_hot - (1 - Q) * M[REF] * T_ref
    return Q * M_ns, y


def snap_temperature(idx, t):
    """The SNAP's FPGA temperature (deg C, `fpga_temp_temp_c`, logged about every
    20 s) interpolated linearly to times `t` (unix)."""
    ft = idx.select().meta[["time_best", "fpga_temp_temp_c"]].dropna()
    return np.interp(t, ft["time_best"], ft["fpga_temp_temp_c"])


def lna3_temperature(idx, t):
    """LNA 3's thermistor (K, tempctrl_lna1 on GP28, measured pull-up)
    interpolated linearly to times `t` (unix)."""
    m = add_temperatures(idx.select().meta)[["time_best", "T_lna3"]].dropna()
    return np.interp(t, m["time_best"], m["T_lna3"])


def vna_sweeps(meta, gap_s=60.0):
    """Start and end times (unix) of each VNA sweep: runs of rows whose switch
    state is a VNA* state, split where they are more than gap_s apart."""
    t = np.sort(meta.loc[meta["rfswitch"].astype(str).str.startswith("VNA"), "time_best"].to_numpy())
    if t.size == 0:
        return np.empty((0, 2))
    cut = np.flatnonzero(np.diff(t) > gap_s)
    starts = np.r_[t[0], t[cut + 1]]
    ends = np.r_[t[cut], t[-1]]
    return np.c_[starts, ends]


#: Integrations this long before a VNA sweep starts (and after it ends) are
#: excluded: the VNA service disturbs the receiver for up to ~17 s before it
#: takes the switch (power +120 % to +280 % in any state), § 12 of the notebook.
SWEEP_GUARD_S = (20.0, 5.0)


def near_sweep(meta, sweeps, guard=SWEEP_GUARD_S):
    """Rows of `meta` within guard[0] s before a sweep's start or guard[1] s
    after its end (sweeps from vna_sweeps)."""
    t = meta["time_best"].to_numpy()
    out = np.zeros(t.size, bool)
    for t0, t1 in sweeps:
        out |= (t > t0 - guard[0]) & (t < t1 + guard[1])
    return out
