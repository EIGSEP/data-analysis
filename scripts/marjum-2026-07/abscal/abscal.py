#!/usr/bin/env python3
"""First-order absolute calibration of Marjum 2026-07 switched-load data.

Reference plane
---------------
The switched reference plane is the **RF switch common port** inside the
receiver box -- the point the RFAMB ambient load and the RFNON noise source
are presented at. Everything between that plane and the antenna terminals
(balun, feed cable, connectors) is NOT corrected here and is absorbed into
the reported T_ant. Radiation efficiency and antenna reflection are likewise
NOT removed: this product is

    T_ant^(sw)(nu, t)  ==  the input temperature at the switch common port
                           when the switch selects RFANT,

not a sky brightness. Converting to T_sky requires Gamma_ant, the noise-wave
parameters, and the efficiency chain -- none of which are in hand (see
UNCERTAINTY.md).

Formalism
---------
Three-state Y-factor, following eigsep_observing.live_status.calibration
(which is the code that ran in the field, so the field dashboard and this
pipeline cannot silently disagree):

    G(nu)     = (P_on - P_amb) / (T_hot - T_amb)          [counts / K]
    T_rx(nu)  = P_amb / G - T_amb
    T_in(nu)  = P / G - T_rx      ==  (P - P_amb)/G + T_amb

with, from the campaign obs_config ``calibration`` block,

    T_ENR = 290 K * 10^((ENR_dB - atten_dB)/10)
    T_hot = T_ns + T_ENR          T_ns  = rfswitch_therm.temp_therm2 + 273.15
    T_amb = tempctrl_load.T_now + 273.15

This is FIRST ORDER. It omits the five noise-wave terms, the antenna and
receiver reflection coefficients, and any bandpass polynomial. It is
correct only to the extent that |Gamma| ~ 0 at the switch plane.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from eigsep_data.paths import campaign_data_dir

ENR_REF_K = 290.0          # IEEE ENR reference temperature
CELSIUS_TO_KELVIN = 273.15
# Point at a campaign with eigsep_data.set_campaign_root(...) or
# EIGSEP_CAMPAIGN_ROOT before importing anything that reads DATA.
DATA = campaign_data_dir()

# The switched receiver. Established empirically: across an RFNON/RFAMB
# dwell, input 4 (box-air) moves by a factor ~2.3 while input 0 (box-gnd)
# moves <2%. Only input 4 is behind the RF switch.
SWITCHED_INPUT = "4"


def _jmeta(f, key):
    if key not in f["metadata"]:
        return None
    v = f["metadata"][key][()]
    return json.loads(v.decode() if isinstance(v, bytes) else v)


def _stream_mean(recs, field_name):
    """Mean of a metadata-stream field, ignoring error/None records."""
    vals = [
        r[field_name]
        for r in (recs or [])
        if isinstance(r, dict)
        and r.get("status") == "update"
        and r.get(field_name) is not None
    ]
    return (float(np.mean(vals)), len(vals)) if vals else (np.nan, 0)


@dataclass
class Dwell:
    """One file's worth of state-resolved spectra."""

    fname: str
    phase: str
    t0: float
    t1: float
    freqs: np.ndarray
    acc_len: float
    dt: float = np.nan          # seconds per integration; PROXY FOR acc_len
    # state -> (n_samples, nchan) raw accumulations
    by_state: dict = field(default_factory=dict)
    t_ns_c: float = np.nan
    t_amb_c: float = np.nan
    t_amb_n: int = 0
    amb_sensor_tripped: bool = False
    n_wrapped: int = 0


def read_file(path, input_key=SWITCHED_INPUT):
    with h5py.File(path, "r") as f:
        states = _jmeta(f, "rfswitch")
        if states is None:
            return None
        states = np.array([str(s) for s in states])
        freqs = f["header/freqs"][:]
        times = f["header/times"][:]
        acc = f["header/acc_cnt"][:]
        if input_key not in f["data"]:
            return None
        d = f["data"][input_key][:, :].astype(np.float64)
        therm = _jmeta(f, "rfswitch_therm")
        tload = _jmeta(f, "tempctrl_load")
        phase = f.attrs.get("filter_phase")

    t_ns_c, _ = _stream_mean(therm, "temp_therm2")
    t_amb_c, n_amb = _stream_mean(tload, "T_now")
    tripped = any(
        r.get("sensor_tripped") for r in (tload or []) if isinstance(r, dict)
    )

    dw = Dwell(
        fname=Path(path).name,
        phase=phase,
        t0=float(times[0]),
        t1=float(times[-1]),
        freqs=freqs,
        acc_len=float(np.median(np.diff(acc))) if len(acc) > 1 else np.nan,
        dt=float(np.median(np.diff(times))) if len(times) > 1 else np.nan,
        t_ns_c=t_ns_c,
        t_amb_c=t_amb_c,
        t_amb_n=n_amb,
        amb_sensor_tripped=tripped,
    )
    # int32 accumulator wrap (natural-experimenter MEMO-006, formerly their
    # self-assigned 002): a wrapped sample reinterprets as a large negative
    # int32. Power is positive by
    # construction, so a negative accumulation is never physical -- mask it
    # rather than let it poison a channel mean. NaN propagates through the
    # nan-aware reductions below.
    n_wrapped = int((d < 0).sum())
    d = np.where(d < 0, np.nan, d)

    dw.n_wrapped = n_wrapped
    for s in np.unique(states):
        if s in ("UNKNOWN", "None", "NONE"):
            continue          # switch in transit -- never calibrate on these
        dw.by_state[s] = d[states == s]
    return dw


def t_enr_k(enr_db, atten_db):
    """Effective excess noise temperature of the padded diode."""
    return ENR_REF_K * 10.0 ** ((float(enr_db) - float(atten_db)) / 10.0)


@dataclass
class CalSolution:
    freqs: np.ndarray
    gain: np.ndarray          # counts / K
    t_rx: np.ndarray          # K
    t_hot: float
    t_amb: float
    t_ns: float               # noise-source pad physical temperature, K
    t_enr: float              # attenuated diode excess, K
    dt: float                 # s per integration (acc_len proxy)
    gain_per_s: np.ndarray    # gain normalised by dt -- THIS is the one that
                              # is comparable across the corr_acc_len doubling
    p_on: np.ndarray
    p_amb: np.ndarray
    n_on: int
    n_amb: int
    var_on: np.ndarray
    var_amb: np.ndarray
    t0: float
    t1: float


def solve_gain_trx(dwells, cal_cfg):
    """Y-factor solve from the pooled RFNON / RFAMB samples of ``dwells``."""
    on = [d.by_state["RFNON"] for d in dwells if "RFNON" in d.by_state]
    amb = [d.by_state["RFAMB"] for d in dwells if "RFAMB" in d.by_state]
    if not on or not amb:
        return None
    # Raw accumulations scale with corr_acc_len, which DOUBLED at
    # 2026-07-15 15:55 UTC. Pooling dwells from either side of that boundary
    # would average incommensurable counts, so refuse.
    dts = np.array([d.dt for d in dwells if np.isfinite(d.dt)])
    if len(dts) and (dts.max() / dts.min() > 1.5):
        raise ValueError(
            f"dwells span a corr_acc_len change (dt {dts.min():.4f}"
            f"..{dts.max():.4f} s); refusing to pool incommensurable counts"
        )
    dt = float(np.median(dts)) if len(dts) else np.nan

    on = np.concatenate(on, axis=0)
    amb = np.concatenate(amb, axis=0)

    t_ns = np.nanmean([d.t_ns_c for d in dwells]) + CELSIUS_TO_KELVIN
    t_amb = np.nanmean([d.t_amb_c for d in dwells]) + CELSIUS_TO_KELVIN
    t_enr = t_enr_k(
        cal_cfg["noise_diode_enr_db"], cal_cfg["noise_source_atten_db"]
    )
    t_hot = t_ns + t_enr
    if not t_hot > t_amb:
        raise ValueError(f"t_hot {t_hot} !> t_amb {t_amb}")

    p_on, p_amb = np.nanmean(on, axis=0), np.nanmean(amb, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_raw = (p_on - p_amb) / (t_hot - t_amb)
        gain = np.where(g_raw > 0, g_raw, np.nan)
        t_rx = p_amb / gain - t_amb
    return CalSolution(
        freqs=dwells[0].freqs,
        gain=gain,
        t_rx=t_rx,
        t_hot=t_hot,
        t_amb=t_amb,
        t_ns=t_ns,
        t_enr=t_enr,
        dt=dt,
        gain_per_s=gain / dt,
        p_on=p_on,
        p_amb=p_amb,
        n_on=len(on),
        n_amb=len(amb),
        var_on=np.nanvar(on, axis=0, ddof=1),
        var_amb=np.nanvar(amb, axis=0, ddof=1),
        t0=min(d.t0 for d in dwells),
        t1=max(d.t1 for d in dwells),
    )


def calibrate(p, sol):
    """Raw accumulation -> input temperature at the switch common port."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return p / sol.gain - sol.t_rx


def propagate(p_ant, sol, n_ant, var_ant, sig_enr_db=0.0, sig_t_amb=0.0,
              sig_t_ns=0.0):
    """Propagate to sigma(T_ant), split by error source. Returns a dict of
    1-sigma contributions in K, all same shape as ``p_ant``.

    Write the estimator in the form where every term is explicit:

        T_ant = dT * N / D + T_amb,
        dT = T_hot - T_amb = T_ns + T_ENR - T_amb,
        N  = P_ant - P_amb,   D = P_on - P_amb,   R = N / D.

    Exact partials:

        dT_ant/dP_ant =  dT / D
        dT_ant/dP_on  = -dT * N / D^2
        dT_ant/dP_amb =  dT * (N - D) / D^2
        dT_ant/dT_ENR =  R                    (T_ENR enters only via dT)
        dT_ant/dT_ns  =  R                    (likewise)
        dT_ant/dT_amb =  1 - R                (T_amb enters dT *and* the offset)

    ``sig_enr_db`` is the 1-sigma uncertainty on the *effective* ENR,
    (ENR_dB - atten_dB). A dB uncertainty is multiplicative:
    sigma(T_ENR) = T_ENR * ln(10)/10 * sig_enr_db.

    Note dT_ant/dT_amb = 1 - R changes sign at R = 1, i.e. where the antenna
    sits exactly at the hot reference. Below that the ambient-load error
    partially cancels; above it, it adds. This is real, not a bug.
    """
    dT = sol.t_hot - sol.t_amb
    D = sol.p_on - sol.p_amb
    N = p_ant - sol.p_amb
    with np.errstate(divide="ignore", invalid="ignore"):
        R = N / D

        sig_p_ant = np.abs(dT / D) * np.sqrt(var_ant / max(n_ant, 1))
        sig_p_on = np.abs(dT * N / D**2) * np.sqrt(sol.var_on / max(sol.n_on, 1))
        sig_p_amb = np.abs(dT * (N - D) / D**2) * np.sqrt(
            sol.var_amb / max(sol.n_amb, 1)
        )

        sig_enr = np.abs(R) * sol.t_enr * (np.log(10.0) / 10.0) * sig_enr_db
        sig_tns = np.abs(R) * sig_t_ns
        sig_tamb = np.abs(1.0 - R) * sig_t_amb

    stat = np.sqrt(sig_p_ant**2 + sig_p_on**2 + sig_p_amb**2)
    syst = np.sqrt(sig_enr**2 + sig_tns**2 + sig_tamb**2)
    return {
        "R": R,
        "P_ant": sig_p_ant,
        "P_on": sig_p_on,
        "P_amb": sig_p_amb,
        "T_ENR": sig_enr,
        "T_ns": sig_tns,
        "T_amb": sig_tamb,
        "stat": stat,
        "syst": syst,
        "total": np.sqrt(stat**2 + syst**2),
    }
