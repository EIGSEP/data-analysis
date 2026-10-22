"""Helpers for bench_4pass.ipynb: the four-pass bench run of 2026-10-20/21.

What belongs to this run (where its files are, how it was wired, the
heater log's phases) and the drift diagnostics the notebook adds. The
measurement itself is ``eigsep_data.noise_source_bench.measure`` and
``eigsep_cal.noise_source``; the run's as-built record is
``data-analysis/notebooks/charlie/bench_runs/2026-10-20_4pass/README.md``.
"""

import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from eigsep_data import switched as sw
from eigsep_data.noise_source_bench import BenchSetup
from eigsep_data.numerics import weighted_mean
from eigsep_data.units import celsius_to_kelvin

DATA = Path("/mnt/data02/eigsep/scratch/switch_bench_tests/10_20")
RECORD = Path("/mnt/data02/eigsep/data-analysis/notebooks/charlie/bench_runs/2026-10-20_4pass")
HEATER_LOG = RECORD / "hot_load_plateaus_20261020T172431Z.log"
#: Lab reflection of the noise-source path at P (VNA on the field board's LNA
#: port, switch on RFNON / RFNOFF), 2026-10-15; the same board as this run.
GAMMA_NS_FILE = Path("/mnt/data02/eigsep/scratch/rfn_sparameters_10_15.npz")
#: The 2026-10-08 interim run, re-measured here for comparison with the same
#: package code and the same wiring as below (GP27, the tempctrl_lna1 stream, on
#: the reference load in both runs; noise_source v0001 took the reference from
#: tempctrl_lna2, LNA 3's thermistor). PREV_ROOT holds tcal v0004 and the lab
#: Gamma_NS file (on the memo branch until it merges).
PREV_DATA = Path("/mnt/data02/eigsep/marjum-2026-07/data/bench/2026-10-08_interim")
PREV_ROOT = Path("/mnt/data02/eigsep-worktrees/memo-004-methods/marjum-2026-07")
#: The thresholds noise_source v0001 used for 2026-10-08 (its plateaus were
#: 15 min, too short for this run's stricter steady limit).
PREV_CHOICES = dict(steady_per_min=1.0, sweep_guard=(20.0, 5.0), band_mhz=(50.0, 245.0),
                    chi2_factor=3.0, max_rel_err=0.05)
TZ = ZoneInfo("America/Los_Angeles")

#: As wired (run record, section 2): hot load on the Amb/Hot Load port (RFAMB,
#: tempctrl_load, GP26), ambient reference on the Feed port (RFANT, GP27, the
#: tempctrl_lna1 stream); LNA 3 is GP28 (the tempctrl_lna2 stream). Measured
#: pull-ups. The noise diode's board thermistor is rfswitch_therm temp_therm2.
SETUP = BenchSetup(
    hot="RFAMB", ref="RFANT", non="RFNON", noff="RFNOFF",
    hot_s11="amb", ref_s11="ant",
    hot_therm=("tempctrl_load_resistance", 9980.0),
    ref_therm=("tempctrl_lna1_resistance", 9970.0),
    board_therm_c="rfswitch_therm_temp_therm2",
    antenna="box-air",
)
ORDER = (SETUP.ref, SETUP.hot, SETUP.non, SETUP.noff)
#: LNA 3's thermistor (tempctrl_lna2, GP28) and the SP1 cable's (potmon GP26,
#: logged as a voltage), with their measured pull-ups.
LNA3_THERM = ("tempctrl_lna2_resistance", 9910.0)
SP1_PULLUP_OHM = 9960.0
#: The measurement's thresholds. A cycle is steady when the hot load changes
#: by less than 0.3 K/min (1 K/min for noise_source v0001): this run's
#: 60-minute plateaus leave enough cycles for the stricter limit.
CHOICES = dict(steady_per_min=0.3, sweep_guard=(20.0, 5.0), band_mhz=(50.0, 245.0),
               chi2_factor=3.0, max_rel_err=0.05)
BANDS = ((50.0, 88.0), (108.0, 150.0), (150.0, 200.0), (200.0, 245.0))

_ACTION = re.compile(r"^(\S+ \S+)Z INFO ACTION (?:pass (\d+) (ambient|plateau \d|cool-down)|exit)")


def phases(log=HEATER_LOG):
    """The heater program's phases from its log: one row per phase with its
    ``pass``, ``phase`` (ambient, plateau 1-3, cool-down) and start and end
    (unix ``t0``, ``t1``)."""
    rows = []
    for line in open(log):
        m = _ACTION.match(line)
        if m:
            t = pd.Timestamp(m.group(1).replace(",", "."), tz="UTC").timestamp()
            rows.append((t, int(m.group(2)) if m.group(2) else 0, m.group(3) or "exit"))
    ph = pd.DataFrame(rows, columns=["t0", "pass", "phase"])
    ph["t1"] = ph["t0"].shift(-1)
    return ph[ph["phase"] != "exit"].reset_index(drop=True)


def label_times(t, ph):
    """Pass number (0 outside the program) and phase name of each time."""
    t = np.asarray(t, float)
    i = np.clip(np.searchsorted(ph["t0"].to_numpy(), t, side="right") - 1, 0, None)
    inside = (t >= ph["t0"].iloc[0]) & (t < ph["t1"].to_numpy()[i])
    return (np.where(inside, ph["pass"].to_numpy()[i], 0),
            np.where(inside, ph["phase"].to_numpy()[i], "outside"))


def sp1_cable_k(volt):
    """The SP1 cable thermistor (K) from potmon's logged voltage: R = R_pullup
    V / (3.3 - V), then Steinhart-Hart (run record, section 3)."""
    v = np.asarray(volt, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = SP1_PULLUP_OHM * v / (3.3 - v)
    return sw.thermistor_k(r, sw.FIRMWARE_PULLUP_OHM)


def temperatures(meta):
    """``meta`` with every bench temperature added: T_hot, T_ref, T_lna3,
    T_board and T_sp1 in K, and the SNAP's FPGA temperature ``fpga_c`` in
    degC."""
    meta = meta.copy()
    meta["T_hot"] = sw.thermistor_k(meta[SETUP.hot_therm[0]], SETUP.hot_therm[1])
    meta["T_ref"] = sw.thermistor_k(meta[SETUP.ref_therm[0]], SETUP.ref_therm[1])
    meta["T_lna3"] = sw.thermistor_k(meta[LNA3_THERM[0]], LNA3_THERM[1])
    meta["T_board"] = celsius_to_kelvin(meta[SETUP.board_therm_c])
    meta["T_sp1"] = sp1_cable_k(meta["potmon_pot_az_voltage"])
    meta["fpga_c"] = pd.to_numeric(meta["fpga_temp_temp_c"], errors="coerce")
    return meta


def interp_column(meta, col, t):
    """A sparsely logged metadata column (e.g. the SNAP's FPGA temperature,
    logged about every 20 s) interpolated linearly to times ``t``."""
    m = meta[["time_best", col]].dropna().sort_values("time_best")
    return np.interp(np.asarray(t, float), m["time_best"].to_numpy(float), m[col].to_numpy(float))


def load_gamma_ns(path=GAMMA_NS_FILE):
    with np.load(path) as g:
        return {"freqs_mhz": g["freqs"] / 1e6, "on": g["RFNON"], "off": g["RFNOFF"]}


def dwells_and_cycles(index, sweeps, choices=CHOICES, dwell_gap_s=5.0):
    """The dwells and cycles ``measure`` builds (its steps 1-2, the same
    package functions and thresholds), with what it does not return: each
    state's rows and spectra, the per-cycle powers and the dwell tables."""
    states, dwells, persistent = {}, {}, None
    for st in ORDER:
        meta, f, spec, _ = sw.state_dwells(index, st, SETUP.antenna, gap_s=dwell_gap_s)
        meta = temperatures(meta)
        flags, pc = sw.rfi_flags(spec, f)
        flags[sw.near_sweep(meta, sweeps, choices["sweep_guard"])] = True
        persistent = pc if persistent is None else persistent | pc
        states[st] = {"meta": meta, "freqs": f, "spec": spec, "flags": flags}
        dwells[st] = sw.dwell_means(meta, spec, flags, columns=("T_hot", "T_ref", "T_lna3", "T_board"))
    cyc, cp = sw.cycles(dwells, ORDER)
    cyc["T_lna3"] = dwells[SETUP.ref][0]["T_lna3"].to_numpy()[cyc[f"i_{SETUP.ref}"].to_numpy()]
    return {"states": states, "dwells": dwells, "cyc": cyc, "cp": cp, "persistent": persistent}


def fit_subset(res, use):
    """T_NON per channel from the cycles ``use`` of a ``measure`` result:
    the same straight line (``eigsep_cal.noise_source``) on a subset."""
    from eigsep_cal.noise_source import fit_source_temperature, source_temperature_terms

    cyc, mm = res["cycles"], res["mismatch"]
    x, y = source_temperature_terms(res["q"], mm["hot"], mm["ref"], mm["ns"],
                                    cyc["T_hot"].to_numpy()[:, None],
                                    cyc["T_ref"].to_numpy()[:, None])
    return fit_source_temperature(x, y, np.asarray(use, bool))


def gain_and_trx(cp, res, t_non):
    """Per cycle and channel, the receiver's gain g (counts/K) and noise
    temperature T_r (K) from the reference load and the noise source alone:
    P_ref = g (M_ref T_ref + T_r) and P_NON = g (M_NS T_NON + T_r), with
    T_NON the run's fit. The hot load is not used, so every cycle counts."""
    cyc, mm = res["cycles"], res["mismatch"]
    t_ref = cyc["T_ref"].to_numpy()[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (cp[SETUP.non] - cp[SETUP.ref]) / (mm["ns"] * t_non[None, :] - mm["ref"] * t_ref)
        t_r = cp[SETUP.ref] / g - mm["ref"] * t_ref
    return g, t_r


def cycle_tnon(res, intercept):
    """Per cycle and channel, T_NON from that cycle alone: (y - a) / x, with
    ``intercept`` a the all-cycle fit's (``source_temperature_terms``).
    Meaningful only where the hot load is far from the reference (the
    plateaus); without subtracting a, y / x would change with the plateau."""
    from eigsep_cal.noise_source import source_temperature_terms

    cyc, mm = res["cycles"], res["mismatch"]
    x, y = source_temperature_terms(res["q"], mm["hot"], mm["ref"], mm["ns"],
                                    cyc["T_hot"].to_numpy()[:, None],
                                    cyc["T_ref"].to_numpy()[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (y - intercept[None, :]) / x


def redwell(D, skip_first):
    """The cycle powers of ``dwells_and_cycles`` again, with the first
    ``skip_first`` integrations of every dwell dropped (same rows, flags and
    cycles otherwise). Returns {state: (ncycle, nchan) powers}, matched to
    ``D["cyc"]`` by dwell id; a dwell left with no integrations gives NaN."""
    cyc, out = D["cyc"], {}
    for st in ORDER:
        s = D["states"][st]
        tab, means = sw.dwell_means(s["meta"], s["spec"], s["flags"], skip_first=skip_first)
        row = {d: i for i, d in enumerate(tab["dwell"].to_numpy())}
        ids = D["dwells"][st][0]["dwell"].to_numpy()[cyc[f"i_{st}"].to_numpy()]
        nan = np.full(means.shape[1], np.nan)
        out[st] = np.array([means[row[d]] if d in row else nan for d in ids])
    return out


def fit_powers(cp, res, use):
    """T_NON per channel from cycle powers ``cp`` (e.g. ``redwell``'s), with
    ``res``'s reflections and temperatures, over the cycles ``use``."""
    from eigsep_cal.noise_source import fit_source_temperature, source_temperature_terms

    cyc, mm = res["cycles"], res["mismatch"]
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (cp[SETUP.hot] - cp[SETUP.ref]) / (cp[SETUP.non] - cp[SETUP.ref])
    x, y = source_temperature_terms(q, mm["hot"], mm["ref"], mm["ns"],
                                    cyc["T_hot"].to_numpy()[:, None],
                                    cyc["T_ref"].to_numpy()[:, None])
    return fit_source_temperature(x, y, np.asarray(use, bool))


def band_wmean(values, sigma, freqs, keep, lo, hi):
    """Inverse-variance weighted mean of ``values`` over the ``keep`` channels
    in [lo, hi) MHz."""
    k = keep & (freqs >= lo) & (freqs < hi)
    return float(weighted_mean(values[k], 1.0 / sigma[k] ** 2))


def band_rows(x, freqs, keep, lo, hi):
    """Per row of ``x`` (rows x channels), the mean over the ``keep`` channels
    in [lo, hi) MHz (equal weights; NaN channels left out)."""
    k = keep & (freqs >= lo) & (freqs < hi)
    return weighted_mean(x[:, k], np.ones(int(k.sum()))[None, :], axis=1)


def local(t):
    """Unix seconds as naive local (PDT) datetimes, for plotting."""
    return pd.to_datetime(np.asarray(t, float), unit="s", utc=True).tz_convert(TZ).tz_localize(None)
