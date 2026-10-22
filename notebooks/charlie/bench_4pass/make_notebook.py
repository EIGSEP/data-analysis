"""Build notebooks/charlie/bench_4pass/bench_4pass.ipynb (cells only; run with nbconvert)."""
import sys

import nbformat as nbf

out = sys.argv[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Noise-source temperature from the 2026-10-20 four-pass bench run, and system drift

**Goal:** measure $T_\\mathrm{NON}$, the noise-source path's temperature at reference plane P, from four passes of the hot-load temperature program, as `bench_tns.ipynb` did for the 2026-10-08 interim run; check that the four passes agree; and measure how the system (gain, receiver temperature, noise source, reflections) drifts over the ~24 h run.

**The run** (record: `data-analysis/notebooks/charlie/bench_runs/2026-10-20_4pass/README.md`): 2026-10-20 10:24 → 10-21 10:26 PDT, field switch board, hot 50 Ω load on `RFAMB` (thermistor `tempctrl_load`), ambient 50 Ω reference on `RFANT` (`tempctrl_lna1`, GP27), lab LNA 14 → 3 → 10, `corr_acc_len` 2²⁸ (1.074 s), 10 s per state in a 1-minute cycle `RFANT → RFAMB → RFNON → RFNOFF → RFSP1 short → RFSP1 open`, VNA every 20 min. Four passes of: ambient 60 min, plateaus at 47 / 67 / 87 °C targets for 60 min each, cool-down 90 min.

**Method:** the same as `bench_tns.ipynb` and `noise_source` v0001, now in the packages (installed from `main`): `eigsep_data.noise_source_bench.measure` (dwells, RFI flags and the VNA-sweep guard, cycles, steady-cycle selection, the mismatch-corrected fit and channel mask) and `eigsep_cal.noise_source` (the model). Run-specific constants and the drift helpers are in `bench_4pass.py` beside this notebook. Times are local (PDT, UTC−7).

**Thermistors are named by pin.** GP26 is the hot load, GP27 the ambient reference load and GP28 LNA 3. In both bench runs' data the GP27 channel is the `tempctrl_lna1` stream and GP28 the `tempctrl_lna2` stream (shown for this run in § 2, and for 2026-10-08 from that run's ambient minutes); the 2026-10-08 config labels them the other way round, and `noise_source` v0001 took its reference temperature from GP28 (LNA 3). The 2026-10-08 run is re-measured here with GP27 (§ 1).

**Steady cycles** are those in which the hot load changes by less than 0.3 K/min (`b4.CHOICES`), stricter than v0001's 1 K/min."""))

C.append(code("""import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline
import eigsep_data, eigsep_cal
from eigsep_data import MetadataIndex, switched as sw
from eigsep_data.noise_source_bench import measure
from eigsep_data.numerics import weighted_mean
from eigsep_data.products import noise_source
from eigsep_cal import S11
from eigsep_cal.conventions import D5_ENBW_HZ
from eigsep_cal.noise_source import excess_from_enr, enr_from_excess, fit_source_temperature, switched_ratio_noise
import bench_4pass as b4

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
S = b4.SETUP
CS = {S.ref: "#0072B2", S.hot: "#D55E00", S.non: "#009E73", S.noff: "#CC79A7", "RFSP1": "#E69F00"}   # colour-blind safe
CK = {"steady": "#000000", "heating": "#D55E00", "cooling": "#56B4E9"}
CP = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7"}                                          # passes
K0 = 273.15
DATASHEET_K = tuple(float(x) for x in excess_from_enr(np.array([30.0, 35.0]), 30.0))   # NC302L, behind the 30 dB pad
loc = b4.local
HHMM = mdates.DateFormatter("%m-%d %H:%M")
print(eigsep_data.__file__, eigsep_cal.__file__, sep="\\n")"""))

C.append(md("## 1. Load the data and run the measurement"))
C.append(code("""idx = MetadataIndex(str(b4.DATA), cache=False, streams="all")
allmeta = b4.temperatures(idx.select().meta.reset_index(drop=True))
ph = b4.phases()
sweeps = sw.vna_sweeps(allmeta)
s11 = S11(str(b4.DATA), year=2026, pattern="*s11_*.h5")
s11.calibrate()
gamma = b4.load_gamma_ns()
res = measure(idx, s11, gamma, S, **b4.CHOICES)
# The same dwells and cycles, with what measure does not return (per-state rows and spectra, cycle powers)
D = b4.dwells_and_cycles(idx, sweeps)
assert np.array_equal(D["cyc"]["t"].to_numpy(), res["cycles"]["t"].to_numpy())
cyc = res["cycles"].copy()
cyc["T_lna3"] = D["cyc"]["T_lna3"].to_numpy()
# The SNAP's FPGA temperature is logged about every 20 s, so a 10 s dwell often has no reading: interpolate it
cyc["fpga_c"] = b4.interp_column(allmeta, "fpga_c", cyc["t"])
cyc["pass"], cyc["phase"] = b4.label_times(cyc["t"].to_numpy(), ph)
cp, Q, mm = D["cp"], res["q"], res["mismatch"]
freqs, good, mask, vb = res["freqs"], res["good"], res["mask"], res["validation"]
steady = (cyc["kind"] == "steady").to_numpy()
inb = (freqs >= b4.CHOICES["band_mhz"][0]) & (freqs <= b4.CHOICES["band_mhz"][1])

t = allmeta["time_best"].to_numpy(float)
print(f"{len(allmeta)} integrations of {allmeta.integration_time.iloc[0]:.3f} s in {allmeta.file.nunique()} files, "
      f"{loc(np.nanmin(t)):%m-%d %H:%M} to {loc(np.nanmax(t)):%m-%d %H:%M} PDT")
print("switch states:", allmeta.rfswitch.value_counts().to_dict())
print(f"{len(sweeps)} VNA sweeps; {len(s11.get_all_s11s('rec')[0])} receiver and {len(s11.get_all_s11s('amb')[0])} load S11 sweeps")
print("rows excluded around sweeps:", vb["rows_excluded_around_sweeps"])
print("in-band samples flagged:", {k: f"{100*v:.2f} %" for k, v in vb["flagged_in_band_fraction"].items()})
print(f"complete cycles: {len(cyc)};", vb["cycles"])
print("cycles per pass and phase (steady only):")
print(cyc[steady].groupby(["pass", "phase"]).size().unstack(fill_value=0))
ft = allmeta[["time_best", "fpga_c"]].dropna()
print(f"SNAP FPGA temperature: {len(ft)} readings, median spacing {np.median(np.diff(ft.time_best)):.1f} s")

# The 2026-10-08 interim run, re-measured with the same code, wiring (GP27 reference) and v0001's thresholds
idx0 = MetadataIndex(str(b4.PREV_DATA), cache=False, streams="all")
s110 = S11(str(b4.PREV_DATA), year=2026, pattern="*s11_*.h5")
s110.calibrate()
res0 = measure(idx0, s110, gamma, S, **b4.PREV_CHOICES)
print("2026-10-08 re-measured:", res0["validation"]["cycles"], "cycles;", res0["validation"]["channels_good"], "good channels")"""))

C.append(md("## 2. Temperatures over the run, and which cycles are steady"))
C.append(code("""fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True, height_ratios=(3, 1.2, 1.6))
tl = loc(allmeta.time_best)
on = allmeta["tempctrl_load_enabled"].astype("boolean").fillna(False).to_numpy(bool)
axs[0].fill_between(tl, 0, 1, where=on, transform=axs[0].get_xaxis_transform(), color="0.9", lw=0, label="heater enabled")
for col, c, lab in (("T_hot", CS[S.hot], "hot load (GP26)"), ("T_ref", CS[S.ref], "ambient reference (GP27)"),
                    ("T_lna3", "#E69F00", "LNA 3 (GP28)"), ("T_board", CS[S.non], "noise-diode board (temp_therm2)"),
                    ("T_sp1", "0.5", "SP1 cable (potmon)")):
    axs[0].plot(tl, allmeta[col] - K0, color=c, lw=0.7, label=lab)
for t0 in ph.loc[ph.phase == "ambient", "t0"]:
    for ax in axs:
        ax.axvline(loc([t0])[0], color="k", lw=0.5, ls=":")
axs[0].set(ylabel="temperature (°C)")
axs[0].legend(fontsize=7, ncol=3, loc="upper right")
f_ok = allmeta.fpga_c.notna()
axs[1].plot(tl[f_ok], allmeta.fpga_c[f_ok], color="0.3", lw=0.7)
axs[1].set(ylabel="SNAP FPGA (°C)")
lim = b4.CHOICES["steady_per_min"]
for k, c in CK.items():
    m = (cyc.kind == k).to_numpy()
    axs[2].plot(loc(cyc.t[m]), cyc.rate_per_min[m], "o", ms=1.5, color=c, label=f"{k} ({m.sum()})")
axs[2].axhspan(-lim, lim, color="0.88", lw=0)
axs[2].set_yscale("symlog", linthresh=lim)
axs[2].set(ylabel="hot load (K/min)", xlabel="PDT")
axs[2].legend(fontsize=7, ncol=3)
axs[2].xaxis.set_major_formatter(HHMM)
plt.tight_layout(); plt.show()"""))
C.append(md("""**Figure 1.** Top: the bench thermistors over the run (measured pull-ups and Steinhart–Hart, run record § 3), heater-enabled periods shaded, the start of each pass's ambient block dotted. Middle: the SNAP's FPGA temperature. Bottom: the hot-load thermistor's rate of change in each switch cycle (symmetric log), by class; only cycles inside the grey band (|dT/dt| < 1 K/min) are *steady* and fitted. The heating ramps and the start of each cool-down fall outside."""))

C.append(md("""### Which thermistor is the reference load?

GP27 is on the ambient reference load and GP28 on LNA 3. In each ambient block the heater has been off for at least 90 min, so the hot load and the reference load sit at room temperature and should read alike; and $Q$, the hot load's power over the noise source's, both measured against the reference, should then be close to zero."""))
C.append(code("""# T_NS as this receiver takes it (Q per kelvin, no reflection correction; § 7), 108–200 MHz, to put Q in kelvin
fs0 = fit_source_temperature(np.repeat((cyc.T_hot - cyc.T_ref).to_numpy()[:, None], freqs.size, axis=1), Q, steady)
mid = good & (freqs >= 108) & (freqs < 200)
tns_mid = float(np.mean(1 / fs0["b"][mid]))
rows = []
for p in range(1, 5):
    a = ph[(ph["pass"] == p) & (ph.phase == "ambient")].iloc[0]
    m = (allmeta.time_best >= a.t1 - 1800) & (allmeta.time_best < a.t1)        # the block's last 30 min
    c = steady & (cyc["pass"].to_numpy() == p) & (cyc.phase.to_numpy() == "ambient") & (cyc.t.to_numpy() >= a.t1 - 1800)
    qb = b4.band_rows(Q[c], freqs, good, 108, 200)
    rows.append({"pass": p, "T_hot (°C)": round(allmeta.T_hot[m].mean() - K0, 2),
                 "GP27 (reference) − hot (K)": round((allmeta.T_ref[m] - allmeta.T_hot[m]).mean(), 2),
                 "GP28 (LNA 3) − hot (K)": round((allmeta.T_lna3[m] - allmeta.T_hot[m]).mean(), 2),
                 "Q, 108–200 MHz (mean over cycles)": round(float(np.nanmean(qb)), 4),
                 "Q × T_NS (K)": round(float(np.nanmean(qb)) * tns_mid, 1),
                 "cycles": int(c.sum())})
pd.DataFrame(rows)"""))
C.append(md("""**Table 1.** Each ambient block's last 30 minutes: the hot load's temperature, the two candidate reference thermistors relative to it, and the mean of $Q$ over that block's steady cycles (unmasked channels, 108–200 MHz), also as a temperature, $Q\,T_\mathrm{NS}$, the power difference between the hot load and the reference in kelvin as this receiver takes it (no reflection correction). Neither thermistor reads the hot load's temperature: the reference load sits on the bench a few kelvin warmer than the hot load's aluminium block, and $Q\,T_\mathrm{NS}$ says by how much in power, independently of either thermistor. GP27's offset is the one close to it; GP28 (LNA 3) runs warmer again."""))

C.append(md("## 3. Is the switch sequence clean?"))
C.append(code("""a = ph[(ph["pass"] == 1) & (ph.phase == "ambient")].iloc[0]
k0 = int(np.flatnonzero(allmeta.time_best.to_numpy() > a.t0 + 1200)[0])                 # 20 min into pass 1
sl = slice(k0, k0 + 2 * 56)                                                             # two 1-minute cycles
i150 = int(np.argmin(abs(freqs - 150)))
spec_win = np.asarray(idx.select(time=(allmeta.time_best[k0] - 1, allmeta.time_best[k0 + 2 * 56] + 1)).load_bundle(antenna=S.antenna, products=[]).data, float)
mw = idx.select(time=(allmeta.time_best[k0] - 1, allmeta.time_best[k0 + 2 * 56] + 1)).meta.reset_index(drop=True)
mw = b4.temperatures(mw)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
tw = loc(mw.time_best)
for st in list(CS) + ["UNKNOWN"]:
    m = (mw.rfswitch == st).to_numpy()
    if st == "RFSP1":
        for term, mk in (("SHORT", "v"), ("OPEN", "^")):
            mt = m & (mw.potmon_sp1_term_name.astype(str).str.upper() == term).to_numpy()
            a0.semilogy(tw[mt], spec_win[mt, i150], mk, ms=3, color=CS[st], label=f"RFSP1 {term.lower()}")
    else:
        a0.semilogy(tw[m], spec_win[m, i150], "o", ms=3, color=CS.get(st, "0.7"), label=st)
a0.set(ylabel=f"power at {freqs[i150]:.1f} MHz (counts)")
a0.legend(fontsize=7, ncol=4)
a1.plot(tw, mw.T_board - K0, color=CS[S.non])
m = (mw.rfswitch == S.non).to_numpy()
a1.plot(tw[m], mw.T_board[m] - K0, "o", ms=3, color=CS[S.non], label="RFNON rows")
a1.set(ylabel="diode board (°C)", xlabel="PDT")
a1.legend(fontsize=7)
a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.tight_layout(); plt.show()"""))
C.append(md("""**Figure 2.** Two switch cycles, 20 minutes into pass 1: raw power of each integration at 150 MHz by state, with the SP1 short and open visits marked by the SP1 controller's reported termination, and integrations that straddle a switch change (`UNKNOWN`, grey; never used). Bottom: the noise-diode board thermistor, which warms during each `RFNON` dwell (the diode heats when on)."""))

C.append(md("## 4. Settling after a switch change"))
C.append(code("""fig, ax = plt.subplots(figsize=(6, 3))
bset = (freqs > 60) & (freqs < 240) & ~((freqs > 88) & (freqs < 108))
for st in b4.ORDER:
    meta, sp = D["states"][st]["meta"], D["states"][st]["spec"]
    rel = []
    for d, g in meta[meta.dwell >= 0].groupby("dwell"):
        x = np.nanmean(sp[g.index][:, bset], axis=1)
        if 5 <= len(x) <= 12:                                   # normal ~10 s visits only
            rel.append(pd.Series(x / x[-3:].mean() - 1, index=g.pos_in_dwell.values))
    r = pd.concat(rel, axis=1)
    ax.plot(r.index, r.mean(axis=1) * 100, "o-", color=CS[st], ms=3, label=f"{st} ({r.shape[1]} dwells)")
ax.set(xlabel="integration within dwell", ylabel="power vs dwell end (%)")
ax.legend(fontsize=7)
plt.tight_layout(); plt.show()"""))
C.append(md("""**Figure 3.** Band power (60–240 MHz without FM) of each integration relative to the last three integrations of its dwell, averaged over the normal (5–12 integration) dwells, per state. Integrations well off their dwell's end mean the switch or the signal chain settles more slowly than one integration."""))

C.append(code("""# T_NON with the first k integrations of every dwell dropped (same rows, flags, cycles and reflections)
fit0 = b4.fit_powers(cp, res, steady)
assert np.allclose(fit0["b"][good], res["t_non_k"][good], rtol=1e-9)          # k = 0 reproduces measure
rows = []
for k in range(0, 5):
    ft = fit0 if k == 0 else b4.fit_powers(b4.redwell(D, k), res, steady)
    row = {"first integrations dropped": k}
    for lo, hi in b4.BANDS:
        row[f"excess {lo:.0f}–{hi:.0f} (K)"] = round(b4.band_wmean(ft["b"] - res["t_off_k"], ft["sb"], freqs, good, lo, hi), 1)
    rows.append(row)
settle = pd.DataFrame(rows).set_index("first integrations dropped")
settle"""))
C.append(md("""**Table 2.** The noise source's excess at P (§ 11) fitted again with the first $k$ integrations of every dwell dropped from the dwell means, everything else unchanged ($k = 0$ is `measure`'s result). If the low first integrations of Figure 3 bias the dwell means, the excess moves with $k$ until the settled part alone remains."""))
C.append(md("## 5. Within-dwell noise against the radiometer equation"))
C.append(code("""meta, sp = D["states"][S.ref]["meta"], D["states"][S.ref]["spec"]
rows = []
for fr in (75, 150, 220):
    i = int(np.argmin(abs(freqs - fr)))
    sd, pred = [], []
    for d, g in meta[meta.dwell >= 0].groupby("dwell"):
        x = sp[g.index, i]
        x = x[x > 0]
        if len(x) > 3:
            sd.append(x.std(ddof=1)); pred.append(x.mean() / np.sqrt(D5_ENBW_HZ * g.integration_time.iloc[0]))
    r = np.array(sd) / np.array(pred)
    rows.append({"MHz": round(freqs[i], 1), "dwells": r.size, "scatter / radiometer, mean": round(r.mean(), 2),
                 "95th pct": round(np.percentile(r, 95), 2), "max": round(r.max(), 1)})
print(f"Δν = {D5_ENBW_HZ/1e3:.1f} kHz (one channel; the PFB's ENBW is not yet measured), τ = {meta.integration_time.iloc[0]:.3f} s")
pd.DataFrame(rows)"""))
C.append(md("""**Table 3.** The scatter of raw power between the integrations of one ~10 s reference dwell, divided by $P/\\sqrt{\\Delta\\nu\\,\\tau}$. Near 1 means the data are radiometer-noise limited within a dwell, with $\\Delta\\nu$ the channel spacing standing in for the filterbank's equivalent noise bandwidth."""))

C.append(md("## 6. Q against the temperature difference"))
C.append(code("""dT = (cyc.T_hot - cyc.T_ref).to_numpy()
X_unc = np.repeat(dT[:, None], freqs.size, axis=1)
fits = {k: fit_source_temperature(X_unc, Q, (cyc.kind == k).to_numpy()) for k in CK}
fs = fits["steady"]
picks = [55, 75, 150, 220]
fig, axs = plt.subplots(2, len(picks), figsize=(11, 4.6), sharex=True, height_ratios=(2, 1))
xx = np.array([np.nanmin(dT), np.nanmax(dT)])
for j, fr in enumerate(picks):
    i = int(np.argmin(abs(freqs - fr)))
    for kind, c in CK.items():
        m = (cyc.kind == kind).to_numpy()
        axs[0, j].plot(dT[m], Q[m, i], ".", ms=2, color=c, label=kind)
        axs[1, j].plot(dT[m], Q[m, i] - fs["a"][i] - fs["b"][i] * dT[m], ".", ms=1.5, color=c)
    axs[0, j].plot(xx, fs["a"][i] + fs["b"][i] * xx, "k-", lw=0.8)
    axs[0, j].set_title(f"{freqs[i]:.1f} MHz: T_NS = {1/fs['b'][i]:.0f} K (steady)", fontsize=8)
    axs[1, j].axhline(0, color="k", lw=0.5)
    axs[1, j].set_xlabel("T_hot − T_ref (K)")
axs[0, 0].set_ylabel("Q"); axs[1, 0].set_ylabel("Q − steady fit")
axs[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.show()"""))
C.append(md("""**Figure 4.** $Q = (P_\\mathrm{hot} - P_\\mathrm{ref})/(P_\\mathrm{NON} - P_\\mathrm{ref})$ per switch cycle against the thermistor temperature difference at four frequencies, coloured by cycle class, with the straight line fitted to the steady cycles of all four passes (no reflection correction yet: its inverse slope is $T_\\mathrm{NS}$ as the bench receiver takes it). Bottom: residuals from that line. The four plateaus of each temperature should land on one cluster if the system is stable between passes."""))
C.append(code("""rows = []
for fr in picks:
    i = int(np.argmin(abs(freqs - fr)))
    row = {"MHz": round(freqs[i], 1)}
    for k in CK:
        g = fits[k]
        row[f"T_NS {k} (K)"] = round(1 / g["b"][i])
        if k != "steady":
            row[f"{k} − steady (σ)"] = round((g["b"][i] - fs["b"][i]) / np.hypot(g["sb"][i], fs["sb"][i]), 1)
    rows.append(row)
print({k: int((cyc.kind == k).sum()) for k in CK}, "cycles per class")
print("ramp check over the good channels (mean z, 95th pct |z|):", vb["ramp_check_heating_cooling_vs_steady"])
pd.DataFrame(rows)"""))
C.append(md("""**Table 4.** The ramp check: $T_\\mathrm{NS}$ fitted to the heating and cooling cycles alone against the steady cycles, in units of their combined error. If the thermistor tracked the load during ramps, all three would agree; a large heating offset means the thermistor lags the heater, which is why only steady cycles are fitted."""))

C.append(md("## 7. $T_\\mathrm{NS}$ across the band (steady cycles, no reflection correction)"))
C.append(code("""tns = 1 / fs["b"]
band = (freqs > 45) & (freqs < 250)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, height_ratios=(2, 1))
a0.plot(freqs[band], tns[band], color="k", lw=0.7, label="steady cycles, all passes, no reflection correction")
a0.axhspan(*DATASHEET_K, color=CS[S.hot], alpha=0.12, label=f"ENR 30–35 dB behind the pad ({DATASHEET_K[0]:.0f}–{DATASHEET_K[1]:.0f} K)")
a0.axvspan(88, 108, color="0.9", label="FM band")
a0.set(ylabel="T_NS (K)", ylim=(0, 1200))
a0.legend(fontsize=7)
a1.plot(freqs[band], fs["a"][band], color="0.3", lw=0.7)
a1.axhline(0, color="k", lw=0.5)
a1.axvspan(88, 108, color="0.9")
a1.set(ylabel="intercept of Q", xlabel="frequency (MHz)")
plt.tight_layout(); plt.show()"""))
C.append(md("""**Figure 5.** $T_\\mathrm{NS}$ per channel from the inverse slope of $Q$ against $T_\\mathrm{hot} - T_\\mathrm{ref}$ over the steady cycles of all passes, before any reflection correction or channel mask (top), and the fit's intercept (bottom; zero if the two loads match and the thermistors are right)."""))

C.append(md("## 8. Reflections at P, and how they drift"))
C.append(code("""fv = np.asarray(s11.freqs_mhz, float)
bf = (fv >= 50) & (fv <= 245)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, (st, dut) in zip(axs[:2], ((S.hot, S.hot_s11), (S.ref, S.ref_s11))):
    for plane, ls in (("dut", ":"), ("lna", "-")):
        ts, g = s11.get_all_s11s(dut, plane)
        for x in g:
            ax.semilogy(fv, np.abs(x), ls, color=CS[st], lw=0.5, alpha=0.6)
        ax.plot([], [], ls, color=CS[st], label=f"@{'DUT' if plane == 'dut' else 'P'} ({len(g)} sweeps)")
    ax.set(title=f"{st} ({'hot load' if st == S.hot else 'ambient reference'})", xlabel="frequency (MHz)")
    ax.legend(fontsize=7)
ts, g = s11.get_all_s11s("rec")
for x in g:
    axs[2].semilogy(fv, np.abs(x), color="0.3", lw=0.5, alpha=0.6)
axs[2].set(title=f"receiver at P ({len(g)} sweeps)", xlabel="frequency (MHz)")
axs[0].set_ylabel("|Γ|")
for ax in axs:
    ax.axvspan(50, 245, color="0.93", zorder=0)
plt.tight_layout(); plt.show()
print("max |Γ| at the loads' own ports (50-245 MHz), all sweeps:", {k: round(v, 4) for k, v in vb["loads_at_their_ports_max_abs_gamma"].items()})"""))
C.append(md("""**Figure 6.** $|\\Gamma|$ of every sweep: the hot load (`RFAMB`) and the reference (`RFANT`) at their own ports (@DUT, dotted) and at plane P through their switch paths (solid), and the receiver at P. The loads de-embed to a few thousandths at their ports if the packaged switch-path characterization fits this board."""))
C.append(code("""fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
T_at = lambda tt, col: np.interp(tt, allmeta.time_best, allmeta[col])
for (dut, plane, lab, c) in ((S.hot_s11, "lna", "hot load at P", CS[S.hot]), (S.ref_s11, "lna", "reference at P", CS[S.ref]),
                             ("rec", "default", "receiver at P", "0.3")):
    ts, g = s11.get_all_s11s(dut, plane)
    ts = np.asarray(ts, float); g = np.asarray(g)
    d = np.mean(np.abs(g[:, bf] - g[:, bf].mean(0)), axis=1)
    axs[0].plot(loc(ts), d, "o-", ms=2.5, lw=0.6, color=c, label=lab)
axs[0].set(ylabel="mean |Γ − Γ̄| (50–245 MHz)")
axs[0].legend(fontsize=7)
ts, g = s11.get_all_s11s(S.hot_s11, "dut")
ts = np.asarray(ts, float)
axs[1].plot(loc(ts), np.mean(np.abs(np.asarray(g)[:, bf]), axis=1), "o-", ms=2.5, lw=0.6, color=CS[S.hot], label="hot load |Γ| at its port")
ax2 = axs[1].twinx(); ax2.grid(False)
ax2.plot(loc(allmeta.time_best), allmeta.T_hot - K0, color=CS[S.hot], lw=0.5, alpha=0.4)
ax2.set_ylabel("hot load (°C)", color=CS[S.hot])
axs[1].set(ylabel="mean |Γ| (50–245 MHz)", xlabel="PDT")
axs[1].legend(fontsize=7, loc="upper left")
axs[1].xaxis.set_major_formatter(HHMM)
plt.tight_layout(); plt.show()
g_hot_port = np.asarray(s11.get_all_s11s(S.hot_s11, "dut")[1])[:, bf]
r = np.corrcoef(np.mean(np.abs(g_hot_port), axis=1), T_at(ts, "T_hot"))[0, 1]
print(f"corr(hot load's mean |Γ| at its port, its temperature) over {len(ts)} sweeps: {r:+.2f}")"""))
C.append(md("""**Figure 7.** Drift of the reflections over the run. Top: each sweep's mean distance from the run-mean $\\Gamma$ at P (50–245 MHz), for the hot load, the reference and the receiver. Bottom: the hot load's mean $|\\Gamma|$ at its own port per sweep, with its temperature (faint, right axis): whether the load's match changes with its temperature."""))

C.append(md("""### The receiver's reflection is ripplier than on 2026-10-08

Figure 6's receiver curves carry a ripple the 2026-10-08 run did not. Below: where in the calibration chain it appears (`raw`, the VNA's uncorrected reading; `vna`, after the internal standards; `default`, at P), how repeatable it is from sweep to sweep, and its delay."""))
C.append(code("""def ripple(s, dut, plane, k=15):
    fv_ = np.asarray(s.freqs_mhz, float); b = (fv_ > 50) & (fv_ < 245)
    g = np.asarray(s.get_all_s11s(dut, plane)[1])
    a = np.abs(g[:, b])
    r = (a - np.array([np.convolve(q, np.ones(k) / k, mode="same") for q in a]))[:, k:-k]
    cc = np.mean([np.corrcoef(r[i], r[i + 1])[0, 1] for i in range(len(r) - 1)])
    return {"mean |Γ|": round(float(a.mean()), 4), "ripple rms": round(float(np.std(r)), 5),
            "sweep-to-sweep std": round(float(np.median(a.std(0))), 5), "ripple corr. between sweeps": round(float(cc), 2)}
rows = [{"run": run, "plane": pl, **ripple(ss, "rec", pl)} for run, ss in (("2026-10-08", s110), ("2026-10-20", s11)) for pl in ("raw", "vna", "default")]
display(pd.DataFrame(rows))
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.4))
for run, ss, c in (("2026-10-08", s110, CS[S.ref]), ("2026-10-20", s11, "k")):
    fv_ = np.asarray(ss.freqs_mhz, float); b = (fv_ > 20) & (fv_ < 245)
    g = np.asarray(ss.get_all_s11s("rec", "raw")[1])
    a0.plot(fv_, np.abs(g).mean(0), color=c, lw=0.8, label=f"{run}, raw, mean of {len(g)} sweeps")
    x = g[:, b] - g[:, b].mean(1, keepdims=True)
    F = np.mean(np.abs(np.fft.fft(x * np.blackman(x.shape[1]), axis=1)) ** 2, axis=0)
    tau = np.fft.fftfreq(x.shape[1], d=(fv_[1] - fv_[0]) * 1e6) * 1e9
    o = np.argsort(tau); pos = tau[o] > 0
    a1.semilogy(tau[o][pos], F[o][pos], color=c, lw=0.8, label=run)
a0.set(xlabel="frequency (MHz)", ylabel="|Γ_rec| (raw)", xlim=(1, 250))
a0.legend(fontsize=7)
a1.set(xlabel="delay (ns)", ylabel="power (raw Γ_rec, 20–245 MHz)", xlim=(0, 120))
a1.legend(fontsize=7)
plt.tight_layout(); plt.show()"""))
C.append(md("""**Table and Figure 7b.** The receiver's reflection in both bench runs. The extra structure is already in the VNA's raw reading, before any calibration, so the de-embedding does not make it, and it repeats from sweep to sweep (ripple correlation 0.97 between consecutive sweeps on 10-20, against 0.2–0.3 on 10-08), so it is a fixed feature of the measurement, not noise. Left: it shows as narrow bumps near 73, 97, 108 and 155 MHz on a curve whose overall shape also differs from 10-08's; right: as broad excess power from ~15 to ~100 ns of delay rather than one reflection. The VNA settings (1000 points, 100 Hz IFBW, −40 dBm on the receiver path) are the same in both runs' headers. Candidates, untested: a changed connection on the receiver side of P (the P → LNA 14 cable or adapters, which the run record has yet to describe), or steady external signals picked up at −40 dBm (two of the bumps fall in the FM band)."""))
C.append(md("## 9. Mismatch factors"))
C.append(code("""fig, ax = plt.subplots(figsize=(9, 3))
for key, st in (("hot", S.hot), ("ref", S.ref)):
    ax.plot(freqs[inb], mm[key][steady][:, inb].mean(0), color=CS[st], lw=0.8, label=f"M {st}")
ax.plot(freqs[inb], mm["ns"][steady][:, inb].mean(0), color=CS[S.non], lw=0.8, label="M_NS (lab Γ_NS)")
ax.plot(freqs[inb], (mm["hot"] - mm["ref"])[steady][:, inb].mean(0) + 1, color="0.4", lw=0.8, ls="--", label="1 + (M_hot − M_ref)")
ax.set(xlabel="frequency (MHz)", ylabel="mismatch factor")
ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
print("M_NS range in band:", [round(x, 3) for x in vb["mismatch_factor_ns_range"]],
      "| |Γ_NS| range:", [round(x, 3) for x in vb["gamma_ns"]["abs_range_in_band"]],
      "| max |Γ_NS on − off|:", round(vb["gamma_ns"]["max_abs_on_minus_off"], 4))"""))
C.append(md("""**Figure 8.** Mismatch factors $M = (1-|\\Gamma|^2)/|1-\\Gamma\\Gamma_\\mathrm{rec}|^2$ at P with the bench receiver, averaged over the steady cycles: the two loads (from this run's sweeps), the noise source (from the 2026-10-15 lab measurement of $\\Gamma_\\mathrm{NS}$ at P on this board), and the hot-minus-reference difference that enters $Q$."""))

C.append(md("## 10. Channel mask"))
C.append(code("""th, tr = cyc.T_hot.to_numpy()[:, None], cyc.T_ref.to_numpy()[:, None]
X_cor = mm["hot"] * th - mm["ref"] * tr
fit_cor = fit_source_temperature(X_cor, Q, steady)
tau = float(allmeta.integration_time.iloc[0])
sq = switched_ratio_noise(cp[S.hot], cp[S.ref], cp[S.non], cyc[f"n_{S.hot}"].to_numpy()[:, None],
                          cyc[f"n_{S.ref}"].to_numpy()[:, None], cyc[f"n_{S.non}"].to_numpy()[:, None], D5_ENBW_HZ, tau)
with np.errstate(invalid="ignore", divide="ignore"):
    z2 = np.where(steady[:, None], (fit_cor["res"] / sq) ** 2, np.nan)
chi2 = np.array([np.nanmean(c) if np.isfinite(c).any() else np.nan for c in z2.T])
print({k: int(v.sum()) for k, v in mask.items()}, "channels masked;", int(good.sum()), "good;",
      f"in-band median χ² {vb['median_chi2_in_band']:.2f} (Q scatters {np.sqrt(vb['median_chi2_in_band']):.1f}× its radiometer noise)")
fig, axs = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
axs[0].semilogy(freqs[inb], chi2[inb], ".", ms=2, color="0.4")
axs[0].semilogy(freqs[mask["chi2"]], chi2[mask["chi2"]], ".", ms=3, color="#D55E00", label=f"masked ({mask['chi2'].sum()})")
axs[0].axhline(b4.CHOICES["chi2_factor"] * vb["median_chi2_in_band"], color="k", lw=0.8, ls="--", label="3 × in-band median")
axs[0].set(ylabel="mean (residual/σ_Q)²")
rel = fit_cor["sb"] / fit_cor["b"]
axs[1].semilogy(freqs[inb], np.abs(rel[inb]), ".", ms=2, color="0.4")
axs[1].semilogy(freqs[mask["rel_err"]], np.abs(rel[mask["rel_err"]]), ".", ms=3, color="#D55E00", label=f"masked ({mask['rel_err'].sum()})")
axs[1].axhline(b4.CHOICES["max_rel_err"], color="k", lw=0.8, ls="--", label="5 %")
axs[1].set(ylabel="slope error / slope")
keep = inb & ~mask["chi2"] & ~mask["rel_err"] & ~mask["persistent"]
axs[2].plot(freqs[keep], 1 / fit_cor["b"][keep], ".", ms=2, color="0.4", label="kept")
axs[2].plot(freqs[mask["persistent"]], 1 / fit_cor["b"][mask["persistent"]], ".", ms=3, color="#D55E00",
            label=f"steady line in some state ({mask['persistent'].sum()})")
axs[2].set(ylabel="T_NS, loads corrected (K)", ylim=(300, 800), xlabel="frequency (MHz)")
for ax in axs:
    ax.legend(fontsize=7)
plt.tight_layout(); plt.show()"""))
C.append(md("""**Figure 9.** The channel mask (as `measure` builds it; masked channels in orange). Top: the mean square of the steady cycles' residuals from the line, in units of $Q$'s radiometer noise. Middle: the slope's fractional error. Bottom: $T_\\mathrm{NS}$ with the loads' reflections corrected, kept channels and channels with a steady line in some state."""))

C.append(md("## 11. The noise source's temperature at P"))
C.append(code("""t_non, s_non, t_off = res["t_non_k"], res["t_non_sigma_k"], res["t_off_k"]
ex = t_non - t_off
prev_ex = res0["t_non_k"] - res0["t_off_k"]
prev_s, prev_good = res0["t_non_sigma_k"], res0["good"]
v0001 = noise_source.load("v0001", root=b4.PREV_ROOT)          # as built: reference from GP28 (LNA 3)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, height_ratios=(2, 1))
a0.axhspan(*DATASHEET_K, color=CS[S.hot], alpha=0.12, label=f"ENR 30–35 dB behind the pad ({DATASHEET_K[0]:.0f}–{DATASHEET_K[1]:.0f} K)")
a0.plot(freqs, np.where(good, 1 / fits["steady"]["b"], np.nan), color="0.75", lw=0.7, label="T_NS from Q, no correction (§ 7)")
a0.plot(freqs, np.where(good, 1 / fit_cor["b"], np.nan), color="0.45", lw=0.7, label="T_NS from Q, loads corrected (§ 10)")
a0.plot(freqs, ex, ".", ms=2, color="k", label="excess at P, T_NON − T_off (this run, 4 passes)")
a0.plot(freqs, prev_ex, ".", ms=2, color=CS[S.ref], alpha=0.6, label="excess at P, 2026-10-08 re-measured (GP27 reference)")
a0.set(ylabel="K", ylim=(0, 1000), xlim=(45, 250))
a0.legend(fontsize=7, loc="lower left")
both = good & prev_good
d = np.where(both, ex - prev_ex, np.nan)
a1.errorbar(freqs[both], d[both], yerr=np.hypot(s_non, prev_s)[both], fmt=".", ms=2, lw=0.4, color="k", ecolor="0.75")
a1.axhline(0, color="k", lw=0.5)
a1.set(ylabel="this run − 10-08 (K)", xlabel="frequency (MHz)")
plt.tight_layout(); plt.show()
print(f"T_off (diode board, noise-off dwells of the steady cycles): {t_off:.2f} K ({t_off - K0:.1f} °C); 10-08: {res0['t_off_k']:.2f} K")
rows = []
for lo, hi in b4.BANDS:
    e = b4.band_wmean(ex, s_non, freqs, good, lo, hi); e0 = b4.band_wmean(prev_ex, prev_s, freqs, prev_good, lo, hi)
    e1 = b4.band_wmean(v0001.excess_k, v0001.t_non_sigma_k, freqs, v0001.good, lo, hi)
    k = good & (freqs >= lo) & (freqs < hi)
    rows.append({"band (MHz)": f"{lo:.0f}–{hi:.0f}", "channels": int(k.sum()),
                 "T_NON (K)": round(b4.band_wmean(t_non, s_non, freqs, good, lo, hi), 1), "excess (K)": round(e, 1),
                 "excess per channel (K)": f"{np.nanmin(ex[k]):.0f}–{np.nanmax(ex[k]):.0f}",
                 "ENR behind the pad (dB)": round(float(enr_from_excess(e, 30.0)), 2),
                 "10-08 excess (K)": round(e0, 1), "difference (%)": round(100 * (e / e0 - 1), 2),
                 "v0001 as built (K)": round(e1, 1)})
pd.DataFrame(rows)"""))
C.append(md("""**Figure 10 and Table 5.** The noise-source path's excess at P from all four passes' steady cycles (black; $T_\\mathrm{NON}$ minus the diode board's temperature in the noise-off dwells), with the stages before it (grey: $T_\\mathrm{NS}$ from $Q$ without and with the loads' reflections, both still absorbing the noise source's own mismatch with the bench receiver) and the 2026-10-08 run re-measured here with its GP27 reference (blue). Bottom: this run minus 10-08, per channel unmasked in both, with the two slope errors combined. The table gives inverse-variance weighted means per band, and `noise_source` v0001 as built (its reference from GP28, LNA 3's thermistor), which sits 5–7 K below the re-measurement."""))

C.append(md("## 12. Do the four passes agree?"))
C.append(code("""pass_fit, rows = {}, []
for p in range(1, 5):
    use = steady & (cyc["pass"].to_numpy() == p)
    ft = b4.fit_subset(res, use)
    noff = use                                  # the board's temperature in this pass's noise-off dwells
    t_off_p = float(weighted_mean(cyc.T_board_off.to_numpy()[noff], np.ones(int(noff.sum()))))
    pass_fit[p] = (ft, t_off_p)
    row = {"pass": p, "steady cycles": int(use.sum()), "board T_off (°C)": round(t_off_p - K0, 2),
           "SNAP FPGA (°C)": round(float(np.nanmean(cyc.fpga_c.to_numpy()[use])), 1),
           "LNA 3 (°C)": round(float(np.nanmean(cyc.T_lna3.to_numpy()[use])) - K0, 2)}
    for lo, hi in b4.BANDS:
        e = b4.band_wmean(ft["b"] - t_off_p, ft["sb"], freqs, good, lo, hi)
        row[f"excess {lo:.0f}–{hi:.0f} (K)"] = round(e, 1)
    rows.append(row)
tab = pd.DataFrame(rows).set_index("pass")
edges = np.arange(48, 246, 4.0); cen = 0.5 * (edges[1:] + edges[:-1])
def binned(v, s):
    out = np.full(cen.size, np.nan)
    for j, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
        k = good & (freqs >= e0) & (freqs < e1) & np.isfinite(v) & np.isfinite(s)
        if k.any():
            out[j] = float(weighted_mean(v[k], 1 / s[k] ** 2))
    return out
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, height_ratios=(1.4, 1))
allb = binned(ex, s_non)
for p, (ft, t_off_p) in pass_fit.items():
    bp = binned(ft["b"] - t_off_p, ft["sb"])
    a0.plot(cen, bp, "o-", ms=2, lw=0.8, color=CP[p], label=f"pass {p}")
    a1.plot(cen, 100 * (bp / allb - 1), "o-", ms=2, lw=0.8, color=CP[p])
a0.plot(cen, allb, "k--", lw=0.8, label="all passes")
a0.set(ylabel="excess at P (K)")
a0.legend(fontsize=7, ncol=5)
a1.axhline(0, color="k", lw=0.5)
a1.set(ylabel="pass − all (%)", xlabel="frequency (MHz)")
plt.tight_layout(); plt.show()
tab"""))
C.append(md("""**Figure 11 and Table 6.** The excess at P fitted to each pass's steady cycles alone (good channels, inverse-variance weighted in 4 MHz bins), and each pass relative to the fit to all four, with the conditions of each pass: the diode board's temperature in the noise-off dwells, the SNAP's FPGA temperature and LNA 3's. The passes ran at different times of day; their spread is the drift test the run record asks for (§ 10), and a trend with the board temperature would be the noise source's temperature coefficient over the room's natural range."""))
C.append(code("""qband = b4.band_rows(Q, freqs, good, 108, 200)
fig, ax = plt.subplots(figsize=(7, 3.6))
pl = cyc.phase.str.startswith("plateau").to_numpy() | (cyc.phase.to_numpy() == "ambient")
for p in range(1, 5):
    m = steady & pl & (cyc["pass"].to_numpy() == p)
    ax.plot(dT[m], qband[m], ".", ms=2, color=CP[p], label=f"pass {p}")
ax.set(xlabel="T_hot − T_ref (K)", ylabel="Q, 108–200 MHz mean")
ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
rows = []
for name in ("ambient", "plateau 1", "plateau 2", "plateau 3"):
    for p in range(1, 5):
        m = steady & (cyc.phase.to_numpy() == name) & (cyc["pass"].to_numpy() == p)
        qb = qband[m]
        rows.append({"phase": name, "pass": p, "cycles": int(m.sum()), "T_hot − T_ref (K)": round(float(np.nanmean(dT[m])), 2),
                     "Q (108–200 MHz)": round(float(np.nanmean(qb)), 4),
                     "Q / (T_hot − T_ref) (1/K)": round(float(np.nanmean(qb)) / float(np.nanmean(dT[m])), 6) if name != "ambient" else np.nan})
pd.DataFrame(rows).pivot(index="pass", columns="phase", values="Q / (T_hot − T_ref) (1/K)")"""))
C.append(md("""**Figure 12 and Table 7.** The quick check of the run record's § 10: $Q$ (108–200 MHz mean over unmasked channels) against the temperature difference for the steady ambient and plateau cycles, by pass; and, per plateau and pass, $Q$ per kelvin of temperature difference. The same plateau should give the same value in every pass. (It falls from plateau to plateau because the line has an intercept: $Q$ is not zero at zero temperature difference.)"""))

C.append(md("""## 13. System drift

The hot load measures the noise source; the reference load and the noise source together measure the receiver. Per switch cycle, with $T_\\mathrm{NON}$ fixed to the all-pass fit,
$$P_\\mathrm{ref} = g\\,(M_\\mathrm{ref}T_\\mathrm{ref} + T_r),\\qquad P_\\mathrm{NON} = g\\,(M_\\mathrm{NS}T_\\mathrm{NON} + T_r)$$
give the receiver's gain $g$ and noise temperature $T_r$ in every cycle, heater on or off (`bench_4pass.gain_and_trx`). Drift in $g$ or $T_r$ is the receiver's (or the SNAP's); drift in the noise source itself would show as $g$ and $T_r$ moving together, and is tested separately in § 13.3 from the hot load on the plateaus. The SNAP's FPGA temperature is interpolated to each cycle from its ~20 s log."""))
C.append(code("""g, t_r = b4.gain_and_trx(cp, res, t_non)
tc = cyc.t.to_numpy()
BD = (108, 200)
gb = b4.band_rows(g, freqs, good, *BD)
trb = b4.band_rows(t_r, freqs, good, *BD)
pref = b4.band_rows(cp[S.ref], freqs, good, *BD)
pnoff = b4.band_rows(cp[S.noff], freqs, good, *BD)
fig, axs = plt.subplots(5, 1, figsize=(10, 9), sharex=True, height_ratios=(1.2, 1.2, 1.2, 1, 1.4))
axs[0].plot(loc(tc), 100 * (pref / np.nanmean(pref) - 1), ".", ms=1.5, color=CS[S.ref], label="P_ref (RFANT)")
axs[0].plot(loc(tc), 100 * (pnoff / np.nanmean(pnoff) - 1), ".", ms=1.5, color=CS[S.noff], label="P_NOFF")
axs[0].set(ylabel="power vs run mean (%)")
axs[0].legend(fontsize=7, markerscale=4)
axs[1].plot(loc(tc), 100 * (gb / np.nanmean(gb) - 1), ".", ms=1.5, color="k")
axs[1].set(ylabel="gain g vs mean (%)")
axs[2].plot(loc(tc), trb, ".", ms=1.5, color="#E69F00")
axs[2].set(ylabel="T_r (K)")
axs[3].plot(loc(tc), cyc.fpga_c, ".", ms=1.5, color="0.3")
axs[3].set(ylabel="SNAP FPGA (°C)")
for col, c, lab in (("T_lna3", "#E69F00", "LNA 3"), ("T_ref", CS[S.ref], "reference load"), ("T_board_off", CS[S.non], "diode board")):
    axs[4].plot(loc(tc), cyc[col] - K0, ".", ms=1.5, color=c, label=lab)
axs[4].set(ylabel="°C", xlabel="PDT")
axs[4].legend(fontsize=7, markerscale=4, ncol=3)
axs[4].xaxis.set_major_formatter(HHMM)
for t0 in ph.loc[ph.phase == "ambient", "t0"]:
    for ax in axs:
        ax.axvline(loc([t0])[0], color="k", lw=0.5, ls=":")
plt.tight_layout(); plt.show()
print(f"{BD[0]}–{BD[1]} MHz, unmasked channels, every complete cycle ({len(tc)}):")
print(f"  gain: peak-to-peak {100*(np.nanmax(gb)-np.nanmin(gb))/np.nanmean(gb):.2f} %, rms {100*np.nanstd(gb)/np.nanmean(gb):.2f} %")
print(f"  T_r: mean {np.nanmean(trb):.1f} K, peak-to-peak {np.nanmax(trb)-np.nanmin(trb):.1f} K, rms {np.nanstd(trb):.1f} K")"""))
C.append(md("""**Figure 13.** The system over the ~24 h run, one point per switch cycle (108–200 MHz, unmasked channels; pass starts dotted): the raw powers of the two ambient-temperature states, the receiver's gain and noise temperature from the reference and the noise source, and the temperatures that could drive them (SNAP FPGA, LNA 3, the reference load, the diode board). $T_r$ here is at plane P and includes the bench LNA's noise; it is the lab receiver's, not the field receiver's."""))
C.append(md("""### 13.3 The noise source itself

Per cycle, $T_\\mathrm{NON}$ from that cycle alone, $(y-a)/x$ in the straight line of § 11 with the all-cycle intercept $a$ (`bench_4pass.cycle_tnon`), is independent of the receiver's gain and noise; it is meaningful only where the hot load is well away from the reference, so only the steady cycles of plateaus 2 and 3 are used. A drift in it is the noise source's (or its path's), and its relation to the diode board's temperature is the noise source's temperature coefficient over this run's range."""))
C.append(code("""fit_all = b4.fit_subset(res, steady)
tn_c = b4.cycle_tnon(res, fit_all["a"])
sel = steady & np.isin(cyc.phase.to_numpy(), ["plateau 2", "plateau 3"])
tnb = b4.band_rows(tn_c - cyc.T_board_off.to_numpy()[:, None], freqs, good, *BD)
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.4), width_ratios=(2, 1))
for p in range(1, 5):
    m = sel & (cyc["pass"].to_numpy() == p)
    a0.plot(loc(tc[m]), tnb[m], ".", ms=2, color=CP[p], label=f"pass {p}")
    a1.plot(cyc.T_board_off.to_numpy()[m] - K0, tnb[m], ".", ms=2, color=CP[p])
a0.set(ylabel="excess at P, per cycle (K)", xlabel="PDT")
a0.xaxis.set_major_formatter(HHMM)
a0.legend(fontsize=7, markerscale=4)
a1.set(xlabel="diode board, noise-off dwell (°C)")
rows = []
for name, ls in (("plateau 2", "--"), ("plateau 3", "-")):
    m = sel & (cyc.phase.to_numpy() == name)
    x = cyc.T_board_off.to_numpy()[m] - K0; y = tnb[m]
    okk = np.isfinite(x) & np.isfinite(y)
    sl, ic = np.polyfit(x[okk], y[okk], 1)
    xx = np.array([x[okk].min(), x[okk].max()])
    a1.plot(xx, ic + sl * xx, ls, color="k", lw=0.8, label=f"{name}: {sl:+.1f} K/K")
    rows.append({"phase": name, "cycles": int(okk.sum()), "excess mean (K)": round(float(np.mean(y[okk])), 1),
                 "rms (K)": round(float(np.std(y[okk])), 1), "slope vs board (K/K)": round(sl, 2),
                 "slope (%/K)": round(100 * sl / float(np.mean(y[okk])), 2), "board range (K)": round(float(np.ptp(x[okk])), 2)})
a1.legend(fontsize=7)
plt.tight_layout(); plt.show()
pd.DataFrame(rows)"""))
C.append(md("""**Figure 14 and Table 8.** The noise source's excess at P from each steady cycle of plateaus 2 and 3 alone (108–200 MHz, unmasked channels), over time by pass (left) and against the diode board's temperature in that cycle's noise-off dwell (right), with a straight line fitted within each plateau. In every pass the plateau-3 cycles sit a few kelvin above the plateau-2 cycles, so the line of § 11 is not exactly straight across the hot load's range (candidates: the hot-load thermistor's scale at high temperature, through its 2-wire pull-up; the hot load's reflection, which changes with its temperature, Fig. 7); a slope fitted across both plateaus would mix that with the board. Within a plateau the board spans only a few tenths of a kelvin, so these slopes bound the noise source's temperature coefficient only loosely."""))

C.append(md("""### 13.4 Is the curvature in the 2026-10-08 run too?

The single-cycle estimate of § 13.3, per plateau, in both runs (each with its own all-cycle intercept; plateaus picked by the hot load's temperature, within 3 K of 47, 67 and 87 °C, steady cycles only). If the line were straight, every plateau would give the same excess."""))
C.append(code("""rows = []
for run, r, steady_r in (("2026-10-08", res0, (res0["cycles"].kind == "steady").to_numpy()), ("2026-10-20", res, steady)):
    fit_r = b4.fit_subset(r, steady_r)
    tn_r = b4.cycle_tnon(r, fit_r["a"])
    c = r["cycles"]
    th = c.T_hot.to_numpy() - K0
    base = None
    for tgt in (47, 67, 87):
        m = steady_r & (np.abs(th - tgt) < 3)
        v = b4.band_rows(tn_r[m] - c.T_board_off.to_numpy()[m][:, None], freqs, r["good"], *BD)
        mean = float(np.nanmean(v))
        base = mean if tgt == 47 else base
        rows.append({"run": run, "plateau (°C)": tgt, "cycles": int(m.sum()), "excess, mean over cycles (K)": round(mean, 1),
                     "rms (K)": round(float(np.nanstd(v)), 1), "vs 47 °C (%)": round(100 * (mean / base - 1), 2)})
pd.DataFrame(rows)"""))
C.append(md("""**Table 9.** The noise source's excess at P (108–200 MHz, unmasked channels) from single steady cycles, by plateau, in both bench runs. A trend with the plateau in both runs, of the same size, points to something common to both (the hot-load thermistor's scale at high temperature, or the hot load's reflection, which follows its temperature); a trend in one run only points to that run."""))
C.append(md("## 14. Power around the VNA sweeps"))
C.append(code("""print(f"{len(sweeps)} VNA sweeps; durations {np.round(sweeps[:, 1] - sweeps[:, 0]).astype(int)[:10]} ... s")
bm = (freqs > 108) & (freqs < 200) & good
pw = {}
for st in b4.ORDER:
    meta, sp = D["states"][st]["meta"], D["states"][st]["spec"]
    fl = sw.rfi_flags(sp, freqs)[0]                     # RFI flags only, without the sweep exclusion
    pw[st] = (meta.time_best.to_numpy(), np.nanmean(np.where(fl[:, bm], np.nan, sp[:, bm]), axis=1))
pick = np.linspace(0, len(sweeps) - 1, 4).astype(int)
fig, axs = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
for ax, (t0, t1) in zip(axs, sweeps[pick]):
    for st in b4.ORDER:
        tt, p = pw[st]
        near = (tt > t0 - 120) & (tt < t1 + 120)
        ref = (tt > t0 - 900) & (tt < t1 + 900) & ~near
        ax.plot(tt[near] - t0, 100 * (p[near] / np.nanmean(p[ref]) - 1), "o", ms=3, color=CS[st], label=st)
    ax.axvspan(0, t1 - t0, color="0.85", label="VNA sweep (VNA* states)")
    ax.axvspan(-b4.CHOICES["sweep_guard"][0], 0, color="#F0E442", alpha=0.4, label="excluded before sweep")
    ax.set(title=f"{loc([t0])[0]:%m-%d %H:%M} PDT", xlabel="seconds from sweep start")
axs[0].set_ylabel("band power vs nearby mean (%)")
axs[0].legend(fontsize=6)
plt.tight_layout(); plt.show()"""))
C.append(md("""**Figure 15.** Band power (108–200 MHz, unmasked channels, RFI-flagged samples left out) of every switched integration within ±2 minutes of four VNA sweeps spread over the run, relative to the same state's mean from 2 to 15 minutes away. The yellow span is the 20 s before each sweep that `measure` excludes (the VNA service disturbs every state there in 2026-10-08); a disturbance outside it would need a wider guard."""))

C.append(md("## 15. In the field"))
C.append(code("""from eigsep_data.calibration.build_tcal import noise_source_tns
import tempfile
z = np.load(b4.PREV_ROOT / "derived/tcal/v0004/solutions.npz")
s11_field = {d: (z[f"s11_{d}_t"], z[f"s11_{d}"], z[f"s11_{d}_epoch"]) for d in ("rec", "amb")}
# This run's excess as a noise_source product (written to a temporary tree), so it gets the product's own smooth fit
with tempfile.TemporaryDirectory() as tmp:
    noise_source.write(tmp, "v9999", freqs=freqs, t_non_k=t_non, t_non_sigma_k=s_non, t_off_k=t_off, good=good,
                       gamma_ns=res["gamma_ns"], provenance={"note": "bench_4pass.ipynb, not a product"})
    ns4 = noise_source.load("v9999", root=tmp)
t_ns4, chk4 = noise_source_tns(ns4, freqs, s11_field, z["therm_k"], (50.0, 200.0))
t_ns_v4 = np.asarray(z["t_ns_k"], float)
fb = (freqs >= 50) & (freqs <= 200)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True, height_ratios=(2, 1))
a0.plot(freqs[fb], t_ns_v4[fb], color=CS[S.ref], label="tcal v0004 (from 10-08's v0001: reference from LNA 3)")
a0.plot(freqs[fb], t_ns4[fb], color="k", label="from this run")
a0.set(ylabel="field T_NS (K)")
a0.legend(fontsize=7)
a1.plot(freqs[fb], 100 * (t_ns4[fb] / t_ns_v4[fb] - 1), color="k")
a1.axhline(0, color="k", lw=0.5)
a1.set(ylabel="this run vs v0004 (%)", xlabel="frequency (MHz)")
plt.tight_layout(); plt.show()
print(f"field T_NS from this run: {np.nanmin(t_ns4[fb]):.0f}–{np.nanmax(t_ns4[fb]):.0f} K over 50–200 MHz; "
      f"change from tcal v0004: {np.nanmin(100*(t_ns4[fb]/t_ns_v4[fb]-1)):+.2f} to {np.nanmax(100*(t_ns4[fb]/t_ns_v4[fb]-1)):+.2f} %")"""))
C.append(md("""**Figure 16.** What this run would change in the field calibration: the noise-source temperature as the field receiver takes it, built exactly as `tcal` v0004 builds its own (`eigsep_data.calibration.build_tcal.noise_source_tns`: the excess's smooth fit, the lab $\\Gamma_\\mathrm{NS}$, the campaign-mean field $\\Gamma_\\mathrm{rec}$ and $\\Gamma_\\mathrm{amb}$, the mean load thermistor), from this run and from 10-08 (v0004's). The calibrated temperature's $T^*-T_\\mathrm{load}$ scales by the same ratio."""))

# The summary is written into the notebook by hand after each run, from that run's outputs.
C.append(md("## Summary\n\n*(written after the run.)*"))

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}})
nbf.write(nb, out)
print("wrote", out, len(C), "cells")
