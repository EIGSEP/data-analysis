"""Assemble the B7 g_rx calibration-toggle checkpoint notebook."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

C = []
M = lambda s: C.append(new_markdown_cell(s))
K = lambda s: C.append(new_code_cell(s))

M(r"""# Applying B7's receiver gain to the beam scan — and why it makes the fit worse

**Milestone.** Aaron asked for rf-calibrator's per-cycle $g_{\rm rx}$ gain
correction to be available in `beam_explorer.ipynb` as an interactive toggle,
not baked in. The toggle is implemented (`apply g_rx cal (B7)`). This checkpoint
records what it does, because the answer is not what the physics predicted and
the default therefore needed justifying.

**Question.** Does dividing `measured_tx` by $g_{\rm rx}(\nu,t)$ improve the
beam fit?

**Answer.** No — it degrades it, substantially and consistently, and *not*
because of interpolation error. Three controls below establish that the
correction is implemented correctly and that the premise itself fails: the
comb amplitude in this window does not scale with receiver gain.

**Scope note.** This dispatch was calibration only. The azimuth-zero /
elevation-convention half is explicitly **not** touched — geometer's
hardware-derived az-zero still tests ~3× worse than the free-fit optimum by
both geometer's terrain MAD and my own beam-RMS check, and that discrepancy is
back with geometer.

**Inputs.** `abscal/trx_phaseC.npz` (B7: 66 switching cycles, per-cycle
$g_{\rm rx}(\nu,t)$ and $T_{\rm rx}(\nu,t)$, Phase C),
`beam_explorer_cache.npz` (rebuilt 2026-09-17 with a `times` axis so the
per-cycle solutions can be interpolated onto samples), `cal_gain.npz` (built by
`build_cal_gain.py`).

**Units.** With the toggle off the measured quantity is raw correlator
accumulator counts. With it on the quantity is counts / $g_{\rm rx}$, so fit
amplitudes change scale by ~$10^{-3}$; that is a unit change, not a result.""")

K(r"""import datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "font.size": 9})

import explorer_model as E

I = int(np.argmin(np.abs(E.CHANS - 712)))
H = E.heading_from_enu(0, 8, -93.5)
print(f"cache: {E.CACHE.name}")
print(f"B7 calibration available: {E.CAL_AVAILABLE}")
print(f"unsolved gap: {E.CAL_GAP[0]} -> {E.CAL_GAP[1]}")
print(f"channels {E.CHANS.size}   samples {E.AZ.size}")""")

M(r"""## 1. Why this should have worked

`measured_tx` is the channel difference
`auto[c] − ½(auto[c−1] + auto[c+1])`. With `auto = g_rx · (T_sky + T_rx)` and
$g_{\rm rx}$ smooth across three adjacent 244 kHz channels, the difference
carries $g_{\rm rx}$ as a **linear multiplicative factor** — the differencing
removes the smooth bandpass, not the gain. So a drift in $g_{\rm rx}$
multiplies the data by a time-varying factor, and the model has exactly one
free amplitude per channel, which cannot absorb it.

That drift is not small.""")

K(r"""u = E.USED[I]
g = E.CAL_GAIN[I]
print(f"ch 712 ({E.FREQS[I]:.2f} MHz), over used samples:")
print(f"  g_rx  {np.nanmin(g[u]):.1f} .. {np.nanmax(g[u]):.1f}   "
      f"ratio {np.nanmax(g[u])/np.nanmin(g[u]):.2f}")

ratios = []
for j in range(len(E.CHANS)):
    gj = E.CAL_GAIN[j][E.USED[j]]
    if np.isfinite(gj).any():
        ratios.append(np.nanmax(gj) / np.nanmin(gj))
ratios = np.array(ratios)
print(f"\nacross all {ratios.size} channels, the within-window g_rx drift ratio:")
print(f"  median {np.median(ratios):.2f}   range "
      f"{ratios.min():.2f} .. {ratios.max():.2f}")
print("\nThis is the standing `receiver_regime_caveat` in beam_fits_v2: a real,")
print("uncorrected multiplicative systematic on the amplitude scale.")""")

M(r"""## 2. What B7 covers here, and the one hole in it""")

K(r"""cal = np.load("cal_gain.npz", allow_pickle=True)
st = np.sort(cal["sol_times"].astype(float))
t = E.TIMES
tv = t[t > 0]
inw = (st >= tv.min()) & (st <= tv.max())

print(f"B7 cycles: {st.size} total, spanning "
      f"{dt.datetime.utcfromtimestamp(st.min())} -> "
      f"{dt.datetime.utcfromtimestamp(st.max())} UTC")
print(f"beam-scan window: {dt.datetime.utcfromtimestamp(tv.min())} -> "
      f"{dt.datetime.utcfromtimestamp(tv.max())} UTC")
print(f"  -> the window sits ENTIRELY inside the calibrated span")
print(f"\ncycles inside the beam-scan window: {int(inw.sum())}")
print("gaps between them [minutes]:", np.round(np.diff(st[inw]) / 60, 1))

print(f"\nused samples with a gain solution: "
      f"{int(E.cal_used(I).sum())} of {int(u.sum())}")
print("  -> nothing is dropped. The 'trailing samples past the last cycle'")
print("     question turned out to be empty: no USED sample falls outside.")

g0 = dt.datetime(2026, 7, 17, 19, 42, 14, tzinfo=dt.timezone.utc).timestamp()
g1 = dt.datetime(2026, 7, 18, 1, 24, 40, tzinfo=dt.timezone.utc).timestamp()
gap = u & (t > g0) & (t < g1)
print(f"\nthe 342-minute hole {E.CAL_GAP[0]} -> {E.CAL_GAP[1]}:")
print(f"  holds {int(gap.sum())} used samples "
      f"({100*gap.sum()/u.sum():.1f}% of them)")
print("  and it is exactly the interval B7 brackets the receiver regime change")
print("  to, so inside it the interpolation is a straight line through an")
print("  unknown transition -- not a measurement.")
print("\nminutes from a used sample to the nearest real cycle:")
for q in (50, 75, 90, 95):
    print(f"   p{q:<3d} {np.nanpercentile(E.CAL_ANCHOR_MIN[u], q):6.1f}")""")

K(r"""fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))

ax = axes[0]
gg = E.CAL_GAIN[I]
ok = u & np.isfinite(gg)
hrs = (t - tv.min()) / 3600.0
ax.plot(hrs[ok], gg[ok], ",", color="#31688e", alpha=0.5)
ax.plot((st[inw] - tv.min()) / 3600.0,
        cal["gain_cycles"][:, I][np.argsort(cal["sol_times"])][inw],
        "o", ms=5, color="#c0392b", label="real cycles")
ax.axvspan((g0 - tv.min()) / 3600.0, (g1 - tv.min()) / 3600.0,
           color="0.85", zorder=0, label="342-min hole")
ax.set_xlabel("hours since scan start")
ax.set_ylabel(r"$g_{\rm rx}$ [counts/s/K]")
ax.set_title("interpolated gain, ch 712", fontsize=9)
ax.legend(fontsize=7, frameon=False)

ax = axes[1]
ax.hist(E.CAL_ANCHOR_MIN[u], bins=40, color="#31688e")
ax.set_xlabel("minutes to nearest real cycle")
ax.set_ylabel("used samples")
ax.set_title("how well anchored the\ncorrection is", fontsize=9)

ax = axes[2]
ax.hist(ratios, bins=25, color="#31688e")
ax.set_xlabel(r"within-window $g_{\rm rx}$ drift ratio")
ax.set_ylabel("channels")
ax.set_title("the drift is large on\nevery channel", fontsize=9)
fig.tight_layout(); plt.show()""")

M(r"""## 3. The result: it degrades the fit

Pure HFSS (shape terms zero), amplitude refit in every case, so the only thing
changing is the data.""")

K(r"""def score(i, use_cal, extra=None, alpha=51.0, arm=1, heading=None):
    m, A = E.model_power(i, heading if heading is not None else H, alpha, arm,
                         [0, 0, 0], [0, 0, 0], apply_cal=use_cal)
    uu = E.cal_used(i) if use_cal else E.USED[i]
    if extra is not None:
        uu = uu & extra
    d = E.data(i, apply_cal=use_cal)[uu]
    r = d - m[uu]
    return (float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(d ** 2))),
            float(A), int(uu.sum()))


r0, a0, n0 = score(I, False)
r1, a1, n1 = score(I, True)
print("ch 712, pure HFSS, amplitude refit")
print(f"{'':12} {'normRMS':>9} {'amplitude':>13} {'n':>7}")
print(f"{'cal OFF':<12} {r0:>9.4f} {a0:>13.4g} {n0:>7d}")
print(f"{'cal ON':<12} {r1:>9.4f} {a1:>13.4g} {n1:>7d}")
print(f"{'delta':<12} {r1-r0:>+9.4f}")

res = {}
for lab, uc in (("OFF", False), ("ON", True)):
    v = []
    for j in range(len(E.CHANS)):
        m, _ = E.model_power(j, E.C["heading_new"], float(E.C["alpha_new"]),
                             int(E.ARMS[j]), [0, 0, 0], [0, 0, 0],
                             apply_cal=uc)
        v.append(E.normalized_rms(j, m, apply_cal=uc))
    res[lab] = np.array(v)
    print(f"\nall 101 channels, cal {lab:<3}: median {np.median(res[lab]):.4f}"
          f"   below 0.5: {int((res[lab]<0.5).sum())}"
          f"   below 0.7: {int((res[lab]<0.7).sum())}")
d = res["ON"] - res["OFF"]
print(f"\nchannels improved by the correction: {int((d<0).sum())}/101 ; "
      f"worsened: {int((d>0).sum())}")""")

K(r"""fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax = axes[0]
ax.scatter(E.FREQS, res["OFF"], s=14, color="#31688e", label="cal OFF")
ax.scatter(E.FREQS, res["ON"], s=14, color="#c0392b", label="cal ON")
ax.set_xlabel("frequency [MHz]"); ax.set_ylabel("normalized RMS")
ax.set_title("per-channel, before and after", fontsize=9)
ax.legend(fontsize=7, frameon=False); ax.set_ylim(0, 1.05)

ax = axes[1]
ax.scatter(ratios, d, s=14, color="#31688e")
ax.axhline(0, color="0.5", ls=":", lw=1)
ax.set_xlabel(r"within-window $g_{\rm rx}$ drift ratio")
ax.set_ylabel("change in normalized RMS")
ax.set_title("more drift 'corrected'\n= more damage", fontsize=9)
fig.tight_layout(); plt.show()
print(f"corr(drift ratio, RMS change) = "
      f"{np.corrcoef(ratios, d)[0,1]:+.3f}")""")

M(r"""## 4. Three controls

The obvious objection is that the correction is simply implemented wrong, or
that the 342-minute interpolation is doing the damage. Both are testable.""")

K(r"""# CONTROL 1 -- a time-CONSTANT gain must be an exact no-op, because the one
# free amplitude absorbs any constant factor. If this fails, the plumbing is
# broken and nothing else here means anything.
keep = E.CAL_GAIN[I].copy()
E.CAL_GAIN[I] = np.where(np.isfinite(keep), np.nanmean(keep), np.nan)
rc, _, _ = score(I, True)
E.CAL_GAIN[I] = keep
print("CONTROL 1 -- constant gain must be a no-op")
print(f"   constant gain applied : {rc:.6f}")
print(f"   no correction at all  : {r0:.6f}")
print(f"   difference            : {abs(rc-r0):.1e}")
print("   -> exact. The division, the masking and the amplitude refit are")
print("      correct, and only the TIME VARIATION of g_rx can matter.")""")

K(r"""# CONTROL 2 -- if the 342-min interpolation were the problem, the correction
# would hurt inside the hole and help where it is well anchored.
print("CONTROL 2 -- by distance to the nearest real cycle")
print(f"{'subset':<20} {'n':>7} {'cal OFF':>9} {'cal ON':>9} {'delta':>9}")
for lo, hi, lab in ((0, 15, "anchor < 15 min"), (0, 30, "anchor < 30 min"),
                    (60, 1e9, "anchor > 60 min")):
    ex = (E.CAL_ANCHOR_MIN >= lo) & (E.CAL_ANCHOR_MIN < hi)
    x0, _, n = score(I, False, ex)
    x1, _, _ = score(I, True, ex)
    print(f"{lab:<20} {n:>7d} {x0:>9.4f} {x1:>9.4f} {x1-x0:>+9.4f}")
print("   -> the correction hurts MOST where g_rx is best known, and HELPS")
print("      where it is least known. That is backwards for an")
print("      interpolation-error explanation.")""")

K(r"""# CONTROL 3 -- scan the exponent. If any exposure to g_rx helped, there would
# be an interior optimum at some p > 0.
print("CONTROL 3 -- divide the data by g_rx**p")
gk = E.CAL_GAIN[I].copy()
print(f"{'p':>8} {'normRMS':>10}")
for p in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0):
    with np.errstate(invalid="ignore"):
        E.CAL_GAIN[I] = np.where(np.isfinite(gk), gk ** p, np.nan)
    x, _, _ = score(I, True)
    print(f"{p:>+8.2f} {x:>10.4f}")
E.CAL_GAIN[I] = gk
print("   -> monotonic across the whole range. The fit prefers MULTIPLYING by")
print("      g_rx (p = -1, RMS 0.347) to leaving it alone (p = 0, 0.391) to")
print("      dividing by it (p = +1, 0.486). There is no interior optimum at")
print("      p > 0, which is what a genuine gain correction would produce.")""")

M(r"""## 5. What this establishes, and what it does not

**Established.**

- The correction is implemented correctly. Control 1 is exact to $10^{-16}$.
- Applying it degrades the fit: ch 712 pure-HFSS **0.3914 → 0.4855**; fleet
  median **0.5176 → 0.5873**; **69 of 101** channels worse; and the damage
  scales with how much $g_{\rm rx}$ drifts on that channel.
- The 342-minute interpolation hole is **not** the cause. Control 2 shows the
  damage is largest on the best-anchored samples and the correction *helps* on
  the worst-anchored ones — the opposite of interpolation error.
- Control 3 is monotonic with no interior optimum, so the data carry no
  positive exposure to $g_{\rm rx}$ at all.

So the premise — that the beam-scan comb amplitude scales as
$g_{\rm rx}^{+1}$ — is not supported.

**A tempting inference, which I am deliberately not making.** A far-field
transmitter signal received through the antenna and the full analog chain
*must* scale with $g_{\rm rx}$. A tone injected downstream of the LNA — the
digital self-comb at 250/128 MHz that `load_v007_data`'s own docstring and
`flags/v0` both point to — would not. That makes this look like evidence on the
open TX-identity dispute.

**This test cannot carry that weight, for one specific reason.** The beam scan
is **time-ordered**: pointing and time are entangled, so *any* time-varying
quantity correlates with the pointing pattern — and the pointing pattern is the
signal being fit. The raw
$\mathrm{corr}(g_{\rm rx}, |{\tt measured\_tx}|) = -0.33$ is therefore
confounded and must not be read as receiver physics. A common thermal driver
moving both LNA gain and radiated comb amplitude would produce the same
signature.

**Deferred findings**

- Settling the above needs a **pointing-controlled** test: regress comb
  amplitude against $g_{\rm rx}$ *within* fixed (az, el) cells, so the beam
  pattern is held constant and only the time variation is free. Not attempted
  here; it is a separate piece of work with a real bearing on TX identity.
- $T_{\rm rx}$ from B7 is loaded into `cal_gain.npz` (`trx_cycles`) but not
  used. A $T_{\rm rx}$-based correction is a different model — additive, not
  multiplicative — and was not part of this dispatch.
- The $\times 1.61$ $g_{\rm rx}$ drift remains a real, uncorrected systematic
  on the absolute amplitude scale. Nothing here fixes it; it establishes that
  dividing it out does not help the *shape* fit.

## Decision requested

1. **Keep the toggle defaulting to OFF.** Applying the correction as specified
   makes the fit worse on 69 of 101 channels, and the controls show that is not
   a bug. I recommend off, with the toggle there for inspection.
2. **Authorize (or decline) the pointing-controlled $g_{\rm rx}$ regression.**
   It is the test that would turn this anomaly into evidence on TX identity,
   which is the largest open question on this dataset.
3. Whether to hand this to `rf-calibrator` — the finding is about how their
   product behaves against the beam data, and they may have an explanation
   (e.g. a known injection point for the comb) that closes it immediately.

**STOPPED AT REVIEW GATE — awaiting Aaron's approval.**""")

nb = new_notebook(cells=C, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, "beam_cal_toggle_checkpoint.ipynb")
print(f"wrote beam_cal_toggle_checkpoint.ipynb ({len(C)} cells)")
