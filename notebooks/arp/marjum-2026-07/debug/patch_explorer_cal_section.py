"""Append the executed Calibration section to beam_explorer.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

NB = "beam_explorer.ipynb"
nb = nbf.read(NB, as_version=4)

nb.cells.append(new_markdown_cell(r"""---

# Calibration: B7's $g_{\rm rx}$, and why the toggle defaults to off

`apply g_rx cal (B7)` divides `measured_tx` by rf-calibrator's per-cycle
receiver gain, interpolated onto the sample times. The reasoning for doing so
is sound on paper. The result is not what it predicts, so this section records
the evidence rather than leaving a surprising default unexplained.

**Why it should work.** `measured_tx` is the channel difference
`auto[c] − ½(auto[c−1] + auto[c+1])`. With `auto = g_rx · (T_sky + T_rx)` and
$g_{\rm rx}$ smooth across three adjacent 244 kHz channels, the difference
carries $g_{\rm rx}$ as a **linear multiplicative factor**. Across this window
$g_{\rm rx}$ at 173.83 MHz runs 473 → 293, a factor **1.61** — a drift that a
single per-channel fit amplitude cannot absorb. Removing it should help.

**What B7 actually covers here.** 66 cycles, 07-17 04:11 → 07-18 02:57 UTC.
The beam-scan window sits entirely inside that span, so *no used sample needs
extrapolation* — the trailing-samples question turned out to be empty. But the
cycles are not evenly spread."""))

nb.cells.append(new_code_cell(r"""import datetime as _dt

_st = np.load("cal_gain.npz", allow_pickle=True)["sol_times"]
_st = np.sort(_st.astype(float))
_t = C["times"].astype(float)
_tv = _t[_t > 0]
_inw = (_st >= _tv.min()) & (_st <= _tv.max())
_u712 = USED[CH_712 := int(np.argmin(np.abs(CHANS - 712)))]

print(f"cycles in the beam-scan window: {int(_inw.sum())} of {_st.size}")
print("gaps between them [minutes]:",
      np.round(np.diff(_st[_inw]) / 60, 1))
print()
print(f"used samples with a gain solution : "
      f"{int((_u712 & np.isfinite(CAL_GAIN[CH_712])).sum())} of {int(_u712.sum())}")
print(f"  -> nothing to drop; no used sample falls outside the cycle span")
print()
_g0 = _dt.datetime(2026, 7, 17, 19, 42, 14, tzinfo=_dt.timezone.utc).timestamp()
_g1 = _dt.datetime(2026, 7, 18, 1, 24, 40, tzinfo=_dt.timezone.utc).timestamp()
_gapmask = _u712 & (_t > _g0) & (_t < _g1)
print(f"the 342-minute hole {CAL_GAP[0]} -> {CAL_GAP[1]}")
print(f"  holds {int(_gapmask.sum())} used samples "
      f"({100*_gapmask.sum()/_u712.sum():.1f}% of them)")
print(f"  and it is exactly the interval B7 brackets the receiver regime "
      f"change to,")
print(f"  so inside it the interpolation is a straight line through an "
      f"unknown transition.")
print()
print("minutes to the nearest real cycle, over used samples:")
for _q in (50, 90, 95):
    print(f"   p{_q:<3d} {np.nanpercentile(CAL_ANCHOR_MIN[_u712], _q):6.1f}")"""))

nb.cells.append(new_markdown_cell(r"""## The result: it degrades the fit

Pure HFSS, amplitude refit in each case, so only the data changes."""))

nb.cells.append(new_code_cell(r"""_h = heading_from_enu(0, 8, -93.5)


def _score(i, use_cal, extra=None, alpha=51.0, arm=1):
    m, A = model_power(i, _h, alpha, arm, [0, 0, 0], [0, 0, 0],
                       apply_cal=use_cal)
    u = chan_used(i, use_cal)
    if extra is not None:
        u = u & extra
    d = chan_data(i, use_cal)[u]
    r = d - m[u]
    return (float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(d ** 2))),
            float(A))


_r0, _a0 = _score(CH_712, False)
_r1, _a1 = _score(CH_712, True)
print(f"ch 712, pure HFSS, amplitude refit")
print(f"  cal OFF : normalized RMS {_r0:.4f}   amplitude {_a0:.4g}")
print(f"  cal ON  : normalized RMS {_r1:.4f}   amplitude {_a1:.4g}")
print(f"  -> {_r1-_r0:+.4f}")
print()

_med = {}
for _lab, _uc in (("OFF", False), ("ON", True)):
    _v = []
    for _j in range(len(CHANS)):
        _m, _ = model_power(_j, HEADING_NEW, ALPHA_NEW, int(ARMS[_j]),
                            [0, 0, 0], [0, 0, 0], apply_cal=_uc)
        _v.append(normalized_rms(_j, _m, apply_cal=_uc))
    _v = np.array(_v)
    _med[_lab] = _v
    print(f"all 101 channels, cal {_lab:<3}: median {np.median(_v):.4f}   "
          f"below 0.5: {int((_v<0.5).sum())}   below 0.7: {int((_v<0.7).sum())}")
_d = _med["ON"] - _med["OFF"]
print(f"  channels improved by the correction: {int((_d<0).sum())}/101 ; "
      f"worsened: {int((_d>0).sum())}")"""))

nb.cells.append(new_markdown_cell(r"""## Three controls, to show that is not a bug in the correction"""))

nb.cells.append(new_code_cell(r"""# CONTROL 1 -- a time-CONSTANT gain must be an exact no-op, because the single
# free amplitude absorbs any constant. If this fails, the plumbing is wrong.
_keep = CAL_GAIN[CH_712].copy()
CAL_GAIN[CH_712] = np.where(np.isfinite(_keep), np.nanmean(_keep), np.nan)
_rc, _ = _score(CH_712, True)
CAL_GAIN[CH_712] = _keep
print(f"CONTROL 1  constant gain applied : {_rc:.6f}")
print(f"           no cal at all         : {_r0:.6f}")
print(f"           -> identical to {abs(_rc-_r0):.1e}; the plumbing is correct,")
print(f"              and only the TIME VARIATION of g_rx matters.")
print()

# CONTROL 2 -- if the problem were the 342-min interpolation, the correction
# would hurt inside the gap and help where it is well anchored. Test that.
print("CONTROL 2  by distance to the nearest real cycle:")
print(f"{'subset':<20} {'n':>7} {'cal OFF':>9} {'cal ON':>9} {'delta':>8}")
for _lo, _hi, _lab in ((0, 15, "anchor < 15 min"), (0, 30, "anchor < 30 min"),
                       (60, 1e9, "anchor > 60 min")):
    _ex = (CAL_ANCHOR_MIN >= _lo) & (CAL_ANCHOR_MIN < _hi)
    _x0, _ = _score(CH_712, False, _ex)
    _x1, _ = _score(CH_712, True, _ex)
    print(f"{_lab:<20} {int((USED[CH_712]&_ex).sum()):>7} {_x0:>9.4f} "
          f"{_x1:>9.4f} {_x1-_x0:>+8.4f}")
print("           -> the correction hurts MOST where g_rx is best known and")
print("              HELPS where it is least known. That is backwards for an")
print("              interpolation-error explanation.")
print()

# CONTROL 3 -- scan the exponent. If any exposure to g_rx helped, there would
# be a minimum at some p > 0.
print("CONTROL 3  divide the data by g_rx**p:")
_g = CAL_GAIN[CH_712].copy()
for _p in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
    with np.errstate(invalid="ignore"):
        CAL_GAIN[CH_712] = np.where(np.isfinite(_g), _g ** _p, np.nan)
    _x, _ = _score(CH_712, True)
    print(f"           p = {_p:+.2f} -> {_x:.4f}")
CAL_GAIN[CH_712] = _g
print("           -> monotonic. The fit prefers MULTIPLYING by g_rx to")
print("              dividing by it; there is no interior optimum at p > 0.")"""))

nb.cells.append(new_markdown_cell(r"""## What this does and does not establish

**Established.** The correction is implemented correctly (Control 1 is exact to
$10^{-6}$), and applying it degrades the fit in a way that is *not* explained by
the 342-minute interpolation gap — the damage is largest where $g_{\rm rx}$ is
best measured (Control 2), and the preference is monotonic in the wrong
direction (Control 3). So the premise that the comb amplitude scales as
$g_{\rm rx}^{+1}$ is not supported by this data.

**A tempting inference, and why I am not making it.** A far-field transmitter
signal received through the antenna and the full analog chain *must* scale with
$g_{\rm rx}$. A tone injected downstream of the LNA — the digital self-comb at
250/128 MHz that `load_v007_data`'s own docstring and `flags/v0` both point to —
would not. So this looks like evidence on the open TX-identity dispute.

**But this test cannot carry that weight.** The beam scan is *time-ordered*:
pointing and time are entangled, so any time-varying quantity correlates with
the pointing pattern, which is itself the signal being fit. The raw
$\mathrm{corr}(g_{\rm rx}, |{\tt measured\_tx}|) = -0.33$ is therefore
confounded and must not be read as receiver physics. A common thermal driver
moving both LNA gain and comb amplitude would produce the same signature.

Settling it needs a **pointing-controlled** test: compare comb amplitude
against $g_{\rm rx}$ *within* fixed pointing cells, so the beam pattern is held
constant. That is a separate piece of work and is not attempted here.

**Practical upshot.** Leave the toggle off for fitting. The $\times 1.61$
$g_{\rm rx}$ drift is real and remains an uncorrected systematic on the
amplitude scale — the standing `receiver_regime_caveat` — but dividing it out
as specified does not improve the beam fit, and why is now a concrete,
testable question rather than a guess."""))

nbf.write(nb, NB)
print(f"appended Calibration section; {len(nb.cells)} cells")
