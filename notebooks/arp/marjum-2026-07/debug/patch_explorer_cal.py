"""Add B7's per-cycle g_rx gain correction to beam_explorer.ipynb as a toggle.

Aaron's requirement: a checkbox he can flip interactively, not baked in --
same pattern as the pure-HFSS lock.

Default is OFF. That is a deliberate choice, not laziness: applying the
correction *degrades* the fit (ch712 0.391 -> 0.486 pure-HFSS; fleet median
0.518 -> 0.587; 69 of 101 channels worse), and the degradation is largest on
exactly the samples where g_rx is best determined. So the physical premise --
that the comb amplitude scales with receiver gain -- is not supported by the
data, and the correction should not be on unless someone is deliberately
looking at it. See the "Calibration" section the patch adds for the evidence.
"""
import nbformat as nbf

NB = "beam_explorer.ipynb"
nb = nbf.read(NB, as_version=4)


def find(pred):
    for j, c in enumerate(nb.cells):
        if pred("".join(c.source)):
            return j
    raise SystemExit("cell not found")


# ---- controls: add the checkbox --------------------------------------------
j = find(lambda s: "purehfss_w = W.Checkbox" in s)
s = "".join(nb.cells[j].source)
anchor = 'purehfss_w = W.Checkbox(value=True, description="pure HFSS (lock shape)")'
assert anchor in s
s = s.replace(anchor, anchor + '''
# B7's per-cycle receiver gain. ON divides measured_tx by g_rx(nu, t),
# interpolated onto the sample times, and drops samples with no solution.
# Default OFF because applying it makes the fit worse -- see the Calibration
# section below. The toggle exists so both can be looked at.
cal_w = W.Checkbox(value=False, description="apply g_rx cal (B7)",
                   disabled=not E_CAL_AVAILABLE)''')
nb.cells[j].source = s

# ---- model cell: expose the cal arrays under the notebook's own names ------
j = find(lambda s: "def heading_from_enu" in s)
s = "".join(nb.cells[j].source)
s = s.replace('print("model ready")', '''# --- B7 receiver-gain calibration -------------------------------------------
# measured_tx is a channel difference, so with auto = g_rx * (T_sky + T_rx)
# and g_rx smooth over three adjacent 244 kHz channels it carries g_rx as a
# linear multiplicative factor. Dividing it out *should* remove a drift the
# single fit amplitude cannot absorb (x1.61 at 173.83 MHz here). Measured, it
# does the opposite -- see the Calibration section.
import os.path as _osp

E_CAL_AVAILABLE = _osp.exists("cal_gain.npz") and "times" in C
CAL_GAIN = None
CAL_ANCHOR_MIN = None
CAL_GAP = (None, None)
if E_CAL_AVAILABLE:
    _cal = np.load("cal_gain.npz", allow_pickle=True)
    TIMES = C["times"].astype(float)
    _st = _cal["sol_times"].astype(float)
    _gc = _cal["gain_cycles"].astype(float)
    CAL_GAIN = np.full((len(CHANS), TIMES.size), np.nan)
    _tok = TIMES > 0
    for _j in range(len(CHANS)):
        _ok = np.isfinite(_gc[:, _j]) & (_gc[:, _j] > 0)
        if _ok.sum() < 2:
            continue
        _s, _g = _st[_ok], _gc[_ok, _j]
        _in = _tok & (TIMES >= _s[0]) & (TIMES <= _s[-1])
        CAL_GAIN[_j, _in] = np.interp(TIMES[_in], _s, _g)
    CAL_ANCHOR_MIN = np.full(TIMES.size, np.nan)
    CAL_ANCHOR_MIN[_tok] = np.min(
        np.abs(TIMES[_tok][:, None] - _st[None, :]), axis=1) / 60.0
    CAL_GAP = (str(_cal["gap_start_utc"]), str(_cal["gap_end_utc"]))
    print(f"B7 gain loaded: {_st.size} cycles; "
          f"unsolved gap {CAL_GAP[0]} -> {CAL_GAP[1]}")
else:
    print("cal_gain.npz not found -- the g_rx toggle will be disabled")


def chan_data(i, apply_cal):
    """Measured quantity: raw accumulator counts, or counts / g_rx."""
    d = Y[i].astype(float)
    if not apply_cal or not E_CAL_AVAILABLE:
        return d
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / CAL_GAIN[i]


def chan_used(i, apply_cal):
    """USED, restricted to samples with a gain solution when cal is on."""
    if not apply_cal or not E_CAL_AVAILABLE:
        return USED[i]
    return USED[i] & np.isfinite(CAL_GAIN[i])


print("model ready")''')

# model_power / normalized_rms must honour it
s = s.replace('''def model_power(ch_index, heading, alpha_deg, arm, shape_re, shape_im,
                gain=None, az_off=0.0, el_off=0.0):''',
              '''def model_power(ch_index, heading, alpha_deg, arm, shape_re, shape_im,
                gain=None, az_off=0.0, el_off=0.0, apply_cal=False):''')
s = s.replace('''    u = USED[ch_index]
    d = Y[ch_index].astype(float)
    if gain is None:''',
              '''    u = chan_used(ch_index, apply_cal)
    d = chan_data(ch_index, apply_cal)
    if gain is None:''')
s = s.replace('''def normalized_rms(ch_index, model, el_cut=180.0):
    u = USED[ch_index] & (np.abs(EL) <= el_cut)
    d = Y[ch_index].astype(float)[u]''',
              '''def normalized_rms(ch_index, model, el_cut=180.0, apply_cal=False):
    u = chan_used(ch_index, apply_cal) & (np.abs(EL) <= el_cut)
    d = chan_data(ch_index, apply_cal)[u]''')
nb.cells[j].source = s

# ---- update(): thread the toggle through -----------------------------------
j = find(lambda s: "def update(**kw):" in s)
s = "".join(nb.cells[j].source)
s = s.replace("    gain = None if autog_w.value else gain_w.value\n    locked = purehfss_w.value",
              "    gain = None if autog_w.value else gain_w.value\n    use_cal = bool(cal_w.value) and E_CAL_AVAILABLE\n    locked = purehfss_w.value")
s = s.replace("""    m, A = model_power(i, heading, alpha_w.value, arm, s_re, s_im,
                       gain=gain, az_off=azoff_w.value, el_off=eloff_w.value)""",
              """    m, A = model_power(i, heading, alpha_w.value, arm, s_re, s_im,
                       gain=gain, az_off=azoff_w.value, el_off=eloff_w.value,
                       apply_cal=use_cal)""")
s = s.replace("    rms = normalized_rms(i, m, el_cut=cut)\n    rms_all = normalized_rms(i, m, el_cut=180.0)",
              "    rms = normalized_rms(i, m, el_cut=cut, apply_cal=use_cal)\n"
              "    rms_all = normalized_rms(i, m, el_cut=180.0, apply_cal=use_cal)")
s = s.replace("    u = USED[i] & (np.abs(EL) <= cut)\n    d = Y[i].astype(float)",
              "    u = chan_used(i, use_cal) & (np.abs(EL) <= cut)\n    d = chan_data(i, use_cal)")
s = s.replace('''        print(f"alpha {alpha_w.value:.2f} deg | az off {azoff_w.value:+.1f} | "
              f"el off {eloff_w.value:+.1f} | {mode}")''',
              '''        print(f"alpha {alpha_w.value:.2f} deg | az off {azoff_w.value:+.1f} | "
              f"el off {eloff_w.value:+.1f} | {mode}")
        if use_cal:
            print(f"g_rx cal ON: data is counts / g_rx, "
                  f"{int(u.sum())} samples with a solution "
                  f"(median {np.nanmedian(CAL_ANCHOR_MIN[u]):.0f} min from a "
                  f"real cycle). NOTE: this makes the fit worse -- see the "
                  f"Calibration section.")
        else:
            print("g_rx cal OFF: data is raw accumulator counts.")''')
s = s.replace("        autog_w, gain_w, elcut_w, purehfss_w] + shape_re + shape_im",
              "        autog_w, gain_w, elcut_w, purehfss_w, cal_w] + shape_re + shape_im")
s = s.replace("    W.HBox([purehfss_w]),", "    W.HBox([purehfss_w, cal_w]),")
nb.cells[j].source = s

# ---- document it ----------------------------------------------------------
j = find(lambda s: "## Controls" in s and "Tips for fitting by hand" in s)
s = "".join(nb.cells[j].source)
s = s.replace("**Tips for fitting by hand**", r"""**`apply g_rx cal (B7)` starts OFF, and that is a finding, not a default.**
The checkbox divides `measured_tx` by rf-calibrator's per-cycle receiver gain
$g_{\rm rx}(\nu,t)$ (`abscal/trx_phaseC.npz`, 66 cycles) interpolated onto the
sample times, and drops samples with no solution. Since `measured_tx` is a
channel *difference* it should carry $g_{\rm rx}$ as a linear factor, so
dividing it out ought to remove a real drift — $\times 1.61$ at 173.83 MHz
across this window, which the single fit amplitude cannot absorb.

**It makes the fit worse.** ch712 pure-HFSS goes 0.391 → 0.486; the fleet
median goes 0.518 → 0.587; 69 of 101 channels degrade. The Calibration section
at the bottom of this notebook has the controls that establish this is not a
bug in the correction. Leave it off for fitting; turn it on to look at the
anomaly.

**Tips for fitting by hand**""")
nb.cells[j].source = s

nbf.write(nb, NB)
print(f"patched {NB}")
