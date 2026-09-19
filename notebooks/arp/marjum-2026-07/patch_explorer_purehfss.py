"""Make "pure HFSS" a real, default, locked mode in beam_explorer.ipynb.

Aaron wants to hand-fit with the physical knobs only -- amplitude, geometry/TX
position, alpha -- with shape1/2/3 held at zero. Before this patch the shape
sliders merely *defaulted* to 0.0 and could be dragged off it; there was a
"zero the shape terms" button but nothing holding them there. That is
"settable to zero as one option among many", not a baseline mode.

This adds a `pure HFSS (lock shape)` checkbox, ON by default, which zeroes the
shape sliders and disables them, so the model evaluated is the unperturbed
HFSS prior unless the checkbox is explicitly cleared.
"""
import nbformat as nbf

NB = "beam_explorer.ipynb"
nb = nbf.read(NB, as_version=4)


def find(pred):
    for j, c in enumerate(nb.cells):
        if pred("".join(c.source)):
            return j
    raise SystemExit("cell not found")


# ---- controls cell: add the checkbox ---------------------------------------
j = find(lambda s: "btn_zero = W.Button" in s)
s = "".join(nb.cells[j].source)
anchor = 'autog_w = W.Checkbox(value=True, description="auto-fit amplitude")'
assert anchor in s
s = s.replace(anchor, anchor + """
# Pure-HFSS baseline. The shape terms are extra, non-physical degrees of
# freedom on top of the HFSS prior; with this checked the model is the
# unperturbed physical prediction and only the physical knobs (amplitude,
# TX direction, alpha) do anything. Default ON: the baseline should be what
# you get without asking for it.
purehfss_w = W.Checkbox(value=True, description="pure HFSS (lock shape)")""")
nb.cells[j].source = s

# ---- update cell: honour the lock ------------------------------------------
j = find(lambda s: "def update(**kw):" in s)
s = "".join(nb.cells[j].source)

anchor = """    gain = None if autog_w.value else gain_w.value
    m, A = model_power(i, heading, alpha_w.value, arm,
                       [s.value for s in shape_re], [s.value for s in shape_im],
                       gain=gain, az_off=azoff_w.value, el_off=eloff_w.value)"""
assert anchor in s, "update() anchor not found"
s = s.replace(anchor, """    gain = None if autog_w.value else gain_w.value
    locked = purehfss_w.value
    if locked:
        # hold the shape terms at zero AND grey them out, so the pure-HFSS
        # baseline cannot be perturbed by a stray drag
        for w in shape_re + shape_im:
            if w.value != 0.0:
                w.unobserve_all()
                w.value = 0.0
                w.observe(lambda change: update(), names="value")
            w.disabled = True
        s_re = [0.0] * len(shape_re)
        s_im = [0.0] * len(shape_im)
    else:
        for w in shape_re + shape_im:
            w.disabled = False
        s_re = [w.value for w in shape_re]
        s_im = [w.value for w in shape_im]
    m, A = model_power(i, heading, alpha_w.value, arm, s_re, s_im,
                       gain=gain, az_off=azoff_w.value, el_off=eloff_w.value)""")

# report the mode in the printed summary
old_print = """        print(f"alpha {alpha_w.value:.2f} deg | az off {azoff_w.value:+.1f} | "
              f"el off {eloff_w.value:+.1f} | shapes Re {[round(s.value,3) for s in shape_re]} "
              f"Im {[round(s.value,3) for s in shape_im]}")"""
assert old_print in s, "print anchor not found"
s = s.replace(old_print, """        mode = ("PURE HFSS (shape terms locked at zero)" if locked
                else f"shapes Re {[round(v,3) for v in s_re]} "
                     f"Im {[round(v,3) for v in s_im]}")
        print(f"alpha {alpha_w.value:.2f} deg | az off {azoff_w.value:+.1f} | "
              f"el off {eloff_w.value:+.1f} | {mode}")""")

# add to the observed controls and the panel layout
s = s.replace("ctrl = [ch_w, arm_w, alpha_w, dE_w, dN_w, dU_w, azoff_w, eloff_w,\n        autog_w, gain_w, elcut_w] + shape_re + shape_im",
              "ctrl = [ch_w, arm_w, alpha_w, dE_w, dN_w, dU_w, azoff_w, eloff_w,\n        autog_w, gain_w, elcut_w, purehfss_w] + shape_re + shape_im")
s = s.replace("    W.HBox([alpha_w, autog_w, gain_w]),",
              "    W.HBox([alpha_w, autog_w, gain_w]),\n    W.HBox([purehfss_w]),")
nb.cells[j].source = s

# ---- document it in the Controls markdown ----------------------------------
j = find(lambda s: "## Controls" in s and "Tips for fitting by hand" in s)
s = "".join(nb.cells[j].source)
s = s.replace("**Tips for fitting by hand**", """**Pure HFSS is the default.** The `pure HFSS (lock shape)` checkbox starts
**on**, which holds `shape1/2/3` at zero and greys them out. In that mode the
model is the unperturbed physical HFSS prediction and the only live knobs are
the physical ones: amplitude, TX direction (E/N/U), and `alpha`. The shape
terms are extra non-physical freedom on top of the prior, not one of this
notebook's intended comparison knobs, so they are locked unless you clear the
box deliberately.

For scale, on the corrected cache the pure-HFSS model reaches a median
normalized RMS of **0.518** across the 101 channels with only the amplitude
free; letting the three shape terms float as well buys **0.469**. So the
non-physical freedom is worth ~0.05 in the median -- worth knowing before you
decide the prior is inadequate.

**Tips for fitting by hand**""")
nb.cells[j].source = s

nbf.write(nb, NB)
print(f"patched {NB}")
