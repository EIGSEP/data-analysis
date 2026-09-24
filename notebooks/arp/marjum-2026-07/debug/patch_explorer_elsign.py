"""Add the corrected elevation-rotation direction to beam_explorer.ipynb.

Christian (2026-09-17): +EL is the right-hand rule about the highline vector
pointing WEST, so at el=0 (antenna up) +EL tips the boresight toward NORTH.

The model rotates elevation about the fixed +x = East shaft, so its +el tips
the boresight toward -N (SOUTH) -- verified directly from the rotation matrix
(boresight_ENU = R.zhat = (0, -sin el, cos el)). The model is backwards.

Exposed as a toggle, defaulting to the CORRECTED convention, because the
correction changes every fitted heading and Aaron should be able to see both.
"""
import nbformat as nbf

NB = "beam_explorer.ipynb"
nb = nbf.read(NB, as_version=4)


def find(pred):
    for j, c in enumerate(nb.cells):
        if pred("".join(c.source)):
            return j
    raise SystemExit("cell not found")


# ---- model cell: thread el_sign through model_power ------------------------
j = find(lambda s: "def model_power(" in s)
s = "".join(nb.cells[j].source)
old = """                gain=None, az_off=0.0, el_off=0.0, apply_cal=False):
    \"\"\"Model power per sample. gain=None -> least-squares amplitude (RMS-optimal).\"\"\"
    cpl = coupling(AZ + az_off, EL + el_off, heading, alpha_deg, arm)"""
assert old in s, "model_power anchor not found"
s = s.replace(old, """                gain=None, az_off=0.0, el_off=0.0, apply_cal=False,
                el_sign=-1):
    \"\"\"Model power per sample. gain=None -> least-squares amplitude (RMS-optimal).

    el_sign = -1 is Christian's hardware convention (+EL tips the boresight
    NORTH); the model's own rotation, about the fixed East shaft, tips it
    SOUTH, so el_sign = +1 reproduces the uncorrected behaviour and every
    number published before 2026-09-17.
    \"\"\"
    cpl = coupling(AZ + az_off, el_sign * EL + el_off, heading, alpha_deg, arm)""")
nb.cells[j].source = s

# ---- controls: the checkbox ------------------------------------------------
j = find(lambda s: "cal_w = W.Checkbox" in s)
s = "".join(nb.cells[j].source)
anchor = 'cal_w = W.Checkbox(value=False, description="apply g_rx cal (B7)",\n                   disabled=not E_CAL_AVAILABLE)'
assert anchor in s, "cal_w anchor not found"
s = s.replace(anchor, anchor + '''
# Elevation rotation direction. ON = Christian's hardware convention (+EL tips
# the boresight NORTH). OFF reproduces the model as it shipped, which tips it
# SOUTH -- i.e. backwards. Default ON, but note that everything published
# before 2026-09-17 used the OFF behaviour.
elsign_w = W.Checkbox(value=True, description="correct el direction")''')
nb.cells[j].source = s

# ---- update(): use it ------------------------------------------------------
j = find(lambda s: "def update(**kw):" in s)
s = "".join(nb.cells[j].source)
s = s.replace("    use_cal = bool(cal_w.value) and E_CAL_AVAILABLE",
              "    use_cal = bool(cal_w.value) and E_CAL_AVAILABLE\n"
              "    el_sign = -1 if elsign_w.value else 1")
s = s.replace("""                       gain=gain, az_off=azoff_w.value, el_off=eloff_w.value,
                       apply_cal=use_cal)""",
              """                       gain=gain, az_off=azoff_w.value, el_off=eloff_w.value,
                       apply_cal=use_cal, el_sign=el_sign)""")
s = s.replace('''        else:
            print("g_rx cal OFF: data is raw accumulator counts.")''',
              '''        else:
            print("g_rx cal OFF: data is raw accumulator counts.")
        print("el direction: " + ("CORRECTED (+EL tips boresight NORTH, "
                                  "Christian's hardware convention)"
                                  if el_sign < 0 else
                                  "as-shipped (+el tips SOUTH -- backwards; "
                                  "matches everything published before "
                                  "2026-09-17)"))''')
s = s.replace("        autog_w, gain_w, elcut_w, purehfss_w, cal_w] + shape_re + shape_im",
              "        autog_w, gain_w, elcut_w, purehfss_w, cal_w,\n"
              "        elsign_w] + shape_re + shape_im")
s = s.replace("    W.HBox([purehfss_w, cal_w]),",
              "    W.HBox([purehfss_w, cal_w, elsign_w]),")
nb.cells[j].source = s

# ---- document it -----------------------------------------------------------
j = find(lambda s: "The arm-to-channel wiring is now known" in s)
s = "".join(nb.cells[j].source)
anchor2 = "- **The arm-to-channel wiring is now known (2026-09-17, Aaron/Christian).**"
assert anchor2 in s
s = s.replace(anchor2, r"""- **The elevation rotation direction in this model is BACKWARDS, and the
  `correct el direction` checkbox (default ON) fixes it.** Christian
  (2026-09-17): `+EL` is the right-hand rule about the highline vector pointing
  **west**, so at `el = 0` it tips the boresight toward **north**. The model
  rotates elevation about the fixed `+x = East` shaft, giving
  `boresight_ENU = (0, −sin el, cos el)` — it tips **south**. Read straight off
  the rotation matrix, so there is no ambiguity.
  - **Effect on the fitted geometry: the transmitter bearing rotates by
    180°.** Verified exact to ~1e-5: flipping `el` is identically
    `(dE, dN) → (−dE, −dN)`. (An earlier note here said it "flips `dN`" — that
    is only the `dE = 0` slice of this more general rule.)
  - **It does *not* reconcile the fit with the survey.** Best-fit heading is
    43.2° from the surveyed direction uncorrected and **43.4° corrected** —
    the flip mirrors the optimum rather than moving it toward nadir. At the
    surveyed heading itself the corrected convention is modestly better
    (RMS 0.3684 vs 0.3856). The fit still wants the transmitter ~87 m
    horizontally (38° off vertical) where the survey puts it 6.9 m (4.2°).
  - **Everything published before 2026-09-17 used the uncorrected sign** —
    `fit_beam_v2`, both review checkpoints. Turning this on changes fitted
    headings; it does not change the arm or calibration findings.

""" + anchor2)
nb.cells[j].source = s

nbf.write(nb, NB)
print(f"patched {NB}")
