"""Stage 2: patch the narrative markdown with the numbers the corrected run
actually produced. Markdown cells carry no outputs, so this does not require
re-execution.
"""
import nbformat as nbf

NB = "beam_fits_v2_review_checkpoint.ipynb"
nb = nbf.read(NB, as_version=4)

REVISION = r"""> ## REVISED 2026-09-17 — corrected sample selection
>
> This checkpoint has been **re-run and replaced in place**. Two defects in the
> sample selection were found after the first issue (see
> `beam_metric_outliers_checkpoint`) and have been fixed here. Every number
> below is from the corrected run.
>
> **1. Calibration data was being fit as if it were beam data.** 23 of the 226
> files in this window are calibration files — the receiver was on a VNA, noise
> source or load, not on the antenna. Nothing in the mask chain was a campaign
> gate, so those samples entered the fit. They carried a **median 71.3% of the
> residual power** across the 101 channels.
>
> The exclusion is built from the **per-sample `metadata/rfswitch` stream**, not
> from the `rfswitch_dominant` column of `curation/file_state.csv`. That column
> is a file-level majority vote, and 14 of the 23 calibration files are
> *majority* `RFANT`, so a dominant-state test keeps them
> (`corr_20260717_191940Z.h5` is `{RFANT: 124, RFNON: 111, RFAMB: 3}`). Going
> per sample also *keeps* 1938 genuine on-antenna samples that a whole-file
> exclusion would have thrown away.
>
> **2. `EL_SOLUTION_GLITCH`.** `pointing_table@v1.2+45f8059` adds a bit marking
> samples where the IMU elevation solution jumped faster than the drive can
> move. Small in aggregate (~0.002 on the median) but locally decisive.
>
> Net effect on the mask: **25352 → 22969 samples** (−2200 off-antenna, −183
> glitch), and on the headline:
>
> | | before | after |
> |---|---|---|
> | median normalized RMS (corrected geometry) | 0.9029 | **0.4691** |
> | channels below 0.5 | 2 | **61** |
> | median normalized RMS (old geometry) | 0.9115 | **0.5502** |
>
> **3. Section 6 is un-blocked and rewritten.** Q8 is closed; `el` is the
> boresight zenith angle. The `|el| ≈ 180` "wrap cluster" is
> boresight-on-transmitter, not an angle-wrap artifact, and is now counted.
>
> `fit_beam_v2.py`, which generated the reports read here, lived only in `/tmp`
> and was unversioned until this revision. It now sits next to this notebook.

"""

s0 = "".join(nb.cells[0].source)
assert "D2 review checkpoint" in s0
nb.cells[0].source = REVISION + s0

DECISION = r"""## 16. Decision requested

**The framing of this section has changed materially since the last issue, and
the previous verdict is withdrawn in place.** It read *"D2 has no defensible
beam measurement."* That conclusion was reached on a sample set in which
calibration data — receiver on a load, not the antenna — supplied a median
71.3% of the residual power. With that removed the fit is far better than this
notebook previously reported.

**What the corrected run shows.** Median normalized RMS against the corrected
geometry is **0.4691**, down from 0.9029, with **61 of 101 channels below
0.5** where previously there were 2. On the single-amplitude-DOF test the
median is **0.5176**, with 65/101 channels agreeing to better than 0.60. This
is a real fit, not a null result. The blunt "no measurement here" verdict was
an artifact of the selection defect.

**What did *not* change, and this is the important part.** The arm-structure
finding is completely unmoved:

| | before | after |
|---|---|---|
| own-arm empirical template | 0.242 | **0.2416** |
| other-arm template | 0.956 | **0.9556** |
| HFSS physical model | 0.670 | **0.6697** |
| blind template beats physical model on | 93/101 | **93/101** |
| arm-to-arm profile correlation | −0.92 | **−0.9211** |

That robustness is itself evidence rather than a coincidence. Section 15 works
on per-azimuth-bin **medians** within `|el| < 10°`; that slice is 10.8%
contaminated, and a median absorbs a 10.8% minority. So the arm result never
depended on the bad samples — it is a property of the data, and the physical
model still fails to reproduce it while a blind empirical template succeeds.

The arm-0/arm-1 gap also survives: median PCA normalized RMS is **0.5505** for
arm 0 against **0.4539** for arm 1 (point-biserial correlation −0.353,
p = 3e-4).

**So the D2 position is now:** the beam fit is usable on a majority of
channels, but the forward model still cannot represent a strong, repeatable
polarization structure that a blind per-arm template captures at 0.24. Three
independent refutations from the previous issue (frame mismatch, arm/pol angle
swap, azimuth registration) are unaffected — each was refuted by ~2 orders of
magnitude, far beyond the factor this correction moves.

**Still unresolved, carried forward unchanged:**

- **The transmitter direction is not measurable from this dataset.** The fitted
  heading sits 86.3° from the surveyed direction and substituting the surveyed
  one changes the median by ~0.0005 (section 1a). The earlier geometry refit
  *improved* RMS while moving 70° away from truth — a false minimum. Nothing
  here changes that; geometry should be taken from the survey, not fitted.
- **TX identity** (section 10). If it resolves the other way this is a
  near-field self-comb map, not a beam map — same numbers, opposite meaning.
- **Receiver regime change** bracketing the window: amplitude scale, not shape.
- **Elevation zero** is bounded to `|offset| ≲ 2°`, and **absolute azimuth zero
  is still open** — the coverage map has no north anchor.

**Recommended next steps** (for Aaron / `experimental-strategist` to rank, not
for me to self-assign):

1. **Re-derive geometry from the survey rather than fitting it**, and re-score.
   This is now the largest remaining known-wrong input.
2. **Treat the per-arm empirical templates as the benchmark** any physical
   model must beat. The 0.24 / 0.67 gap is the real modelling target, and it is
   now the dominant residual rather than one defect among several.
3. Hand the calibration-window leak and the per-sample `rfswitch` mask to
   `data-archivist` as durable facts about the data, so no other analysis
   repeats this.

**STOPPED AT REVIEW GATE — awaiting Aaron's approval.**"""

assert "16. Decision requested" in "".join(nb.cells[56].source)
nb.cells[56].source = DECISION

nbf.write(NB and NB, nb) if False else nbf.write(nb, NB)
print("stage 2 patched")
