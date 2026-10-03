# noisewave — tier-2 receiver noise-wave fits (study, in progress)

Fits receiver noise-wave parameters to the field calibrators of the
07-18 rx-B spare-port blocks: the ambient load and the SP1 cable terminated
open and short, with T_NS fixed, following memo M003 § 7.1 (tier 2) and
`eigsep_cal/docs/api.md` § 4.5. Inputs come from `tcal` v0002 (Q per
integration) and the rx-B S11 sweep at plane P.

| Script | What it does |
|---|---|
| `fit_tier2.py` | The M003 tier-2 model (smooth 7-term T_unc, T_cos, T_sin, T_L): synthetic recovery, held-out prediction, block repeatability, SP1-temperature sensitivity, antenna comparison with tier 1. |
| `fit_models.py` | Four models compared: smooth; with a delay on T_cos − i T_sin; intrinsic (Bucher et al.); intrinsic with a delay. |
| `diagnose2.py` | With block-to-block errors: a correction to the receiver's Gamma, linear drift of the parameters, and a delay on T_unc, each held-out tested. |
| `diagnose.py` | Block-to-block versus within-block scatter of Q, model B refitted with block-to-block errors, and whether the fit wants a correction to the spare port's measured Gamma (held-out tested). |

Status, 2026-10-03: the smooth model fails (chi2/dof ~ 1e4, held-out errors
of hundreds of K). A delay of ~8 ns on the noise waves cuts the held-out
error tenfold (median 22 K). Within-block errors understate the
block-to-block scatter of the open/short Q 38–51x (about 2 K); with
block-to-block errors the delay model has chi2/dof 114, residuals ~20 K. A
correction to the spare port's Gamma lowers chi2 only with an unphysical
|delta| ~ 0.25 and worsens held-out prediction, so it is not an S11 error at
that port. Not a product yet.

Second round (diagnose2.py): drift over the 40 minutes changes nothing. A
smooth correction to Gamma_rec (median 0.047, max 0.15) halves chi2 (to 49)
and improves held-out prediction (short 21 -> 12 K median). The measured
Gamma_rec is repeatable to ~0.001 across rx-A and rx-B, so this would be an
accuracy error of the receiver's S11 (plausible at the level the two switch
files differed), not a change in time. A ripple on T_unc at 72 ns does about
as well and is not obviously physical. Field data cannot separate the two; a
bench measurement of the receiver's reflection at plane P can.

## Next steps for characterizing the receiver

The field data have taken this as far as they can: the delay model is the
best description, and about 20 K of residual remains that either a ~0.05
accuracy error in Gamma_rec or a 72 ns ripple on T_unc can absorb. What would
settle it, in priority order:

**Bench measurements (the LNAs and switches survived the deployment):**

1. **Receiver reflection at plane P.** Measure Gamma_rec with the bench VNA
   both directly at the LNA input and through the VNARF switch path the field
   sweeps used. If the field value is off by ~0.05, the Gamma_rec calibration
   (most likely the VNARF path characterization) is wrong; if it agrees, the
   residual is real receiver structure in T_unc.
2. **The switch-path characterization of VNARF**, re-measured by Charlie's
   corrected method and checked against item 1. The two switch files
   differed by ~0.07 in port reflection, the size of error at issue.
3. **The path between P and the LNA input.** Find the physical cable or
   trace and its length: the fitted ~8 ns delay predicts about 4 ns one way
   (~0.8 m of coax). A measured length that matches would make the delay
   model physical rather than fitted.
4. **The noise source:** its ENR, and the VNANON/RFNON switch paths, which
   were never characterized. These fix T_NS, which tier 2 holds at the
   nameplate 917 K, and they bear on memo 004's open factor of 2.4.
5. **A lab receiver calibration** (M003 tier 3): hot and ambient loads plus
   an open/short cable at a known, logged temperature, through the same
   switch. This determines T_NS and the noise waves together and removes the
   assumption that the SP1 cable sits at the load temperature.

**Analysis, once bench data exist:**

- Fix Gamma_rec at its bench value and refit; check that the held-out
  termination is then predicted to the ~2 K block-to-block repeatability.
- If the residual survives, model T_unc's structure physically (for
  example, the delay of item 3) rather than with a free ripple.
- Add the 07-16 01:15 SP1/RFNOFF block as an independent epoch.
- Then promote the solver to `eigsep_cal` (`fit_noise_waves`, api.md
  stage 3b) and build a tier-2 `tcal` version. Until then tier-2 antenna
  temperatures must not replace tier 1.

**For the next deployment** (M003 § 8): a heated RFAMB for temperature
contrast, SP1 open/short kept in the schedule with longer dwells, a second
cable of different length on SP2, and logged temperatures for the SP1 cable
and the LNA.

## Recent changes

- 2026-10-03: next steps for characterizing the receiver written down.
- 2026-10-03: added; study in progress.
