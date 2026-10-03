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
| `diagnose.py` | Block-to-block versus within-block scatter of Q, model B refitted with block-to-block errors, and whether the fit wants a correction to the spare port's measured Gamma (held-out tested). |

Status, 2026-10-03: the smooth model fails (chi2/dof ~ 1e4, held-out errors
of hundreds of K). A delay of ~8 ns on the noise waves cuts the held-out
error tenfold (median 22 K). Within-block errors understate the
block-to-block scatter of the open/short Q 38–51x (about 2 K); with
block-to-block errors the delay model has chi2/dof 114, residuals ~20 K. A
correction to the spare port's Gamma lowers chi2 only with an unphysical
|delta| ~ 0.25 and worsens held-out prediction, so it is not an S11 error at
that port. Not a product yet.

## Recent changes

- 2026-10-03: added; study in progress.
