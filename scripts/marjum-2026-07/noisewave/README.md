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

Status, 2026-10-03: the smooth model fails (chi2/dof ~ 1e4, held-out errors
of hundreds of K). A delay of ~8 ns on the noise waves cuts the held-out
error tenfold (median 22 K) but leaves chi2/dof ~ 2e3. Not a product yet.

## Recent changes

- 2026-10-03: added; study in progress.
