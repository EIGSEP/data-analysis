# tcal — field temperature calibration product

Builds and validates `marjum-2026-07/derived/tcal/<version>/`, the
box-air kelvin scale that `eigsep_data`'s `tcal` product applies when a
bundle is loaded (`Bundle.calibrated`, `Bundle.t_star`).

| Script | What it does |
|---|---|
| `build_tcal.py` | Collects the calibration inputs on their own cadences and writes `solutions.npz` and `manifest.json`: RFAMB/RFNON visit spectra (weighted means, masked with the flag product's RFI bits) with their receiver regime, field S11 at plane P on correlator channels, load-thermistor readings, and the noise-source excess `t_ns_k`: the assumed 917 K, or with `--noise-source <version>` (v0004 on) a spectrum built from the bench-measured `noise_source` product and the field reflections. It computes no temperatures. |
| `bracket_cost.py` | What wider interpolation limits cost in temperature, by leave-out over calibration brackets and S11 holds, and whether the 07-17 16:22 → 19:19 hole can be bridged; writes `bracket_cost.json`. The limits in `build_tcal.py` come from it. |
| `validate_tcal.py` | Leave-one-out test of linear versus nearest interpolation, a plumbing check through `load_bundle`, comparison with Christian's T12 calibration, and coverage; writes `validation.json`. |

The equations live in `eigsep_cal.dicke` and the per-row interpolation
and limits in `eigsep_data.products.tcal`. What the product is, its
coverage and its validation are in the product's own `README.md`.

## Recent changes

- 2026-10-20: `build_tcal.py --noise-source <version>` builds `t_ns_k` as one value per channel from `derived/noise_source/<version>/` (the bench-measured noise source, converted to the field receiver), for `tcal` v0004; without the option it still writes the assumed 917 K. `validate_tcal.py` and `bracket_cost.py` take either form.
- 2026-10-05: the builder reads `flags@v3`, the release of `v3-beta.3` (same files); a
  rebuild reproduces `tcal` v0003 exactly.
- 2026-10-05: `tcal` v0003. Calibration visits are masked with the campaign
  flag product `flags@v3-beta.3`'s RFI bits (0x74), which now flags
  calibration rows; this replaces v0002's own > 5 sigma stopgap. Wrapped
  samples and dropped integrations are still masked directly, and the
  contaminated-visit test stays as an independent check. Coverage now counts
  a row as calibrated when >= 99 % of its 50-200 MHz channels are, because
  two clock-aligned channels (125.0 and 187.5 MHz) are fully flagged in a few
  visits.
- 2026-10-01: `tcal` v0002. Each calibration visit is now a masked,
  integration-time-weighted mean rather than a median (AGENTS.md, Analysis
  methodology). The mask is built per sample (wraps, dropped integrations,
  > 5 robust sigma), because `flags@v2` runs no detector on calibration
  rows.
- 2026-10-01: `tcal` v0001. `build_tcal.py` drops calibration visits that
  disagree with their neighbours in more than 100 channels; three
  visits contaminated by box-air's 1.000 MHz comb EMI (memo 004 §4.2) had
  corrupted v0000. `validate_tcal.py` now
  reports worst cases and per-channel counts, which v0000's band-median
  checks missed. All three scripts take `--products` to work on a
  campaign tree other than the raw-data root.
- 2026-10-01: added, for `tcal` v0000.
