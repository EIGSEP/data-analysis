# pointing — Marjum 2026-07 pointing-table generator

Builds the campaign pointing solution: fuses the per-sample motor/IMU
telemetry embedded in the correlator files into one az/el/height timeline
with quality flags. This is the source of `pointing_table.parquet` /
`pointing_table.schema.json` — the campaign repo's copy at
`~/projects/eigsep/marjum-2026-07/curation/` is regenerated from here, not
the other way around; treat this directory as upstream of that one.

## Pipeline, in run order

| Script | Does |
|---|---|
| `extract.py` | Pulls the per-sample `metadata/<sensor>` streams out of the correlator files. |
| `fuse.py` | Fuses the extracted streams into one az/el/height timeline, per this campaign's empirically-established sensor roles (which sensor is trusted for which axis/era). |
| `build_table.py` | Builds the final table: fused az/el/height, per-sample uncertainty, raw per-sensor values for audit, and quality flags (schema in `../../../../../marjum-2026-07/curation/pointing_table.schema.json`). |
| `make_residual_figures.py` | Residual figures for the pointing-table memo (`MEMO-002`). |
| `tests/` | `python -m pytest tests` (needs `picohost`): v6's tilt and elevation on synthetic sweeps calibrated with `picohost.imu_geometry.fit_el_calibration`, through `build_table.build`. |
| `scan_imu_liveness.py` | Diagnostic: IMU sensor liveness across the campaign — distinct from (and a correction to a near-misreading of) the curation mode table's `az_alive`/`el_alive`, which is motor telemetry presence, not IMU liveness. |

## Recent changes

- 2026-11-04: v6 adds `tilt_deg` (the axle's tilt from horizontal, `TxGeometryModel`'s tau, per sample) and `tilt_sigma_deg` from the along-axle component of the averaged `imu_el` acceleration, with the axle fitted on the 07-17 raster and the sign fixed by the raster beam fit's +4.74 deg. `el_deg` is recomputed from the same accelerations through the campaign's firmware calibration, which removes the recorder's linear-mean wrap glitch near +/-180 deg (flag bit 15 `EL_WRAP_CORRECTED`); bit 14 `IMU_NOT_QUASI_STATIC` marks | |a|/g - 1 | > 2 %. `imu_el_deg`, `quality` and every other column are unchanged. The generator now imports `picohost.imu_geometry`.
- 2026-10-28: v5 negates the derived azimuth (`AZ_SENSE = -1` in `build_table.py`) because the azimuth is left-handed about the boresight (memos 002, 003). `az_deg` now goes into `eigsep_base.rotations.mount_rotation` as is; the raw motor/pot columns keep the sensors' sense.
- 2026-09-15 (`software-engineer`): added this file (README-convention
  retrofit, fleet-wide consolidation pass) — this generates the pointing
  table touched throughout B15's schema-v1.1 work and had no README.
