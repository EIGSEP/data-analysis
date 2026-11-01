# memo_efield/supporting — tilt scan, error budget, phase jumps

The analyses behind memo Figs. 5-7 (axle-tilt scan, error budget, phase jumps)
and the memo sections that quote them. They were run in October 2026 under the
**right-handed** azimuth and v0013's el zero (+0.93 deg), before memo 003
adopted the left-handed one. The memo labels them that way and they have not
been redone. They are kept as run: only the file paths were changed, to be
relative to this directory. They have not been re-executed since the move, so
expect to touch them up if you do. The recorded outputs that are small enough
are committed: `geomscan2.json`, `compare.json`, `step4.json`.

Run from this directory with `inputs.npz` linked here
(`ln -s ../inputs.npz .`) and `../../selfcal_phase.npz` made by
`../../selfcal_phase.py`.

| Step | Script | Output |
|---|---|---|
| Closure g seeds, geometry scan (2026-10-21) | `python geomscan2.py .` | `geomscan2.json` |
| Tilt x handedness x positions scan | `python run_compare.py` (imports `common.py`, `tilt.py`, `geomfit2.py`) | `compare.json` (Fig. 5) |
| Error budget, corrections in turn | `python step4.py .` | `step4.json` (Fig. 6 quotes its medians) |
| Self-cal on each half of the band | `python selfcal_half.py lo`, `... hi` | `selfcal_lo.npz`, `selfcal_hi.npz` |
| Box-vs-drive elevation lag | `POINTING_TABLE=... python lag_vs_time.py` | `lag_join.npz` |
| Block phases and jumps | `python jumps_vs_lag.py` | `block_series.npz` |
| IMU accelerometer through the raster | `DEPLOYMENT5_DATA=... python extract_imu.py` | `imu_el_raster.npz` |
| Box along-axle tilt | `python imu_swing.py` | `imu_tilt.npz` (Fig. 7) |

`../make_figures.py` reads `compare.json`, `block_series.npz`, `imu_tilt.npz`
and `lag_join.npz` from `EFIELD_SUPPORT`, by default this directory.
