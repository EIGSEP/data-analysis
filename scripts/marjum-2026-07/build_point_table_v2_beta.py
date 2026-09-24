"""Build marjum-2026-07/point_table_v2-beta.npz: commanded azimuth, IMU elevation.

On 07-17 pointing_table@v1.2 records a -27.35 deg azimuth slip over
20:41:24-20:53:43. Beam fits show the potentiometer slipped, not the platform.
The unfitted HFSS beam fits far better with commanded azimuth: held-out test
fractional RMS 0.071 against 0.197 with pointing_table azimuth
(data-analysis/notebooks/arp/marjum-2026-07/single_ch_beam_fit.ipynb, 156.25 MHz).
Consecutive el sweeps also keep changing where the pot says the platform
stalled. The mount never drives az and el together (Aaron, 2026-09-23).

This product covers the 07-17 beam-scan raster (20:25:00-21:30:00 UTC):
- az_deg = motor_az_deg + AZ_OFFSET, with AZ_OFFSET the median of
  pointing_table az - motor az over 20:26:30-20:41:00. Before the slip, pot and
  motor agree with a 1.0 deg std, so the azimuth zero stays the potentiometer
  calibration zero, as in pointing_table. The motor is exactly constant through
  each el sweep and steps 5.0018 deg between sweeps.
- el_deg = pointing_table el_deg (gravity-referenced IMU, 0.24 deg 1-sigma).
  motor_el_deg scatters 15 deg against it and is not used.

Exploratory single-study code; run from anywhere.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / 'marjum-2026-07/data').is_dir())
CAMPAIGN = ROOT / 'marjum-2026-07'
SOURCE_TABLE = CAMPAIGN / 'curation/pointing_table.parquet'
WINDOW_UTC = ('2026-07-17 20:25:00', '2026-07-17 21:30:00')
PRE_SLIP_UTC = ('2026-07-17 20:26:30', '2026-07-17 20:41:00')
VERSION = 'v2-beta'


def wrap(deg):
    return (np.asarray(deg) + 180.) % 360. - 180.


def git_state(path):
    commit = subprocess.run(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(path), 'status', '--porcelain', '--', str(path)],
                           capture_output=True, text=True).stdout.strip()
    return commit, bool(dirty)


def main(output):
    table = pq.read_table(SOURCE_TABLE)
    source_provenance = json.loads(table.schema.metadata[b'eigsep_provenance'])
    t0, t1 = (pd.Timestamp(s, tz='UTC').timestamp() for s in WINDOW_UTC)
    df = table.to_pandas()
    df = df[(df.t_utc_s >= t0) & (df.t_utc_s <= t1)].sort_values('t_utc').reset_index(drop=True)
    assert df.t_utc.is_unique
    ok = (df.quality == 'ok').to_numpy()
    motor = df.motor_az_deg.to_numpy()
    assert np.isfinite(motor[ok]).all()

    p0, p1 = (pd.Timestamp(s, tz='UTC').timestamp() for s in PRE_SLIP_UTC)
    pre = ok & (df.t_utc_s >= p0).to_numpy() & (df.t_utc_s < p1).to_numpy()
    difference = wrap(df.az_deg.to_numpy() - motor)
    az_offset = float(np.median(difference[pre]))
    az_offset_std = float(np.std(difference[pre]))
    az = wrap(motor + az_offset)
    az[~np.isfinite(motor)] = np.nan

    sweep = np.full(len(df), -1, dtype=np.int32)
    ok_rows = np.flatnonzero(ok)
    sweep[ok_rows] = np.cumsum(np.r_[True, np.abs(np.diff(motor[ok_rows])) > 1e-6]) - 1
    change = wrap(az - df.az_deg.to_numpy())

    commit, dirty = git_state(SOURCE.parent)
    provenance = dict(
        product='point_table', version=VERSION, campaign='marjum-2026-07',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        generator=str(SOURCE.relative_to(ROOT)), generator_commit=commit, generator_dirty=dirty,
        source=str(SOURCE_TABLE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE_TABLE.read_bytes()).hexdigest(),
        source_compact=source_provenance['compact'],
        supersedes='point_table v1-beta (per-sweep constant potentiometer azimuth)',
        window_utc=list(WINDOW_UTC), join_key='t_utc (int64 UTC nanoseconds), as in pointing_table',
        azimuth='commanded: motor_az_deg + az_offset_deg',
        az_offset_deg=az_offset,
        az_offset_rule='median of pointing_table az_deg - motor_az_deg over quality==ok rows in '
                       + ' to '.join(PRE_SLIP_UTC) + ' UTC, before the slip',
        az_offset_pre_slip_std_deg=az_offset_std,
        azimuth_zero='potentiometer calibration zero, as in pointing_table; not tied to true north. '
                     'Beam fits carry their own az-zero offset.',
        elevation='pointing_table el_deg (gravity-referenced IMU); motor_el_deg not used',
        evidence='single_ch_beam_fit.ipynb (data-analysis, 156.25 MHz): the unfitted HFSS beam has held-out '
                 'test fractional RMS 0.071 with commanded az against 0.197 with pointing_table az; '
                 'where the pot reports a stall, consecutive sweeps change by a median of 15%, against 10.5% '
                 'across normal steps.',
        basis='Aaron, 2026-09-23/24: the drives never run together and are locked when idle. The '
              'potentiometer, not the platform, slipped. Commanded pointing is the new default.',
        units=dict(t_utc='ns since 1970-01-01 UTC', t_utc_s='s since 1970-01-01 UTC',
                   az_deg='deg, potentiometer-zero azimuth', el_deg='deg, pointing_table elevation convention'),
        nulls='az_deg NaN where motor_az_deg is NaN; el_deg NaN where pointing_table is NaN; '
              'sweep = -1 on non-ok rows',
        summary=dict(rows=int(len(df)), ok_rows=int(ok.sum()), sweeps=int(sweep.max() + 1),
                     az_offset_deg=az_offset,
                     change_from_pointing_table_before_slip_median_deg=float(np.median(np.abs(change[pre]))),
                     change_from_pointing_table_after_2054_median_deg=float(np.median(np.abs(
                         change[ok & (df.t_utc_s >= pd.Timestamp('2026-07-17 20:54', tz='UTC').timestamp()).to_numpy()])))),
        caveats=['Beta: 07-17 beam-scan raster only.',
                 'Commanded azimuth assumes the platform followed the motor. Discrete platform slip during '
                 'az steps would not be seen; none is indicated in this raster.',
                 'quality, flags and sigmas are copied from pointing_table and describe its sensor-fused az, '
                 'not commanded az.'])
    np.savez_compressed(
        output,
        t_utc=df.t_utc.to_numpy(np.int64), t_utc_s=df.t_utc_s.to_numpy(float),
        az_deg=az, el_deg=df.el_deg.to_numpy(float), az_deg_v1=df.az_deg.to_numpy(float),
        motor_az_deg=motor, el_sigma_deg=df.el_sigma_deg.to_numpy(float),
        quality=df.quality.to_numpy(str), flags=df['flags'].to_numpy(np.int32),
        source_file=df.file.to_numpy(str), sweep=sweep,
        provenance=np.array(json.dumps(provenance, indent=1)))
    print(json.dumps(provenance['summary'], indent=1))
    print('wrote', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CAMPAIGN / 'point_table_v2-beta.npz')
    main(parser.parse_args().output)
