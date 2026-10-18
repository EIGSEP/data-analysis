"""Per-integration housekeeping streams (box-air correlator metadata: thermistors, currents,
IMU, motors, pot, lidar, switch state) for 07-15 00:30 - 07-18 03:30 UTC, saved as npz for
``gain_drivers.py``.  usage: hk_streams.py OUT.npz"""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
from eigsep_data import MetadataIndex
from eigsep_data.clock import to_unix_time
idx = MetadataIndex(Path(os.environ['EIGSEP_CAMPAIGN_ROOT']) / 'data')
s = idx.select(time=("2026-07-15T00:30:00Z", "2026-07-18T03:30:00Z"))
m = s.meta
cols = ['imu_el_accel_x', 'imu_el_accel_y', 'imu_el_accel_z', 'imu_el_el_deg', 'lidar_distance_m',
        'motor_az_pos', 'motor_az_target_pos', 'motor_el_pos', 'motor_el_target_pos',
        'potmon_pot_az_voltage', 'rfswitch_therm_temp_therm0', 'rfswitch_therm_temp_therm1',
        'rfswitch_therm_temp_therm2', 'system_current_current_a', 'tempctrl_load_T_now', 'tempctrl_load_active',
        'integration_time']
out = {c: pd.to_numeric(m[c], errors='coerce').to_numpy(float) for c in cols}
out['t'] = np.array([to_unix_time(x) for x in m.time_best]) if not np.issubdtype(m.time_best.dtype, np.number) else m.time_best.to_numpy(float)
out['rf_ant'] = (m.rfswitch == 'RFANT').to_numpy()
out['rfswitch'] = m.rfswitch.astype(str).to_numpy()
o = np.argsort(out['t'])
np.savez(sys.argv[1], **{k: v[o] for k, v in out.items()})
print(len(o), 'rows', pd.Series(out['rfswitch']).value_counts().to_dict())
