"""Box-vs-drive elevation lag through the 07-17 raster, joined to the integrations in inputs.npz."""
import numpy as np, pandas as pd
T = pd.read_parquet(__import__("os").environ["POINTING_TABLE"], columns=["t_utc_s", "motor_el_deg", "imu_el_deg", "el_deg", "motor_az_deg", "flags"])
z = np.load("inputs.npz"); t = z["times"]
T = T[(T.t_utc_s > t.min() - 60) & (T.t_utc_s < t.max() + 60)].sort_values("t_utc_s")
i = np.clip(np.searchsorted(T.t_utc_s.values, t), 1, len(T) - 1)
i -= (np.abs(T.t_utc_s.values[i - 1] - t) < np.abs(T.t_utc_s.values[i] - t))
dtj = np.abs(T.t_utc_s.values[i] - t); print("join |dt| median/max s", np.median(dtj), dtj.max())
mel, iel = T.motor_el_deg.values[i], T.imu_el_deg.values[i]
lag = (mel - iel + 180) % 360 - 180          # drive ahead of box > 0 (sign checked below)
vel = np.gradient(np.unwrap(np.radians(mel)), t) * 180 / np.pi
np.savez("lag_join.npz", t=t, lag=lag, mel=mel, iel=iel, vel=vel, flags=T["flags"].values[i])
tm = (t - t.min()) / 60
print("motor el speed deg/s median", np.median(np.abs(vel)).round(2), " sign fractions +/-:", (vel > 0).mean().round(2), (vel < 0).mean().round(2))
for a in range(0, 62, 5):
    m = (tm >= a) & (tm < a + 5) & np.isfinite(lag)
    if not m.any(): continue
    up, dn = m & (vel > 0), m & (vel < 0)
    print(f"min {a:2d}-{a+5:2d}: lag median {np.median(lag[m]):+6.1f}  p95 |lag| {np.percentile(np.abs(lag[m]), 95):5.1f}   up-moving {np.median(lag[up]) if up.any() else np.nan:+6.1f}  down-moving {np.median(lag[dn]) if dn.any() else np.nan:+6.1f}")
