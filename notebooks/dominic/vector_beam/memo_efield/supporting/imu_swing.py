"""Along-axle tilt from imu_el through the raster: is there a post-slew swing, and does it track the phase jumps?"""
import numpy as np
from scipy.stats import spearmanr
I = np.load("imu_el_raster.npz"); t, a = I["t"], (I["acc"] - I["bias"]) / I["scale"]  # in g
ok = np.all(np.isfinite(a), 1); t, a = t[ok], a[ok]; o = np.argsort(t); t, a = t[o], a[o]
S = np.load("lag_join.npz"); t0, t1 = S["t"].min(), S["t"].max()
r = (t >= t0) & (t <= t1)
ev, V = np.linalg.eigh(np.cov(a[r].T)); n = V[:, 0]                      # least-variance axis = elevation axle
print("variances along eigvecs (g^2):", ev.round(5), " |a| median", np.median(np.linalg.norm(a[r], axis=1)).round(3))
ax = a @ n; tilt = np.degrees(np.arcsin(np.clip(ax, -1, 1)))
dt = np.median(np.diff(t[r])); print(f"sample spacing {dt:.2f} s; along-axle tilt over raster: median {np.median(tilt[r]):+.2f}°, sd {tilt[r].std():.2f}°")
tm = (t - t0) / 60
print("\nslew at 20:25:59 = raster start - ~0 s. Along-axle tilt per window (median, sd, p2-p98 range):")
for lo_, hi_ in ((-6, -3), (-3, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 62)):
    m = (tm >= lo_) & (tm < hi_)
    if m.sum() > 5: print(f"  min {lo_:+3d}..{hi_:+3d}: median {np.median(tilt[m]):+6.2f}°  sd {tilt[m].std():5.2f}°  range {np.ptp(np.percentile(tilt[m],[2,98])):5.2f}°  n={m.sum()}")
# per self-cal block: tilt sd and mean change, vs |jump|
B = np.load("block_series.npz"); J = np.load("lag_join.npz"); ti = J["t"]; nb = len(ti) // 20
bt = [(ti[i*20], ti[min((i+1)*20, len(ti)) - 1]) for i in range(nb)]
bsd = np.array([tilt[(t >= s) & (t <= e)].std() if ((t >= s) & (t <= e)).sum() > 3 else np.nan for s, e in bt])
bmu = np.array([tilt[(t >= s) & (t <= e)].mean() if ((t >= s) & (t <= e)).sum() > 3 else np.nan for s, e in bt])
dmu = np.r_[np.nan, np.diff(bmu)]; jump = B["jump"]; okj = np.isfinite(jump)
for name, x in (("tilt sd in block", bsd), ("|tilt mean change|", np.abs(dmu)), ("tilt mean", bmu)):
    m = okj & np.isfinite(x); rr, p = spearmanr(np.abs(jump[m]), x[m]); print(f"Spearman |jump| vs {name:20s}: r={rr:+.3f} p={p:.3g} n={m.sum()}")
np.savez("imu_tilt.npz", t=t, tilt=tilt, axis=n, bsd=bsd, bmu=bmu)
