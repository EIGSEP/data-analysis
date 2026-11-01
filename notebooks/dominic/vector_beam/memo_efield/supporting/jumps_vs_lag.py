"""Do self-cal phase jumps track the box-vs-drive elevation lag?"""
import numpy as np
from scipy.stats import spearmanr
J = np.load("lag_join.npz"); s = np.load("../../selfcal_phase.npz")
lo, hi = np.load("selfcal_lo.npz")["phase"], np.load("selfcal_hi.npz")["phase"]
B = 20; t = J["t"]; n = len(t) // B; tm = (t - t.min()) / 60
def bl(x, f=np.nanmedian):
    return np.array([f(x[i*B:(i+1)*B]) if np.isfinite(x[i*B:(i+1)*B]).any() else np.nan for i in range(n)])
w = lambda x: np.degrees(np.angle(np.exp(1j * x)))
P = bl(s["phase"]); btm = bl(tm, np.mean); lag = bl(J["lag"]); lagsd = bl(J["lag"], np.nanstd)
el = bl(J["iel"]); dlag = np.r_[np.nan, np.diff(lag)]
jump = np.r_[np.nan, w(np.diff(P))]  # this block minus previous
ok = np.isfinite(jump)
print(f"constrained consecutive-block jumps: {ok.sum()} of {n}")
print("\nper 5-min window: |jump| median, n, lag median, lag sd-in-block median, frac blocks constrained")
for a in range(0, 62, 5):
    m = (btm >= a) & (btm < a + 5)
    k = m & ok
    print(f"  min {a:2d}-{a+5:2d}: |jump| {np.median(np.abs(jump[k])) if k.any() else np.nan:5.1f}° (n={k.sum():2d})  lag {np.nanmedian(lag[m]):+5.1f}  lag sd {np.nanmedian(lagsd[m]):4.1f}  constrained {np.isfinite(P[m]).mean():.2f}")
for name, x in (("lag", lag), ("|lag change| between blocks", np.abs(dlag)), ("lag sd within block", lagsd), ("time", btm)):
    m = ok & np.isfinite(x); r, p = spearmanr(np.abs(jump[m]), x[m]); print(f"Spearman |jump| vs {name:30s}: r={r:+.3f} p={p:.3g} n={m.sum()}")
m = ok & np.isfinite(dlag); r, p = spearmanr(jump[m], dlag[m]); print(f"Spearman signed jump vs signed lag change: r={r:+.3f} p={p:.3g}")
np.savez("block_series.npz", btm=btm, P=P, jump=jump, lag=lag, lagsd=lagsd, dlag=dlag, el=el, Plo=bl(lo), Phi=bl(hi))
