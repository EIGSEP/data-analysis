"""What drives the sub-hour gain? Common-mode 20-min log-gain knots of a global fit (rx-A,
temperature regressor already separated), minus a 3-h smooth, against housekeeping streams
projected onto the same knots, in two disjoint stretches; and event-triggered averages of the
3-h-knot fit's residual around calibration cycles and pointing moves.
usage: gain_drivers.py FIT20.json FIT3H_resid.npz BINNED.npz HK.npz (from hk_streams.py) [OUT.json]"""
import sys, json
import numpy as np
J = json.load(open(sys.argv[1])); R3 = np.load(sys.argv[2]); B = np.load(sys.argv[3]); M = np.load(sys.argv[4], allow_pickle=True)
FREQ = [f for f in (117.2, 125.0, 132.8, 140.6, 148.4, 156.2, 164.1, 171.9, 179.7, 203.1)]
res = {round(r['freq_mhz'], 1): r['all'] for r in J['results']}
OUT = {'corr': {}, 'events': {}, 'regress': {}}
use = [f for f in FREQ if not res[f]['diverged']]
# --- common-mode fast log gain at the knots, per rx-A segment
gk = res[use[0]]['gain_knots']; seg = np.array(gk['segment']); kt = np.array(gk['t'])
sel = np.array([s.startswith('rx-A') for s in seg])
kt, seg = kt[sel], seg[sel]
L = np.array([np.log(np.array(res[f]['gain_knots']['gain'])[sel]) for f in use])
for s in np.unique(seg):
    m = seg == s; L[:, m] -= L[:, m].mean(1, keepdims=True)
def detrend(x, tk, sg, knot=3 * 3600.0):
    """x minus its piecewise-linear fit with knots every `knot` s, per segment."""
    out = np.full_like(x, np.nan)
    for s in np.unique(sg):
        m = np.flatnonzero((sg == s) & np.isfinite(x))
        if m.size < 4: continue
        t = tk[m]; k = np.arange(t.min(), t.max() + knot, knot)
        A = np.stack([np.interp(t, k, np.eye(k.size)[j]) for j in range(k.size)], 1)
        c, *_ = np.linalg.lstsq(A, x[m], rcond=None); out[m] = x[m] - A @ c
    return out
Lf = np.array([detrend(l, kt, seg) for l in L])
cm = np.nanmean(Lf, 0)
half_a, half_b = np.nanmean(Lf[0::2], 0), np.nanmean(Lf[1::2], 0)
ok = np.isfinite(half_a) & np.isfinite(half_b)
print(f'frequencies {use}; {kt.size} rx-A knots')
print(f'fast common-mode log gain: rms {100 * np.nanstd(cm):.2f}%; corr between disjoint frequency halves {np.corrcoef(half_a[ok], half_b[ok])[0, 1]:+.2f}')
OUT.update(freqs=use, n_knots=int(kt.size), fast_rms=float(np.nanstd(cm)), halves_corr=float(np.corrcoef(half_a[ok], half_b[ok])[0, 1]),
           knot_t=kt.tolist(), knot_seg=seg.tolist(), fast_cm=cm.tolist())
# --- housekeeping per 60-s bin, then projected onto the knots with triangular weights
tm = M['t']; i0 = np.searchsorted(tm, B['t0']); i1 = np.searchsorted(tm, B['t1'], 'right')
def binmean(v):
    c = np.r_[0, np.nancumsum(np.where(np.isfinite(v), v, 0))]; n = np.r_[0, np.cumsum(np.isfinite(v))]
    with np.errstate(invalid='ignore', divide='ignore'):
        return (c[i1] - c[i0]) / (n[i1] - n[i0])
d = lambda v: np.r_[0, np.abs(np.diff(v))]
moving_az = (np.r_[0, np.abs(np.diff(M['motor_az_pos']))] > 0).astype(float)
moving_el = (np.r_[0, np.abs(np.diff(M['motor_el_pos']))] > 0).astype(float)
cal = np.isin(M['rfswitch'], ['RFAMB', 'RFNON', 'RFNOFF', 'RFSP1']).astype(float)
vna = np.char.startswith(M['rfswitch'].astype(str), 'VNA').astype(float)
streams = {
    'therm0 (switch)': M['rfswitch_therm_temp_therm0'], 'therm1': M['rfswitch_therm_temp_therm1'],
    'therm2': M['rfswitch_therm_temp_therm2'], 'load T': M['tempctrl_load_T_now'],
    'load heater active': M['tempctrl_load_active'], 'system current': M['system_current_current_a'],
    'accel x': M['imu_el_accel_x'], 'accel y': M['imu_el_accel_y'], 'accel z': M['imu_el_accel_z'],
    '|d accel x|': d(M['imu_el_accel_x']), 'imu el': M['imu_el_el_deg'], 'lidar distance': M['lidar_distance_m'],
    'az motor moving': moving_az, 'el motor moving': moving_el, 'pot az voltage': M['potmon_pot_az_voltage'],
}
binned = {k: binmean(v) for k, v in streams.items()}
# Switch states other than the antenna are not in antenna bins: count them over each bin's span and the gap before it.
cum = lambda v: np.r_[0, np.cumsum(v)]
gap0 = np.searchsorted(tm, np.r_[B['t0'][0], B['t1'][:-1]])
for name, v in (('cal-cycle rows before bin', cal), ('VNA rows before bin', vna)):
    c = cum(v); binned[name] = (c[i0] - c[gap0]).astype(float)
binned['flag occupancy'] = np.nanmean(B['nsamp'] / B['n'][:, None], 1)
t = B['t']; rxA = B['regime'] == 'rx-A'
segb = np.array([f'{r}|{int(round(x * 1e3))}' for r, x in zip(B['regime'], B['tau'])])
def onknots(x):
    out = np.full(kt.size, np.nan)
    for j in range(kt.size):
        same = np.flatnonzero(seg == seg[j])
        lo = kt[same[same < j][-1]] if (same < j).any() else kt[j] - 1200.0
        hi = kt[same[same > j][0]] if (same > j).any() else kt[j] + 1200.0
        m = rxA & (segb == seg[j]) & (t >= lo) & (t <= hi) & np.isfinite(x)
        if m.sum() < 3: continue
        w = np.where(t[m] < kt[j], (t[m] - lo) / (kt[j] - lo), (hi - t[m]) / (hi - kt[j]))
        out[j] = np.sum(w * x[m]) / np.sum(w)
    return out
stretch = {'07-15/16 (30 m)': kt < np.datetime64('2026-07-16T20:00').astype('datetime64[s]').astype(int),
           '07-17 (30/87.5/91 m)': kt >= np.datetime64('2026-07-16T20:00').astype('datetime64[s]').astype(int)}
print('\ncorr of fast common-mode log gain with each stream (both detrended with 3-h knots), lag 0 and best of +-2 knots')
print(f'{"stream":28s}' + ''.join(f'{k:>34s}' for k in stretch))
for name, x in binned.items():
    xk = detrend(onknots(x), kt, seg)
    row = f'{name:28s}'
    for sn, sm in stretch.items():
        m = sm & np.isfinite(xk) & np.isfinite(cm)
        if m.sum() < 10 or np.nanstd(xk[m]) == 0:
            row += f'{"--":>34s}'; continue
        r0 = np.corrcoef(cm[m], xk[m])[0, 1]
        OUT['corr'].setdefault(name, {})[sn] = {'r': float(r0), 'n': int(m.sum())}
        idx = np.flatnonzero(m); best = (0, r0)
        for lag in (-2, -1, 1, 2):
            j = idx + lag; ok2 = (j >= 0) & (j < kt.size)
            a, b = cm[idx[ok2]], xk[j[ok2]]; f2 = np.isfinite(a) & np.isfinite(b)
            if f2.sum() > 10:
                r = np.corrcoef(a[f2], b[f2])[0, 1]
                if abs(r) > abs(best[1]): best = (lag, r)
        OUT['corr'][name][sn].update(best_r=float(best[1]), best_lag=int(best[0]))
        row += f'   r {r0:+.2f} (n {m.sum():3d}) best {best[1]:+.2f}@{best[0]:+d}'
    print(row)
# --- event-triggered averages of the 3-h fit's common-mode residual (1-min bins)
f3 = np.round(R3['freqs_mhz'], 1); jj = [list(f3).index(f) for f in use if f in f3]
q = np.nanmean(R3['resid_frac'][jj], 0); tq = R3['t']; okq = R3['kept'] & np.isfinite(q) & (R3['regime'] == 'rx-A')
def events(mask, min_gap=600):
    on = np.flatnonzero(np.diff(mask.astype(int)) == -1) + 1          # end of a non-antenna stretch
    ev = tm[on]; return ev[np.r_[True, np.diff(ev) > min_gap]] if ev.size else ev
mv = (moving_az + moving_el) > 0
evs = {'cal-cycle end': events(cal > 0), 'VNA end': events(vna > 0), 'motor move end': events(mv)}
rng = np.random.default_rng(0)
print('\nevent-triggered mean of the 3-h fit residual (mean 117-203 MHz, %): window relative to the event')
wins = [(-30, -10), (-10, 0), (0, 10), (10, 30), (30, 60)]
print(f'{"event":18s}  n   ' + ''.join(f'{f"[{a},{b}) min":>13s}' for a, b in wins))
for name, ev in list(evs.items()) + [('random times', rng.uniform(tq[okq].min(), tq[okq].max(), 300))]:
    row = []; n = 0
    for a, b in wins:
        vals = [np.mean(q[okq & (tq >= e + 60 * a) & (tq < e + 60 * b)]) for e in ev
                if (okq & (tq >= e + 60 * a) & (tq < e + 60 * b)).sum() >= 3]
        row.append(f'{100 * np.mean(vals):+6.2f}±{100 * np.std(vals) / max(np.sqrt(len(vals)), 1):.2f}' if vals else '     --     ')
        n = max(n, len(vals))
        OUT['events'].setdefault(name, []).append({'win_min': [a, b], 'mean': float(np.mean(vals)) if vals else None,
            'sem': float(np.std(vals) / max(np.sqrt(len(vals)), 1)) if vals else None, 'n': len(vals)})
    print(f'{name:18s} {n:3d}   ' + ' '.join(row))

# --- fast gain regressed on the fast switch temperature (and its rate), per stretch
X = {k: detrend(onknots(binned[k]), kt, seg) for k in ('therm0 (switch)', 'therm1')}
X['d therm0/dt'] = detrend(np.r_[0, np.diff(onknots(binned['therm0 (switch)']))] / 20.0, kt, seg)
print('\nfast common-mode log gain regressed on fast temperatures (3-h detrended): slope and R2')
for sn, sm in list(stretch.items()) + [('both', np.ones(kt.size, bool))]:
    for names in (['therm0 (switch)'], ['therm1'], ['therm0 (switch)', 'd therm0/dt']):
        A = np.stack([X[n] for n in names], 1); m = sm & np.isfinite(cm) & np.isfinite(A).all(1)
        Am = np.c_[A[m], np.ones(m.sum())]
        c, *_ = np.linalg.lstsq(Am, cm[m], rcond=None)
        r2 = 1 - np.var(cm[m] - Am @ c) / np.var(cm[m])
        OUT['regress'].setdefault(sn, {})['+'.join(names)] = {'slopes': c[:-1].tolist(), 'r2': float(r2), 'n': int(m.sum()),
            'fast_T_rms': float(np.nanstd(X[names[0]][m]))}
        print(f'  {sn:22s} {"+".join(names):30s} n {m.sum():3d}  R2 {r2:.2f}  ' +
              ' '.join(f'{100 * x:+.2f}%/{"(K/min)" if "dt" in n else "K"}' for x, n in zip(c, names)))

X0 = X['therm0 (switch)']
OUT['fast_therm0'] = [None if not np.isfinite(v) else float(v) for v in X0]
OUT['stretch_of_knot'] = [next(k for k, v in stretch.items() if v[j]) for j in range(kt.size)]
if len(sys.argv) > 5:
    json.dump(OUT, open(sys.argv[5], 'w'))
