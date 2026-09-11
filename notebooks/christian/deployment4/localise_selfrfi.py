"""Gain-free localisation of the deployment-4 duty-cycled self-RFI emitter,
plus the duty-cycle timing signature.

Two analyses that were NOT in the existing notebooks:

1. **Comb-referenced emitter power.** `noise_dropouts_selfRFI.ipynb` compares
   antennas by the *fraction of each antenna's own total power* the emitter
   contributes. That is not a field-strength comparison — it folds in each
   antenna's bandpass, gain and sky coupling. `radiated_selfRFI_lines.ipynb`
   does it correctly for the narrowband lines, by referencing the line power to
   the received calibration comb, which cancels receiver gain. This applies the
   same gain-free reference to the broadband duty-cycled emitter.

   Because the comb is radiated from the *ground* transmitter, the ratio is
   antenna-independent for a source beside the transmitter, and much larger on
   the bowtie for a source on the platform.

2. **Duty-cycle timing.** Whether the interval between off-windows is a fixed
   period (digital scheduler) or varies with thermal load (thermostat).

Inputs: `cache_dropout_{metadata,passA,passB}.npz`, built by
`noise_dropouts_selfRFI.ipynb`. No raw data needed.

viv2 (correlator key 0) is excluded throughout: its front-end is dead (~37 dB
low, both pols, no comb visible), so its flatness is an artefact and proves
nothing about near-field geometry.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.ndimage import median_filter

MTN = ZoneInfo("America/Denver")
D4 = "/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment4"
SWAP = datetime(2025, 7, 18, 18, 0, tzinfo=MTN).timestamp()
DF = 0.244140625

# bowtie = suspended; ground = viv1-N; 3 = viv1-E (two pols of one ground Vivaldi)
ANTS = ["bowtie", "ground", "3"]
LABEL = {"bowtie": "suspended bowtie", "ground": "viv1-N (ground)", "3": "viv1-E (ground)"}


def antenna_mask(t, sw_t, sw_s, guard=5.0):
    ix = np.clip(np.searchsorted(sw_t, t, side="right") - 1, 0, len(sw_t) - 1)
    ant = sw_s[ix] == 0
    chg = sw_t[1:][np.diff(sw_s) != 0]
    j = np.searchsorted(chg, t)
    for off in (-1, 0):
        jj = np.clip(j + off, 0, len(chg) - 1)
        ant &= np.abs(t - chg[jj]) > guard
    return ant


# =========================================================== 1. localisation ==
z = np.load(f"{D4}/cache_dropout_passB.npz")
nchan = z["sin_bowtie"].shape[1]
freq = np.arange(nchan) * DF
comb = np.arange(nchan) % 16 == 0      # deployment-4 comb sits on residue 0
nontx = ~comb
ok = {p: (z[f"cin_{p}"].min(axis=1) >= 50) & (z[f"cout_{p}"].min(axis=1) >= 100)
      for p in ANTS}

emitter, combpow = {}, {}
for p in ANTS:
    on = np.nanmedian(z[f"sout_{p}"][ok[p]] / z[f"cout_{p}"][ok[p]], axis=0)
    off = np.nanmedian(z[f"sin_{p}"][ok[p]] / z[f"cin_{p}"][ok[p]], axis=0)
    emitter[p] = on - off
    # comb power = comb channel minus the mean of its non-comb neighbours
    cont = np.array([np.nanmean(on[max(0, c - 6):c + 7][nontx[max(0, c - 6):c + 7]])
                     for c in np.nonzero(comb)[0]])
    combpow[p] = on[comb] - cont

comb_idx = np.nonzero(comb)[0]


def ratio(p, f0, f1):
    b = nontx & (freq >= f0) & (freq < f1)
    sel = (freq[comb_idx] >= f0) & (freq[comb_idx] < f1)
    pc = np.nansum(combpow[p][sel])
    return np.nansum(emitter[p][b]) / pc if pc > 0 else np.nan


print("=" * 74)
print("1. GAIN-FREE LOCALISATION — emitter power referenced to the received comb")
print("=" * 74)
print("The comb is radiated from the GROUND transmitter. A source beside it gives")
print("an antenna-independent ratio; a source on the platform gives a much larger")
print("ratio on the bowtie.\n")
print(f"{'band [MHz]':>12} {'bowtie':>9} {'viv1-N':>9} {'viv1-E':>9}   nearer")
for f0 in range(30, 170, 20):
    r = {p: ratio(p, f0, f0 + 20) for p in ANTS}
    g = np.nanmean([r["ground"], r["3"]])
    v = "platform" if r["bowtie"] > 3 * g else ("GROUND" if g > 3 * r["bowtie"] else "ambiguous")
    print(f"{f0:5d}-{f0 + 20:<6d} {r['bowtie']:9.3f} {r['ground']:9.3f} {r['3']:9.3f}   {v}")

print()
for f0, f1 in [(40, 65), (100, 150), (30, 160)]:
    r = {p: ratio(p, f0, f1) for p in ANTS}
    g = np.nanmean([r["ground"], r["3"]])
    print(f"  {f0}-{f1} MHz: bowtie/ground = {r['bowtie'] / g:.1f}x")

print("\n  For comparison, the same statistic for the 244 MHz clock line")
print("  (radiated_selfRFI_lines.ipynb): bowtie 0.0017, viv1-E 1.14, viv1-N 0.26")
print("  -> ~700x stronger on the ground antennas.")

# ================================================================ 2. timing ==
m = np.load(f"{D4}/cache_dropout_metadata.npz")
sw_t, sw_s = m["sw_t"], m["sw_s"]
a = np.load(f"{D4}/cache_dropout_passA.npz")
times_c, bow2, bow4 = a["times_c"], a["bow2"], a["bow4"]
ant = antenna_mask(times_c, sw_t, sw_s)
bow = np.where(times_c < SWAP, bow4, bow2).astype(float)
bow[~ant] = np.nan
good = np.isfinite(bow)
fi = np.interp(times_c, times_c[good], bow[good])


def census(win_min, thr=-0.7):
    base = median_filter(fi, size=int(win_min * 60 / 1.07), mode="nearest")
    with np.errstate(all="ignore"):
        r = 10 * np.log10(fi / base)
    r[~good] = np.nan
    low = np.nan_to_num(r) < thr
    d = np.diff(low.astype(int))
    st, en = np.nonzero(d == 1)[0] + 1, np.nonzero(d == -1)[0] + 1
    runs = []
    for s, e in zip(st, en):
        if runs and times_c[s] - times_c[runs[-1][1]] < 120:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    return np.array([(times_c[s], times_c[e] - times_c[s])
                     for s, e in runs if 120 < times_c[e] - times_c[s] < 600])


def hours(ts):
    return np.array([datetime.fromtimestamp(t, MTN).hour
                     + datetime.fromtimestamp(t, MTN).minute / 60 for t in ts])


print("\n" + "=" * 74)
print("2. DUTY-CYCLE TIMING — scheduler or thermal load?")
print("=" * 74)
print("A digital scheduler gives a tight, time-of-day-invariant period. A")
print("thermostat cycles faster under higher heat load.\n")
print(f"{'detrend':>8} {'N':>4} |  median start-to-start interval [min] by local time")
print(f"{'window':>8} {'':>4} |   00-06    06-12    12-18    18-24")
for win in (15, 25, 35, 60, 90):
    ev = census(win)
    h = hours(ev[:, 0])
    g = np.diff(ev[:, 0]) / 60.0
    gh, sel = h[:-1], np.diff(ev[:, 0]) / 60.0 < 120
    row = []
    for h0 in (0, 6, 12, 18):
        k = sel & (gh >= h0) & (gh < h0 + 6)
        row.append(f"{np.median(g[k]):7.1f}" if k.sum() > 2 else "      -")
    print(f"{win:>6} m {len(ev):>4} | {' '.join(row)}")

ev = census(35)
h = hours(ev[:, 0])
g = np.diff(ev[:, 0]) / 60.0
gh, sel = h[:-1], g < 120
print("\nNight only (20:30-09:30 MDT, free of human activity), 35 min window:")
for lo, hi, lbl in [(20.5, 24, "20:30-24:00"), (0, 3, "00:00-03:00"),
                    (3, 6, "03:00-06:00"), (6, 9.5, "06:00-09:30")]:
    k = sel & (gh >= lo) & (gh < hi)
    if k.sum() > 1:
        print(f"  {lbl}  n={k.sum():2d}  median interval {np.median(g[k]):5.1f} min")

night = (h >= 20.5) | (h <= 9.5)
print(f"\nOFF duration is nearly constant while the interval varies ~4x:")
for lbl, k in [("night", night), ("day  ", ~night)]:
    print(f"  {lbl}  median {np.median(ev[k, 1]):3.0f} s  "
          f"IQR [{np.percentile(ev[k, 1], 25):.0f}, {np.percentile(ev[k, 1], 75):.0f}]")

# ============================================================ 3. tempctrl ===
tc_v = m["tc_v"]
print("\n" + "=" * 74)
print("3. ON-PLATFORM THERMAL CONTROLLER")
print("=" * 74)
print(f"  {len(m['tc_t'])} reports, columns [A_drive, A_active, B_drive, B_active]")
print(f"  unique rows: {np.unique(tc_v, axis=0).tolist()}")
print(f"  value changes across the deployment: "
      f"{np.count_nonzero(np.any(np.diff(tc_v, axis=0) != 0, axis=1))}")
print("  -> the platform's own thermal control never ran. Excluded as the emitter.")
print(f"\n  motor stream: {len(m['mot_t'])} record(s) -> no motor power-state telemetry")
