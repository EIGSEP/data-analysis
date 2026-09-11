"""Standalone reproduction of the deployment-4 self-RFI floor numbers.

Re-derives, from the cached arrays only, the quantities the manuscript would
quote:

  * the event census and the duty cycle (floor present ~89 per cent of the time)
  * the floor as a fraction of total measured power, per antenna, per band

The live notebook is gone; this rebuilds cells 17 and 20 of
`noise_dropouts_selfRFI-checkpoint.ipynb` from `cache_dropout_{metadata,passA,
passB}.npz`.

SCOPE OF THE CHECK: the caches were built from the raw deployment-4 correlator
files, which are not on this machine. So this verifies the *reduction* --- the
detrending, thresholding, event selection, per-event paired baselining and the
median across events --- and confirms the prose matches what the code produces.
It does NOT re-verify the extraction from raw HDF5.
"""

from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.ndimage import median_filter

MTN = ZoneInfo("America/Denver")
D4 = "/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment4"
CKPT = D4  # caches live beside the notebook, not in the checkpoint dir
SWAP = datetime(2025, 7, 18, 18, 0, tzinfo=MTN).timestamp()
DF = 0.244140625  # MHz per channel


def loc(t, fmt="%a %m-%d %H:%M:%S"):
    return datetime.fromtimestamp(t, tz=MTN).strftime(fmt)


def antenna_mask(times, sw_t, sw_s, guard=5.0):
    """sw_state == 0 (antenna), away from switch transitions."""
    ix = np.clip(np.searchsorted(sw_t, times, side="right") - 1, 0, len(sw_t) - 1)
    ant = sw_s[ix] == 0
    chg_t = sw_t[1:][np.diff(sw_s) != 0]
    j = np.searchsorted(chg_t, times)
    for off in (-1, 0):
        jj = np.clip(j + off, 0, len(chg_t) - 1)
        ant &= np.abs(times - chg_t[jj]) > guard
    return ant


# arp's by-eye list, copied verbatim from the notebook, used only as a recall check
ARP = [
    (1752826741, 240), (1752829296, 239), (1752836820, 240), (1752839293, 244),
    (1752847809, 239), (1752849929, 239), (1752998738, 239), (1753002924, 239),
    (1753006892, 266), (1753008681, 239), (1753010172, 240), (1753011706, 257),
    (1753013096, 240), (1753014466, 240), (1753016215, 240), (1753017894, 239),
    (1753019562, 248), (1753021191, 239), (1753023264, 241),
]

meta = np.load(f"{CKPT}/cache_dropout_metadata.npz")
sw_t, sw_s = meta["sw_t"], meta["sw_s"]

passA = np.load(f"{CKPT}/cache_dropout_passA.npz")
times_c, bow2, bow4 = passA["times_c"], passA["bow2"], passA["bow4"]

print(f"metadata: {len(sw_t)} rfswitch reports")
print(f"passA:    {len(times_c)} integrations, "
      f"{loc(times_c[0])} -> {loc(times_c[-1])}")

# ---------------------------------------------------------------- cell 17 ----
ant_c = antenna_mask(times_c, sw_t, sw_s)
bow = np.where(times_c < SWAP, bow4, bow2).astype(float)
bow[~ant_c] = np.nan
good = np.isfinite(bow)
fi = np.interp(times_c, times_c[good], bow[good])
base = median_filter(fi, size=int(35 * 60 / 1.07), mode="nearest")
with np.errstate(all="ignore"):
    r_det = 10 * np.log10(fi / base)
r_det[~good] = np.nan

low = np.nan_to_num(r_det) < -0.7
dd_ = np.diff(low.astype(int))
starts, ends = np.nonzero(dd_ == 1)[0] + 1, np.nonzero(dd_ == -1)[0] + 1
runs = []
for s, e in zip(starts, ends):
    if runs and times_c[s] - times_c[runs[-1][1]] < 120:
        runs[-1][1] = e
    else:
        runs.append([s, e])
events = np.array([
    (times_c[s], times_c[e] - times_c[s],
     np.nanmedian(np.where(r_det[s:e] < -0.7, r_det[s:e], np.nan)))
    for s, e in runs if 120 < times_c[e] - times_c[s] < 600
])

hours = np.array([
    datetime.fromtimestamp(t, tz=MTN).hour + datetime.fromtimestamp(t, tz=MTN).minute / 60
    for t in events[:, 0]
])
is_night = (hours >= 20.5) | (hours <= 9.5)
matched = sum(bool(np.any(np.abs(events[:, 0] - t0) < 150)) for t0, _ in ARP)
gaps = np.diff(events[:, 0]) / 60
per_day = Counter(loc(t, "%a %m-%d") for t in events[:, 0])

print("\n--- census (cell 17) ---")
print(f"{len(events)} events; per day: "
      f"{dict(sorted(per_day.items(), key=lambda kv: kv[0][4:]))}")
print(f"{is_night.sum()} at night (20:30-09:30 MDT), {(~is_night).sum()} in work hours")
print(f"duration: median {np.median(events[:, 1]):.0f} s, "
      f"10-90%: [{np.percentile(events[:, 1], 10):.0f}, "
      f"{np.percentile(events[:, 1], 90):.0f}] s")
print(f"depth (dip band): median {np.nanmedian(events[:, 2]):.2f} dB")
print(f"gaps between events: median {np.median(gaps[gaps < 120]):.0f} min (of gaps < 2 h)")
print(f"arp events recovered: {matched}/{len(ARP)}")

# duty cycle: fraction of antenna-state time inside a detected event
in_ev = np.zeros(len(times_c), bool)
for t0, dur, _ in events:
    in_ev |= (times_c >= t0) & (times_c <= t0 + dur)
night_s = (hours[:0], )  # placeholder, recomputed below per-sample
samp_hours = np.array([
    datetime.fromtimestamp(t, tz=MTN).hour + datetime.fromtimestamp(t, tz=MTN).minute / 60
    for t in times_c
])
samp_night = (samp_hours >= 20.5) | (samp_hours <= 9.5)
duty_all = in_ev[ant_c].mean()
duty_night = in_ev[ant_c & samp_night].mean()
print(f"\nfraction of antenna-state time in dropout (floor OFF): "
      f"{100 * duty_all:.1f}% overall, {100 * duty_night:.1f}% in night hours "
      f"-> floor PRESENT ~{100 * (1 - duty_all):.0f}% of the time")

# ---------------------------------------------------------------- cell 20 ----
PHYS = ["bowtie", "ground", "0", "3"]
passB = np.load(f"{CKPT}/cache_dropout_passB.npz")
S_IN = {p: passB[f"sin_{p}"] for p in PHYS}
S_OUT = {p: passB[f"sout_{p}"] for p in PHYS}
C_IN = {p: passB[f"cin_{p}"] for p in PHYS}
C_OUT = {p: passB[f"cout_{p}"] for p in PHYS}

nE_cached = S_IN["bowtie"].shape[0]
nchan = S_IN["bowtie"].shape[1]
freq = np.arange(nchan) * DF
nontx = np.arange(nchan) % 16 != 0  # deployment-4 comb sits on residue 0

print(f"\n--- level (cell 20) ---")
print(f"passB: {nE_cached} night events x {nchan} channels")
if nE_cached == int(is_night.sum()):
    print(f"CONSISTENT: cached event count matches the {is_night.sum()} night "
          f"events reproduced above")
else:
    print(f"*** MISMATCH: cache has {nE_cached}, reproduction gives "
          f"{is_night.sum()} -- the cache is stale relative to this census ***")

FRAC = {}
for p in PHYS:
    with np.errstate(all="ignore"):
        fr = 1 - (S_IN[p] / C_IN[p]) / (S_OUT[p] / C_OUT[p])
    ok_ev = (C_IN[p].min(axis=1) >= 50) & (C_OUT[p].min(axis=1) >= 100)
    FRAC[p] = (np.nanmedian(fr[ok_ev], axis=0), ok_ev.sum())

print()
for f0, f1 in [(40, 65), (110, 150)]:
    b = nontx & (freq >= f0) & (freq <= f1)
    for p in ["bowtie", "3", "0"]:
        frac, nok = FRAC[p]
        print(f"{p:>7} {f0}-{f1} MHz: median {100 * np.nanmedian(frac[b]):5.1f}%, "
              f"peak {100 * np.nanmax(frac[b]):5.1f}% of total power  "
              f"({nok} events)")

# extra: the broadest defensible single band, and the 40-150 claim
print()
for f0, f1 in [(40, 150), (50, 250)]:
    b = nontx & (freq >= f0) & (freq <= f1)
    frac, _ = FRAC["bowtie"]
    print(f" bowtie {f0}-{f1} MHz (incl. FM): median "
          f"{100 * np.nanmedian(frac[b]):5.1f}%, "
          f"peak {100 * np.nanmax(frac[b]):5.1f}%")

# where the floor is actually detectable at all
frac_b, _ = FRAC["bowtie"]
det = nontx & (frac_b > 0.02)
if det.any():
    print(f"\n bowtie: floor > 2% of total power over "
          f"{freq[det].min():.0f}-{freq[det].max():.0f} MHz "
          f"({det.sum()} of {nontx.sum()} non-comb channels)")

# --------------------------------------------------------------- band shape --
# The prose "of order ten per cent" is a summary of two bands. Check how it
# behaves across the manuscript's actual 50-250 MHz operating band, and answer
# the reviewer's fourth descriptor (impact on usable observing bandwidth)
# directly.
print("\n--- floor vs frequency, bowtie (median over 38 night events) ---")
for f0 in range(50, 250, 25):
    b = nontx & (freq >= f0) & (freq < f0 + 25)
    print(f"  {f0:3d}-{f0 + 25:3d} MHz: median {100 * np.nanmedian(frac_b[b]):5.1f}%  "
          f"peak {100 * np.nanmax(frac_b[b]):5.1f}%")

band = nontx & (freq >= 50) & (freq <= 250)
for thr in (0.01, 0.02, 0.05, 0.10):
    hit = band & (frac_b > thr)
    print(f"\n  floor > {100 * thr:4.1f}% of total power in "
          f"{100 * hit.sum() / band.sum():5.1f}% of non-comb channels over 50-250 MHz")
