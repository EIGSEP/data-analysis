"""Which tones' continuum estimates are safe to plot as sky modulation.

The rotation figure gains a second panel showing the continuum that is subtracted
from each tone -- the sky and receiver power, carried by the channels with no
injected signal in them. Plotted raw it is wrong: a dozen channels between 110
and 240 MHz swing 1-3 dB through the turn, far more than the sky does, because
they carry a terrestrial transmitter rather than sky.

The control is the stationary ground copy. It does not rotate, so anything it
also saw change over the same integrations changed in time rather than with
antenna orientation. A tone's continuum is plotted only where the ground copy is
steady on the same flanking channels.

This script fixes the threshold, shows that the quoted median does not depend on
it, and records the one thing the test cannot do.

Reduction is identical to build_nb.py; only the continuum panel is new.
"""
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path("/home/christian/Documents/research/eigsep/data-analysis")
SIDECAR = REPO / "notebooks/christian/deployment5/motor_scan_20260717_key4.h5"
KEY0 = REPO / "notebooks/christian/beam_modulation/key0_raster.npz"
OUT = Path(__file__).resolve().parent

COMB_RESIDUE = 8
COMB_SPILL = (7, 8, 9, 15, 0, 1)
FM_BAND = (86.0, 110.0)
BIN_DEG = 5.0
MAX_ROUGHNESS = 0.25
NSIG = 3.0

with h5py.File(SIDECAR, "r") as f:
    d4 = f["data4"][:].astype(np.float64)
    t = f["times"][:]
    deg = 1.0 / f.attrs["counts_per_deg"]
    az = f["az_counts"][:] * deg
    el = f["el_counts"][:] * deg
    freq = f["freqs"][:]

d4[d4 <= 0] = np.nan
nchan = d4.shape[1]
chan = np.arange(nchan)
is_fm = (freq > FM_BAND[0]) & (freq < FM_BAND[1])

d0 = np.load(KEY0)["d0"].astype(np.float64)
d0[d0 <= 0] = np.nan

turns = np.where(np.diff(np.sign(np.diff(el))) != 0)[0] + 1
rotations = [(a, b) for a, b in zip(np.r_[0, turns], np.r_[turns, len(el)]) if b - a > 100]
rot_az = np.array([np.nanmedian(az[a:b]) for a, b in rotations])
ROT = int(np.argmin(np.abs(rot_az - (-90.0))))
A, B = rotations[ROT]
finite = np.isfinite(d4[A:B]).mean(0) > 0.9

edges = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
centres = 0.5 * (edges[:-1] + edges[1:])
bin_idx = np.digitize(el, edges) - 1
ZERO = int(np.argmin(np.abs(centres)))


def to_db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(x)


def flanking(c):
    """Offsets +-3..+-6: eight non-comb channels, clear of the comb spillover.

    No FM condition, unlike the version this exploration started from. Only one
    tone in the band, 111.3 MHz, has a flanker inside the 86-110 MHz guard
    window; keeping the exception moves that one curve by 0.1 dB and costs a
    seven-channel special case in the caption.
    """
    return [c + o for o in range(-6, 7)
            if 3 <= abs(o) <= 6 and 0 <= c + o < nchan and finite[c + o]
            and (c + o) % 16 not in COMB_SPILL]


def binned(series, a, b):
    s = series[a:b]
    idx = bin_idx[a:b]
    mu = np.full(len(centres), np.nan)
    for j in range(len(centres)):
        v = s[(idx == j) & np.isfinite(s)]
        if v.size:
            mu[j] = np.median(v)
    return mu


def subtract(c):
    return d4[:, c] - np.nanmedian(d4[:, flanking(c)], axis=1)


def noise_sigma(c):
    sds = [np.nanstd(binned(subtract(cp), A, B)) for cp in flanking(c) if flanking(cp)]
    return np.nanmedian(sds) if sds else np.inf


def excess_profile(c):
    mu = binned(subtract(c), A, B)
    ok = mu > NSIG * noise_sigma(c)
    pos = mu > 0
    ref = np.nanmedian(mu[ZERO - 1:ZERO + 2])
    full = np.where(pos, to_db(np.where(pos, mu, np.nan) / ref), np.nan)
    return full, ok


def referenced(data, chans):
    """Binned power in `chans`, in dB relative to the value at 0 deg."""
    mu = binned(np.nanmedian(data[:, chans], axis=1), A, B)
    return to_db(mu / np.nanmedian(mu[ZERO - 1:ZERO + 2]))


def swing(v):
    return np.ptp(v[np.isfinite(v)])


tones = []
for c in chan[(chan % 16 == COMB_RESIDUE) & (freq > 50) & (freq < 250) & finite & ~is_fm]:
    nb = flanking(c)
    rough = np.nanmedian(np.abs(np.diff(binned(to_db(np.nanmedian(d4[:, nb], 1)), A, B), 2)))
    if rough >= MAX_ROUGHNESS:
        continue
    prof, ok = excess_profile(c)
    if not ok[ZERO] or ok.sum() < 20:
        continue
    tones.append(c)

print(f"rotation {ROT}: azimuth {rot_az[ROT]:+.0f} deg, {B - A} integrations, "
      f"{(t[B - 1] - t[A]):.0f} s")
print(f"{len(tones)} tones in panel (a), "
      f"{freq[tones[0]]:.1f}-{freq[tones[-1]]:.1f} MHz\n")

air = {c: referenced(d4, flanking(c)) for c in tones}     # suspended, rotating
gnd = {c: referenced(d0, flanking(c)) for c in tones}     # stationary control

# ------------------------------------------------- 1. where the threshold goes
print("-- ground-copy swing per tone, sorted (dB) --")
order = sorted(tones, key=lambda c: swing(gnd[c]))
vals = np.array([swing(gnd[c]) for c in order])
for c, v in zip(order, vals):
    print(f"   {freq[c]:7.1f} MHz   gnd {v:5.2f}   air {swing(air[c]):5.2f}")
jump = int(np.argmax(np.diff(vals)))
step = int(np.searchsorted(vals, 0.35))
print(f"\n   values run {vals[0]:.2f}-{vals[step - 1]:.2f} and then "
      f"{vals[step]:.2f}-{vals[-1]:.2f}: a step of "
      f"{vals[step] - vals[step - 1]:.2f} dB at the threshold.")
print(f"   Be careful how much weight that carries. It is NOT the largest break in")
print(f"   the distribution -- that is {vals[jump]:.2f} -> {vals[jump + 1]:.2f} dB, "
      f"after {jump + 1} tones -- and neighbouring")
print(f"   spacings are comparable, so 'the threshold sits in a natural gap' is more")
print(f"   than the data supports. The defence that does hold is the stability scan below.")

GND_MAX = 0.35
keep = [c for c in tones if swing(gnd[c]) < GND_MAX]
drop = [c for c in tones if swing(gnd[c]) >= GND_MAX]
kept_swing = np.array([swing(air[c]) for c in keep])
print(f"\n   GND_MAX = {GND_MAX}: {len(keep)} of {len(tones)} tones, "
      f"{freq[keep[0]]:.1f}-{freq[keep[-1]]:.1f} MHz")
print(f"   sky swing {kept_swing.min():.2f}-{kept_swing.max():.2f} dB "
      f"(median {np.median(kept_swing):.2f}, "
      f"{100 * (10 ** (np.median(kept_swing) / 10) - 1):.0f} per cent in power)")
print(f"   median ground-copy swing of the kept population is "
      f"{np.median([swing(gnd[c]) for c in keep]):.2f} dB, so the gap is only a "
      f"factor of {GND_MAX / np.median([swing(gnd[c]) for c in keep]):.1f} above it")

# ------------------------------------- 2. the quoted median against the choice
print("\n-- does the quoted median depend on the threshold? --")
print("   thr   tones   median sky swing   band")
for thr in (1.00, 0.60, 0.35, 0.25, 0.20, 0.15, 0.12):
    k = [c for c in tones if swing(gnd[c]) < thr]
    if not k:
        continue
    m = np.median([swing(air[c]) for c in k])
    print(f"   {thr:4.2f}   {len(k):5d}   {m:14.2f} dB   "
          f"{freq[k[0]]:5.1f}-{freq[k[-1]]:5.1f} MHz")
print("   The median moves by little across a factor of eight in threshold. The rise")
print("   at the tight end is frequency coverage, not cleanliness: only the low tones")
print("   survive there, and those have the deepest sky modulation anyway.")

# ------------------------------- 3. what the cut does and does not say about (a)
print("\n-- what the cut costs panel (a) --")


def margin_db(c, bins=None):
    """Tone power above its own continuum, in dB, per rotation-angle bin."""
    tone_b = binned(d4[:, c], A, B)
    cont_b = binned(np.nanmedian(d4[:, flanking(c)], axis=1), A, B)
    exc = tone_b - cont_b
    m = np.isfinite(exc) & np.isfinite(cont_b) & (exc > 0)
    if bins is not None:
        m &= bins
    return to_db(exc[m] / cont_b[m])


def worst_shift(c, margins):
    """Curve shift if the entire ground-copy swing were error on the continuum."""
    err = 10 ** (swing(gnd[c]) / 10) - 1.0
    return abs(to_db(1.0 + err / 10 ** (margins / 10))).max()


def margin_at_zero(c):
    cont_b = binned(np.nanmedian(d4[:, flanking(c)], axis=1), A, B)
    exc0 = binned(d4[:, c], A, B)[ZERO] - cont_b[ZERO]
    return to_db(exc0 / cont_b[ZERO])


peak = np.array([margin_at_zero(c) for c in drop])
print(f"   at 0 deg, the response peak, the {len(drop)} discarded tones sit "
      f"{peak.min():.1f}-{peak.max():.1f} dB above their own continuum,")
print(f"   so there the whole ground-copy swing moves those curves by at most "
      f"{max(worst_shift(c, np.array([m])) for c, m in zip(drop, peak)):.3f} dB.")

print("\n   BUT that is the most favourable point of the turn, and it is not the")
print("   number the argument needs. Over the bins panel (a) actually draws and")
print("   quotes depths from, the margin closes as the tone goes into its null:")
sig_lo, sig_shift = [], []
for c in drop:
    _, ok = excess_profile(c)
    mg = margin_db(c, bins=ok)
    if mg.size:
        sig_lo.append((freq[c], mg.min(), worst_shift(c, mg)))
sig_lo.sort(key=lambda r: r[1])
print(f"   margin over significant bins : {min(r[1] for r in sig_lo):.1f} to "
      f"{max(r[1] for r in sig_lo):.1f} dB")
print(f"   worst-case curve shift       : {max(r[2] for r in sig_lo):.2f} dB")
print("   tightest three:")
for f_, lo, sh in sig_lo[:3]:
    print(f"      {f_:7.1f} MHz   margin {lo:5.1f} dB   shift up to {sh:.2f} dB")
print("\n   So the honest claim is bounded to the top of the range: the contamination")
print("   the cut removes is negligible against panel (a)'s signal near the peak, and")
print("   is NOT negligible near the null. That is consistent with what the paper")
print("   already says -- below 158 MHz the tone reaches the noise before the null")
print("   does and those depths are reported as upper limits -- but it means the")
print("   claim must be stated at the peak, not for the curve as a whole.")

# ------------------------------------ 4. what the test cannot do: steady + fixed
print("\n-- the blind spot: a source steady in time but fixed in direction --")
print("   It is constant on a stationary receiver, so it passes, yet a rotating")
print("   antenna modulates it exactly like a point source. The aeronautical band")
print("   shows both halves, and the ground copy splits them where the allocation does:")
print("\n   band              suspended   ground copy")
for lo, hi, what in ((70.0, 86.0, "clean sky"),
                     (110.0, 118.0, "navigation beacons, continuous"),
                     (118.0, 137.0, "airband voice, intermittent"),
                     (137.0, 145.0, "back to normal")):
    sel = [c for c in tones if lo <= freq[c] < hi]
    if not sel:
        continue
    print(f"   {lo:5.0f}-{hi:5.0f} MHz   {np.median([swing(air[c]) for c in sel]):6.2f} dB"
          f"   {np.median([swing(gnd[c]) for c in sel]):6.2f} dB   {what}")
print("\n   channel by channel on the ground copy, through the transition")
print("   (the continuum channels themselves, not the tones they feed):")
band_ch = [c for c in chan if 109.0 <= freq[c] <= 122.0
           and finite[c] and c % 16 not in COMB_SPILL]
for c in band_ch:
    v = referenced(d0, [c])
    print(f"      {freq[c]:7.2f} MHz   gnd {swing(v):5.2f}")
print("   The intermittent half is dropped; the continuous half is not, which is why")
print("   the deepest curves left in the panel sit just above 110 MHz.")
low = [c for c in tones if 110.0 <= freq[c] < 118.0]
print(f"   Not one channel dragging a median: every tone from 110 to 118 MHz modulates")
print(f"   {min(swing(air[c]) for c in low):.1f}-{max(swing(air[c]) for c in low):.1f} dB, "
      f"decaying with distance from the band, so an outlier test over the eight")
print("   flankers would find nothing.")

# ----------------------------------------------------------------- the figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for c in keep:
    ax[0].plot(centres, air[c], color="C0", lw=0.8, alpha=0.8)
for c in drop:
    ax[0].plot(centres, air[c], color="C3", lw=0.8, alpha=0.8)
ax[0].set_xlim(-180, 180)
ax[0].set_xticks([-180, -90, 0, 90, 180])
ax[0].axhline(0.0, color="0.75", lw=0.5, ls=":")
ax[0].grid(alpha=0.25, lw=0.4)
ax[0].set_xlabel("Platform rotation angle [deg]")
ax[0].set_ylabel("Continuum estimate rel. $0^\\circ$ [dB]")
ax[0].set_title(f"blue: kept ({len(keep)})   red: ground copy moved too ({len(drop)})",
                fontsize=9)

ax[1].semilogy(freq[tones], [swing(gnd[c]) for c in tones], "o", ms=4, color="0.3")
ax[1].axhline(GND_MAX, color="C3", lw=1.0, ls="--")
ax[1].text(60, GND_MAX * 1.1, f"GND_MAX = {GND_MAX} dB", fontsize=8, color="C3")
ax[1].axvspan(110, 137, color="C1", alpha=0.15)
ax[1].text(112, 0.045, "aeronautical", fontsize=8, color="C1")
ax[1].grid(alpha=0.25, lw=0.4, which="both")
ax[1].set_xlabel("Frequency [MHz]")
ax[1].set_ylabel("Ground-copy swing over the turn [dB]")

fig.tight_layout()
fig.savefig(OUT / "61_ground_copy_cut.png", dpi=140)
print(f"\nsaved {OUT / '61_ground_copy_cut.png'}")
