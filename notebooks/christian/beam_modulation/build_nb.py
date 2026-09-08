import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Rotation response with signal injection — 2026-07-17 motor raster

Figure for the EIGSEP instrument paper: **how the deployed instrument responds to platform
rotation while an external signal is injected**. Both are novelties of the experiment.

## Setup

The 2026-07-17 motor raster (`run_tag == "motor_scan"`, 20:28–21:28 UTC) rotated the
suspended antenna through 53 full $\pm180^\circ$ turns of the elevation axis, stepping
azimuth $5^\circ$ between turns.

Two receivers ran simultaneously. From `header/wiring`, they are **identical receive-only
copies** — the only difference is that one is suspended:

| key | name | mounting |
|---|---|---|
| 4 | `box-air` | suspended, on the motorised platform |
| 0 | `box-gnd` | fixed on the ground |

A **separate ground-based comb transmitter** injects tones every 16 channels
(3.906 MHz spacing). It is off all day and switches on at ~18:14 UTC, about two hours
before the raster — it was turned on deliberately for this run. `box-gnd` therefore acts
as a free monitor: it sees the same injected signal but never moves, so anything it
records is transmitter drift or common-mode gain, not rotation.

## What the figure shows

One full rotation at azimuth $-90^\circ$. Each curve is one injected tone, coloured by
frequency, with the sky continuum subtracted from its channel. The response varies by
8.1–33.1 dB through the turn (median 25.4 dB across tones), with deep nulls near
$\pm90^\circ$ and maxima at $0^\circ$ and $\pm180^\circ$.

## Things that are easy to get wrong

- **The comb sits on channel residue 8** (`ch % 16 == 8`), not 15/0/1. A second comb sits
  on residue 0. Masking the wrong residues puts every transmitter tone inside a nominally
  clean band.
- **The two combs are the transmitter's two polarizations, not a strong and a weak set.**
  Their coupling to the single-polarization bowtie trades off in antiphase as the azimuth
  is stepped: residue 8 nulls at azimuth $-130^\circ$ exactly where residue 0 peaks, and
  the reverse near $-45^\circ$ (`explore/54`). At the $-90^\circ$ azimuth this figure uses,
  residue 8 leads residue 0 by 7.9 dB, so residue 8 is the aligned polarization here.
- **Raw channel power is tone + sky continuum.** A raw curve flattens onto that pedestal
  once the tone becomes weak, so the depth it can show is capped by the tone-to-continuum
  ratio, which rises from 0.2 dB at 53 MHz to 25 dB at 197 MHz. The frequency ordering of
  raw curves is therefore **the transmitter link budget, not the beam sharpening**.
  Subtracting the continuum removes that ordering and extends the usable band down to
  56.6 MHz.
- **The continuum estimate has to be local.** DPSS models spanning the band were scored
  against transmitter-off data, where the comb channels carry continuum only and the truth
  is known (`explore/45`–`47`), and lost to the flanking median: 2.5 per cent against
  1.6 per cent error at the comb channels. The bandpass has structure on a few MHz that a
  band-spanning basis cannot follow, and over a window narrow enough to track it a DPSS
  basis supports less than one mode and reduces to that same local average. Fitting in
  linear rather than log power is far worse again (28 per cent).
- **A bin's own scatter is not its error bar.** Each 5° bin holds one to three
  integrations. Pooling within-bin variances is no better — across a 5° bin the response
  itself changes by several dB near the shoulders, so that measures the gradient rather
  than the noise. The noise is taken instead from a null test on the flanking channels,
  which carry the same continuum and the same estimator error but no injected tone.
- **Only the first half of the scan is well behaved.** Rotations 0–29 (az $-180^\circ$ to
  $-35^\circ$) vary smoothly with azimuth; rotations 30–51 are ~6× rougher and include
  implausible ~50 dB depths. The azimuth potentiometer does not explain this — its offset
  from the motor counts is a near-constant $-14^\circ$ to $-19^\circ$ — and the platform is
  suspended with the IMU erroring during the scan, so commanded azimuth is not a precise
  attitude measurement. Unresolved; revisit before using the second half.
- **Do not fold many rotations together.** Azimuth is stepped deliberately and the response
  genuinely depends on it, so folding 4 rotations blunts the 189 MHz null by ~3 dB and makes
  the curves visibly jagged. One rotation at one azimuth is both cleaner and simpler.
""")

code(r"""
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

REPO = Path.cwd().resolve()
while not (REPO / "data" / "deployment5_filtered").is_dir():
    if REPO.parent == REPO:
        raise FileNotFoundError("data/deployment5_filtered not found above cwd")
    REPO = REPO.parent

DATA_DIR = REPO / "data" / "deployment5_filtered"
SIDECAR = REPO / "notebooks" / "christian" / "deployment5" / "motor_scan_20260717_key4.h5"
OUTDIR = REPO / "notebooks" / "christian" / "beam_modulation"
KEY0_CACHE = OUTDIR / "key0_raster.npz"

COMB_RESIDUE = 8              # comb tones sit on ch % 16 == 8
COMB_SPILL = (7, 8, 9, 15, 0, 1)   # both combs, plus their +-1 spillover
FM_BAND = (86.0, 110.0)       # broadcast FM plus a margin, excluded throughout
BIN_DEG = 5.0                 # rotation-angle bin width
MAX_ROUGHNESS = 0.25          # dB, point-to-point roughness of the flanking channels
NSIG = 3.0                    # significance the binned excess must reach to count
""")

md(r"""
`MAX_ROUGHNESS` sits in a gap in the distribution: flanking roughness runs smoothly from
0.013 to 0.174 dB and then jumps to 0.499 dB (244.1 MHz) and 0.953 dB (248.0 MHz), so this
threshold removes those two channels and nothing else. A tighter cut mattered when the
figure plotted raw power and the continuum set the floor; with the continuum subtracted and
the tones well above it, a noisy continuum estimate barely moves the excess.
""")

code(r"""
with h5py.File(SIDECAR, "r") as f:
    d4 = f["data4"][:].astype(np.float64)          # suspended receiver, key 4
    t = f["times"][:]
    deg = 1.0 / f.attrs["counts_per_deg"]
    az = f["az_counts"][:] * deg
    el = f["el_counts"][:] * deg                   # the rotation axis
    freq = f["freqs"][:]                           # MHz, 0.244140625 spacing
    file_names = [n.decode() for n in f["file_names"][:]]

d4[d4 <= 0] = np.nan
nchan = d4.shape[1]
chan = np.arange(nchan)
is_fm = (freq > FM_BAND[0]) & (freq < FM_BAND[1])
tmin = (t - t[0]) / 60.0

print(f"{d4.shape[0]} integrations x {nchan} channels, {tmin[-1]:.1f} min contiguous")
print(f"rotation axis {np.nanmin(el):.0f}..{np.nanmax(el):.0f} deg, "
      f"azimuth {np.nanmin(az):.0f}..{np.nanmax(az):.0f} deg "
      f"(the raster was stopped before completing the full +-180 deg azimuth range)")
""")

md(r"""
### The stationary receiver

Key 0 is not in the sidecar, so pull it from the filtered files that make up the raster.
It is used only as a control: it never moves, so it separates rotation response from
transmitter drift and common-mode gain.
""")

code(r"""
def load_key0():
    if KEY0_CACHE.exists():
        return np.load(KEY0_CACHE)["d0"].astype(np.float64)
    blocks = []
    for name in file_names:
        with h5py.File(DATA_DIR / name, "r") as f:
            blocks.append(f["data/0"][:])
    d = np.concatenate(blocks)
    np.savez_compressed(KEY0_CACHE, d0=d.astype(np.float32))
    return d.astype(np.float64)


d0 = load_key0()
d0[d0 <= 0] = np.nan
assert d0.shape == d4.shape, f"key 0 shape {d0.shape} != key 4 shape {d4.shape}"
print(f"stationary receiver loaded: {d0.shape}")
""")

md(r"""
### Splitting the scan into individual rotations

Elevation is the fast axis, so turning points of `el` bound each full rotation.
""")

code(r"""
turns = np.where(np.diff(np.sign(np.diff(el))) != 0)[0] + 1
rotations = [(a, b) for a, b in zip(np.r_[0, turns], np.r_[turns, len(el)]) if b - a > 100]
rot_az = np.array([np.nanmedian(az[a:b]) for a, b in rotations])

print(f"{len(rotations)} full rotations, "
      f"{np.median([b - a for a, b in rotations]):.0f} integrations each "
      f"({np.median([tmin[b - 1] - tmin[a] for a, b in rotations]) * 60:.0f} s)")

# the figure uses a single rotation at azimuth -90 deg, inside the well-behaved first half
ROT = int(np.argmin(np.abs(rot_az - (-90.0))))
A, B = rotations[ROT]
print(f"using rotation {ROT}: azimuth {rot_az[ROT]:+.0f} deg, "
      f"t = {tmin[A]:.1f}-{tmin[B - 1]:.1f} min, {B - A} integrations")

# a channel counts as usable if it is finite through the rotation being plotted, rather
# than over the whole raster, where a single bad sample would discard it outright
finite = np.isfinite(d4[A:B]).mean(0) > 0.9
print(f"{finite.sum()} of {nchan} channels usable through this rotation")
""")

md(r"""
### Subtracting the continuum

A tone channel holds the injected tone *and* the sky continuum. Raw power therefore
flattens onto the continuum once the tone becomes weak, and the trough a curve reaches is
set by its tone-to-continuum ratio rather than by the antenna. The continuum under each
tone is estimated from the median of the non-comb channels three to six channels either
side — close enough to track the few-MHz bandpass structure that defeats a band-spanning
DPSS model, far enough to clear the comb spillover.
""")

code(r"""
edges = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
centres = 0.5 * (edges[:-1] + edges[1:])
bin_idx = np.digitize(el, edges) - 1
ZERO = int(np.argmin(np.abs(centres)))


def to_db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(x)


def flanking(c):
    # non-comb, non-FM channels either side of c, clear of the comb spillover
    return [c + o for o in range(-6, 7)
            if 3 <= abs(o) <= 6 and 0 <= c + o < nchan and finite[c + o]
            and (c + o) % 16 not in COMB_SPILL and not is_fm[c + o]]


def binned(series, a, b):
    # median of the series in each rotation-angle bin
    s = series[a:b]
    idx = bin_idx[a:b]
    mu = np.full(len(centres), np.nan)
    for j in range(len(centres)):
        v = s[(idx == j) & np.isfinite(s)]
        if v.size:
            mu[j] = np.median(v)
    return mu


def subtract(c):
    # continuum-subtracted power in channel c, per integration
    return d4[:, c] - np.nanmedian(d4[:, flanking(c)], axis=1)
""")

md(r"""
### How noisy is a binned excess?

Not something a bin can tell us: each 5° bin holds one to three integrations, so its own
scatter is far too noisy to use as an error bar and would reject bins at random part-way
down the slope. Pooling the within-bin variances fails the other way — across 5° the
response itself changes by several dB near the shoulders, so the pooled number measures the
gradient, not the noise.

Instead run the identical reduction on the clean channels flanking the tone. They carry the
same continuum and the same estimator error but no injected tone, so the scatter of their
binned residual about zero is the noise floor for that tone's excess. It varies from $-7$
to $-37$ dB relative to each tone's own response at $0^\circ$, so no single level applies
to all of them.
""")

code(r"""
def noise_sigma(c):
    sds = [np.nanstd(binned(subtract(cp), A, B)) for cp in flanking(c) if flanking(cp)]
    return np.nanmedian(sds) if sds else np.inf


def excess_profile(c):
    # continuum-subtracted tone power per bin, in dB relative to 0 deg, plus the mask
    # marking where the excess is significant
    mu = binned(subtract(c), A, B)
    ok = mu > NSIG * noise_sigma(c)
    pos = mu > 0
    ref = np.nanmedian(mu[ZERO - 1:ZERO + 2])
    full = np.where(pos, to_db(np.where(pos, mu, np.nan) / ref), np.nan)
    return full, ok
""")

md(r"""
### Selecting tones

A tone is used if its flanking channels are not RFI-ridden and its excess is significant at
$0^\circ$ and over at least 20 of the 72 bins. There is deliberately **no** smoothness cut
on a tone's own profile: one was tried, and the secondary dip near $-45^\circ$ that would
trip it turns out to be present in every tone from 201 to 240 MHz, deepening progressively
with frequency (`explore/57`), so it is structure in the response rather than a channel
glitch.

The band limit is the instrument band; the roughness cut removes the DTV-contaminated
channels above ~225 MHz on its own.
""")

code(r"""
in_band = [c for c in chan[(chan % 16 == COMB_RESIDUE) & (freq > 50) & (freq < 250)]]
candidates = [c for c in in_band if finite[c] and not is_fm[c] and flanking(c)]
tones, rejected = [], []
for c in candidates:
    nb = flanking(c)
    rough = np.nanmedian(np.abs(np.diff(binned(to_db(np.nanmedian(d4[:, nb], 1)), A, B), 2)))
    if rough >= MAX_ROUGHNESS:
        rejected.append((freq[c], f"RFI, flanking roughness {rough:.3f} dB")); continue
    prof, ok = excess_profile(c)
    if not ok[ZERO] or ok.sum() < 20:
        rejected.append((freq[c], f"excess significant in only {int(ok.sum())} bins")); continue
    tones.append(c)

tone_freq = freq[np.array(tones)]
profs = {c: excess_profile(c) for c in tones}
print(f"{len(in_band)} comb channels of this polarization in the band; {len(tones)} used, "
      f"{tone_freq.min():.1f}-{tone_freq.max():.1f} MHz")
print(f"{len(in_band) - len(candidates)} dropped in the FM band, where there are no clean "
      f"flanking channels for a continuum estimate")
for f_, why in rejected:
    print(f"    {f_:7.1f} MHz  {why}")
""")

md(r"""
### Controls

1. **Stationary receiver on the same tones** — bounds transmitter drift and common-mode gain.
2. **Raw depth on the same tones** — how much of the modulation the pedestal was hiding.
""")

code(r"""
sig = [np.where(profs[c][1], profs[c][0], np.nan) for c in tones]
sub_depth = np.array([np.nanmax(m) - np.nanmin(m) for m in sig])     # peak-to-trough
below_zero = np.array([-np.nanmin(m) for m in sig])                  # drop below 0 deg
raw_depth = np.array([np.ptp(binned(to_db(d4[:, c]), A, B)[
    np.isfinite(binned(to_db(d4[:, c]), A, B))]) for c in tones])

gnd_cont = np.nanmedian(d0[:, [o for c in tones for o in flanking(c)]], axis=1)
gnd_mu = binned(np.nanmedian(d0[:, tones], axis=1) - gnd_cont, A, B)
gnd = to_db(gnd_mu / np.nanmedian(gnd_mu[ZERO - 1:ZERO + 2]))

print(f"subtracted, peak-to-trough : {sub_depth.min():.1f}-{sub_depth.max():.1f} dB "
      f"(median {np.median(sub_depth):.1f})")
print(f"subtracted, below 0 deg    : {below_zero.min():.1f}-{below_zero.max():.1f} dB "
      f"(median {np.median(below_zero):.1f})")
print(f"raw, peak-to-trough        : {raw_depth.min():.1f}-{raw_depth.max():.1f} dB "
      f"(median {np.median(raw_depth):.1f})")
print(f"stationary receiver        : {np.nanmax(gnd) - np.nanmin(gnd):.2f} dB")
print()
print("The 0 deg reference is not quite each curve's maximum -- the maxima at +-180 deg sit")
print("a median of 1.3 dB above it -- so the drop below 0 deg is smaller than the")
print("peak-to-trough. Quote them consistently; the paper uses peak-to-trough.")

hi = [c for c in tones if freq[c] > 158]
print(f"\n{sum(1 for c in tones if profs[c][1].all())} tones never drop below the noise "
      f"anywhere in the sweep;")
print(f"above 158 MHz that is {sum(1 for c in hi if profs[c][1].all())} of {len(hi)}, so "
      f"those nulls are measured.")
print("Below it the tone reaches the noise before the null does and the depths are upper")
print("limits on the power received in the null.")
""")

md(r"""
## The figure

Curves are drawn wherever the subtracted power is positive, so each one runs past its own
detection threshold and into the noise rather than stopping dead at a per-curve limit the
reader cannot decode. Where the tone falls to the level of the continuum the subtraction
scatters about zero and the curve breaks — 0.6 per cent of bins, almost all within 30° of
the nulls.

The axis is clipped at $-33$ dB, just below the deepest significant point over all tones
($-31.3$ dB), so nothing measured is hidden. Left to autoscale it runs to $-42$ dB to
accommodate noise excursions on two tones (76.2 and 111.3 MHz, the two least reliable
continuum estimates in the set), which compresses the part of the plot that carries the
result.
""")

code(r"""
norm = Normalize(tone_freq.min(), tone_freq.max())
smap = ScalarMappable(norm, plt.cm.plasma)

fig, ax = plt.subplots(figsize=(5.2, 3.8))
for c in tones:
    ax.plot(centres, profs[c][0], color=smap.to_rgba(freq[c]), lw=0.9, alpha=0.95)

ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
ax.set_xlim(-180, 180)
ax.set_ylim(-33, 4)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.grid(alpha=0.25, lw=0.5)
ax.set_xlabel("Platform rotation angle [deg]", fontsize=9)
ax.set_ylabel("Injected tone power relative to $0^\\circ$ [dB]", fontsize=9)
ax.tick_params(labelsize=8)

cb = fig.colorbar(smap, ax=ax, pad=0.02)
cb.set_label("Frequency [MHz]", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUTDIR / f"beam_modulation.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUTDIR / 'beam_modulation.pdf'}")
""")

md(r"""
The version in the paper is rendered at the single-column figure size by
`docs/render_beam_modulation.py` in the manuscript repo, which carries the same reduction.

## Caption

> Response of the suspended EIGSEP antenna to platform rotation, measured against an
> injected signal during the July 2026 deployment. The curves represent the received power
> of the bowtie antenna as a function of rotation angle, binned in $5^\circ$ steps. Each
> curve corresponds to one transmitted tone, coloured by the transmission frequency. Tones
> in the FM broadcast band are excluded, along with those close enough to its edges that
> the channels used to estimate the continuum fall inside the band. The sky continuum is
> subtracted from each tone channel and the power is referenced to the power at $0^\circ$
> rotation. Each curve is drawn wherever the continuum-subtracted power is positive; where
> the tone falls to the level of the continuum the subtraction scatters about zero and the
> curve breaks. The vertical axis is clipped at $-33$ dB, just below the deepest point at
> which any tone is still significantly detected, so the tails reaching that limit are
> noise. The depths quoted in the text use only those bins in which a tone stays above its
> own noise level, which differs from tone to tone.

## Numbers quoted above

| quantity | value |
|---|---|
| tones used | 42, 56.6–240.2 MHz, 3.906 MHz spacing |
| in-band comb channels of this polarization | 51: 42 used, 6 in the FM band, 2 RFI-rough (244, 248 MHz), 1 not detected (52.7 MHz) |
| rotation shown | one full turn, azimuth $-90^\circ$, 128 integrations, 68 s |
| polarization | residue 8, aligned at this azimuth, 7.9 dB above residue 0 |
| depth, subtracted, peak-to-trough | 8.1–33.1 dB (median 25.4) |
| depth, subtracted, below $0^\circ$ | 6.9–31.3 dB (median 24.5) |
| depth, raw, peak-to-trough | 1.7–29.5 dB (median 18.2) |
| depth, stationary receiver | 0.03 dB |
| nulls measured rather than limited | above 158 MHz, 17 of 18 tones |

## Why the FM band cannot be recovered

The injected tone is perfectly detectable there — it rises 5.7–8.5 dB above its local
neighbours when the transmitter switches on, comparable to the tones just above the band.
It is unusable for a different reason: FM broadcast arrives from a fixed direction on the
horizon, so the rotation modulates it in the same way as it modulates the injected tone.
The non-comb channels there swing up to 7.3 dB through a turn against 1.1–3.0 dB in clean
bands (`explore/60`), so the contaminant is degenerate with the signal and no continuum
estimate can separate them.
""")

nb["cells"] = C
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(nb, "beam_modulation.ipynb")
print("wrote beam_modulation.ipynb")
