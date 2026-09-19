#!/usr/bin/env python3
"""Generate rfi_flag_prototype.ipynb -- a stage-by-stage evaluation of the
current RFI flagging prototype.

Same pattern as the other build_*.py here: this script owns the content, the
.ipynb is a build artifact, render_rfi_proto.sh turns it into a PDF.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "rfi_flag_prototype.ipynb")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ------------------------------------------------------------------- intro

md(r"""
# RFI flagging prototype — stage-by-stage evaluation

**Status: prototype. Not a product; writes nothing.** This notebook evaluates
what each stage of `rfi_proto.py` currently does on real campaign data, one
stage at a time: what it tests, where its threshold comes from, how much it
flags, and whether it is working.

I/O goes through `eigsep_data`. A `Selection` goes in and an
`eigsep_data.bundle.Bundle` comes out with `b.flags` filled in, so everything
here uses the same objects as the rest of the package. **Existing flags are
never loaded** — every `load_bundle` call passes `products=[]`, because a
detector seeded from `flags/v0` or `flags/v2` could not tell you what it found
on its own.

## The stages

| bit | stage | axis | threshold from |
|---|---|---|---|
| 0,1 | off-sky, overflow | — | switch state; sign of the auto |
| 2 | a priori bands | frequency | known emitter table |
| 3 | transient | time, per channel | Gaussian, `model/√N` |
| 4,5,6 | broadband / gross power / meteor | whole integration | band occupancy; empirical |
| 7 | coherence | per pixel | Rayleigh, `1/√(2N)` |
| 8,9 | persistent line, occupancy | frequency | Rayleigh floor vs neighbours |
| 10,11 | retracted, residual | per pixel | second pass: overturned, or auto-only evidence |

Two detectors see different things — a transient has time structure, a
persistent tone does not — and a second pass re-decides once a continuum model
exists. Thresholds are calibrated rather than tuned: `nsig` defaults to 6 on
all three z-scores.
""")

code(r"""
import os, sys, glob, json
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import rfi_proto as P
from eigsep_data import MetadataIndex

# %matplotlib widget works here. Under ipympl a figure renders at a FIXED
# pixel size rather than scaling to the output area, so the inline defaults
# give a canvas wider than the JupyterLab panel; INTERACTIVE shrinks it.
INTERACTIVE = "ipympl" in matplotlib.get_backend().lower()
plt.rcParams.update({"figure.dpi": 90 if INTERACTIVE else 110,
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def fsize(w, h):
    # Figure size, scaled down when the widget backend is active.
    return (w * 0.78, h * 0.85) if INTERACTIVE else (w, h)


def cbars(fig, ax):
    # Give EVERY panel a colorbar slot, shown or not. A colorbar on some
    # panels and not others steals width from those axes only, so
    # sharex/sharey panels stop lining up -- and under ipympl the shared zoom
    # then visibly disagrees between them. Returns add(im, i), which reveals
    # slot i.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    axf = np.atleast_1d(ax).ravel()
    slots = []
    for a_ in axf:
        c_ = make_axes_locatable(a_).append_axes("right", size="4%", pad=0.04)
        c_.set_visible(False)
        slots.append(c_)

    def add(im, i):
        slots[i].set_visible(True)
        fig.colorbar(im, cax=slots[i])
    return add


def widget_finish(fig):
    if INTERACTIVE:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_position = "right"

CAMPAIGN = os.environ.get("MARJUM_DATA_ROOT",
                          "/mnt/data02/eigsep/marjum-2026-07")
idx = MetadataIndex(os.path.join(CAMPAIGN, "data"))
idx.rebuild()                       # uses the sidecar cache when it exists
print(f"index: {len(idx.table):,} integrations over "
      f"{idx.table.file.nunique()} files"
      f"{' (from cache)' if idx.from_cache else ''}")


def one_file(fname):
    return idx.select(files=(fname, fname))


REF = "corr_20260717_034832Z.h5"      # phase C, comb off, on sky throughout
b = P.flag_bundle(one_file(REF))
print()
print(b.summary())
print(f"\nreturned {type(b).__module__}.{type(b).__name__}; "
      f"b.flags is {b.flags.dtype}, {b.flags.shape}")
""")

# --------------------------------------------------------------- the floors

md(r"""
## 0. What every threshold rests on

Before any stage: the noise floors. Both are computed, not fitted.

- **auto**: a power spectrum averaged over $N$ samples has fractional noise
  $1/\sqrt N$, so the scatter about a continuum model is $P/\sqrt N$.
- **cross**: $|r| = |V_{ab}|/\sqrt{P_aP_b}$ is Rayleigh with
  $\sigma = 1/\sqrt{2N}$, and $P(|r| > z\sigma) = e^{-z^2/2}$.

$N = \Delta\nu\,\tau$ comes from the index's own `integration_time` and
`nchan` columns, per row — it halves before the 07-15 accumulator change, so
it is never hardcoded.
""")

code(r"""
info = b.provenance["products"]["flags"]
n_samp = info["n_independent_samples"]
print(f"N = dnu * tau                 {n_samp:,.0f}")
print(f"sigma_theory = 1/sqrt(2N)     {P.rayleigh_sigma_theory(n_samp):.6g}")
print(f"auto fractional noise 1/sqrt(N) {1/np.sqrt(n_samp):.5f}")
print(f"\nfitted sigma_hat / sigma_theory (cross): "
      f"{info['sigma_hat_over_theory']:.2f}")
print(f"model_inflation (auto, measured/radiometric): "
      f"{info['model_inflation_median']:.2f}")
print(f"\ncross quantisation ok: {info['cross_quantisation_ok']}  "
      f"(|V|=0 fraction {info['cross_zero_fraction']:.3f})")
print(f"thresholds: {json.dumps({k: round(float(v), 2) for k, v in info['thresholds'].items()})}")
""")

code(r"""
coh = b.products["flags"]["coherence"]
fr = b.freqs_mhz
in_band, excl, quiet = P.band_masks(fr)
clean = in_band & ~excl
v = coh[:, clean].ravel()
sig_hat = np.quantile(v, 0.25) / np.sqrt(-2 * np.log(0.75))

fig, ax = plt.subplots(1, 2, figsize=fsize(11, 3.2))
q = np.linspace(0.02, 0.98, 60)
ax[0].plot(q, np.quantile(v, q) / sig_hat, lw=1.2, label="measured")
ax[0].plot(q, np.sqrt(-2 * np.log(1 - q)), "k--", lw=1, label="Rayleigh")
ax[0].set(xlabel="quantile", ylabel=r"$|r| / \hat\sigma$",
          title="cross: Rayleigh at the bottom, RFI in the tail")
ax[0].legend(fontsize=7)

infl = b.products["flags"]["model_inflation"]
ax[1].plot(fr[np.isfinite(infl)], infl[np.isfinite(infl)], lw=0.7)
ax[1].axhline(1.0, color="k", ls="--", lw=1)
ax[1].set(xlabel="Frequency [MHz]", ylabel="measured / radiometric",
          ylim=(0, 4), title="auto: residual scatter vs the radiometric floor")
fig.tight_layout()

print("cross, low quantiles vs Rayleigh:")
for qq in (0.10, 0.25, 0.50, 0.90, 0.99):
    print(f"   q={qq:4.2f}   measured {np.quantile(v, qq)/sig_hat:6.3f}   "
          f"Rayleigh {np.sqrt(-2*np.log(1-qq)):6.3f}")
""")

md(r"""
**Assessment.** The auto floor is real: the residual's per-channel scatter sits
within a few percent of $P/\sqrt N$ across the band, so `model_inflation` ≈ 1
and the auto z-score means what it says.

The cross floor does not behave as well. The bottom of the distribution is
Rayleigh to ~1%, which is why $\hat\sigma$ is fitted on the 25th percentile and
not the median — but the fitted value sits a factor ~2 above the radiometric
prediction. Either the cross achieves ~20% of ideal sensitivity or there is a
common-mode floor between the two antennas. **Unresolved**, and it means the
coherence z is a calibrated *relative* statistic, not an absolute one.
""")

# ---------------------------------------------------------------- stage 2

md(r"""
## Stage 2 — a priori bands

Known emitters masked *before* any statistic is computed, so they cannot
contaminate a fitted scale or drag the continuum fit. FM, ORBCOMM, the two
clock harmonics, and (when asked) the self-comb teeth ± 1 channel.

These are still reported as flagged. Excluded from the statistics, not from
the product.
""")

code(r"""
m = b.flags
apr = (m & P.BIT_A_PRIORI) > 0
print(f"a priori channels: {apr[0].sum()} of {fr.size} "
      f"({apr[0].mean():.1%} of the full axis, "
      f"{apr[0][in_band].mean():.1%} of the analysis band)")
for name, (lo, hi) in P.A_PRIORI_BANDS.items():
    print(f"   {name:<8} {lo:6.1f}-{hi:6.1f} MHz   "
          f"{int(((fr >= lo) & (fr <= hi)).sum()):3d} channels")
print(f"   clock    {[f'{fr[c]:.3f}' for c in P.CLOCK_HARMONIC_CHANS]} MHz")

# what the self-comb option costs, in the era where it applies
bc = P.flag_bundle(one_file("corr_20260717_201113Z.h5"), self_comb=True)
ac = (bc.flags & P.BIT_A_PRIORI) > 0
ibc = (bc.freqs_mhz >= 45) & (bc.freqs_mhz <= 235)
print(f"\nself-comb era, teeth +/-1 channel: a priori covers "
      f"{ac[0][ibc].mean():.1%} of the analysis band")
""")

md(r"""
**Assessment.** Working, and the cost is visible rather than buried. In the
self-comb era the mask removes 3 of every 8 channels — 37.5% of the analysis
band — because the channelizer leaks the teeth into their neighbours. That is a
hardware problem, not a flagging one, and no threshold choice makes it go away.
""")

# ---------------------------------------------------------------- stage 3

md(r"""
## Stage 3 — transient, model-free on the raw autos

Per channel along time: a running median (width 9) removes slow gain drift,
and what is left is compared to $\max(P/\sqrt N,\ \mathrm{MAD})$, one-sided.
No model, no cross, and it runs over the **full** axis — channels outside
45–235 MHz are where the cleanest instrumental diagnostics live.

Channels carrying essentially no power are not testable at all (below ~25 MHz
the median is 0–0.5 counts, so every scale estimate collapses); `min_counts`
excludes them.
""")

code(r"""
prod = b.products["flags"]
tr = (m & P.BIT_TRANSIENT) > 0
za, zb = prod["z_transient_a"], prod["z_transient_b"]
print(f"transient pixels: {tr.sum()} ({tr.mean():.4f} of the full axis)")
print(f"   integrations with any: {int(tr.any(axis=1).sum())} of {tr.shape[0]}")
print(f"   channels with any:     {int(tr.any(axis=0).sum())} of {fr.size}")
print(f"\nper antenna (a = box-gnd, b = box-air):")
print(f"   a: {int((za > 6).sum()):6d} pixels over threshold")
print(f"   b: {int((zb > 6).sum()):6d} pixels over threshold")

rows = np.argsort(tr.sum(axis=1))[::-1][:5]
print("\nbusiest integrations:")
for t in rows:
    js = np.where(tr[t])[0]
    top = js[np.argsort(np.maximum(za, zb)[t, js])[::-1][:6]]
    print(f"   int {t:3d}: {len(js):3d} channels; strongest "
          + ", ".join(f"{fr[j]:.3f} MHz (z={np.maximum(za, zb)[t, j]:.3g})"
                      for j in top))
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=fsize(12, 3.6), sharex=True, sharey=True)
add = cbars(fig, ax)
ext = [fr[0], fr[-1], m.shape[0], 0]
kw = dict(aspect="auto", interpolation="nearest", extent=ext)
im = ax[0].imshow(np.clip(np.maximum(za, zb), 0, 12), cmap="viridis",
                  vmin=0, vmax=12, **kw)
ax[0].set(title="max transient $z$ over the two autos", xlabel="Frequency [MHz]",
          ylabel="integration")
add(im, 0)
ev = np.zeros(m.shape)
ev[za > 6] = 0.5
ev[zb > 6] = 1.0
im = ax[1].imshow(ev, cmap="Oranges", vmin=0, vmax=1, **kw)
ax[1].set(title="which antenna (light = box-gnd, dark = box-air)",
          xlabel="Frequency [MHz]")
add(im, 1)
for a_ in ax:
    a_.axvspan(45, 235, color="k", alpha=0.04)
widget_finish(fig)
""")

md(r"""
**Assessment.** Working, and it is the stage that covers what the cross cannot:
it needs neither a partner antenna nor a model, so it is the only detector that
functions where the cross is quantisation-limited.

Two things it finds on this file that nothing else would. A set of simultaneous
single-integration tones whose channel indices are exactly 49 and 207 mod 256 —
two families 62.5 MHz ($f_s/8$) apart, mirror images about the grid, i.e. an
aliased clock artifact rather than sky RFI. And broadband excursions confined to
**one antenna**, which the coherence stage is blind to by construction.
""")

# -------------------------------------------------------------- stage 4-6

md(r"""
## Stages 4–6 — whole-integration tests

Three statistics that condemn an entire integration rather than a pixel,
because a contaminated row cannot be partially trusted.

- **broadband** (bit 4): more than `broadband_frac` of the band tripped the
  transient test at once. A few-percent excess across hundreds of channels
  leaves most of them individually below threshold, and letting those constrain
  the continuum is how one bad row reaches every other row's residual.
- **gross power** (bit 5): the median over 236–249 MHz, where the bandpass has
  rolled off ~3 decades but a broadband transient still lands. Median, not
  mean, because that band has a narrow emitter in it.
- **meteor** (bit 6): FM and DTV rising together, each normalised by a
  reference continuum (108–136, 155–174, 216–235 MHz) so common gain drift
  divides out.
""")

code(r"""
for bit, name in ((P.BIT_BROADBAND, "broadband"), (P.BIT_GROSS, "gross power"),
                  (P.BIT_METEOR, "meteor")):
    rows_ = np.where((m & bit).any(axis=1))[0]
    print(f"{name:<12} {len(rows_):3d} integrations  {rows_[:12]}")

exc = prod["meteor_excess"]
fig, ax = plt.subplots(2, 1, figsize=fsize(11, 4.4), sharex=True)
for nm, e_ in exc.items():
    ax[0].plot(e_, lw=0.9, label=nm)
ax[0].axhline(P.DEFAULTS["meteor_nsig"], color="k", ls=":", lw=1)
ax[0].set(ylabel="excess [MAD]", yscale="symlog",
          title="meteor statistic: FM and DTV vs the reference continuum")
ax[0].legend(fontsize=7, ncol=3)

trf = ((m & P.BIT_TRANSIENT) > 0)[:, in_band].mean(axis=1)
ax[1].plot(trf, lw=0.9)
ax[1].axhline(P.DEFAULTS["broadband_frac"], color="C3", ls="--", lw=1,
              label=f"broadband_frac = {P.DEFAULTS['broadband_frac']}")
ax[1].set(xlabel="integration", ylabel="in-band transient fraction",
          title="broadband statistic", yscale="log")
ax[1].legend(fontsize=7)
fig.tight_layout()

print(f"\nin-band transient fraction: median {np.median(trf):.4f}, "
      f"p99 {np.percentile(trf, 99):.3f}, max {trf.max():.3f}")
""")

md(r"""
**Assessment.** The broadband threshold is not delicate — the in-band transient
fraction has median 0.004 and p99 0.17, so 0.15 selects a handful of rows and
0.25 selects none. That is the behaviour you want from a morphological cut.

The meteor statistic is the weakest of the three and **fires on things it was
not built for**: the single-integration clock-comb events trip it, because they
are not propagation but they do lift the transmitter bands relative to the
reference. It should probably require the FM/DTV rise to occur *without* a
simultaneous transient elsewhere. Not fixed.
""")

# ---------------------------------------------------------------- stage 7

md(r"""
## Stage 7 — coherence

$|V_{ab}|/\sqrt{P_aP_b}$: the fraction of the two autos' amplitude that is
actually correlated between the antennas. Receiver noise is independent and
averages down in the cross; a common interferer does not. It is also naturally
normalised — bandpass, reflection ripple and the accumulator change all divide
out — so one threshold means the same thing across the band.

$\hat\sigma$ is fitted **per channel** on a low quantile. That absorbs a
frequency-dependent common-mode floor, at a price paid in stage 8.
""")

code(r"""
cohbit = (m & P.BIT_COHERENT) > 0
mcoh = np.median(coh, axis=0)
idx_ib = np.where(in_band)[0]
top = idx_ib[np.argsort(mcoh[idx_ib])[::-1][:16]]

fig, ax = plt.subplots(figsize=fsize(11, 3.2))
ax.semilogy(fr[in_band], mcoh[in_band], lw=0.7, color="0.3")
for lo, hi in P.A_PRIORI_BANDS.values():
    ax.axvspan(lo, hi, color="C0", alpha=0.10)
for ch in top:
    ax.annotate(f"{fr[ch]:.1f}", (fr[ch], mcoh[ch]), fontsize=6,
                rotation=90, ha="center", va="bottom")
ax.set(xlabel="Frequency [MHz]", ylabel="median coherence",
       title="median coherence; shaded = a priori bands")
fig.tight_layout()

print(f"coherence-flagged pixels: {cohbit.mean():.4f} of the full axis")
print("\nhighest median coherence in band:")
for ch in sorted(top)[:12]:
    tag = ""
    if ch in P.CLOCK_HARMONIC_CHANS:
        tag = "  <- clock harmonic"
    elif 88 <= fr[ch] <= 108:
        tag = "  FM"
    elif 136.5 <= fr[ch] <= 138.5:
        tag = "  ORBCOMM"
    print(f"   {fr[ch]:8.3f} MHz   C = {mcoh[ch]:.3f}{tag}")
""")

md(r"""
**Assessment.** Working where the cross is usable, and it identifies real
emitters with no model and no tuning: the FM carriers, ORBCOMM, and 125.000 /
187.500 MHz — exactly $f_s/2$ and $3f_s/4$, instrumental. The 125 MHz line is
one that `flags/v0` calls CLEAN 86–92% of the time.

Two failure modes, both real:

1. **Quantisation.** In phase A the cross has 21–38% of in-band samples with
   $|V|$ *exactly* zero, so $\hat\sigma$ collapses and the stage reports a clean
   sky. `quantisation_ok()` refuses those files; roughly the first two campaign
   days have no coherence path at all.
2. **Self-suppression.** Because $\hat\sigma$ is fitted per channel, a channel
   contaminated at *every* sample simply gets a bigger $\hat\sigma$ and stops
   being an outlier against itself. In the self-comb era the whole band is busy
   enough that the stage goes quiet where it should be loudest.
""")

# -------------------------------------------------------------- stage 8-9

md(r"""
## Stages 8–9 — persistent lines and occupancy

Stage 7 cannot see a tone that is on at every sample. Along frequency it is
obvious: that channel's fitted floor stands above its neighbours'. Stage 8
median-filters $\hat\sigma(\nu)$ and flags the channels that stand proud.

Stage 9 promotes any channel flagged in more than `occ_frac` of its valid
samples to a whole-channel kill — a ragged partial mask on a mostly-on emitter
is worse than useless, because downstream averaging then sees a biased subset.
""")

code(r"""
line = (m & P.BIT_LINE) > 0
occb = (m & P.BIT_OCCUPANCY) > 0
lch = np.where(line[0])[0]
och = np.where(occb[0])[0]
print(f"persistent lines: {len(lch)} channels   occupancy kills: {len(och)}")
print(f"overlap: {len(set(lch) & set(och))}")
print("\nline channels (MHz):")
print("  ", np.round(fr[lch], 2).tolist()[:44])

fig, ax = plt.subplots(figsize=fsize(11, 3.0))
sh = prod["sigma_hat"]
ax.semilogy(fr[in_band], sh[in_band], lw=0.7, label=r"$\hat\sigma$ per channel")
ax.plot(fr[lch], sh[lch], "r.", ms=4, label="flagged as a line")
ax.set(xlabel="Frequency [MHz]", ylabel=r"$\hat\sigma$",
       title="stage 8: the per-channel Rayleigh floor, and what stands above it")
ax.legend(fontsize=7)
fig.tight_layout()
widget_finish(fig)
""")

md(r"""
**Assessment.** This is the piece B16 has no equivalent of, and it is doing the
work: the line channels are the FM carriers, ORBCOMM, the clock harmonics,
152.6, 161.9/162.4 and a block near 230–234 MHz — narrow clusters, not smeared.

152.6 and 161.9/162.4 are real, persistent and **unattributed**. 152.6 sits
outside `detectors.BAND_FAN = (148, 152)`, so calling it the box fan would be an
identification the data does not support.
""")

# --------------------------------------------------------------- stage 10

md(r"""
## Stage 10 — second pass: re-deciding after the DPSS fit

A flag raised before the continuum is modelled is provisional. With a 40 ns
DPSS model fitted on what survived stage 9, the auto can be asked an
independent question — does this sample still stand above the *radiometric*
noise? — and the answer is allowed to be no.

`z_coh` alone does not carry a flag through, or the second pass would be
decoration:

| condition | outcome |
|---|---|
| $z_{\rm res} > t_{hi}$ | flag — auto evidence |
| $z_{\rm coh} > t_{lo}$ and $z_{\rm res} > t_{lo}$ | flag — corroborated |
| $z_{\rm coh} > t_{hi}$, $z_{\rm res} \le t_{lo}$ | **retract** |

"Correlated" is not the same claim as "interference"; only the auto settles it.
""")

code(r"""
print(f"dpss_ok        {info['dpss_ok']}")
print(f"second pass    {info['second_pass']}")
print(f"retracted      {info.get('n_retracted')} pixels")
print(f"promoted       {info.get('n_promoted')} pixels")
print(f"model inflation (median) {info['model_inflation_median']:.2f}")

zres = prod["z_residual"]
fitted = ((m & (P.RFI_BITS | P.UNUSABLE_BITS)) == 0)
fig, ax = plt.subplots(1, 2, figsize=fsize(12, 3.6), sharex=True, sharey=True)
add = cbars(fig, ax)
cmap_ = plt.get_cmap("bwr").copy()
cmap_.set_bad("0.85")
im = ax[0].imshow(np.where(fitted, np.clip(zres, -12, 12), np.nan),
                  cmap=cmap_, vmin=-12, vmax=12, **kw)
ax[0].set(title="$z_{res}$ on pixels that constrained the fit",
          xlabel="Frequency [MHz]", ylabel="integration")
add(im, 0)
im = ax[1].imshow(((m & P.BIT_RETRACTED) > 0), cmap="Reds", vmin=0, vmax=1, **kw)
ax[1].set(title="retracted by the second pass", xlabel="Frequency [MHz]")
add(im, 1)
widget_finish(fig)

occ_r = (np.abs(zres) > 6)[:, in_band] & fitted[:, in_band]
print(f"\n|z_res| > 6 among fit-constraining pixels: {occ_r.mean():.4f}")
f_ib = fr[in_band]
o = occ_r.mean(axis=0)
print("worst remaining channels:")
for j in np.argsort(o)[::-1][:6]:
    print(f"   {f_ib[j]:8.3f} MHz   occ={o[j]:.2f}   "
          f"median z={np.median(zres[:, in_band][:, j]):+7.1f}")
""")

md(r"""
**Assessment.** Trustworthy only where the continuum model fits. `model_inflation`
is the diagnostic: near 1 the auto test is calibrated and promotes modestly;
where the 40 ns model cannot follow the data the residual is full of *model
error* rather than noise and the test over-promotes. §12 shows that directly
across campaign conditions.

It can also decline to run — `hera_filters` returns an all-zero model rather
than raising when a wide contiguous block is zero-weighted at a band edge, and
`dpss_ok` catches that.

What survives on this file is **not RFI**: a smooth ~2.5 MHz-wide, 2.2%-deep
depression at 146 MHz, plus the two band edges. A 2.5 MHz feature is ~200 ns in
delay, outside the 40 ns window by construction, so the model cannot represent
it. That is a real spectral feature, and whether it gets modelled or carried as
a known systematic is a decision for the signal side.
""")

# -------------------------------------------------------------- composite

md(r"""
## 11. The composite mask
""")

code(r"""
print(f"{'bit':>4}  {'name':<12}{'full axis':>11}{'in band':>10}")
print("-" * 40)
for s in P.FLAG_BITS["bits"]:
    hit = (m & s["value"]) > 0
    print(f"{s['bit']:>4}  {s['name']:<12}{hit.mean():>11.4f}"
          f"{hit[:, in_band].mean():>10.4f}")
print("-" * 40)
print(f"      {'any RFI bit':<12}{((m & P.RFI_BITS) > 0).mean():>11.4f}"
      f"{((m[:, in_band] & P.RFI_BITS) > 0).mean():>10.4f}")
print(f"      {'unusable':<12}{((m & P.UNUSABLE_BITS) > 0).mean():>11.4f}"
      f"{((m[:, in_band] & P.UNUSABLE_BITS) > 0).mean():>10.4f}")
print(f"      {'clean':<12}{(m == 0).mean():>11.4f}"
      f"{(m[:, in_band] == 0).mean():>10.4f}")
""")

code(r"""
bits_show = [(P.BIT_A_PRIORI, "a priori"), (P.BIT_TRANSIENT, "transient"),
             (P.BIT_BROADBAND | P.BIT_GROSS | P.BIT_METEOR, "whole-integration"),
             (P.BIT_COHERENT, "coherent"),
             (P.BIT_LINE | P.BIT_OCCUPANCY, "line + occupancy"),
             (P.RFI_BITS, "any RFI bit")]
fig, ax = plt.subplots(2, 3, figsize=fsize(13, 6), sharex=True, sharey=True)
for a_, (bit, name) in zip(ax.ravel(), bits_show):
    hit = (m & bit) > 0
    a_.imshow(hit, cmap="Greys", vmin=0, vmax=1, **kw)
    a_.set_title(f"{name} ({hit.mean():.1%})", fontsize=9)
for a_ in ax[-1]:
    a_.set_xlabel("Frequency [MHz]")
for a_ in ax[:, 0]:
    a_.set_ylabel("integration")
fig.suptitle(f"{REF} — antenna {b.provenance['antenna']} "
             f"(input {b.provenance['keys'][0]})", fontsize=10)
fig.tight_layout()
widget_finish(fig)
""")

code(r"""
prof_f, prof_t = P.measure_margin(b)
print("does the excess reach beyond a flagged channel?")
print("offset   median z (channel)   median z (time)")
for off in sorted(prof_f):
    a_ = prof_f[off]
    b_ = prof_t.get(off, np.nan)
    fa = "  (core)" if not np.isfinite(a_) else f"{a_:>8.2f}"
    fb = "  (core)" if not np.isfinite(b_) else f"{b_:>8.2f}"
    print(f"  {off:>3}      {fa}          {fb}")
""")

md(r"""
Flat at $z \approx 1$ from offset 1 on both axes — the immediate neighbours of a
flagged channel, and the integrations either side of a flagged sample, are
already at the noise floor. There is no shoulder to catch, which is why
dilation defaults to 0.
""")

md(r"""
## 11b. DPSS on the masked data — residual with and without the mask

The point of the whole exercise: fit a continuum to data the mask has cleaned,
and look at what is left. Shown on the **sky antenna** (box-air) for three
files that differ in how busy they are.

Each row is one file: the raw auto, the residual `data - model` with **no**
mask applied, and the same residual with the mask applied. The middle panel is
what the continuum fit has to cope with; the right panel is what a downstream
analysis would actually see.
""")

code(r"""
SKY = [
    ("corr_20260717_034832Z.h5", "comb off", False),
    ("corr_20260716_002210Z.h5", "event window", False),
    ("corr_20260717_201113Z.h5", "self-comb on", True),
]
sky_b = []
for fn_, lab, sc in SKY:
    # box-air is the sky-facing antenna; box-gnd is its partner for the cross.
    bb = P.flag_bundle(one_file(fn_), antenna="box-air", partner="box-gnd",
                       self_comb=sc)
    sky_b.append((lab, fn_, bb))
    ii = bb.provenance["products"]["flags"]
    mm = bb.flags
    ibb = (bb.freqs_mhz >= 45) & (bb.freqs_mhz <= 235)
    print(f"{lab:<15} input {bb.provenance['keys'][0]}  dpss_ok={ii['dpss_ok']}  "
          f"infl={ii.get('model_inflation_median', float('nan')):.2f}  "
          f"masked={((mm[:, ibb] & P.RFI_BITS) > 0).mean():.3f}")
""")

code(r"""
def residual_rows(entries, figsize=None):
    # Widget-safe: no `return fig` (ipympl would render it a second time as a
    # static PNG), every panel gets a colorbar slot so the shared axes keep
    # equal widths, and a smaller canvas under ipympl, which renders at a
    # fixed pixel size instead of scaling to the output area.
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    interactive = INTERACTIVE
    n = len(entries)
    if figsize is None:
        figsize = fsize(12.8, 2.9 * n)
    fig, ax = plt.subplots(n, 3, figsize=figsize, sharex=True,
                           squeeze=False, dpi=90 if interactive else None)
    cax = np.empty(ax.shape, dtype=object)
    for r_ in range(n):
        for c_ in range(3):
            cax[r_, c_] = make_axes_locatable(ax[r_, c_]).append_axes(
                "right", size="4%", pad=0.04)
            cax[r_, c_].set_visible(False)

    for r_, (lab, fn_, bb) in enumerate(entries):
        f_ = bb.freqs_mhz
        ibb = (f_ >= P.BAND_ANALYSIS[0]) & (f_ <= P.BAND_ANALYSIS[1])
        prod_ = bb.products["flags"]
        mdl_ = prod_.get("smooth_model")
        ext_ = [f_[ibb][0], f_[ibb][-1], bb.flags.shape[0], 0]
        kw_ = dict(aspect="auto", interpolation="nearest", extent=ext_)

        d_ = np.asarray(bb.data, float)[:, ibb]
        im = ax[r_, 0].imshow(np.log10(np.maximum(d_, 1)), cmap="plasma", **kw_)
        cax[r_, 0].set_visible(True)
        fig.colorbar(im, cax=cax[r_, 0])
        ax[r_, 0].set_ylabel(f"{lab}\nintegration", fontsize=8)

        if mdl_ is None:
            for c_ in (1, 2):
                ax[r_, c_].text(0.5, 0.5, "DPSS fit declined",
                                ha="center", va="center",
                                transform=ax[r_, c_].transAxes, fontsize=9)
            continue

        res_ = d_ - mdl_[:, ibb]
        # One colour scale for both residual panels, set on the MASKED data so
        # the comparison is like-for-like: scaling the left panel to its own
        # (RFI-dominated) range would make the right panel look empty for
        # reasons of normalisation rather than content.
        keep_ = ((bb.flags[:, ibb] & (P.RFI_BITS | P.UNUSABLE_BITS)) == 0)
        v_ = np.percentile(np.abs(res_[keep_]), 99) if keep_.any() else 1.0

        im = ax[r_, 1].imshow(res_, cmap="bwr", vmin=-v_, vmax=v_, **kw_)
        cax[r_, 1].set_visible(True)
        fig.colorbar(im, cax=cax[r_, 1])

        cmap_ = plt.get_cmap("bwr").copy()
        cmap_.set_bad("0.85")
        im = ax[r_, 2].imshow(np.where(keep_, res_, np.nan), cmap=cmap_,
                              vmin=-v_, vmax=v_, **kw_)
        cax[r_, 2].set_visible(True)
        fig.colorbar(im, cax=cax[r_, 2])

    for c_, ttl in enumerate(("raw auto, $\\log_{10}$",
                              "residual, NO mask",
                              "residual, mask applied")):
        ax[0, c_].set_title(ttl, fontsize=9)
    for a_ in ax[-1]:
        a_.set_xlabel("Frequency [MHz]")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.08,
                        wspace=0.30, hspace=0.18)
    if interactive:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_position = "right"


residual_rows(sky_b)
""")

code(r"""
print(f"{'file':<15}{'rms resid':>12}{'rms masked':>12}{'|z|>6 all':>11}"
      f"{'|z|>6 kept':>12}")
print("-" * 62)
for lab, fn_, bb in sky_b:
    prod_ = bb.products["flags"]
    if prod_.get("smooth_model") is None:
        print(f"{lab:<15}  DPSS fit declined")
        continue
    f_ = bb.freqs_mhz
    ibb = (f_ >= P.BAND_ANALYSIS[0]) & (f_ <= P.BAND_ANALYSIS[1])
    d_ = np.asarray(bb.data, float)[:, ibb]
    res_ = d_ - prod_["smooth_model"][:, ibb]
    keep_ = ((bb.flags[:, ibb] & (P.RFI_BITS | P.UNUSABLE_BITS)) == 0)
    z_ = prod_["z_residual"][:, ibb]
    print(f"{lab:<15}{np.std(res_):>12.4g}{np.std(res_[keep_]):>12.4g}"
          f"{(np.abs(z_) > 6).mean():>11.4f}"
          f"{((np.abs(z_) > 6) & keep_).sum() / max(keep_.sum(), 1):>12.4f}")
""")

md(r"""
**Assessment.** The middle and right panels are the before/after that matters.
Unmasked, the residual is dominated by the lines the fit deliberately did not
absorb — they are still in `data - model` because the model refused them, which
is the correct behaviour and the reason the panel looks so busy. With the mask
applied, what remains is the noise the continuum could not follow.

The `rms masked` column is the honest summary: on the comb-off file it drops by
roughly two orders of magnitude relative to the unmasked residual. On the event
file the DPSS fit declines to run (`dpss_ok` False) because pass 1 zero-weights
a wide contiguous block, so there is no residual to show — that is the guard
working, not a gap in the figure.

Colour scales are set from the *masked* residual and shared across both panels,
so the right-hand panel is not made to look empty by its own normalisation.
""")

# -------------------------------------------------------------- campaign

md(r"""
## 12. The same stages across campaign conditions

One file per condition: both wiring phases, both accumulator lengths, comb on
and off, a labelled event window.
""")

code(r"""
# Phase B needs explicit keys: those files' own `input_to_ant` header declares
# {0: box-air, 2: box-gnd, 4: viv-N, 5: viv-E} while the live data keys are
# ['3','35','4','5'] -- neither named antenna is present and load_bundle
# correctly refuses. The header is stale for that wiring. Per
# curation/select_files.py PHASE_INPUTS, input 3 is box-gnd and 5 is the mux
# copy of box-air.
SAMPLES = [
    ("corr_20260712_220026Z.h5", "A start", False, {}),
    ("corr_20260713_023613Z.h5", "A short acc", False, {}),
    ("corr_20260714_120032Z.h5", "B (keys 3/5)", False,
     dict(key="3", partner_key="5")),
    ("corr_20260716_002210Z.h5", "C event window", False, {}),
    ("corr_20260717_034832Z.h5", "C comb off", False, {}),
    ("corr_20260717_201113Z.h5", "C self-comb on", True, {}),
]
rows_out = []
for fn_, lab, sc, extra in SAMPLES:
    try:
        bb = P.flag_bundle(one_file(fn_), self_comb=sc, **extra)
    except Exception as e:
        print(f"{lab}: {type(e).__name__}: {e}")
        continue
    rows_out.append((lab, fn_, bb))

hdr = (f"{'condition':<17}{'sw?':>5}{'xquant':>7}{'infl':>6}{'fit':>6}"
       f"{'apri':>7}{'trans':>7}{'coh':>7}{'resid':>7}{'line':>7}"
       f"{'anyRFI':>8}{'retr':>7}")
print(hdr)
print("-" * len(hdr))
for lab, fn_, bb in rows_out:
    mm = bb.flags
    ii = bb.provenance["products"]["flags"]
    ibb = (bb.freqs_mhz >= 45) & (bb.freqs_mhz <= 235)
    f_ = lambda bit: ((mm[:, ibb] & bit) > 0).mean()
    infl = ii.get("model_inflation_median", np.nan)
    print(f"{lab:<17}{ii['rfswitch_missing_rows'] == 0 and 'yes' or 'NO':>5}"
          f"{str(ii['cross_quantisation_ok']):>7}"
          f"{infl if np.isfinite(infl) else float('nan'):>6.2f}"
          f"{str(ii.get('dpss_ok')):>6}"
          f"{f_(P.BIT_A_PRIORI):>7.3f}{f_(P.BIT_TRANSIENT):>7.3f}"
          f"{f_(P.BIT_COHERENT):>7.3f}{f_(P.BIT_RESIDUAL):>7.3f}"
          f"{f_(P.BIT_LINE):>7.3f}{f_(P.RFI_BITS):>8.3f}"
          f"{f_(P.BIT_RETRACTED):>7.3f}")
""")

code(r"""
n_s = len(rows_out)
fig, ax = plt.subplots(2, n_s, figsize=fsize(2.3 * n_s, 5.6),
                       sharex=True, sharey=True, squeeze=False)
for k, (lab, fn_, bb) in enumerate(rows_out):
    f_ = bb.freqs_mhz
    ext_ = [f_[0], f_[-1], bb.flags.shape[0], 0]
    kw_ = dict(aspect="auto", interpolation="nearest", extent=ext_)
    ax[0, k].imshow(np.log10(np.maximum(bb.data, 1)), cmap="plasma", **kw_)
    ax[0, k].set_title(f"{lab}\n{fn_[5:18]}", fontsize=7.5)
    ax[1, k].imshow((bb.flags & P.RFI_BITS) > 0, cmap="Greys",
                    vmin=0, vmax=1, **kw_)
    ax[1, k].set_xlabel("Freq [MHz]", fontsize=8)
for a_, lb in zip(ax[:, 0], ("raw auto", "any RFI bit")):
    a_.set_ylabel(lb, fontsize=9)
fig.tight_layout()
widget_finish(fig)
""")

md(r"""
**Assessment across conditions.**

- **`sw?` NO in both phase-A rows.** The `rfswitch` stream is absent from the
  early files — 240 rows of `MISSING` each. `on_sky` treats unknown as *on
  sky* (`switch_unknown="assume_sky"`), because the strict reading marks every
  row off-sky and silently disables every detector downstream; the first two
  campaign days then come back clean, which is an absent measurement wearing a
  result's clothes.
- **`xquant` False in both phase-A rows.** The cross is quantisation-limited
  there and the coherence stage is correctly disabled, so `coh` is 0 and the
  per-pixel work falls to `trans` and `resid` — the auto-only path. Not a small
  corner case: it is roughly the first two campaign days.
- **Phase B needs explicit input keys.** Its files' `input_to_ant` header is
  stale (it names inputs that are not live), so the antenna-name path fails
  there and the caller has to say `key="3", partner_key="5"`. A product would
  need that mapping recorded somewhere authoritative rather than passed by
  hand.
- **`infl` sorts the second pass into two regimes.** Near 1 (comb-off, quiet)
  it is calibrated. Well above 1 (phase B, self-comb) the 40 ns continuum
  cannot follow the data and the auto test reads model error as excess power.
  **This is the main open problem**, and it is a continuum-model problem, not a
  threshold one.
- **`coh` collapses in the self-comb era** — the self-suppression of stage 7,
  visible as a number.
- **`resid` carries phase A and phase B.** Bit 11 is the second pass's own
  evidence and needs no cross, which is what keeps those eras covered at all.
""")

# ------------------------------------------------------------ limitations

md(r"""
## 13. Where this stands

Working and calibrated: stages 2, 3, 4, 8, 9. Their thresholds come from the
radiometer equation and a stated false-alarm rate rather than tuning, and they
behave consistently across campaign conditions.

Known limitations, in the order I would fix them:

1. **The second pass is only trustworthy where `model_inflation` ≈ 1.** In
   phase B and the self-comb era it over-promotes. Fixing that means a
   continuum model that fits those eras, not a different threshold.
2. **No completeness measurement.** `flagging/validate.py` has the harness —
   labelled-window recall, false-positive rate on quiet data, an injection
   curve giving the 50%/90% detection thresholds — and none of it has been
   pointed at this. Everything above is descriptive; there is no signal-loss
   number.
3. **Stage 7 self-suppresses on persistently contaminated channels**, because
   $\hat\sigma$ is fitted per channel. Stage 8 covers the steady case; a channel
   contaminated most-but-not-all of the time is covered by neither.
4. **The meteor statistic fires on non-propagation events.** It should require
   the FM/DTV rise to happen without a simultaneous transient elsewhere.
5. **Output is a union across antennas.** v0's product is per-input, and a
   box-air-only event currently masks box-gnd data too. Conservative, but it
   discards good data on a clean antenna — a decision to make deliberately
   before this is written in the `flags/<version>/` schema.
6. **The cross noise floor is a factor ~2 above radiometric and unexplained.**
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
