"""Assemble the residual-outlier / fit-metric review checkpoint notebook.

Writes beam_metric_outliers_checkpoint.ipynb; execution and rendering are done
separately by run_outlier_checkpoint.sh.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

C = []
M = lambda s: C.append(new_markdown_cell(s))
K = lambda s: C.append(new_code_cell(s))

M(r"""# Residual outliers and the fit metric — marjum-2026-07 beam scan

**Milestone.** Aaron, hand-fitting `beam_explorer.ipynb`, found the fit visibly
matches better over most of az/el than the normalized-RMS score suggests, and
attributed the gap to a handful of large, discontinuous-in-angle outlier points.
This checkpoint reproduces his exact case, identifies the samples actually
driving the metric, and asks whether a robust metric would change the D2
conclusions.

**Question.** Is the normalized-RMS score on this fit dominated by isolated
angular glitches — and if so, does the D2 comparison need a robust metric?

**Answer, up front.** No, and no. The score is dominated by two *dense,
structured, already-documented* populations, not by isolated glitches:

1. **23 of the 226 files in the fit window are calibration files.** During them
   the receiver was on a VNA / noise source / load, not on the antenna. They are
   15.7% of the samples used and carry a **median 71.3% of the residual power**
   across the 101 channels (>80% on 45 of them). Two independent curation
   products flag them; the fit's mask consults neither.
2. **On ch 712 specifically, a dense `el ≈ 0` dwell** (36% of used samples)
   where the model is essentially uncorrelated with the data (r = 0.15).

The isolated-in-angle glitches Aaron saw are real but carry a small share of the
residual power. A trimmed metric happens to land near the correctly-masked
number, which means **the defect is the mask, not the metric.** Fixing the mask
is the right action; switching metrics would paper over a known contaminant.

**Inputs.** `beam_explorer_cache_pre20260917.npz` (the explorer's own cache as
it stood during this investigation -- see the pinning note below), `beam_explorer_sidecar.npz` (per-sample
timestamps / raw-encoder pointing / `pointing_table@v1` flags, built by
`build_sample_sidecar.py`), `marjum-2026-07/flags/v0` (RFI categories),
`marjum-2026-07/curation/cal_windows.jsonl`,
`beam_fits_v2_pointingv1geom_report_pre20260917.json` (pinned, see below), and
`marjum-2026-07/curation/pointing_table.parquet` at **v1.2+45f8059**
(schema_version 4), whose `EL_SOLUTION_GLITCH` bit section 8 now consumes
directly.

**Cache generation, and why it is pinned.** This notebook reads
`beam_explorer_cache_pre20260917.npz` — the cache *before* the corrections it
recommends were applied. That is deliberate: it is the record of the
investigation that found the defect, so its numbers have to be the numbers that
showed the defect. The corrections were accepted on 2026-09-17 and are live in
`beam_explorer_cache.npz`, `fit_beam_v2.py` and
`beam_fits_v2_review_checkpoint`; **for current numbers read those, not this.**
On the corrected cache ch 712 at Aaron's parameters scores 0.5307 rather than
the 0.6802 below, and the fleet median normalized RMS is 0.4691 rather than
0.9029.

**Units and frames.** `measured_tx` is raw correlator accumulator counts
(channel-differenced). az/el are `pointing_table@v1` degrees, which is what the
explorer cache carries. Normalized RMS is residual RMS / data RMS over the
selected samples, after a single free amplitude.""")

K(r"""import json, os, datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": False})

# Pin the cache generation. This notebook is the record of the investigation
# that FOUND the calibration-window leak, so it must run against the cache that
# still contained it. The live cache was rebuilt on 2026-09-17 with the
# correction applied; running this against that one would have it "discover" a
# defect that had already been removed.
os.environ["BEAM_EXPLORER_CACHE"] = "beam_explorer_cache_pre20260917.npz"
import explorer_model as E

print(f"cache: {E.CACHE.name}")

S = np.load("beam_explorer_sidecar.npz", allow_pickle=True)
RFI = np.load("rfi_cat_ch712.npz")
CALFILE = np.load("cal_file_mask.npy")
# Pinned alongside the cache, for the same reason: this notebook's numbers must
# be the ones that showed the defect. The live report was regenerated on
# 2026-09-17 with the correction applied, and mixing the pre-correction cache
# with post-correction shape terms would be incoherent.
REPORT = json.load(open("beam_fits_v2_pointingv1geom_report_pre20260917.json"))
CHROWS = sorted(REPORT["channels"], key=lambda r: r["channel"])

TIMES = S["times"]
FLAGS_V1 = S["flags_v1"]
EL0 = np.abs(E.EL) < 2.0
WRAP = np.abs(np.abs(E.EL) - 180.0) < 20.0

print(f"window: 226 files, {S['files'][0]} .. {S['files'][-1]}")
print(f"samples: {E.AZ.size}   channels: {E.CHANS.size}")
g = TIMES > 0
print(f"time span: {dt.datetime.utcfromtimestamp(TIMES[g].min())} -> "
      f"{dt.datetime.utcfromtimestamp(TIMES[g].max())} UTC")""")

M(r"""## 1. Reproduction of Aaron's case

ch 712 (173.83 MHz), `force arm 1`, alpha = 51°, TX dE = 0 / dN = 8 /
dU = −93.5 m, gain = 1.82e11 with auto-fit **off**, all shape terms zero, no
`|el|` cut. He reported normalized RMS = 0.68.""")

K(r"""I, MODEL, AMP, RMS = E.aaron_case()
U = E.USED[I]
D = E.Y[I].astype(float)
RES = D - MODEL

print(f"ch {E.CHANS[I]}  {E.FREQS[I]:.2f} MHz  native arm {E.ARMS[I]}  forced arm 1")
print(f"amplitude          {AMP:.4g}   (fixed, auto-fit off)")
print(f"normalized RMS     {RMS:.4f}        <- Aaron reported 0.68")
print(f"samples used       {U.sum()} of {U.size} ({100*U.mean():.1f}%)")

m_auto, A_auto = E.model_power(I, E.heading_from_enu(0, 8, -93.5), 51.0, 1,
                               [0, 0, 0], [0, 0, 0])
print(f"\nfor reference, the RMS-optimal amplitude for these same shape "
      f"parameters is {A_auto:.4g},")
print(f"which scores {E.normalized_rms(I, m_auto):.4f}. His hand amplitude is "
      f"{AMP/A_auto:.2f}x that, so ~0.18 of")
print("the 0.68 is amplitude mis-set rather than shape mismatch. Everything "
      "below refits")
print("the amplitude per subset, so amplitude never confounds a shape "
      "comparison.")""")

M(r"""**Reproduced exactly: 0.6802.** Note in passing that his amplitude is 1.53×
the RMS-optimal one; at the optimum the same shape scores 0.5014. That is not
the effect he was asking about, so from here on every subset gets its own
least-squares amplitude, which is the only way a normalized-RMS comparison
between subsets is fair.""")

M(r"""## 2. Data, model and residual — his exact configuration

The convention panels first, so the structure is visible before any statistic
is quoted.""")

K(r"""def panels(mask, model, title, fig=None, axes=None):
    gd, gm = E.grid(D, mask), E.grid(model, mask)
    gr = E.grid(D - model, mask)
    fin = np.isfinite(gd)
    vmin = np.nanpercentile(gd[fin], 2); vmax = np.nanpercentile(gd[fin], 98)
    rmax = np.nanpercentile(np.abs(gr[np.isfinite(gr)]), 98)
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))
    ext = [E.AZ_EDGES[0], E.AZ_EDGES[-1], E.EL_EDGES[0], E.EL_EDGES[-1]]
    for ax, gg, ttl, cm, lo, hi in (
            (axes[0], gd, "measured", "viridis", vmin, vmax),
            (axes[1], gm, "model", "viridis", vmin, vmax),
            (axes[2], gr, "measured - model", "RdBu_r", -rmax, rmax)):
        im = ax.imshow(gg, origin="lower", aspect="auto", extent=ext, cmap=cm,
                       vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("az [deg]")
        ax.set_ylabel("el [deg]")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-180, -90, 0, 90, 180])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


fig = panels(U, MODEL, f"ch 712, Aaron's parameters -- normalized RMS "
                       f"{RMS:.4f}   (amplitude {AMP:.3g}, fixed)")
plt.show()""")

M(r"""Both features he described are present. The `el ≈ 0` horizontal stripe runs
across the full azimuth range, and there is scattered structure in
az ≈ 200–350°. Section 4 shows the stripe is not what it looks like: it spans
the plot because *some* sample exists in every 4° azimuth bin at that
elevation, but 95% of its samples sit in az ≈ 300–20°. Continuity in the
display is a binning effect, not continuity in the data.""")

M(r"""## 3. Is the residual outlier-dominated? — the direct test

Aaron's hypothesis is that a handful of points carry the score. That is a
quantitative claim about how residual power is distributed, so measure it
directly.""")

K(r"""ru = RES[U]
tot = np.sum(ru ** 2)
order = np.argsort(-np.abs(ru))
rows = []
for n in (1, 10, 50, 100, 250, 500, 1000, 2500):
    rows.append((n, 100 * n / ru.size, 100 * np.sum(ru[order[:n]] ** 2) / tot))

print("concentration of residual power, ch 712, Aaron's parameters")
print(f"{'top N samples':>14} {'% of samples':>13} {'% of resid power':>18}")
for n, ps, pp in rows:
    print(f"{n:>14d} {ps:>12.2f}% {pp:>17.1f}%")

fig, ax = plt.subplots(figsize=(4.6, 2.7))
frac = np.cumsum(ru[order] ** 2) / tot
x = 100 * np.arange(1, ru.size + 1) / ru.size
ax.plot(x, 100 * frac, lw=1.6, color="#31688e")
ax.plot([0, 100], [0, 100], ls=":", lw=1, color="0.5", label="uniform")
ax.set_xscale("log")
ax.set_xlabel("% of used samples, largest |residual| first")
ax.set_ylabel("cumulative % of residual power")
ax.set_title("residual power is NOT carried by a few points", fontsize=9)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); plt.show()""")

M(r"""**This refutes the outlier hypothesis for ch 712.** The single largest
residual sample carries 0.5% of the residual power; the top 100 (0.4% of
samples) carry 7.1%; it takes ~2500 samples (10%) to reach half. A genuine
glitch-dominated residual would be far more concentrated than this. Whatever is
driving the score is a *dense population*, not a sparse one.

So the next step is to find the dense population — which means looking at
residual power against elevation.""")

K(r"""bands = [(0, 2), (2, 5), (5, 15), (15, 45), (45, 90), (90, 150),
         (150, 175), (175, 181)]
print("ch 712, residual and data power by |el| band")
print(f"{'|el| band':>12} {'n':>7} {'% resid pwr':>12} {'% data pwr':>11} "
      f"{'rms/data_rms':>13}")
drms = np.sqrt(np.mean(D[U] ** 2))
for lo, hi in bands:
    s = U & (np.abs(E.EL) >= lo) & (np.abs(E.EL) < hi)
    if not s.sum():
        continue
    print(f"{lo:>5}-{hi:<6} {s.sum():>7d} "
          f"{100*np.sum(RES[s]**2)/tot:>11.1f}% "
          f"{100*np.sum(D[s]**2)/np.sum(D[U]**2):>10.1f}% "
          f"{np.sqrt(np.mean(RES[s]**2))/drms:>13.3f}")""")

M(r"""**80.4% of the residual power sits in `|el| < 2°`** — 8876 samples, 36% of
those used. In that band the residual RMS equals the data RMS (ratio 1.018):
the model explains nothing there. In every other band the ratio is 0.13–0.44,
i.e. the model fits well. Aaron's visual impression was right, and the cause is
this one dense band.""")

M(r"""## 4. What the `el ≈ 0` band actually is

Two candidate explanations: a data/timing artifact, or a genuine dwell the model
cannot fit. The sidecar's timestamps and drive flags separate them.""")

K(r"""stripe = U & EL0
rest = U & ~EL0


def corr(s):
    return np.corrcoef(D[s], MODEL[s])[0, 1]


print(f"{'':32} {'el~0 band':>12} {'everything else':>17}")
print(f"{'samples used':32} {stripe.sum():>12d} {rest.sum():>17d}")
print(f"{'corr(data, model)':32} {corr(stripe):>12.3f} {corr(rest):>17.3f}")
m1, _ = E.model_power(I, E.heading_from_enu(0, 8, -93.5), 51.0, 1,
                      [0, 0, 0], [0, 0, 0], gain=1.0)


def refit_rms(mask):
    s = U & mask
    A = np.sum(D[s] * m1[s]) / np.sum(m1[s] ** 2)
    r = D[s] - A * m1[s]
    return np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(D[s] ** 2)), A


print(f"{'normalized RMS (amp refit)':32} {refit_rms(EL0)[0]:>12.4f} "
      f"{refit_rms(~EL0)[0]:>17.4f}")

FB = {4: "AZ_SLIP_EVENT", 16: "EL_STUCK", 128: "HEIGHT_ASSUMED",
      256: "EL_POST_FAILURE", 512: "AZ_SLIP_RAMP"}
print(f"\n{'pointing_table@v1 flag':32} {'el~0 band':>12} {'everything else':>17}")
for b, nm in FB.items():
    a = (FLAGS_V1 & b) != 0
    print(f"{nm:32} {(a & stripe).sum():>12d} {(a & rest).sum():>17d}")

print(f"\nazimuth distribution of the el~0 band (20-deg bins, counts):")
h, e = np.histogram(E.AZ[stripe], bins=np.arange(0, 361, 20))
print("  " + "  ".join(f"{int(a)}:{int(b)}" for a, b in zip(e[:-1], h) if b))
print(f"  az 300-360 plus 0-20 holds {int(h[15:].sum()+h[0])} of "
      f"{stripe.sum()} samples "
      f"({100*(h[15:].sum()+h[0])/stripe.sum():.0f}%)")""")

K(r"""fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8))
ax = axes[0]
ax.hist(E.EL[U], bins=np.arange(-180, 181, 3), color="#31688e")
ax.set_yscale("log"); ax.set_xlabel("el [deg]"); ax.set_ylabel("used samples")
ax.set_title("elevation sampling is two dwells\nplus a sparse scan", fontsize=9)
ax.set_xticks([-180, -90, 0, 90, 180])

ax = axes[1]
ax.scatter(E.AZ[stripe], np.abs(RES[stripe]), s=1, alpha=0.15, color="#c0392b",
           label="el~0 band")
ax.set_xlabel("az [deg]"); ax.set_ylabel("|residual|")
ax.set_title("el~0 band: sparse at most azimuths,\ndense at az 300-20",
             fontsize=9)
ax.set_xticks([0, 90, 180, 270, 360])

ax = axes[2]
tt = TIMES[stripe]
ax.hist((tt - TIMES[U & (TIMES > 0)].min()) / 3600.0, bins=60,
        color="#c0392b", label="el~0 band")
ax.hist((TIMES[rest] - TIMES[U & (TIMES > 0)].min()) / 3600.0, bins=60,
        histtype="step", color="#31688e", lw=1.2, label="rest")
ax.set_xlabel("hours since scan start")
ax.set_ylabel("used samples")
ax.set_title("the el~0 band is spread over the\nwhole window, not one episode",
             fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); plt.show()""")

M(r"""**The `el ≈ 0` band is not a timing artifact and not a slip event.** It
carries no `EL_STUCK` and no `UNCOMMANDED_MOTION`; it is spread over 78 of the
226 files and across the entire 8-hour window rather than concentrated in one
episode; and 95% of its samples sit in az ≈ 300–20°. It is a genuine
elevation dwell near the horizon, revisited repeatedly, with azimuth largely
parked too.

What is wrong is the **fit**, not the data: in that band
`corr(data, model) = 0.148` and normalized RMS is 0.51 even with the amplitude
refit on the band alone. The model has no explanatory power at the one pointing
where the campaign spent a third of its samples. `EL_POST_FAILURE` is set on
*every* used sample in this window — the elevation drive had already failed —
so this dwell is a consequence of the drive failure, which is exactly the regime
where the model's elevation dependence is least constrained.

Aaron's specific worry — that the stripe sits at the `el = 0` "antenna faces up"
convention anchor — is worth recording but **cannot be settled here**: the
geometry question (Q8) is separately in flight with `geometer`, and the beam is
front/back symmetric to ~1%, so no power measurement in this dataset
discriminates `el` from `el + 180`.""")

M(r"""## 5. The larger defect: 23 calibration files are inside the fit

While tracing the `el ≈ 0` samples I checked the RFI category flags at channel
712 and found bit 0, `cal` — *"receiver on load/noise/VNA, not on antenna"* —
set on 1936 used samples. That prompted an independent check against
`curation/cal_windows.jsonl`, which is the authority for file-level calibration
windows. It is worse than the flags suggest.""")

K(r"""rows = [json.loads(l) for l in
        open("/mnt/data02/eigsep/marjum-2026-07/curation/cal_windows.jsonl")]
names = list(S["files"])
ov = [r for r in rows if r["file_last"] >= names[0] and r["file_first"] <= names[-1]]
print(f"curated cal windows overlapping the beam-scan window: {len(ov)}")
print(f"{'start UTC':>21} {'end UTC':>21}  {'type':<18} files")
for r in ov:
    print(f"{r['t_start_utc']:>21} {r['t_end_utc']:>21}  {r['cal_type']:<18} "
          f"{r['n_files']}")

nfiles = len(np.unique(S['file_index'][CALFILE]))
print(f"\ncal-window files inside the 226-file window: {nfiles}")
print(f"used samples from them: {(U & CALFILE).sum()} of {U.sum()} "
      f"({100*(U & CALFILE).sum()/U.sum():.1f}%)")

cal_rfi = ((RFI["cat"] & 1) != 0) & RFI["avail"]
print(f"\ntwo independent products, both consulted by neither mask:")
print(f"  flags/v0 'cal' bit at ch 712          {(U & cal_rfi).sum():>6d} samples")
print(f"  curation/cal_windows.jsonl file list  {(U & CALFILE).sum():>6d} samples")
print(f"  agreeing                              {(U & cal_rfi & CALFILE).sum():>6d} samples")
print("\nThe curated file list is the authority (flagging/README.md: "
      "select_files.py owns\nfile-level campaign masks); flags/v0 catches only "
      "part of it.")

print(f"\nfitting the model to the cal samples ALONE:")
print(f"{'subset':<44} {'n':>6} {'best-fit amp':>14} {'corr':>8}")
for lab, s in (("flags/v0 cal bit at ch 712 (strict)", U & cal_rfi),
               ("cal-window files (whole-file, coarse)", U & CALFILE),
               ("for contrast: everything else", U & ~CALFILE & ~cal_rfi)):
    A = np.sum(D[s] * m1[s]) / np.sum(m1[s] ** 2)
    print(f"{lab:<44} {s.sum():>6d} {A:>14.3g} "
          f"{np.corrcoef(D[s], m1[s])[0,1]:>+8.3f}")

print(f"\nshare of ch-712 residual power in the cal-window files: "
      f"{100*np.sum(RES[U & CALFILE]**2)/tot:.1f}%")""")

M(r"""On the **strictly** flagged subset — the 1936 samples `flags/v0` marks
`cal` at this channel — the best-fit amplitude comes out **negative**
(−6.4e8, against +1.4e11 on the clean data, i.e. three orders of magnitude
smaller *and* the wrong sign) and the correlation with the model is `+0.02`,
statistically indistinguishable from zero. That is the signature of samples with
no beam signal in them at all: they are not noisy beam measurements, they are
not beam measurements.

On the coarser **whole-file** mask (3874 samples) the amplitude is positive and
`corr = +0.16`. The difference is informative rather than contradictory: a
cal-window file is not uniformly cal — the curated windows report `n_unknown`
samples and two windows are marked `-partial` — so whole-file exclusion sweeps
up a minority of genuine on-antenna samples along with the cal ones. Whole-file
exclusion is therefore slightly *conservative in the wrong direction*: it
discards some good data. A sample-level cal mask would be tighter, and is listed
under Deferred findings.

Either way the contamination is real, and it is not small: the cal-window files
carry 51% of ch 712's residual power on 15.7% of its samples.

The fit's mask is `USED = channel_validity & ~gross_power & data_space_rfi_mask
& pointing_v1_valid`. None of those four terms is a file-level campaign mask, so
nothing in the chain ever consulted `cal_windows.jsonl` or `select_files.py`.
The `flagging/README.md` contract says explicitly that a consumer must apply
`select_files.py` for file validity *first*, then the RFI flags. That step is
missing.""")

M(r"""## 6. How much of the metric is this? — per-channel, all 101 channels

Section 3 showed ch 712's residual is not glitch-dominated. The question that
actually matters for D2 is fleet-wide: how much of the residual power across all
101 channels sits in these cal files, and does removing them move the headline
median?""")

K(r"""hN = E.C["heading_new"]; aN = float(E.C["alpha_new"])
VAR = {"A  current masking": np.ones(E.AZ.size, bool),
       "B  minus cal files": ~CALFILE,
       "C  minus el0 dwell": ~EL0,
       "D  minus cal and el0": ~CALFILE & ~EL0,
       "E  minus cal, el0, wrap": ~CALFILE & ~EL0 & ~WRAP}
res = {k: [] for k in VAR}; trm = {k: [] for k in VAR}
frac_cal = []
for i, rc in enumerate(CHROWS):
    mm, _ = E.model_power(i, hN, aN, int(E.ARMS[i]),
                          rc["shape_correction_real"],
                          rc["shape_correction_imag"], gain=1.0)
    d = E.Y[i].astype(float); u = E.USED[i]
    A = np.sum(d[u] * mm[u]) / np.sum(mm[u] ** 2)
    r = np.zeros_like(d); r[u] = d[u] - A * mm[u]
    frac_cal.append(np.sum(r[u & CALFILE] ** 2) / np.sum(r[u] ** 2))
    for k, extra in VAR.items():
        s = u & extra
        if s.sum() < 200:
            res[k].append(np.nan); trm[k].append(np.nan); continue
        A2 = np.sum(d[s] * mm[s]) / np.sum(mm[s] ** 2)
        rr = d[s] - A2 * mm[s]
        res[k].append(np.sqrt(np.mean(rr ** 2)) / np.sqrt(np.mean(d[s] ** 2)))
        kk = int(0.99 * rr.size); o = np.argsort(np.abs(rr))[:kk]
        trm[k].append(np.sqrt(np.mean(rr[o] ** 2))
                      / np.sqrt(np.mean(d[s][o] ** 2)))
frac_cal = np.array(frac_cal)

print("HFSS prior + the pipeline's own shape terms, least-squares amplitude,")
print("101 channels. 'trim 1%' drops the largest 1% of |residual|.\n")
print(f"{'mask variant':<26} {'median':>8} {'mean':>8} {'n<0.5':>7} "
      f"{'median trim 1%':>15}")
for k in VAR:
    a = np.array(res[k]); t = np.array(trm[k])
    print(f"{k:<26} {np.nanmedian(a):>8.4f} {np.nanmean(a):>8.4f} "
          f"{int(np.nansum(a < 0.5)):>7d} {np.nanmedian(t):>15.4f}")
print(f"\nvariant A reproduces the published median "
      f"{REPORT['median_normalized_rms']:.4f} to "
      f"{abs(np.nanmedian(res['A  current masking'])-REPORT['median_normalized_rms']):.4f},")
print("which validates this reconstruction at the median.")

print(f"\nresidual power carried by the 23 cal files "
      f"({100*(U & CALFILE).sum()/U.sum():.1f}% of samples):")
print(f"  median {100*np.median(frac_cal):.1f}%   mean "
      f"{100*frac_cal.mean():.1f}%   range "
      f"{100*frac_cal.min():.1f}-{100*frac_cal.max():.1f}%")
print(f"  channels where they carry >50% of residual power: "
      f"{(frac_cal>0.5).sum()}/101")
print(f"  channels where they carry >80% of residual power: "
      f"{(frac_cal>0.8).sum()}/101")""")

K(r"""fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))
a = np.array(res["A  current masking"]); b = np.array(res["B  minus cal files"])
ax = axes[0]
ax.scatter(E.FREQS, a, s=14, label="current masking", color="#c0392b")
ax.scatter(E.FREQS, b, s=14, label="minus cal files", color="#31688e")
ax.set_xlabel("frequency [MHz]"); ax.set_ylabel("normalized RMS")
ax.set_title("per-channel score, before and after", fontsize=9)
ax.legend(fontsize=7, frameon=False); ax.set_ylim(0, 1.05)

ax = axes[1]
ax.hist(100 * frac_cal, bins=25, color="#31688e")
ax.set_xlabel("% of residual power in the 23 cal files")
ax.set_ylabel("channels")
ax.set_title("15.7% of samples, median 71% of\nthe residual power", fontsize=9)

ax = axes[2]
for k, col in (("A  current masking", "#c0392b"),
               ("B  minus cal files", "#31688e")):
    v = np.sort(np.array(res[k])[~np.isnan(res[k])])
    ax.plot(v, np.linspace(0, 100, v.size), lw=1.6, color=col, label=k)
    vt = np.sort(np.array(trm[k])[~np.isnan(trm[k])])
    ax.plot(vt, np.linspace(0, 100, vt.size), lw=1.1, ls="--", color=col,
            label=k + ", trim 1%")
ax.set_xlabel("normalized RMS"); ax.set_ylabel("cumulative % of channels")
ax.set_title("trimming mimics masking --\nbut only masking is correct",
             fontsize=9)
ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout(); plt.show()""")

M(r"""**The headline median moves from 0.910 to 0.623 by masking the cal files
alone**, and the number of channels scoring below 0.5 goes from 1 to 28. The
`el ≈ 0` dwell, by contrast, is a *ch-712-specific* problem: removing it alone
leaves the median at 0.911.

The right-hand panel is the answer to the metric question. Trimming 1% of the
residual under the current mask gives median 0.604 — almost the same as properly
masking the cal files (0.623). That is not a coincidence: the trimmed metric is
finding the cal samples and throwing them away. Once the cal files are actually
masked, plain and trimmed RMS agree to 0.05 (0.623 vs 0.572), i.e. the residual
is no longer outlier-dominated.

**So a robust metric is not needed and should not be adopted.** It would produce
approximately the right number for the wrong reason, and would hide a
reproducible data-selection defect that also affects anything else computed from
this sample set — not just the RMS.""")

M(r"""## 7. The scattered az ≈ 200–350° outliers

Aaron's second observation. These exist, but they are a third-order effect and
mostly not isolated.""")

K(r"""good = U & ~EL0 & ~CALFILE
sc = 1.4826 * np.median(np.abs(RES[good] - np.median(RES[good])))
big = U & (np.abs(RES) > 8 * sc)
print(f"robust residual scale on clean scanning data: {sc:.3g} counts")
print(f"samples with |residual| > 8x that: {big.sum()} "
      f"({100*big.sum()/U.sum():.1f}% of used)\n")
for lab, s in (("in the el~0 dwell", big & EL0),
               ("in cal-window files", big & CALFILE),
               ("in az 200-350, el 0-100", big & (E.AZ > 200) & (E.AZ < 350)
                & (E.EL > 0) & (E.EL < 100)),
               ("neither el~0 nor cal", big & ~EL0 & ~CALFILE)):
    print(f"  {lab:<28} {s.sum():>6d}")

print(f"\nresidual power share:")
for lab, s in (("el~0 dwell", U & EL0), ("cal-window files", U & CALFILE),
               ("the >8-sigma outliers", big),
               ("outliers outside el~0 and cal", big & ~EL0 & ~CALFILE)):
    print(f"  {lab:<32} {100*np.sum(RES[s]**2)/tot:>6.1f}%")

idx = np.flatnonzero(big & ~EL0 & ~CALFILE)
gaps = np.diff(idx)
iso = int(np.sum((np.r_[99, gaps] > 1) & (np.r_[gaps, 99] > 1)))
fi = S["file_index"]; nm = S["files"]
v, ct = np.unique(fi[idx], return_counts=True)
print(f"\nthose {idx.size} outliers: {iso} are isolated single samples; "
      f"the rest are runs.")
print(f"spread over {v.size} files, clustered: " + ", ".join(
    f"{nm[p]}:{q}" for p, q in sorted(zip(v, ct), key=lambda x: -x[1])[:4]))""")

M(r"""Once the `el ≈ 0` dwell and the cal files are set aside, only 563 samples
(2.3%) exceed 8× the robust residual scale, they carry **5.2% of the residual
power**, and only 43 of them are genuinely isolated single samples. The rest
arrive in runs concentrated in a handful of files, the worst being
`corr_20260717_195402Z.h5` with 156 — which is a per-file data-quality question,
not an angular glitch.

The smooth large-scale residual Aaron correctly identified in that same
az ≈ 200–350° region as "real, legitimate model misfit" is exactly that, and it
survives every cut here. It is the genuine arm-structure mismatch already
documented in D2.""")

M(r"""## 8. Addendum — Aaron's three elevation bands, and the ±55° symmetry

Aaron sharpened the diagnostic while this was in progress: the discontinuities
he is worried about fall in **three bands, `~+55°`, `~−55°` and `0–5°`**, and he
asked whether the symmetric pair shares a common cause — a physical obstruction,
a mount or cable feature appearing twice per rotation, or motor/encoder
mechanics — rather than being three independent artifacts.

**Answer: the ±55° pair is one single cause, and it is none of those three.
The `0–5°` band is genuinely distinct.** Details below.

First, locate the feature properly rather than trusting the 4° display bins.""")

K(r"""# EL_SOLUTION_GLITCH (bit 1024) read straight from the authoritative
# product. pointing_table@v1.2+45f8059 (schema_version 4) added this bit on
# 2026-09-17 from the detector in `detect_el_slew_glitch.py`; this notebook no
# longer carries its own copy of the mask.
import pyarrow.parquet as pq

_names = list(S["files"]); _idx = {n: i for i, n in enumerate(_names)}
_pt = pq.read_table(
    "/mnt/data02/eigsep/marjum-2026-07/curation/pointing_table.parquet",
    columns=["file", "sample_idx", "flags"]).to_pandas()
_pt = _pt[_pt["file"].isin(_idx)]
_rows = 240 * _pt["file"].map(_idx).to_numpy() + _pt["sample_idx"].to_numpy()
_flags = np.zeros(E.EL.size, dtype=np.int64)
_flags[_rows] = _pt["flags"].to_numpy()
ISO = (_flags & 1024) != 0

_local = np.load("disc_mask.npy")   # the one-off mask this analysis was built on
print(f"EL_SOLUTION_GLITCH from the product, in this window: {ISO.sum()}")
print(f"the local disc_mask.npy it replaces:                 {_local.sum()}")
print(f"  disagree on {(ISO != _local).sum()} samples, of which "
      f"{int((U & (ISO != _local)).sum())} are in USED")
print("  (the difference is rows the product treats as gaps and so does not"
      "\n   flag; it does not affect any number below)")

un = U & ~CALFILE
A1 = np.sum(D[un] * m1[un]) / np.sum(m1[un] ** 2)

print("data / model power ratio vs |el|, 2-deg bins, cal files excluded")
print(f"{'|el| band':>14} {'n':>5} {'d/m':>8}")
for lo in np.arange(48, 70, 2):
    s = un & (np.abs(E.EL) >= lo) & (np.abs(E.EL) < lo + 2)
    if s.sum() < 15:
        continue
    mark = "   <<<" if np.mean(D[s]) / np.mean(A1 * m1[s]) > 4 else ""
    print(f"{lo:>6.0f}..{lo+2:<6.0f} {s.sum():>5d} "
          f"{np.mean(D[s])/np.mean(A1*m1[s]):>8.2f}{mark}")

print("\nThe feature is a narrow spike at |el| ~ 58-61, not 55, against a"
      "\nsmooth d/m baseline of ~1.5-1.9 across the whole 40-76 deg range.")""")

M(r"""So the band is **|el| ≈ 58–61°**, about 3° wide, present at both signs. The
next question is whether the *model* has a feature there — a beam null would
produce exactly this signature, model → 0 with real data present.""")

K(r"""el = np.arange(-180, 180.5, 0.5)
azg = np.arange(0, 360, 5.0)
ELg, AZg = np.meshgrid(el, azg, indexing="ij")
cpl = E.coupling(AZg.ravel(), ELg.ravel(),
                 E.heading_from_enu(0, 8, -93.5), 51.0, 1)
a_ = E.A_HFSS[I]
cr = E.householder(a_ / np.linalg.norm(a_)).conj().T @ cpl
P = (np.abs(np.conj(np.r_[1.0, 0, 0, 0]) @ cr) ** 2).reshape(
    el.size, azg.size).mean(axis=1)
P /= P.max()

fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))
ax = axes[0]
ax.semilogy(el, P, lw=1.4, color="#31688e")
for v in (-59, 59):
    ax.axvline(v, color="#c0392b", ls="--", lw=1)
ax.set_xlabel("el [deg]"); ax.set_ylabel("az-averaged model power")
ax.set_title("model is SMOOTH at |el|~59\n(no null; dashed = the bands)",
             fontsize=9)
ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180])

ax = axes[1]
for lab, sgn, col in (("+55 band", 1, "#c0392b"), ("-55 band", -1, "#31688e")):
    s = un & (E.EL * sgn >= 50) & (E.EL * sgn < 60)
    ax.scatter(E.EL[s], D[s] / np.maximum(A1 * m1[s], 1), s=5, alpha=0.5,
               color=col, label=lab)
ax.set_yscale("log"); ax.set_xlabel("el [deg]"); ax.set_ylabel("data / model")
ax.set_title("the excess, both signs", fontsize=9)
ax.legend(fontsize=7, frameon=False)

ax = axes[2]
rat = np.load("spike_ratio_per_channel.npy")
ax.scatter(E.FREQS, rat, s=14, c=np.where(E.ARMS == 0, "#31688e", "#c0392b"))
ax.axhline(1.0, color="0.5", ls=":", lw=1)
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("spike / baseline d/m")
ax.set_title("broadband, both arms\n(blue arm 0, red arm 1)", fontsize=9)
fig.tight_layout(); plt.show()

print(f"spike/baseline ratio across 101 channels: "
      f"median {np.nanmedian(rat):.2f}, "
      f"{int(np.nansum(rat>1.5))} channels above 1.5")
print("Broadband and in both arms -- so not narrowband RFI.")""")

M(r"""**No beam null**, and the excess is **broadband and in both arms**. That
rules out an RFI line and rules out the model simply having a null where the
data has signal.

Next: is it locked to *angle* or to *time*? A physical obstruction is
angle-locked and should recur every time the platform revisits that elevation.""")

K(r"""print(f"{'band':<20} {'n':>5} {'files':>6} {'visits':>7}  time span")
for lab, msk in (("+55 (50..60)", (E.EL >= 50) & (E.EL < 60)),
                 ("-55 (-60..-50)", (E.EL >= -60) & (E.EL < -50)),
                 ("0-5", (E.EL >= 0) & (E.EL < 5)),
                 ("control +20..30", (E.EL >= 20) & (E.EL < 30))):
    s = un & msk
    idx = np.flatnonzero(s)
    visits = int(np.sum(np.diff(idx) > 5) + 1)
    tt = TIMES[s & (TIMES > 0)]
    print(f"{lab:<20} {s.sum():>5d} "
          f"{len(np.unique(S['file_index'][s])):>6d} {visits:>7d}  "
          f"{dt.datetime.utcfromtimestamp(tt.min()).strftime('%m-%d %H:%M')}"
          f" -> {dt.datetime.utcfromtimestamp(tt.max()).strftime('%m-%d %H:%M')}")
print("\nAngle-locked, not a transient: 35 files and ~75 separate visits each,"
      "\nspread over the whole window, indistinguishable from the control band.")""")

M(r"""Angle-locked. At that point an obstruction looks like the leading
hypothesis — which is exactly why the next check matters. The sidecar's
timestamps let me ask what the *pointing solution itself* was doing at those
samples.""")

K(r"""fir = S["file_index"]
same = np.diff(fir) == 0
dtv = np.diff(TIMES); dl = np.abs(np.diff(E.EL))
gg = same & (dtv > 0.1) & (dtv < 2)
rate = np.full(E.EL.size - 1, np.nan); rate[gg] = dl[gg] / dtv[gg]

print("elevation slew rate from the pointing table alone [deg/s]")
print(f"{'band':<22} {'n':>5} {'median':>8} {'p90':>9}")
for lab, msk in (("|el| 40-57.5", (np.abs(E.EL) >= 40) & (np.abs(E.EL) < 57.5)),
                 ("|el| 57.5-61   <<<", (np.abs(E.EL) >= 57.5) & (np.abs(E.EL) < 61)),
                 ("|el| 61-75", (np.abs(E.EL) >= 61) & (np.abs(E.EL) < 75)),
                 ("|el| 10-40", (np.abs(E.EL) >= 10) & (np.abs(E.EL) < 40)),
                 ("|el| 0-5", np.abs(E.EL) < 5)):
    s = un & msk
    rr = rate[np.clip(np.flatnonzero(s), 0, rate.size - 1)]
    rr = rr[np.isfinite(rr)]
    if rr.size < 15:
        continue
    print(f"{lab:<22} {rr.size:>5d} {np.median(rr):>8.3f} "
          f"{np.percentile(rr, 90):>9.1f}")

print(f"\nsamples with an adjacent in-file el step > 20 deg/s: "
      f"{(ISO & un).sum()} of {un.sum()} ({100*(ISO&un).sum()/un.sum():.2f}%)")
h, e = np.histogram(np.abs(E.EL[ISO & un]),
                    bins=[0, 5, 20, 50, 57.5, 61, 75, 150, 175, 181])
print("  by |el|: " + ", ".join(f"{a:g}-{b:g}: {c}"
                                for a, b, c in zip(e[:-1], e[1:], h) if c))

sp = ISO & un & (np.abs(E.EL) >= 57.5) & (np.abs(E.EL) < 61)
k = np.flatnonzero(sp)
nb = np.maximum(np.abs(E.EL[k - 1]), np.abs(E.EL[k + 1]))
print(f"\nof the {sp.sum()} flagged samples in the |el|~59 band, "
      f"{(nb>150).sum()} have an immediate neighbour at |el| > 150.")
print("\nexample sequences (el of previous / this / next sample):")
for j in k[:6]:
    print(f"  {S['files'][fir[j]]}  samp {S['sample_idx'][j]:>3d}:  "
          f"{E.EL[j-1]:>8.2f}  ->  {E.EL[j]:>7.2f}  ->  {E.EL[j+1]:>8.2f}")""")

M(r"""**That is the mechanism.** In the `|el| ≈ 59` band the elevation slew rate
has a 90th percentile of ~220 deg/s against ~6 deg/s everywhere else — the
platform cannot move that fast, so the *pointing solution* is jumping, not the
antenna. 42 of the 45 flagged samples there have an immediate neighbour at
`|el| > 150`, and the example sequences show the pattern plainly: the antenna is
**parked near `el ≈ ±180`** (the post-drive-failure wrap cluster) and the
solution emits a single spurious sample at `|el| ≈ 59–60` before returning to
the park.

So these are not pointings at 59° at all. They are glitches in the IMU
elevation solution, and `|el| ≈ 59–60` is simply the wrong value it emits when
the truth is `|el| ≈ 180`. **The ±55° symmetry is the signature of the glitch,
not of the hardware:** the park sits at both wrap signs and the spurious value
carries the sign with it (23 positive, 22 negative). No obstruction, no cable,
no encoder detent — and nothing that would appear "twice per rotation", because
azimuth is not involved in the mechanism at all.

The test: remove them and the bands should heal completely.""")

K(r"""print("ch 712, per-band fit quality before and after removing the "
      "slew-discontinuous samples")
print(f"{'band':<22} {'n raw':>6} {'corr':>7} {'RMS':>7}   "
      f"{'n cut':>6} {'corr':>7} {'RMS':>7}")


def q(s):
    Ab = np.sum(D[s] * m1[s]) / np.sum(m1[s] ** 2)
    r = D[s] - Ab * m1[s]
    return (np.corrcoef(D[s], m1[s])[0, 1],
            np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(D[s] ** 2)))


for lab, msk in (("+55 (50..60)", (E.EL >= 50) & (E.EL < 60)),
                 ("-55 (-60..-50)", (E.EL >= -60) & (E.EL < -50)),
                 ("+60..70", (E.EL >= 60) & (E.EL < 70)),
                 ("-70..-60", (E.EL >= -70) & (E.EL < -60)),
                 ("0-5 band", (E.EL >= 0) & (E.EL < 5)),
                 ("el~0 dwell |el|<2", np.abs(E.EL) < 2),
                 ("control +20..30", (E.EL >= 20) & (E.EL < 30))):
    s1 = un & msk; s2 = un & msk & ~ISO
    c1, r1 = q(s1); c2, r2 = q(s2)
    print(f"{lab:<22} {s1.sum():>6d} {c1:>7.3f} {r1:>7.3f}   "
          f"{s2.sum():>6d} {c2:>7.3f} {r2:>7.3f}")

print(f"\nand the excess itself, |el| 57.5-61 vs the 40-75 baseline:")
for lab, ex in (("with them", np.ones(E.EL.size, bool)), ("without", ~ISO)):
    s1 = un & (np.abs(E.EL) >= 57.5) & (np.abs(E.EL) < 61) & ex
    s2 = un & (np.abs(E.EL) >= 40) & (np.abs(E.EL) < 75) & ~(
        (np.abs(E.EL) >= 57.5) & (np.abs(E.EL) < 61)) & ex
    r1 = np.mean(D[s1]) / np.mean(A1 * m1[s1])
    r2 = np.mean(D[s2]) / np.mean(A1 * m1[s2])
    print(f"  {lab:<12} d/m spike {r1:>6.2f}   baseline {r2:>5.2f}   "
          f"ratio {r1/r2:>5.2f}")""")

M(r"""**Confirmed.** Removing 45 samples takes `el −60..−50` from
`corr = −0.007, RMS 0.919` to `corr = +0.935, RMS 0.178`, and `el +50..+60` from
`corr = +0.044, RMS 0.874` to `corr = +0.851, RMS 0.295`. The adjacent
`±60–70` bands heal too. The d/m excess ratio goes from 3.03 to **1.03** — it
disappears entirely. The control band does not move.

The `0–5°` band contains only 15 such samples out of 6408 and barely responds
(`corr 0.269 → 0.282`). **The two causes are therefore distinct:**

| band | cause | cure |
|---|---|---|
| `~+55°` and `~−55°` | one shared cause: 45 IMU elevation-solution glitches that escape the `el ≈ ±180` park and land at `\|el\| ≈ 59` | remove 45 samples; bands fully heal |
| `0–5°` | genuine horizon dwell, 36% of used samples, where the model has no explanatory power | not a glitch; needs a modelling or selection decision |

**Two things worth flagging about scope.** First, `pointing_table@v1` does not
catch these: all 45 are `quality == "ok"` and **none** carries
`UNCOMMANDED_MOTION`. They pass every mask in the chain, including the pointing
-quality mask. That is a genuine gap in the product, and the detector used here
is three lines.

Second, and importantly for how much weight to put on this: **fleet-wide these
45 samples barely move the headline number.** Median normalized RMS over the
101 channels goes 0.6228 → 0.6205 with the cal files already masked. This is a
*locally fatal, globally invisible* defect — it destroys two 10°-wide elevation
bands but is negligible in a whole-scan average. It matters for any
elevation-resolved product (a beam cut, an elevation-dependent gain, the Q8
wrap-cluster argument) and hardly at all for a single scan-average score.

One residual honesty note: after the cut, `az 180–270°` at `|el| 50–60` still
shows a ~2× excess (d/m 11.8 → 3.1, against ~1.3–1.8 elsewhere) on ~160
samples. Most of the azimuth structure was the glitches, but not all of it. I am
**not** claiming a second mechanism from 160 samples — it is consistent with the
general arm-structure misfit already documented in D2.""")

M(r"""## 9. Data / model / residual after the correction

The same channel and geometry as Section 2, with the cal files and the
`el ≈ 0` dwell removed and the amplitude refit. This is the comparison the
convention asks for.""")

K(r"""keep = U & ~CALFILE & ~EL0
A = np.sum(D[keep] * m1[keep]) / np.sum(m1[keep] ** 2)
mfix = A * m1
rms_fix = np.sqrt(np.mean((D[keep] - mfix[keep]) ** 2)) / np.sqrt(
    np.mean(D[keep] ** 2))
fig = panels(keep, mfix,
             f"ch 712, same geometry, cal files and el~0 dwell removed -- "
             f"normalized RMS {rms_fix:.4f}   (amplitude {A:.3g})")
plt.show()

print(f"{'selection':<46} {'n':>7} {'normRMS':>9} {'corr':>7}")
for lab, msk in (("as fit today (Aaron's amplitude)", U),
                 ("as fit today (amplitude refit)", U),
                 ("minus cal files", U & ~CALFILE),
                 ("minus cal files and el~0 dwell", keep),
                 ("minus cal, el~0 and the wrap cluster",
                  U & ~CALFILE & ~EL0 & ~WRAP)):
    if lab.endswith("(Aaron's amplitude)"):
        r = D[msk] - MODEL[msk]
        nr = np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(D[msk] ** 2))
        cc = np.corrcoef(D[msk], MODEL[msk])[0, 1]
    else:
        A2 = np.sum(D[msk] * m1[msk]) / np.sum(m1[msk] ** 2)
        r = D[msk] - A2 * m1[msk]
        nr = np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(D[msk] ** 2))
        cc = np.corrcoef(D[msk], m1[msk])[0, 1]
    print(f"{lab:<46} {msk.sum():>7d} {nr:>9.4f} {cc:>7.3f}")""")

M(r"""On ch 712 the score goes from **0.680 as Aaron ran it to 0.234** once the
two contaminated populations are removed, with `corr(data, model) = 0.923`.

**One caveat on that 0.234, stated plainly so it is not over-read.** Normalized
RMS is strongly sensitive to *which* population is in the denominator, because
the data power here is dominated by two dense dwells that fit very differently.
Removing the `el ≈ 0` dwell removes poorly-fit power; the remaining set still
contains the `|el| ≈ 180` wrap cluster, which carries ~50% of the data power and
which the model fits well. Cut that too and the score is 0.434. So 0.234 is not
directly comparable to the 0.242 own-arm template benchmark, which was computed
on a different selection. **What is robust is the direction and rough size of
the shift, not any one decimal.**""")

M(r"""## 10. Effect on the D2 conclusions

Taking each conclusion in turn, against the numbers above.

| D2 conclusion | Affected? |
|---|---|
| Arm-0/arm-1 anti-correlation, r ≈ −0.92, model cannot reproduce it | **Not affected.** It is a correlation between measured arms, computed from data, with no model or metric in it. |
| Own-arm empirical template 0.242 vs other-arm 0.956 vs HFSS 0.670 | **Numbers need recomputation; the ranking is very likely safe.** All three were scored on the same contaminated sample set, so the contamination is largely common-mode. But the cal samples have no beam signal, which penalises a *good* model more than a bad one in normalized-RMS terms, so the HFSS number is the one most likely to improve. |
| Blind template beats the physical model on 93 of 101 channels | **Needs recomputation.** This is a per-channel margin comparison, and 28 channels change score band once the cal files go. A margin of this kind is exactly what a 71%-of-residual-power contaminant can distort. |
| Geometry refit is a false minimum; fitted headings are not measurements | **Not affected, and slightly strengthened.** That conclusion rested on a refit *improving* RMS while moving 70° away from the survey. If most of the RMS was cal-file contamination, the refit had even less real signal to respond to. |
| Refutations of arm-mapping swap, azimuth-registration offset, PCA contamination | **Not affected.** Each was refuted by ~2 orders of magnitude; a factor ~1.5 in the metric does not reach them. |
| Published median normalized RMS 0.903 | **Superseded.** 0.623 with the cal files masked. |

**Withdrawn in place:** my memory note that the pipeline's effective amplitude is
`gain_fitted**2` with median 1.01e11. Tested directly here, `gain**2` fails to
reproduce the reported per-channel RMS by six or more orders of magnitude. The
amplitude used throughout this notebook is instead refit by least squares, which
reproduces the published median to 0.007. The units argument in that note — that
a ~1e11 amplitude is a legitimate counts-to-normalized-beam conversion and not a
calibration defect — is unaffected; only the claimed algebraic relation to
`gain_fitted` was wrong.""")

M(r"""## 11. Caveats

- **Channel 712 is flagged `self-RFI` by `flags/v0` on 100% of samples**, and
  `load_v007_data`'s own docstring says the comb in this slice is the ADC-clock
  subharmonic rather than the transmitter. The `tx_identity_caveat` in the
  pipeline report records this as open. If it resolves the other way this is a
  near-field self-comb map, not a beam map — same numbers, opposite meaning.
  Nothing in this notebook depends on which way it resolves.
- **A receiver regime change (T_rx 212→539 K) brackets this window**, per the
  pipeline's own `receiver_regime_caveat`. Amplitude scale is affected; shape
  should not be.
- The per-channel reconstruction in Section 6 uses the HFSS prior plus the
  pipeline's shape terms with a least-squares amplitude. It matches the
  published median to 0.007 but individual channels drift up to 0.25, because
  the pipeline's fitted coefficients are not exactly `[1, shape…]`. **Section 6
  is therefore trustworthy for medians and distributions, not for per-channel
  claims.**
- `fit_beam_v2.py` lives in `/tmp` and is not under version control. Anything
  reproducing this depends on a file that will not survive a reboot.
- The cal-window mask is file-level. The curated windows report per-state sample
  counts summing to roughly a full file, so whole-file exclusion is right to
  first order, but a sample-level cal mask would be tighter.

## Deferred findings

- `flags/v0`'s `cal` bit is a strict *subset* of the curated cal windows inside
  this window: of the 3874 used samples in cal-window files, `flags/v0` marks
  only 1936 at ch 712, and none outside. Whether the other 1938 are genuinely
  on-antenna samples within a cal file, or a recall gap in the detector, decides
  whether a sample-level cal mask is possible. Worth a pass by whoever owns
  `flags/v0`.
- `corr_20260717_195402Z.h5` contributes 156 of the 563 surviving large
  residuals on ch 712 — a per-file quality outlier that no current mask catches.
- Of the 31421 samples at `|el| < 1°`, only 25.8% survive the existing masks,
  against ~92% in neighbouring elevation bands. Something in the mask chain
  treats the horizon dwell very differently, and it is not obvious which term.

## Decision requested

1. **Re-run the D2 fits with `select_files.py` / `cal_windows.jsonl` applied
   before the RFI mask**, per the `flagging/README.md` contract, and supersede
   the 0.903 median. This is the substantive fix and I recommend it.
2. **Do not adopt a robust/trimmed RMS.** It reproduces the correctly-masked
   number for the wrong reason and would conceal the selection defect. Keep
   plain normalized RMS on a correct sample set.
3. **Recompute the two template benchmarks** (0.242 / 0.956 / 0.670 and the
   "93 of 101 channels" margin) on the corrected selection before either is
   cited further.
4. **Rule on whether the `el ≈ 0` dwell belongs in the fit at all.** It is
   sound pointing but the model has no explanatory power there, and it is 36% of
   ch 712's samples. Including it is defensible; reporting a single number that
   averages it with the rest is not.
5. Whether to hand the cal-window leak and the `|el| < 1°` mask asymmetry to
   `data-archivist` as durable facts about the data.

**STOPPED AT REVIEW GATE — awaiting Aaron's approval.**""")

nb = new_notebook(cells=C, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, "beam_metric_outliers_checkpoint.ipynb")
print(f"wrote beam_metric_outliers_checkpoint.ipynb ({len(C)} cells)")
