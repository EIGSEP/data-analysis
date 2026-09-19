"""Build lidar_explorer.ipynb -- the interactive LIDAR fit explorer.

Usage: _build_lidar_explorer.py [OUT.ipynb]
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'lidar_explorer.ipynb')

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(r"""# Interactive LIDAR fit explorer — marjum-2026-07

Twiddle the sliders, watch the predicted range profile and the residual respond.
Every free parameter in the fit is exposed, including the ones the memo held
fixed.

**Run this cell-by-cell top to bottom, then use the controls at the bottom.**

### What you need locally

| | |
|---|---|
| Python packages | `numpy`, `matplotlib`, `ipywidgets` (and JupyterLab/Notebook) |
| Data | `lidar_explorer_cache.npz` (0.3 MB) — must sit next to this notebook |

**Nothing else.** No `eigsep_terrain` install, no 320 MB DEM mosaic, no pointing
table. `build_lidar_explorer_cache.py` cut the DEM down to the 2.5 MB subtile the
rays can reach and reduced the pointing table to its 665 good returns. The ray
march is reimplemented here in ~15 lines of numpy, and the cache builder
**verifies that reimplementation reproduces `marjum_lidar_constraint.march`
exactly** (max difference 0 m at four widely-separated parameter settings)
before writing the cache — so this notebook cannot silently drift away from the
memo it exists to let you interrogate.

Install what is missing with:

```
pip install numpy matplotlib ipywidgets
```

### What this is for

Two questions in the memo are worth pushing on by hand rather than taking on
faith:

1. **The east/north position.** The memo held `dE`/`dN` fixed at the
   photogrammetric value and then, when asked, found that letting them move
   absorbs some of the second-dip mismatch but does not reproduce across data
   subsets. The sliders let you see the trade directly — in particular how
   `daz` and `dN` swap against each other.
2. **The DEM-error hypothesis.** The memo argues the residual second-dip depth
   is terrain-model error, because a rigid translation can displace a dip but
   not deepen one. There is a `DEM bias` slider so you can try to fix it that
   way and see what it costs elsewhere.""")

code(r"""import json

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display

C = np.load("lidar_explorer_cache.npz", allow_pickle=True)

DEM = C["dem"].astype(np.float64)          # int16 subtile, metres
ROW0, COL0 = int(C["dem_row0"]), int(C["dem_col0"])
RES, SE, SN = float(C["dem_res"]), float(C["dem_se"]), float(C["dem_sn"])

EL, AZ, D, T = C["el"], C["az"], C["d"], C["t"]
DWELL, SWEEP = C["dwell"].astype(bool), C["sweep"].astype(bool)
NEARF, PLUMB = C["nearfield"].astype(bool), C["plumb"].astype(bool)
TMIN = T.min()

ANT = C["antenna"].astype(float)
STEP0, RMAX0 = float(C["march_step"]), float(C["march_rmax"])
EL_PLUMB, EL_DWELL = float(C["el_nadir_plumb"]), float(C["el_nadir_dwell"])
PUB = json.loads(str(C["published"]))
PLUMBFIT = json.loads(str(C["plumb_fit"]))
ENCFG = json.loads(str(C["en_configs"]))
WIN = json.loads(str(C["windows"]))

print("DEM subtile %s at %.2f m  (%.1f MB in memory)" % (DEM.shape, RES, DEM.nbytes / 1e6))
print("returns %d  ->  dwell %d | sweep %d | near-field %d (plumb cluster %d)"
      % (len(EL), DWELL.sum(), SWEEP.sum(), NEARF.sum(), PLUMB.sum()))
print("antenna prior (ENU, m)", ANT)
print("published fit:", PUB)
print("plumb fit    :", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in PLUMBFIT.items()})""")

md(r"""## The model

Two independent pieces of geometry, both exposed below.

**The terrain ray march.** The LIDAR sits at 90° to the antenna boresight, so
only the LIDAR arm matters here — a LIDAR ground return is *not* evidence the
antenna pointed at the ground. The ray's zenith angle is `el − el_nadir + 180`
and its azimuth is `az + daz`. March it from the antenna position in `step`
increments and take the first cell at or below the ray.

**The plumb line.** The near-field returns are the LIDAR striking the PVC frame
carrying the antenna box. The frame hangs under gravity, so a ray at angle `chi`
to it strikes it at `x / sin(chi)`. That is what measures `el_nadir` — and the
link checkbox ties the two panels together, so moving the plumb line moves the
terrain model too. That coupling *is* the §4 result; watch what a degree does.

Note `x/sin(chi)` and `h/cos(chi−90)` are the same function, so this fit cannot
distinguish a vertical line from a plane perpendicular to it. It measures the
direction, not the shape.""")

code(r"""def ground(e, n, bias=0.0):
    # Nearest-neighbour DEM elevation; NaN outside the subtile.
    c = np.round((np.asarray(e) - SE) / RES).astype(np.int64) - COL0
    r = np.round((np.asarray(n) - SN) / RES).astype(np.int64) - ROW0
    ok = (c >= 0) & (c < DEM.shape[1]) & (r >= 0) & (r < DEM.shape[0])
    out = np.full(np.shape(e), np.nan)
    out[ok] = DEM[r[ok], c[ok]] + bias
    return out


def march(origin, el, az_true, el_nadir, step=STEP0, rmax=RMAX0, bias=0.0,
          return_points=False):
    # First DEM intersection range for each ray; NaN where the ray misses.
    chi = np.radians(el - el_nadir + 180.0)       # zenith angle of the LIDAR ray
    a = np.radians(az_true)                       # azimuth, clockwise from north
    v = np.stack([np.sin(chi) * np.sin(a), np.sin(chi) * np.cos(a), np.cos(chi)], -1)
    r = np.arange(step, rmax, step)
    p = np.asarray(origin)[None, None, :] + v[:, None, :] * r[None, :, None]
    below = p[..., 2] <= ground(p[..., 0], p[..., 1], bias)
    rng = np.where(below.any(1), r[np.argmax(below, 1)], np.nan)
    if not return_points:
        return rng
    return rng, np.asarray(origin)[None, :] + v * rng[:, None]


def plumb_model(el, x, el_vertical):
    return x / np.abs(np.sin(np.radians(el - el_vertical)))


# Self-check against the values the memo publishes.
_p = march(ANT + np.array([0., 0., PUB["dU_m"]]), EL[SWEEP], AZ[SWEEP] + PUB["daz_deg"],
           PUB["el_nadir_deg"])
_r = D[SWEEP] - _p
_shape = (D[SWEEP] > 80) & (D[SWEEP] < 130)
print("self-check at the published fit:")
print("  rays hitting terrain %d / %d" % (np.isfinite(_p).sum(), SWEEP.sum()))
print("  shape-set MAD %.3f m   (memo: %.3f)"
      % (np.nanmedian(np.abs(_r[_shape])), ENCFG["published"]["MAD"]))
print("  W2 median     %+.3f m   (memo: %+.3f)"
      % (np.nanmedian(_r[_shape & (EL[SWEEP] >= WIN["W2"][0]) & (EL[SWEEP] < WIN["W2"][1])]),
         ENCFG["published"]["W2"]))""")

md(r"""## Hillshade background

The map panel is optional and off by default — it is the slowest thing here.
The hillshade is rendered once at half resolution and reused; only the ray
footprint is redrawn.""")

code(r"""def hillshade(z, res, az_deg=315.0, alt_deg=45.0):
    gy, gx = np.gradient(z, res, res)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, al = np.radians(360.0 - az_deg + 90.0), np.radians(alt_deg)
    v = (np.sin(al) * np.sin(slope)
         + np.cos(al) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(v, 0, 1)


_sub = DEM[::2, ::2]
HS = hillshade(_sub, RES * 2)
HS_EXTENT = (COL0 * RES + SE, (COL0 + DEM.shape[1]) * RES + SE,
             ROW0 * RES + SN, (ROW0 + DEM.shape[0]) * RES + SN)
print("hillshade %s, extent E %.0f..%.0f  N %.0f..%.0f"
      % (HS.shape, *HS_EXTENT))""")

md(r"""## Controls

**Tips for exploring by hand**

- **`el_nadir` is the dominant knob.** It moves the fit about 1 m of apparent
  antenna height per degree, and it is what changed the memo's answer between
  revisions 1 and 2. The plumb line measures it as 92.25° ± 1.4°; revision 1
  assumed 87.68°. Try both.
- **Watch `daz` and `dN` trade.** Push `dN` to −6 and re-snap `daz`: it drops
  from ~43° to ~36°. That coupling is why the azimuth registration cannot be
  quoted tighter than ~36–47° once E/N is free.
- **Score on the shape set** (measured range 80–130 m). That window is chosen
  from the *measured* values alone so it cannot favour a model. The W2 readout
  is the second dip — the feature under discussion.
- **`DEM bias` shifts the whole terrain up or down.** It is deliberately a blunt
  instrument: if the second-dip mismatch were a simple terrain offset this would
  fix it. Watch what it does to W1 and W3 while you try.
- **`snap daz/dU`** grid-searches the azimuth and height at your current E/N and
  `el_nadir` (~2 s). Use it after moving E/N so you compare best-achievable
  against best-achievable rather than against a stale azimuth.
- **Near-field samples are excluded** from scoring by the `near-field cut`; they
  are the support frame, not terrain. Drop the cut to 0 to see what including
  them does.""")

code(r"""def _fs(desc, val, lo, hi, st, fmt=".2f"):
    return W.FloatSlider(value=val, min=lo, max=hi, step=st, description=desc,
                         continuous_update=False, readout_format=fmt,
                         style={"description_width": "130px"},
                         layout=W.Layout(width="420px"))


daz_w = _fs("daz (deg)", PUB["daz_deg"], 0.0, 90.0, 0.25)
eln_w = _fs("el_nadir (deg)", PUB["el_nadir_deg"], 80.0, 100.0, 0.05)
dU_w = _fs("antenna dU (m)", PUB["dU_m"], -10.0, 10.0, 0.05)
dE_w = _fs("antenna dE (m)", 0.0, -20.0, 20.0, 0.25)
dN_w = _fs("antenna dN (m)", 0.0, -20.0, 20.0, 0.25)
bias_w = _fs("DEM bias (m)", 0.0, -15.0, 15.0, 0.25)

px_w = _fs("plumb x (m)", PLUMBFIT["offset_x_m"], 0.02, 1.00, 0.005, ".3f")
link_w = W.Checkbox(value=True, description="link plumb to el_nadir",
                    indent=False, layout=W.Layout(width="260px"))
pel_w = _fs("plumb el_vertical", PLUMBFIT["el_vertical_deg"], -100.0, -76.0, 0.05)

nf_w = _fs("near-field cut (m)", float(C["nearfield_max"]), 0.0, 40.0, 1.0, ".0f")
lo_w = _fs("score range lo (m)", 80.0, 0.0, 200.0, 5.0, ".0f")
hi_w = _fs("score range hi (m)", 130.0, 10.0, 260.0, 5.0, ".0f")
step_w = W.Dropdown(options=[("0.25 m (slow, precise)", 0.25), ("0.5 m (default)", 0.5),
                             ("1.0 m (fast)", 1.0)], value=STEP0, description="march step",
                    style={"description_width": "130px"}, layout=W.Layout(width="320px"))
xaxis_w = W.Dropdown(options=[("elevation", "el"), ("time", "t")], value="el",
                     description="x axis", style={"description_width": "130px"},
                     layout=W.Layout(width="320px"))
pop_w = W.SelectMultiple(options=[("sweep", "sweep"), ("dwell", "dwell"),
                                  ("near-field", "near")],
                         value=("sweep",), description="show",
                         style={"description_width": "130px"},
                         layout=W.Layout(width="320px", height="70px"))
map_w = W.Checkbox(value=False, description="show map panel (slower)", indent=False,
                   layout=W.Layout(width="260px"))
plumb_w = W.Checkbox(value=True, description="show plumb panel", indent=False,
                     layout=W.Layout(width="260px"))

snap_b = W.Button(description="snap daz/dU", button_style="info",
                  tooltip="grid-search daz and dU at the current E/N and el_nadir")
pub_b = W.Button(description="reset: published")
rev1_b = W.Button(description="reset: revision 1")
en_b = W.Button(description="reset: E/N refit")
print("widgets built")""")

code(r"""def masks(nf_cut, lo, hi):
    near = D < nf_cut
    dwell = DWELL & ~near
    sweep = SWEEP & ~near
    score = sweep & (D > lo) & (D < hi)
    return near, dwell, sweep, score


def evaluate(p):
    near, dwell, sweep, score = masks(p["nf"], p["lo"], p["hi"])
    o = ANT + np.array([p["dE"], p["dN"], p["dU"]])
    sel = sweep | dwell
    pred = np.full(len(EL), np.nan)
    pred[sel] = march(o, EL[sel], AZ[sel] + p["daz"], p["eln"],
                      step=p["step"], bias=p["bias"])
    resid = D - pred
    out = dict(pred=pred, resid=resid, near=near, dwell=dwell,
               sweep=sweep, score=score, origin=o)
    out["nhit"] = int(np.isfinite(pred[sweep]).sum())
    out["mad"] = float(np.nanmedian(np.abs(resid[score]))) if score.sum() else np.nan
    for k, (a, b) in WIN.items():
        m = score & (EL >= a) & (EL < b)
        out[k] = float(np.nanmedian(resid[m])) if m.sum() else np.nan
    out["dwell_med"] = (float(np.nanmedian(resid[dwell])) if dwell.sum() else np.nan)
    return out


def params():
    eln = eln_w.value
    return dict(daz=daz_w.value, eln=eln, dU=dU_w.value, dE=dE_w.value,
                dN=dN_w.value, bias=bias_w.value, step=step_w.value,
                nf=nf_w.value, lo=lo_w.value, hi=hi_w.value,
                px=px_w.value,
                pel=(eln - 180.0) if link_w.value else pel_w.value)


print("evaluation helpers ready")""")

code(r"""out = W.Output()
_busy = {"on": False}


def redraw(*_):
    if _busy["on"]:
        return
    p = params()
    R = evaluate(p)
    npan = 2 + int(plumb_w.value)
    with out:
        out.clear_output(wait=True)
        fig = plt.figure(figsize=(12, 4.8 + 2.8 * (npan - 2)))
        gs = fig.add_gridspec(npan, 1, height_ratios=[2, 1] + [1.6] * (npan - 2),
                              hspace=.55)
        ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        sel = {"sweep": R["sweep"], "dwell": R["dwell"], "near": R["near"]}
        cols = {"sweep": "tab:blue", "dwell": "tab:orange", "near": "tab:red"}
        for name in pop_w.value:
            m = sel[name]
            if not m.sum():
                continue
            x = EL[m] if xaxis_w.value == "el" else (T[m] - TMIN) / 60.0
            k = np.argsort(x)
            xs = x[k]
            # Break the connecting line across large gaps in x; otherwise the
            # two near-field clusters (el ~ -103 and ~ +74) get joined by a
            # line straight across the plot, which looks like data.
            gap = np.r_[False, np.diff(xs) > (5.0 if xaxis_w.value == "el" else 1.0)]
            def brk(v):
                v = np.asarray(v, float).copy()
                v[gap] = np.nan
                return v
            ax0.plot(xs, brk(D[m][k]), "o-", ms=4, lw=.7, color=cols[name],
                     label="%s measured" % name)
            if name != "near":
                ax0.plot(xs, brk(R["pred"][m][k]), "x--", ms=4, lw=.7, color="k",
                         label="%s model" % name)
                ax1.plot(xs, brk(R["resid"][m][k]), "o-", ms=3, lw=.7, color=cols[name])
        if xaxis_w.value == "el":
            for k_, c_ in (("W1", "0.92"), ("W2", "0.82"), ("W3", "0.95")):
                for a_ in (ax0, ax1):
                    a_.axvspan(*WIN[k_], color=c_, zorder=0)
        ax1.axhline(0, color="0.6", lw=.8)
        ax0.set_yscale("log"); ax0.set_ylabel("range [m]")
        ax0.legend(fontsize=6, ncol=3, loc="upper left")
        ax0.set_title("daz %.2f | el_nadir %.2f | dE %+.2f dN %+.2f dU %+.2f | DEM bias %+.2f"
                      % (p["daz"], p["eln"], p["dE"], p["dN"], p["dU"], p["bias"]),
                      fontsize=10)
        ax1.set_ylabel("meas - model [m]"); ax1.set_ylim(-20, 20)
        ax1.set_xlabel("el [deg]" if xaxis_w.value == "el" else "minutes")

        row = 2
        if plumb_w.value:
            axp = fig.add_subplot(gs[row]); row += 1
            ep, rp = EL[R["near"] & (EL < -90)], D[R["near"] & (EL < -90)]
            if len(ep):
                es = np.linspace(ep.min() - 1.5, ep.max() + 1.5, 300)
                axp.plot(ep, rp, "o", ms=5, color="tab:red", label="near-field, up-looking")
                axp.plot(es, plumb_model(es, p["px"], p["pel"]), "-", color="k", lw=1.2,
                         label="plumb x=%.3f m, vertical at el=%.2f" % (p["px"], p["pel"]))
                rms = float(np.sqrt(np.mean((plumb_model(ep, p["px"], p["pel"]) - rp) ** 2)))
                axp.set_title("plumb line (near-field, up-looking): rms %.3f m   "
                              "best published %.3f m" % (rms, PLUMBFIT["rms_resid_m"]),
                              fontsize=9, pad=6)
                axp.set_ylim(0, max(3.0, rp.max() * 1.6))
                axp.legend(fontsize=6); axp.set_xlabel("el [deg]"); axp.set_ylabel("range [m]")
        plt.show()

    with out:
        if map_w.value:
            # Own figure: an equal-aspect map squashed into a wide gridspec row
            # comes out postage-stamp sized.
            figm, axm = plt.subplots(figsize=(7.2, 6.6))
            axm.imshow(HS, cmap="gray", origin="lower", extent=HS_EXTENT, zorder=0)
            m = R["sweep"]
            _, pts = march(R["origin"], EL[m], AZ[m] + p["daz"], p["eln"],
                           step=p["step"], bias=p["bias"], return_points=True)
            good = np.isfinite(pts[:, 0])
            sc = axm.scatter(pts[good, 0], pts[good, 1],
                             c=np.clip(np.abs(R["resid"][m][good]), 0, 20),
                             s=12, cmap="autumn_r", edgecolors="k", linewidths=.2, zorder=3)
            figm.colorbar(sc, ax=axm, label="|resid| [m]")
            axm.plot(R["origin"][0], R["origin"][1], "b*", ms=14, mec="k", zorder=4)
            axm.plot(ANT[0], ANT[1], "c+", ms=10, mew=2, zorder=4)
            lo_e, hi_e = np.nanpercentile(pts[good, 0], [0, 100])
            lo_n, hi_n = np.nanpercentile(pts[good, 1], [0, 100])
            axm.set_xlim(min(lo_e, ANT[0]) - 40, max(hi_e, ANT[0]) + 40)
            axm.set_ylim(min(lo_n, ANT[1]) - 40, max(hi_n, ANT[1]) + 40)
            axm.set_aspect("equal"); axm.set_xlabel("East [m]"); axm.set_ylabel("North [m]")
            axm.set_title("ray footprint; cyan + is the photogrammetric antenna, "
                          "blue star the tuned one", fontsize=9)
            plt.show()

        print("rays hitting terrain %d / %d      scored %d"
              % (R["nhit"], int(R["sweep"].sum()), int(R["score"].sum())))
        print("shape MAD   %7.3f m      (published fit: %.3f)" % (R["mad"], ENCFG["published"]["MAD"]))
        print("W1 median   %+7.3f m      (published: %+.3f)  el %g-%g"
              % (R["W1"], ENCFG["published"]["W1"], *WIN["W1"]))
        print("W2 median   %+7.3f m      (published: %+.3f)  el %g-%g   <-- the second dip"
              % (R["W2"], ENCFG["published"]["W2"], *WIN["W2"]))
        print("W3 median   %+7.3f m      (published: %+.3f)  el %g-%g"
              % (R["W3"], ENCFG["published"]["W3"], *WIN["W3"]))
        print("dwell median residual %+.3f m" % R["dwell_med"])


def on_snap(_):
    # Marches ONLY the scored rays, not the dwell as well: the objective never
    # looks at the dwell, and including it made this button take ~16 s.
    _busy["on"] = True
    snap_b.description = "snapping..."
    try:
        p = params()
        _, _, _, score = masks(p["nf"], p["lo"], p["hi"])
        e_s, a_s, d_s = EL[score], AZ[score], D[score]
        best, bv = None, np.inf
        for a in np.arange(max(0., p["daz"] - 12), p["daz"] + 12.01, 1.0):
            for u in np.arange(p["dU"] - 3, p["dU"] + 3.01, 0.25):
                o = ANT + np.array([p["dE"], p["dN"], u])
                pr = march(o, e_s, a_s + a, p["eln"], step=p["step"], bias=p["bias"])
                ok = np.isfinite(pr)
                if ok.sum() < 0.6 * len(e_s):
                    continue
                v = float(np.median(np.abs(d_s[ok] - pr[ok])))
                if v < bv:
                    best, bv = (a, u), v
        if best:
            daz_w.value, dU_w.value = float(best[0]), float(best[1])
    finally:
        _busy["on"] = False
        snap_b.description = "snap daz/dU"
    redraw()


def _set(daz, eln, dU, dE, dN):
    _busy["on"] = True
    daz_w.value, eln_w.value, dU_w.value, dE_w.value, dN_w.value = daz, eln, dU, dE, dN
    bias_w.value = 0.0
    _busy["on"] = False
    redraw()


snap_b.on_click(on_snap)
pub_b.on_click(lambda _: _set(PUB["daz_deg"], PUB["el_nadir_deg"], PUB["dU_m"], 0., 0.))
rev1_b.on_click(lambda _: _set(34.0, EL_DWELL, -1.25, 0., 0.))
en_b.on_click(lambda _: _set(ENCFG["en_refit"]["daz"], EL_PLUMB, ENCFG["en_refit"]["dU"],
                             ENCFG["en_refit"]["dE"], ENCFG["en_refit"]["dN"]))

for w in (daz_w, eln_w, dU_w, dE_w, dN_w, bias_w, px_w, pel_w, link_w,
          nf_w, lo_w, hi_w, step_w, xaxis_w, pop_w, map_w, plumb_w):
    w.observe(redraw, names="value")

ui = W.VBox([
    W.HTML("<b>Geometry</b>"),
    W.HBox([W.VBox([daz_w, eln_w, dU_w]), W.VBox([dE_w, dN_w, bias_w])]),
    W.HTML("<b>Plumb line (near-field frame returns)</b>"),
    W.HBox([W.VBox([px_w, pel_w]), W.VBox([link_w, plumb_w])]),
    W.HTML("<b>Selection and display</b>"),
    W.HBox([W.VBox([nf_w, lo_w, hi_w]), W.VBox([step_w, xaxis_w, pop_w]), W.VBox([map_w])]),
    W.HBox([snap_b, pub_b, rev1_b, en_b]),
])
display(ui, out)
redraw()""")

md(r"""### Three things worth trying

1. **Reset: revision 1**, then **reset: published**. That is the +4.6° change in
   `el_nadir` from the plumb line. Watch the dwell median residual change sign
   (−1.47 → +1.32 m) and the second dip shift.
2. **Reset: published**, then drag `dN` to −6 and press **snap daz/dU**. W2
   improves from −6.4 to about −3.9 and `daz` falls to ~36°. Then look at the
   deep part of the second dip near `el` 118: it is still ~6 m short. That is
   the point the memo makes — a translation displaces a dip, it does not deepen
   one.
3. **Try to close that gap with `DEM bias`.** You can null W2, but watch W1 and
   W3 go the other way as you do it. If you find a setting that flattens all
   three at once, the memo's DEM-error conclusion is wrong and I want to know.""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'arp', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python', 'version': '3.11'}}
nbf.write(nb, OUT)
print('wrote', OUT)
