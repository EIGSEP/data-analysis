"""Build notebooks/charlie/s11_figures/s11_figures.ipynb (drafts of memo 004's S11 figures)."""
import sys

import nbformat as nbf

out = sys.argv[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# Memo 004's S11 figures (drafts)

Drafts of the figures that show how field S11 sweeps are calibrated to plane P, for review before they go into memo 004's `make_products.py`. Main text: the switch-path S-parameters. Appendix: the internal standards, one antenna and one receiver sweep through every plane, and a closure check of the switch-path file against its raw captures. Next to § 7: the noise-source path's reflection at P. The procedure itself is in Charlie's notes (`memos/memo-004-.../sparam_characterization.tex`) and `EIGSEP.github.io/notebooks/characterizing_system_2026.ipynb`; this notebook draws from the same inputs, now tracked in `marjum-2026-07/data/s11/cal_materials/` (the lab captures in `lab/`), with `eigsep_cal` (installed `main`).

Lab captures were taken on 2026-09-12; the files' names (2026-06-21, 2026-06-29) come from the recording computer's wrong clock. Figures are drawn at the memo's text width (6.5 in) with 8 pt text."""))
C.append(code("""import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import eigsep_cal, eigsep_cal.s11
from eigsep_cal import S11, calkit

CAMPAIGN = Path("/mnt/data02/eigsep-worktrees/memo-004-methods/marjum-2026-07")   # memo branch, until it merges
RAW = Path("/mnt/data02/eigsep/marjum-2026-07")                                     # field S11 sweeps (gitignored bulk)
CAL = CAMPAIGN / "data/s11/cal_materials"
LAB = CAL / "lab"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.25,
                     "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.bbox": "tight"})
W = 6.5                                                  # memo text width, inches
BAND = (50, 245)                                         # the calibration band, MHz
SRC = {"ANT": "#0072B2", "AMB": "#D55E00", "SP1": "#009E73"}
db = lambda x: 20 * np.log10(np.abs(x))
deg = lambda x: np.degrees(np.angle(x))
turns = lambda x: np.unwrap(np.angle(x)) / (2 * np.pi)      # unwrapped phase, in turns
def shade(ax):
    ax.axvspan(*BAND, color="0.94", lw=0, zorder=0)
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
print(eigsep_cal.__file__)"""))

C.append(md("## 1. The switch paths (main text, § 4.3)"))
C.append(code("""sp = dict(np.load(CAL / "switch_sparams.npz"))           # rows: S11, S12·S21, S22 (port 1 at @VNA or P, port 2 at @DUT)
f_sw = np.load(CAL / "vna_internal_osl_20260629T090825Z.npz")["freqs"] / 1e6
order = ["VNAANT", "VNAAMB", "VNASP1", "VNARF", "RFANT", "RFAMB", "RFSP1"]
def style(name):
    if name == "VNARF":
        return dict(color="k", ls="-", lw=1.1)
    return dict(color=SRC[name.replace("VNA", "").replace("RF", "")], ls="-" if name.startswith("VNA") else "--")
fig, axs = plt.subplots(3, 1, figsize=(W, 5.4), sharex=True, constrained_layout=True)
for name in order:
    s = sp[name]
    for ax, row in zip(axs, (1, 0, 2)):
        ax.plot(f_sw, db(s[row]), label=name, **style(name))
axs[0].set(ylabel="|S12·S21| (dB)", title="transmission (path loss)")
axs[1].set(ylabel="|S11| (dB)", title="reflection at port 1 (@VNA or P)")
axs[2].set(ylabel="|S22| (dB)", title="reflection at port 2 (@DUT; P for VNARF)", xlabel="frequency (MHz)", xlim=(1, 250))
for ax in axs:
    shade(ax)
axs[0].legend(ncol=4, loc="lower left")
save(fig, "switch_paths"); plt.show()
b = (f_sw >= BAND[0]) & (f_sw <= BAND[1])
for name in order:
    s = sp[name][:, b]
    print(f"{name:7s} {BAND[0]}-{BAND[1]} MHz: loss {-db(s[1]).max():.2f}–{-db(s[1]).min():.2f} dB; |S11| ≤ {db(s[0]).max():.1f} dB; |S22| ≤ {db(s[2]).max():.1f} dB")"""))
C.append(md("""**Figure (main text).** The seven characterized switch paths (`switch_sparams.npz`, packaged in `eigsep_cal`): transmission $|S_{12}S_{21}|$, the round-trip loss of the path (top), and the reflections at each end (middle, bottom). Colour is the source port (antenna, ambient load, SP1); solid paths join the VNA port @VNA to a source port @DUT (`VNA*`), dashed paths join the LNA port P to it (`RF*`), and `VNARF` (black) joins @VNA to P. The calibration band is shaded. Each was solved in the lab (2026-09-12) from S911T open, short and load standards measured through the path, after removing the VNA and its cable."""))

C.append(md("## 2. The internal standards (appendix)"))
C.append(code("""d = dict(np.load(CAL / "vna_internal_osl_20260629T090825Z.npz"))
f_osl = d["freqs"]
S911T = calkit.S911T(freq_Hz=f_osl)
man = np.array([d["MANUALO"], d["MANUALS"], d["MANUALL"]])         # S911T on the free end of the VNA cable
intl = np.array([d["VNAO"], d["VNAS"], d["VNAL"]])                  # the matrix's internal standards, through the cable
vna_net = calkit.network_sparams(gamma_true=S911T.std_gamma, gamma_meas=man)
intl_cal = calkit.de_embed_sparams(sparams=vna_net, gamma_prime=intl)
pk = np.load(eigsep_cal.s11.PACKAGED_FILES[2026]["osldata"])
d_pk = np.abs(np.array([pk["O"], pk["S"], pk["L"]]) - intl_cal).max()
print(f"rederived internal standards vs eigsep_cal's packaged file: max |difference| {d_pk:.1e}")
f = f_osl / 1e6
fig, axs = plt.subplots(2, 3, figsize=(W, 3.6), sharex=True, constrained_layout=True)
for j, (lab, c) in enumerate(zip("OSL", ("#0072B2", "#D55E00", "#009E73"))):
    for x, ls, name in ((intl[j], ":", "measured through the cable"), (intl_cal[j], "-", "at @VNA (calibrated)")):
        axs[0, j].plot(f, db(x), ls, color=c, label=name)
        axs[1, j].plot(f, turns(x), ls, color=c)
    axs[0, j].set_title({"O": "open", "S": "short", "L": "load"}[lab])
    axs[1, j].set_xlabel("frequency (MHz)")
    for ax in axs[:, j]:
        shade(ax)
axs[0, 0].set_ylabel("|Γ| (dB)"); axs[1, 0].set_ylabel("phase, unwrapped (turns)")
axs[1, 0].legend(*axs[0, 0].get_legend_handles_labels(), loc="lower left")
axs[0, 2].set_ylim(-40, 0)
save(fig, "internal_standards"); plt.show()"""))
C.append(md("""**Figure (appendix).** The switch matrix's internal open, short and load, as the field sweeps use them. Dotted: measured through the VNA cable; solid: at the matrix's VNA port @VNA, after removing the VNA-and-cable network solved from S911T standards on the cable's free end (lab, 2026-09-12). Top: magnitude; bottom: phase, unwrapped, in turns. Each field sweep solves its VNA network against these calibrated values. Rederiving them from the capture reproduces `eigsep_cal`'s packaged file to the printed difference."""))

C.append(md("## 3. One sweep through every plane (appendix)"))
C.append(code("""s = S11(RAW / "data/s11").calibrate(CAL / "switch_sparams.npz", eigsep_cal.s11.PACKAGED_FILES[2026]["osldata"])
fv = np.asarray(s.freqs_mhz, float)
t_night = 1784259000.0                                   # 2026-07-17 ~03:30 UTC, the 16/17 Jul night
ta, _ = s.get_all_s11s("ant", "default"); ia = int(np.argmin(np.abs(np.asarray(ta, float) - t_night)))
tr, _ = s.get_all_s11s("rec", "default"); ir = int(np.argmin(np.abs(np.asarray(tr, float) - t_night)))
PL = {"raw": ("raw VNA reading", "0.6", ":"), "vna": ("@VNA", "#CC79A7", "-"), "dut": ("@DUT", "#0072B2", "-"),
      "lna": ("P", "k", "-"), "default": ("P", "k", "-")}
fig, axs = plt.subplots(2, 2, figsize=(W, 4.0), sharex=True, constrained_layout=True)
for j, (dut, i, planes, title) in enumerate((("ant", ia, ("raw", "vna", "dut", "lna"), "antenna"),
                                             ("rec", ir, ("raw", "vna", "default"), "receiver"))):
    t_, _ = s.get_all_s11s(dut, "default")
    for pl in planes:
        g = np.asarray(s.get_all_s11s(dut, pl)[1])[i]
        lab, c, ls = PL[pl]
        axs[0, j].plot(fv, db(g), ls, color=c, label=lab)
        axs[1, j].plot(fv, turns(g), ls, color=c)
    stamp = np.datetime64(int(np.asarray(t_, float)[i]), "s")
    axs[0, j].set_title(f"{title}, sweep of {stamp} UTC")
    axs[1, j].set_xlabel("frequency (MHz)")
    for ax in axs[:, j]:
        shade(ax)
axs[0, 0].set_ylabel("|Γ| (dB)"); axs[1, 0].set_ylabel("phase, unwrapped (turns)")
axs[0, 0].legend(loc="lower left"); axs[0, 1].legend(loc="lower left")
save(fig, "planes_example"); plt.show()"""))
C.append(md("""**Figure (appendix).** One field sweep of the antenna and one of the receiver, from the 16/17 July night, at each stage of `eigsep_cal.S11`. Antenna (left): the raw VNA reading; at @VNA after removing the VNA network solved from that sweep's internal standards; at @DUT, the antenna port, after removing the `VNAANT` path; and at P after adding the `RFANT` path. Receiver (right): raw, at @VNA, and at P after removing `VNARF`. Top: magnitude; bottom: phase, unwrapped, in turns (its slope is the electrical length each step removes or adds)."""))

C.append(md("## 4. Closure: the switch-path file from its raw captures (appendix)"))
C.append(code("""def read_osl_set(fname):
    with h5py.File(fname, "r") as h:
        raw = {k: h["raw"][k][:] for k in h["raw"]}
        freqs = h["freqs"][:]
        meta = {k: dict(h["metadata_snapshot"][k].attrs) for k in h["metadata_snapshot"]}
    return raw, freqs, meta
files = sorted((LAB / "switch_paths").glob("vna_oslset_*.h5"))
direct = [p for p in files if "VNA_DIRECT" in p.name][0]
raw, fq, _ = read_osl_set(direct)
S911T_sw = calkit.S911T(freq_Hz=fq)
net = calkit.network_sparams(gamma_true=S911T_sw.std_gamma, gamma_meas=np.array([raw["O"], raw["S"], raw["L"]]))
rows = []
for p in files:
    if p == direct:
        continue
    raw, _, meta = read_osl_set(p)
    st = json.loads(meta["L"]["rfswitch"])["sw_state_name"] if isinstance(meta["L"]["rfswitch"], str) else meta["L"]["rfswitch"]["sw_state_name"]
    osl = calkit.de_embed_sparams(sparams=net, gamma_prime=np.array([raw["O"], raw["S"], raw["L"]]))
    rederived = calkit.network_sparams(gamma_true=S911T_sw.std_gamma, gamma_meas=osl)
    rows.append((st, float(np.abs(rederived - sp[st]).max())))
for st, v in rows:
    print(f"{st:7s}: rederived vs switch_sparams.npz, max |difference| {v:.1e}")"""))
C.append(md("""Each switch path solved again from its raw lab captures (`cal_materials/lab/switch_paths/`, 2026-09-12): the VNA-and-cable network from the S911T standards on the cable's free end, removed from the standards measured through each path, which then fix the path's three S-parameters. Every path reproduces the tracked `switch_sparams.npz` to rounding (~1e-15), so the memo quotes this as a number rather than a figure."""))

C.append(md("## 5. The noise-source path's reflection at P (next to § 7)"))
C.append(code("""lab = dict(np.load(LAB / "20261015_131917_vna_data.npz"))
fl = lab["freqs"]
net_l = calkit.network_sparams(gamma_true=calkit.S911T(freq_Hz=fl).std_gamma, gamma_meas=lab["vna"])
g_on = calkit.de_embed_sparams(sparams=net_l, gamma_prime=lab["LNA_NON"])
g_off = calkit.de_embed_sparams(sparams=net_l, gamma_prime=lab["LNA_NOFF"])
rfn = np.load(CAL / "rfn_sparameters_10_15.npz")
print(f"rederived vs rfn_sparameters_10_15.npz: max |difference| {max(np.abs(g_on - rfn['RFNON']).max(), np.abs(g_off - rfn['RFNOFF']).max()):.1e}")
f = fl / 1e6
b = (f >= BAND[0]) & (f <= BAND[1])
fig, axs = plt.subplots(3, 1, figsize=(W, 4.2), sharex=True, constrained_layout=True, height_ratios=(2, 1.4, 1))
axs[0].plot(f, np.abs(g_on), color="#009E73", label="RFNON (diode on)")
axs[0].plot(f, np.abs(g_off), "--", color="#CC79A7", label="RFNOFF (diode off)")
axs[0].set(ylabel="|Γ_NS|")
axs[0].legend()
axs[1].plot(f, deg(g_on), color="#009E73"); axs[1].plot(f, deg(g_off), "--", color="#CC79A7")
axs[1].set(ylabel="phase (deg)")
axs[2].plot(f, np.abs(g_on - g_off), color="k")
axs[2].set(ylabel="|on − off|", xlabel="frequency (MHz)", xlim=(1, 250))
for ax in axs:
    shade(ax)
save(fig, "gamma_ns"); plt.show()
print(f"{BAND[0]}-{BAND[1]} MHz: |Γ_NS| {np.abs(g_on[b]).min():.3f}–{np.abs(g_on[b]).max():.3f}; max |on − off| {np.abs(g_on - g_off)[b].max():.4f}")"""))
C.append(md("""**Figure (next to § 7).** The noise-source path's reflection at P, measured in the lab on 2026-10-15 with the VNA (its cable calibrated with S911T standards at the free end) on the field switch board's LNA port: with the switch on `RFNON` (solid) and `RFNOFF` (dashed), magnitude, phase, and the magnitude of their difference (bottom). The two agree to a few thousandths, as they should with the diode behind a 30 dB pad, so one $\\Gamma_\\mathrm{NS}$ serves both states. The bench measurement of § 7 and `tcal` v0004 use it."""))

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}})
nbf.write(nb, out)
print("wrote", out)
