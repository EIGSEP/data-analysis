"""
Every figure of the E-field memo and its number macros.

Reads output/scores.json and output/beams_maps.npz (compute_beams.py). The tilt
scan and phase-jump figures read the outputs of the scripts in supporting/ from
``EFIELD_SUPPORT`` (default supporting/): compare.json,
block_series.npz, imu_tilt.npz, lag_join.npz; each is skipped if
its inputs are missing. The error budget quotes supporting/step4.json.

Writes output/figures/*.pdf and output/values.tex (the memo's values.tex).
"""

import json
import os

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mounts import NSIDE, OUT, power_scale, tooth_freq  # noqa: E402

FIG = OUT / "figures"
SUPPORT = os.environ.get("EFIELD_SUPPORT", OUT.parent / "supporting")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                     "savefig.bbox": "tight"})
macros = {}


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-12))


def sky(values, pix):
    m = np.full(hp.nside2npix(NSIDE), hp.UNSEEN)
    m[pix] = values
    return m


def fig_maps():
    """Per map tooth: |u_x|^2, |u_y|^2, |u_z|^2 and total, solve vs HFSS vs difference."""
    B = np.load(OUT / "beams_maps.npz")
    for i, d in enumerate(B["map_d"]):
        d = int(d)
        pix, u, t = B[f"pix_{d}"], B[f"u_{d}"], B[f"truth_{d}"]
        # compare_to_truth's single complex factor loses amplitude when per-pixel phases
        # differ from HFSS; for maps, scale power by one real least-squares factor instead.
        u = u * np.sqrt(power_scale(u, t))
        ref = np.max(np.sum(np.abs(t) ** 2, 1))
        rows = [
            ("$|u_x|^2$", np.abs(u[:, 0]) ** 2, np.abs(t[:, 0]) ** 2),
            ("$|u_y|^2$", np.abs(u[:, 1]) ** 2, np.abs(t[:, 1]) ** 2),
            ("$|u_z|^2$", np.abs(u[:, 2]) ** 2, np.abs(t[:, 2]) ** 2),
            ("total gain", np.sum(np.abs(u) ** 2, 1), np.sum(np.abs(t) ** 2, 1)),
        ]
        fig = plt.figure(figsize=(7.2, 7.6))
        for r, (lab, rec, hf) in enumerate(rows):
            a, b = db(rec / ref), db(hf / ref)
            for col, vals, title, lim, cmap in ((1, a, "E-field solve", 30, "viridis"),
                                                (2, b, "HFSS", 30, "viridis"),
                                                (3, a - b, "solve $-$ HFSS", 6, "RdBu_r")):
                hp.mollview(sky(vals, pix), sub=(4, 3, 3 * r + col), fig=fig.number,
                            min=-lim, max=0 if col < 3 else lim, cmap=cmap, title=f"{lab}, {title}",
                            unit="dB", cbar=True, badcolor="0.85", notext=True)
        fig.suptitle(f"{tooth_freq(d):.1f} MHz (tooth d{d}, arm {d % 2}), body frame, "
                     "boresight at top", y=0.995)
        fig.savefig(FIG / f"maps_d{d}.pdf")
        plt.close(fig)
        ok = np.isfinite(a) & np.isfinite(b)
        tag = chr(65 + i)
        macros[f"mapNpix{tag}"] = f"{len(pix)}"
        macros[f"mapRmsDb{tag}"] = (
            f"{np.sqrt(np.average((a - b)[ok] ** 2, weights=10 ** (b[ok] / 10))):.2f}")


def fig_scores():
    """Per-tooth correlation with HFSS for x, y, z and total power under the three mounts."""
    S = json.load(open(OUT / "scores.json"))
    f = np.array([r["freq"] for r in S])
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    mounts = (("R", "0.65", "o", "right-handed"), ("L0", "C1", "^", "left-handed, no tilt"),
              ("L", "C0", "s", "left-handed, tilt +4.7$^\\circ$"))
    for ax, comp in zip(axes, ("x", "y", "z", "power")):
        for tag, col, mk, lab in mounts:
            v = np.array([np.nan if r[tag] is None else r[tag][comp] for r in S])
            ax.plot(f, v, mk, ms=3, color=col, label=lab)
            macros[f"med{comp.capitalize()}{tag.replace('0', 'zero')}"] = f"{np.nanmedian(v):.3f}"
        ax.set_title(comp if comp != "power" else "total power")
        ax.set_xlabel("frequency [MHz]")
        ax.set_ylim(0.75, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("correlation with HFSS")
    axes[3].legend(loc="lower left", fontsize=6)
    fig.savefig(FIG / "scores_vs_freq.pdf")
    plt.close(fig)
    macros["nTeeth"] = str(len(S))


def fig_tilt():
    """Tilt scans: HFSS-free consistency and HFSS rms vs axle tilt, both handednesses."""
    C = json.load(open(os.path.join(SUPPORT, "compare.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.35})
    for key, col, lab in (("R_mean", "C3", "right-handed"), ("L_mean", "C0", "left-handed")):
        t = np.array([r["tau1"] for r in C[key]])
        cons = np.array([(r["cons_even"] + r["cons_odd"]) / 2 for r in C[key]])
        h = np.array([r["hfss"] for r in C[key]])
        axes[0].plot(t, cons, "-o", ms=3, color=col, label=lab)
        axes[1].plot(t, h, "-o", ms=3, color=col, label=lab)
        macros[f"tiltHfss{key[0]}"] = f"{t[np.argmin(h)]:+.0f}"
    axes[0].set_ylabel("HFSS-free consistency")
    axes[1].set_ylabel("HFSS held-out rms")
    for ax in axes:
        ax.axvline(5.87, color="0.5", ls="--", lw=0.8)
        ax.text(6.1, ax.get_ylim()[1], "imu_el", va="top", fontsize=7, color="0.4")
        ax.set_xlabel(r"axle tilt $\tau_1$ [deg] (positive = WNW end low)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.savefig(FIG / "tilt_scan.pdf")
    plt.close(fig)


def fig_budget():
    """Error budget steps (supporting/step4.json, medians over 29 teeth)."""
    stages = ["v0013\n(HFSS-tuned)", "v0005\nsurvey", "+ axle\ntilt", "+ roll-axis\ntilt",
              "+ arm\npolarization"]
    free = [0.115, 0.130, 0.098, 0.097, 0.090]
    hfss = [0.115, 0.130, 0.089, 0.088, 0.081]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    x = np.arange(len(stages))
    ax.plot(x, free, "-o", ms=4, color="C0", label="tilt 3.3$^\\circ$ (HFSS-free)")
    ax.plot(x, hfss, "-o", ms=4, color="0.55", label="tilt 5.7$^\\circ$ (HFSS)")
    ax.axhline(0.052, color="0.4", ls=":", lw=1)
    ax.text(0, 0.055, "white-noise floor 0.052", fontsize=7, color="0.3")
    ax.set_xticks(x, stages, fontsize=7)
    ax.set_ylabel("held-out stripe rms")
    ax.set_ylim(0.04, 0.14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(FIG / "error_budget.pdf")
    plt.close(fig)


def fig_jumps():
    """Self-cal block phase, low vs high half band, and the box's along-axle tilt through the raster."""
    S = np.load(os.path.join(SUPPORT, "block_series.npz"))
    I = np.load(os.path.join(SUPPORT, "imu_tilt.npz"))
    J = np.load(os.path.join(SUPPORT, "lag_join.npz"))
    t0 = J["t"].min()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    btm = S["btm"]
    axes[0].plot(btm, np.degrees(S["Plo"]), ".", ms=3, color="0.5", label="low half (184.6 MHz)")
    axes[0].plot(btm, np.degrees(S["Phi"]), ".", ms=3, color="C0", label="high half (212.9 MHz)")
    axes[0].set_ylabel("block phase [deg]")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].plot(btm, np.abs(S["jump"]), ".", ms=3, color="C3")
    axes[1].set_ylabel("|jump| [deg]")
    axes[2].plot((I["t"] - t0) / 60, I["tilt"], lw=0.5, color="0.3")
    axes[2].set_ylabel("along-axle tilt [deg]")
    axes[2].set_xlabel("minutes after 20:26:00 UTC, 2026-07-17")
    for ax in axes:
        ax.axvspan(20, 40, color="C1", alpha=0.08)
        ax.grid(alpha=0.3)
    axes[2].set_xlim(-1, 63)
    fig.savefig(FIG / "phase_jumps.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for make in (fig_maps, fig_scores, fig_tilt, fig_budget, fig_jumps):
        try:
            make()
        except FileNotFoundError as err:
            print(f"skipped {make.__name__}: {err}")
    with open(OUT / "values.tex", "w") as fh:
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("\n".join(f"{k} = {v}" for k, v in sorted(macros.items())))
