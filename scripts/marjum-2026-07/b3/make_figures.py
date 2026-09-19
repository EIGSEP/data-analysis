"""B3 figures: beam chromaticity residual vs mode count, and the mode budget."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b5b4ae"
SURF = "#fcfcfb"
TARGET = 1e-4

BAND_LABEL = {
    "trough_60_100": "60–100 MHz (trough)",
    "cosmology_50_110": "50–110 MHz (cosmology)",
    "midband_50_130": "50–130 MHz",
    "fullband_50_250": "50–250 MHz (full)",
}
PROD_LABEL = {"bowtie_beam": "bowtie_beam (nside 32, 3.9 MHz)",
              "eigsep_bowtie_v000": "eigsep_bowtie_v000 (nside 64, 1 MHz)"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="major", color=MUTED, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)


def fig_residuals(r):
    bands = ["trough_60_100", "cosmology_50_110", "midband_50_130",
             "fullband_50_250"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True,
                             facecolor=SURF)
    for ax, b in zip(axes, bands):
        style(ax)
        bd = r["bands"][b]
        for prod, col, mk in (("bowtie_beam", S1, "o"),
                              ("eigsep_bowtie_v000", S2, "s")):
            if prod not in bd:
                continue
            y = np.array(bd[prod]["resid_rms_frac"])
            k = np.arange(1, y.size + 1)
            ax.semilogy(k, y, color=col, lw=2, marker=mk, ms=5,
                        markerfacecolor=SURF, markeredgewidth=1.6, zorder=3)
            n = bd[prod]["N_ant_rms"]
            if n:
                ax.plot([n], [y[n - 1]], marker=mk, ms=10, color=col,
                        markerfacecolor=col, zorder=4)
        ax.axhline(TARGET, color=INK, lw=1.4, ls="--", zorder=2)
        na = bd["bowtie_beam"]["N_ant_rms"]
        ax.set_title(BAND_LABEL[b], fontsize=10, color=INK, pad=8)
        ax.text(0.96, 0.93, f"$N_{{\\rm ant}}={na}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, color=INK, weight="bold")
        ax.set_xlim(0.5, 16.5)
        ax.set_xlabel("spectral modes retained  $K$", fontsize=9, color=INK2)
    axes[0].set_ylabel("fractional reconstruction residual\n"
                       "$\\|B-B_K\\|/\\|B\\|$", fontsize=9, color=INK2)
    axes[0].set_ylim(1e-8, 1)
    # direct labels (contrast relief + identity without color alone)
    axes[-1].text(16.2, 4e-3, "bowtie_beam", color=S1, fontsize=9,
                  weight="bold", ha="right")
    axes[-1].text(16.2, 2.2e-5, "eigsep_bowtie_v000", color=S2, fontsize=9,
                  weight="bold", ha="right", va="top")
    axes[-1].text(16.2, TARGET * 1.4, "1 part in $10^4$", color=INK,
                  fontsize=9, ha="right", va="bottom")
    # the v000 plateau is its float32 storage precision, not a physical floor
    axes[0].text(16.2, 1.6e-6, "float32 floor of v000", color=MUTED,
                 fontsize=8, ha="right", va="bottom", style="italic")
    fig.suptitle("EIGSEP bowtie beam: how many spectral modes describe the "
                 "chromaticity?", fontsize=12, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(HERE / "b3_fig1_residual_vs_modes.png", dpi=160,
                facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def fig_budget(r):
    bands = ["trough_60_100", "cosmology_50_110", "midband_50_130",
             "fullband_50_250"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    style(ax1); style(ax2)

    x = np.arange(len(bands))
    nant = [r["bands"][b]["bowtie_beam"]["N_ant_rms"] for b in bands]
    ax1.bar(x, nant, width=0.55, color=S1, zorder=3)
    for xi, v in zip(x, nant):
        ax1.text(xi, v + 0.15, str(v), ha="center", va="bottom",
                 fontsize=11, color=INK, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([BAND_LABEL[b].split(" (")[0] for b in bands],
                        fontsize=9, rotation=12, ha="right")
    ax1.set_ylabel("$N_{\\rm ant}$  (modes to 1 part in $10^4$)",
                   fontsize=9, color=INK2)
    ax1.set_title("Beam modes needed, by band", fontsize=10, color=INK, pad=8)
    ax1.set_ylim(0, max(nant) + 1.6)

    frac = [r["mode_budget"][b]["per_product"]["bowtie_beam"]
            ["frac_dof_remaining"] * 100 for b in bands]
    ax2.bar(x, frac, width=0.55, color=S3, zorder=3)
    for xi, v in zip(x, frac):
        ax2.text(xi, v + 0.4, f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=10, color=INK, weight="bold")
    bl = r["mode_budget"]["_bloom_reference"]["frac_dof_remaining"] * 100
    ax2.axhline(bl, color=S2, lw=2, ls="--", zorder=4)
    ax2.text(len(bands) - 0.45, bl - 1.2, f"BLOOM: {bl:.1f}%", color=S2,
             fontsize=9, ha="right", va="top", weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([BAND_LABEL[b].split(" (")[0] for b in bands],
                        fontsize=9, rotation=12, ha="right")
    ax2.set_ylabel("channels left for the signal  (%)", fontsize=9, color=INK2)
    ax2.set_title("Mode budget at $N_{\\rm fg}=5$:  "
                  "$N_{\\rm modes}=N_{\\rm ant}N_{\\rm fg}$",
                  fontsize=10, color=INK, pad=8)
    ax2.set_ylim(80, 100)
    fig.tight_layout()
    fig.savefig(HERE / "b3_fig2_mode_budget.png", dpi=160, facecolor=SURF,
                bbox_inches="tight")
    plt.close(fig)


def main():
    r = json.loads((HERE / "b3_mode_budget.json").read_text())
    fig_residuals(r)
    fig_budget(r)
    print("wrote b3_fig1_residual_vs_modes.png, b3_fig2_mode_budget.png")


if __name__ == "__main__":
    main()
