"""Fit and iteratively validate a broad v007 TX-frequency consensus."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from eigsep_data.beam_mapping import HFSSBeamSet
from eigsep_data.beam_mapping.diagnostics import fit_v007_beam_joint, load_v007_data


def _score(fit):
    return float(np.sqrt(np.mean(fit.channel_residual_rms ** 2)))


def _best_of_starts(data, beam, channels, starts):
    fits = []
    for start in starts:
        first = fit_v007_beam_joint(
            data, beam, initial=start, beam_channels=channels)
        fits.append(first)
        fits.append(fit_v007_beam_joint(
            data, beam, initial=np.r_[first.heading, first.alpha_deg],
            beam_channels=channels))
    return min(fits, key=_score)


def fit_consensus(data, beam, channels, starts, max_rms=0.60,
                  max_prune_iterations=3):
    """Jointly fit channels and prune any inconsistent with the final geometry."""
    channels = list(map(int, channels))
    rejected = []
    fit = None
    for _ in range(max_prune_iterations + 1):
        fit = _best_of_starts(data, beam, channels, starts)
        bad = np.asarray(channels)[fit.channel_residual_rms > max_rms].tolist()
        if not bad:
            break
        rejected.extend({"channel": int(ch), "reason": "final_rms"}
                        for ch in bad)
        channels = [ch for ch in channels if ch not in bad]
        starts = [np.r_[fit.heading, fit.alpha_deg]]
    return fit, channels, rejected


def make_summary(fit, output):
    freq = fit.frequency_mhz
    arms = np.array([(int(ch) // 8) % 2 for ch in fit.data_cols])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             constrained_layout=True)
    for arm, marker in ((0, "o"), (1, "s")):
        use = arms == arm
        axes[0].scatter(freq[use], fit.channel_residual_rms[use], marker=marker,
                        label=f"TX arm {arm}")
        axes[1].scatter(freq[use], fit.channel_reduced_chisq[use], marker=marker)
        axes[2].scatter(freq[use], fit.gains[use], marker=marker)
    axes[0].axhline(0.6, color="k", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Normalized residual RMS")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Reduced chi-squared")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Profiled gain")
    axes[2].set_xlabel("Frequency [MHz]")
    fig.suptitle(
        f"v007 multi-frequency consensus: {fit.n_channels} channels; "
        f"heading={np.array2string(fit.heading, precision=3)}; "
        f"alpha={fit.alpha_deg:.1f} deg; shared flags={fit.n_flagged}")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("beam_file")
    ap.add_argument("screen_json")
    ap.add_argument("--max-rms", type=float, default=0.60)
    ap.add_argument("--output-json", default="v007_multichannel_consensus.json")
    ap.add_argument("--output-plot", default="v007_multichannel_consensus.png")
    args = ap.parse_args()
    with open(args.screen_json) as stream:
        screen = json.load(stream)
    data = load_v007_data(args.data_path)
    beam = HFSSBeamSet.from_npz(args.beam_file)
    screen_start = np.r_[screen["consensus_heading"], screen["consensus_alpha_deg"]]
    seed_fit = fit_v007_beam_joint(
        data, beam, beam_channels=[536, 544, 552])
    starts = [screen_start, np.r_[seed_fit.heading, seed_fit.alpha_deg],
              np.array([0.0, 0.0, -1.0, 60.0])]
    fit, channels, rejected = fit_consensus(
        data, beam, screen["recommended_channels"], starts, args.max_rms)
    make_summary(fit, args.output_plot)
    report = {
        "channels": channels,
        "frequencies_mhz": fit.frequency_mhz.tolist(),
        "heading": fit.heading.tolist(),
        "alpha_deg": float(fit.alpha_deg),
        "shared_time_flags": int(fit.n_flagged),
        "joint_normalized_rms": _score(fit),
        "flag_provenance": fit.n_flagged_by_channel.tolist(),
        "normalized_rms": fit.channel_residual_rms.tolist(),
        "reduced_chisq": float(fit.reduced_chisq),
        "channel_reduced_chisq": fit.channel_reduced_chisq.tolist(),
        "gains": fit.gains.tolist(),
        "rejected": rejected,
        "summary_plot": args.output_plot,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    with open(args.output_json, "w") as stream:
        stream.write(rendered + "\n")
