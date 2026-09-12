"""Incrementally admit TX channels into the v007 polarized-beam consensus."""

import argparse
import json

import numpy as np

from tx_beam_sim import HFSSBeamSet
from v007_beam_diagnostic import fit_v007_beam_joint, load_v007_data


def _heading_separation_deg(a, b):
    cosine = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _angle_difference_deg(a, b, period=180.0):
    return float(abs((a - b + period / 2) % period - period / 2))


def grow_consensus(data, beam, seed_channels=(536, 544, 552),
                   candidates=(528, 560, 520, 568, 512, 576),
                   threshold=None, clip_sigma=5.0,
                   max_candidate_rms=0.60, max_heading_shift_deg=10.0,
                   max_alpha_shift_deg=10.0,
                   max_existing_rms_increase=0.05, min_samples=1000):
    """Try candidates in order and admit only channels consistent with the fit."""
    consensus = list(seed_channels)
    current = fit_v007_beam_joint(
        data, beam, threshold=threshold, beam_channels=consensus,
        clip_sigma=clip_sigma)
    trials = []
    for candidate in candidates:
        trial_channels = consensus + [int(candidate)]
        initial = np.r_[current.heading, current.alpha_deg]
        trial = fit_v007_beam_joint(
            data, beam, threshold=threshold, initial=initial,
            beam_channels=trial_channels, clip_sigma=clip_sigma)
        heading_shift = _heading_separation_deg(current.heading, trial.heading)
        alpha_shift = _angle_difference_deg(current.alpha_deg, trial.alpha_deg)
        existing_increase = float(np.max(
            trial.channel_residual_rms[:len(consensus)]
            - current.channel_residual_rms))
        candidate_rms = float(trial.channel_residual_rms[-1])
        candidate_samples = int(trial.n_samples_by_channel[-1])
        reasons = []
        if candidate_rms > max_candidate_rms:
            reasons.append("candidate_rms")
        if candidate_samples < min_samples:
            reasons.append("sample_count")
        if heading_shift > max_heading_shift_deg:
            reasons.append("heading_shift")
        if alpha_shift > max_alpha_shift_deg:
            reasons.append("alpha_shift")
        if existing_increase > max_existing_rms_increase:
            reasons.append("existing_rms_increase")
        accepted = not reasons
        trials.append({
            "candidate": int(candidate),
            "frequency_mhz": float(data["freqs"][candidate]),
            "tx_arm": int((candidate // 8) % 2),
            "accepted": accepted,
            "reasons": reasons,
            "candidate_rms": candidate_rms,
            "candidate_reduced_chisq": float(
                trial.channel_reduced_chisq[-1]),
            "candidate_samples": candidate_samples,
            "heading_shift_deg": heading_shift,
            "alpha_shift_deg": alpha_shift,
            "max_existing_rms_increase": existing_increase,
            "shared_time_flags": int(trial.n_flagged),
        })
        if accepted:
            consensus.append(int(candidate))
            current = trial
    return current, {
        "seed_channels": list(seed_channels),
        "accepted_channels": consensus,
        "trials": trials,
        "final_heading": current.heading.tolist(),
        "final_alpha_deg": float(current.alpha_deg),
        "final_residual_rms": current.channel_residual_rms.tolist(),
        "final_reduced_chisq": float(current.reduced_chisq),
        "final_shared_time_flags": int(current.n_flagged),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("beam_file")
    ap.add_argument("--seed-channels", type=int, nargs="+",
                    default=[536, 544, 552])
    ap.add_argument("--candidates", type=int, nargs="+",
                    default=[528, 560, 520, 568, 512, 576])
    ap.add_argument("--clip-sigma", type=float, default=5.0)
    ap.add_argument("--output-json")
    args = ap.parse_args()
    loaded = load_v007_data(args.data_path)
    hfss = HFSSBeamSet.from_npz(args.beam_file)
    _, report = grow_consensus(
        loaded, hfss, args.seed_channels, args.candidates,
        clip_sigma=args.clip_sigma)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w") as stream:
            stream.write(rendered + "\n")
