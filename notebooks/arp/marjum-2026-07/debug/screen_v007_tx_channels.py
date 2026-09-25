"""Screen all HFSS-supported TX tones against a fixed consensus geometry."""

import argparse
import json

import numpy as np

from eigsep_data.beam_mapping import TransmitterGeometry
from eigsep_data.beam_mapping import HFSSBeamSet, simulate_hfss
from eigsep_data.beam_mapping.diagnostics import (
    _beam_slice_at_frequency,
    channel_validity_masks,
    fit_v007_beam_joint,
    gross_power_time_flags,
    load_v007_data,
    tx_arm_for_channel,
)


def score_channel(data, beam, consensus_fit, channel,
                  gross_reference_percentile=99.0,
                  gross_outlier_factor=5.0):
    """Profile one channel's gain at fixed consensus heading/polarization."""
    channel = int(channel)
    gross, _, _ = gross_power_time_flags(
        data, [channel], gross_reference_percentile, gross_outlier_factor)
    shared_flags = consensus_fit.flagged_mask | gross
    valid = channel_validity_masks(
        data, [channel], threshold=None,
        shared_time_flags=shared_flags)[:, 0]
    y = data["measured_tx"][:, channel].astype(float)
    sigma = data["measured_sigma"][:, channel].astype(float)
    frequency = float(data["freqs"][channel])
    one_beam = _beam_slice_at_frequency(beam, frequency)
    geometry = TransmitterGeometry(
        consensus_fit.heading, consensus_fit.alpha_deg)
    arm = tx_arm_for_channel(channel)
    model, _ = simulate_hfss(
        one_beam, data["az_deg"], data["el_deg"], geometry,
        np.full(data["az_deg"].size, arm, dtype=int))
    model = model[0]
    gain = max(0.0, np.sum(y[valid] * model[valid]) /
               max(np.sum(model[valid] ** 2), 1e-30))
    residual = y[valid] - gain * model[valid]
    initial_rms = float(np.sqrt(np.mean(y[valid] ** 2)))
    rms = float(np.sqrt(np.mean(residual ** 2)) / initial_rms)
    chisq = float(np.sum((residual / sigma[valid]) ** 2))
    reduced_chisq = chisq / max(int(valid.sum()) - 1, 1)
    return {
        "channel": channel,
        "frequency_mhz": frequency,
        "tx_arm": arm,
        "n_samples": int(valid.sum()),
        "n_gross_flags": int(gross.sum()),
        "gain": gain,
        "normalized_rms": rms,
        "reduced_chisq": reduced_chisq,
        "initial_rms": initial_rms,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("beam_file")
    ap.add_argument("--consensus-channels", type=int, nargs="+", default=[
        504, 520, 528, 536, 544, 552, 560, 568, 576, 584])
    ap.add_argument("--max-rms", type=float, default=0.60)
    ap.add_argument("--output-json", default="v007_tx_channel_screen.json")
    args = ap.parse_args()
    data = load_v007_data(args.data_path)
    beam = HFSSBeamSet.from_npz(args.beam_file)
    consensus = fit_v007_beam_joint(
        data, beam, beam_channels=args.consensus_channels)
    df = float(data["freqs"][1] - data["freqs"][0])
    first = int(np.ceil(beam.freqs_mhz.min() / df / 8.0) * 8)
    last = int(np.floor(beam.freqs_mhz.max() / df / 8.0) * 8)
    channels = np.arange(first, last + 1, 8, dtype=int)
    scores = [score_channel(data, beam, consensus, ch) for ch in channels]
    recommended = [row["channel"] for row in scores
                   if row["normalized_rms"] <= args.max_rms]
    report = {
        "consensus_channels": list(args.consensus_channels),
        "consensus_heading": consensus.heading.tolist(),
        "consensus_alpha_deg": float(consensus.alpha_deg),
        "consensus_shared_flags": int(consensus.n_flagged),
        "max_rms": args.max_rms,
        "recommended_channels": recommended,
        "scores": scores,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    with open(args.output_json, "w") as stream:
        stream.write(rendered + "\n")
