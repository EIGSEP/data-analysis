"""Do comb-OFF files inside the v007 fit slice corrupt the fit?

``load_v007_data`` takes files[-185:-150] on faith.  Three of those 35 files
carry no TX comb at all (see check_comb_presence_scan.py), so their spectra are
sky + RFI only.  Nothing in the fit checks for that: the validity mask tests
finiteness and the gross-power flagger only rejects *large* excursions, while a
comb-OFF sample is small.  Measure how many survive and what dropping them does.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_data.beam_mapping import HFSSBeamSet
from eigsep_data.beam_mapping.diagnostics import (
    channel_validity_masks,
    fit_v007_beam_joint,
    gross_power_time_flags,
    load_v007_data,
)

from screen_v007_tx_channels import score_channel

CONSENSUS = [504, 520, 528, 536, 544, 552, 560, 568, 576, 584]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--beam", default="../../../hfss_beam_maps/bowtie_beam.npz")
    ap.add_argument("--presence", default="comb_presence_beam_scan.json")
    args = ap.parse_args()

    presence = {r["file"]: r for r in
                json.load(open(args.presence))["files"]}
    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))[-185:-150]
    off_idx = [i for i, f in enumerate(files)
               if not presence[Path(f).name]["comb_on"]]
    print(f"{len(off_idx)} comb-OFF files of {len(files)}: "
          f"{[Path(files[i]).name for i in off_idx]}")

    data = load_v007_data(args.data)
    n = data["measured_tx"].shape[0]
    comb_off = np.zeros(n, bool)
    for i in off_idx:
        comb_off[240 * i:240 * (i + 1)] = True
    # only rows that actually hold data (load pads each file to 240)
    real = data["times"] > 0
    print(f"comb-OFF spectra: {int((comb_off & real).sum())} of "
          f"{int(real.sum())} real spectra "
          f"({100 * (comb_off & real).sum() / real.sum():.1f}%)")

    channels = np.asarray(CONSENSUS, int)
    gross, _, _ = gross_power_time_flags(data, channels)
    valid = channel_validity_masks(data, channels, shared_time_flags=gross)
    survive = comb_off & real & valid.any(axis=1)
    print(f"comb-OFF spectra surviving existing flagging: "
          f"{int(survive.sum())} "
          f"({100 * survive.sum() / max((comb_off & real).sum(), 1):.1f}%)")

    y = data["measured_tx"]
    on_lvl = np.median(np.abs(y[real & ~comb_off][:, channels]))
    off_lvl = np.median(np.abs(y[real & comb_off][:, channels]))
    print(f"median |measured_tx| on consensus channels: "
          f"comb-ON {on_lvl:.4g}, comb-OFF {off_lvl:.4g} "
          f"(ratio {off_lvl / max(on_lvl, 1e-30):.4f})")

    beam = HFSSBeamSet.from_npz(args.beam)
    print("\nrefitting consensus geometry with and without the comb-OFF files ...")
    full = fit_v007_beam_joint(data, beam, beam_channels=CONSENSUS)
    clean = dict(data)
    clean["times"] = np.where(comb_off, 0.0, data["times"])
    cut = fit_v007_beam_joint(clean, beam, beam_channels=CONSENSUS)
    dot = float(np.clip(np.dot(full.heading, cut.heading), -1, 1))
    print(f"  as-is    heading {np.round(full.heading, 6)} "
          f"alpha {full.alpha_deg:.4f} flags {full.n_flagged}")
    print(f"  comb-cut heading {np.round(cut.heading, 6)} "
          f"alpha {cut.alpha_deg:.4f} flags {cut.n_flagged}")
    print(f"  heading shift {np.degrees(np.arccos(dot)):.4f} deg, "
          f"alpha shift {abs(full.alpha_deg - cut.alpha_deg):.4f} deg")

    print(f"\n{'ch':>5} {'gain as-is':>11} {'gain cut':>11} {'ratio':>7} "
          f"{'nrms as-is':>11} {'nrms cut':>10}")
    for ch in CONSENSUS:
        a = score_channel(data, beam, full, ch)
        b = score_channel(clean, beam, cut, ch)
        print(f"{ch:5d} {a['gain']:11.5f} {b['gain']:11.5f} "
              f"{a['gain'] / max(b['gain'], 1e-30):7.3f} "
              f"{a['normalized_rms']:11.4f} {b['normalized_rms']:10.4f}")


if __name__ == "__main__":
    main()
