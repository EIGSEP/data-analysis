"""Rescore TX channels with the int32 wrap repaired, to size the bias.

Rebuilds the v007 data dict exactly as ``load_v007_data`` does but with
negative (wrapped) accumulator samples reinterpreted as uint32, then reruns
``score_channel`` on both versions for a handful of channels.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from eigsep_data.beam_mapping.diagnostics import (
    MOTOR_CAL,
    _records_to_array,
    fit_v007_beam_joint,
    load_v007_data,
    radiometer_difference_sigma,
)
from eigsep_data.beam_mapping import HFSSBeamSet
from eigsep_observing import io

from screen_v007_tx_channels import score_channel

CONSENSUS = [504, 520, 528, 536, 544, 552, 560, 568, 576, 584]


def load_repaired(data_path):
    """load_v007_data with single int32 accumulator wraps undone."""
    files = sorted(glob.glob(str(Path(data_path) / "*.h5")))[-185:-150]
    n = len(files) * 240
    pot = np.zeros(n, dtype=np.float32)
    azm = np.zeros(n, dtype=np.float32)
    elm = np.zeros(n, dtype=np.float32)
    times = np.zeros(n, dtype=np.float64)
    accel = np.zeros((n, 3), dtype=np.float32)
    measured_tx = np.zeros((n, 1024), dtype=np.float64)
    measured_sigma = np.zeros((n, 1024), dtype=np.float64)
    freqs = None
    n_repaired = 0
    for i, filename in enumerate(files):
        dat, header, metadata = io.read_hdf5(filename)
        nt = len(header["times"])
        sl = slice(240 * i, 240 * i + nt)
        times[sl] = header["times"]
        accel[sl] = _records_to_array(metadata.get("imu_el"),
                                      ["accel_x", "accel_y", "accel_z"], nt)
        pot[sl] = _records_to_array(metadata.get("potmon"),
                                    ["pot_az_angle"], nt)
        azm[sl], elm[sl] = _records_to_array(
            metadata.get("motor"), ["az_pos", "el_pos"], nt).T
        raw = np.asarray(dat["4"])
        auto = raw.astype(np.float64)
        neg = auto < 0
        n_repaired += int(neg.sum())
        auto[neg] += 2 ** 32
        measured_tx[sl, 1:-1] = auto[:, 1:-1] - 0.5 * (
            auto[:, :-2] + auto[:, 2:])
        bandwidth_hz = abs(float(header["freqs"][1] - header["freqs"][0])) * 1e6
        integration_s = float(header.get("integration_time", np.median(
            np.diff(np.asarray(header["times"], float)))))
        measured_sigma[sl, 1:-1] = radiometer_difference_sigma(
            auto[:, 1:-1], auto[:, :-2], auto[:, 2:],
            bandwidth_hz, integration_s)
        if freqs is None:
            freqs = np.asarray(header["freqs"])
    print(f"repaired {n_repaired} wrapped samples across {len(files)} files")
    return {
        "files": files, "times": times, "accel": accel, "pot": pot,
        "az_deg": azm * MOTOR_CAL, "el_deg": elm * MOTOR_CAL,
        "measured_tx": measured_tx, "measured_sigma": measured_sigma,
        "freqs": freqs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--beam", default="../../../hfss_beam_maps/bowtie_beam.npz")
    ap.add_argument("--channels", type=int, nargs="+",
                    default=[552, 712, 720, 728, 736, 744])
    args = ap.parse_args()

    beam = HFSSBeamSet.from_npz(args.beam)
    raw_data = load_v007_data(args.data)
    fix_data = load_repaired(args.data)

    print("\nfitting consensus geometry on each version ...")
    raw_cons = fit_v007_beam_joint(raw_data, beam, beam_channels=CONSENSUS)
    fix_cons = fit_v007_beam_joint(fix_data, beam, beam_channels=CONSENSUS)
    print(f"  as-is    heading {np.round(raw_cons.heading, 6)} "
          f"alpha {raw_cons.alpha_deg:.4f} flags {raw_cons.n_flagged}")
    print(f"  repaired heading {np.round(fix_cons.heading, 6)} "
          f"alpha {fix_cons.alpha_deg:.4f} flags {fix_cons.n_flagged}")

    hdr = (f"{'ch':>5} {'MHz':>8} | {'gain':>10} {'nrms':>7} {'red chi2':>13} "
           f"{'init rms':>11} | {'gain':>10} {'nrms':>7} {'red chi2':>13} "
           f"{'init rms':>11} | {'gain ratio':>10}")
    print("\n" + " " * 15 + "as-is" + " " * 43 + "repaired")
    print(hdr)
    print("-" * len(hdr))
    for ch in args.channels:
        a = score_channel(raw_data, beam, raw_cons, ch)
        b = score_channel(fix_data, beam, fix_cons, ch)
        print(f"{ch:5d} {a['frequency_mhz']:8.3f} | "
              f"{a['gain']:10.4f} {a['normalized_rms']:7.4f} "
              f"{a['reduced_chisq']:13.4g} {a['initial_rms']:11.4g} | "
              f"{b['gain']:10.4f} {b['normalized_rms']:7.4f} "
              f"{b['reduced_chisq']:13.4g} {b['initial_rms']:11.4g} | "
              f"{a['gain'] / max(b['gain'], 1e-30):10.2f}")


if __name__ == "__main__":
    main()
