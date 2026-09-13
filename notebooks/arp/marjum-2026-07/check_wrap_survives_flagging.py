"""Do the wrap-contaminated samples survive the existing v007 flagging?

The wrap injects +2**31 into measured_tx (half of a 2**32 neighbour error,
via the second-difference baseline).  gross_power_time_flags should reject
anything that far above the 99th-percentile beam peak -- but only if the
contaminated channel is in the channel list it is given.  Verify end to end.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from eigsep_data.beam_mapping.diagnostics import (
    channel_validity_masks,
    gross_power_time_flags,
    load_v007_data,
)
from eigsep_observing import io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    data = load_v007_data(args.data)

    # recover which (time, channel) entries are wrap-contaminated
    files = sorted(glob.glob(str(Path(args.data) / "*.h5")))[-185:-150]
    wrapped = np.zeros((len(files) * 240, 1024), bool)
    for i, filename in enumerate(files):
        dat, header, _ = io.read_hdf5(filename)
        raw = np.asarray(dat["4"])
        wrapped[240 * i:240 * i + raw.shape[0]] = raw < 0
    contam = np.zeros_like(wrapped)
    contam[:, 1:-1] = wrapped[:, 1:-1] | wrapped[:, :-2] | wrapped[:, 2:]

    sel = np.arange(504, 785, 8)
    consensus = [504, 520, 528, 536, 544, 552, 560, 568, 576, 584]

    for label, channels in [("consensus set", consensus),
                            ("full comb selection", sel.tolist()),
                            ("single channel 552", [552]),
                            ("single channel 728", [728])]:
        channels = np.asarray(channels, int)
        gross, _, _ = gross_power_time_flags(data, channels)
        valid = channel_validity_masks(data, channels,
                                       shared_time_flags=gross)
        bad = contam[:, channels]
        survived = bad & valid
        print(f"\n--- {label} ({channels.size} channels) ---")
        print(f"  contaminated samples in this set : {int(bad.sum())}")
        print(f"  gross-flagged times              : {int(gross.sum())} "
              f"of {gross.size}")
        print(f"  contaminated AND still valid     : {int(survived.sum())}")
        if survived.any():
            ti, ci = np.nonzero(survived)
            for t, c in zip(ti[:10], ci[:10]):
                ch = channels[c]
                y = data["measured_tx"][t, ch]
                sig = data["measured_sigma"][t, ch]
                col = data["measured_tx"][valid[:, c], ch]
                print(f"    t={t} ch={ch}: y={y:.4g} "
                      f"(median valid {np.median(col):.4g}), sigma={sig:.4g}, "
                      f"chi contribution = {(y / sig) ** 2:.4g}")
        else:
            print("    -> fully rejected by existing flagging")


if __name__ == "__main__":
    main()
