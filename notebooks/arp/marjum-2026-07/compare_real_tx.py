"""Compare one measured TX file with the 92.5 m-above-ground HFSS model."""

import argparse
import numpy as np

from rotation_beam import TransmitterGeometry
from tx_beam_sim import HFSSBeamSet, ground_heading, simulate_hfss


def _metadata_series(records, name, n):
    out = np.full(n, np.nan)
    for i, rec in enumerate(records or []):
        if rec is not None and name in rec:
            out[i] = rec[name]
    good = np.flatnonzero(np.isfinite(out))
    if not good.size:
        return np.zeros(n)
    return np.interp(np.arange(n), good, out[good])


def main(data_file, beam_file, key=None):
    from eigsep_observing import io
    data, header, metadata = io.read_hdf5(data_file)
    if key is None:
        key = "2" if "2" in data else ("4" if "4" in data else sorted(data)[0])
    auto = np.asarray(data[key], float)
    n = auto.shape[0]
    tx_ch = np.arange(16, auto.shape[1], 16)
    observed = np.median(auto[:, tx_ch] - .5 * (auto[:, tx_ch-1] + auto[:, tx_ch+1]), axis=0)
    motor = metadata.get("motor", [])
    az = _metadata_series(motor, "az_pos", n) * (180.0 / 1.13e4)
    el = _metadata_series(motor, "el_pos", n) * (180.0 / 1.13e4)
    beam = HFSSBeamSet.from_npz(beam_file)
    geom = TransmitterGeometry(ground_heading(0, 0, 92.5), alpha_deg=0)
    # Beam slices start at 46.875 MHz, corresponding to correlator channel 192.
    beam_ch = np.rint((beam.freqs_mhz - tx_ch[0] * header["freqs"][1]) / (16 * header["freqs"][1])).astype(int)
    use = (beam_ch >= 0) & (beam_ch < len(tx_ch))
    pred = np.empty(beam.beam_cart.shape[0])
    for fi in range(pred.size):
        arm = fi % 2
        pred[fi] = simulate_hfss(beam, [np.nanmedian(az)], [np.nanmedian(el)], geom, [arm])[0][fi, 0]
    measured_at_beam = observed[np.clip(beam_ch, 0, len(observed)-1)]
    measured_at_beam = measured_at_beam[use]
    pred = pred[use]
    good = np.isfinite(measured_at_beam) & (measured_at_beam > 0) & np.isfinite(pred) & (pred > 0)
    x, y = np.log10(pred[good]), np.log10(measured_at_beam[good])
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    print(f"file={data_file} key={key} samples={n} valid_tx_channels={good.sum()}")
    print(f"median pointing az/el={np.nanmedian(az):.3f}/{np.nanmedian(el):.3f} deg")
    print(f"log-spectrum correlation={corr:.4f}, fitted log10(data)= {slope:.4f}*log10(model)+{intercept:.4f}")
    return corr, slope, intercept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("beam_file")
    ap.add_argument("--key")
    args = ap.parse_args()
    main(args.data_file, args.beam_file, args.key)
