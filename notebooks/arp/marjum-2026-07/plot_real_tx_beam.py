"""Generate a measured-versus-HFSS TX beam diagnostic image."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rotation_beam import TransmitterGeometry
from tx_beam_sim import HFSSBeamSet, ground_heading, simulate_hfss


def _metadata_series(records, name, n):
    out = np.full(n, np.nan)
    for i, rec in enumerate(records or []):
        if rec is not None and name in rec:
            out[i] = rec[name]
    good = np.flatnonzero(np.isfinite(out))
    return np.interp(np.arange(n), good, out[good]) if good.size else np.zeros(n)


def make_diagnostic(data_file, beam_file, output, key=None, height_m=92.5):
    from eigsep_observing import io

    data, header, metadata = io.read_hdf5(data_file)
    key = key or ("2" if "2" in data else "4")
    auto = np.asarray(data[key], float)
    tx_ch = np.arange(16, auto.shape[1], 16)
    measured = auto[:, tx_ch] - .5 * (auto[:, tx_ch-1] + auto[:, tx_ch+1])
    beam = HFSSBeamSet.from_npz(beam_file)
    motor = metadata.get("motor", [])
    cal = 180.0 / 1.13e4
    az = _metadata_series(motor, "az_pos", auto.shape[0]) * cal
    el = _metadata_series(motor, "el_pos", auto.shape[0]) * cal
    geom = TransmitterGeometry(ground_heading(0, 0, height_m), 0.0)
    model = np.empty((auto.shape[0], beam.beam_cart.shape[0]))
    for fi in range(model.shape[1]):
        model[:, fi] = simulate_hfss(
            beam, az, el, geom, np.full(auto.shape[0], fi % 2)
        )[0][fi]
    df_mhz = float(header["freqs"][1] - header["freqs"][0])
    model_ch = np.rint((beam.freqs_mhz - tx_ch[0] * df_mhz) / (16 * df_mhz)).astype(int)
    valid = (model_ch >= 0) & (model_ch < measured.shape[1])
    obs = measured[:, model_ch[valid]]
    mdl = model[:, valid]
    good = (obs > 0) & (mdl > 0) & np.isfinite(obs) & np.isfinite(mdl)
    scale = float(np.median(obs[good] / mdl[good])) if np.any(good) else 1.0
    mdl *= scale
    corr = float(np.corrcoef(np.log10(obs[good]), np.log10(mdl[good]))[0, 1]) if np.sum(good) > 2 else np.nan

    freq_obs = header["freqs"][tx_ch[0]:tx_ch[-1]+1:16]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = (freq_obs[0], freq_obs[-1], 0, measured.shape[0])
    v = np.nanpercentile(np.abs(measured), [5, 99.5])
    ax[0].imshow(np.log10(np.maximum(np.abs(measured), 1)), aspect="auto", origin="lower", extent=extent)
    ax[0].set_title("Measured TX spikes")
    ax[1].imshow(np.log10(np.maximum(np.abs(mdl), 1)), aspect="auto", origin="lower", extent=extent)
    ax[1].set_title("HFSS model (scaled)")
    ax[2].scatter(np.log10(obs[good]), np.log10(mdl[good]), s=5, alpha=.4)
    lim = ax[2].get_xlim()
    ax[2].plot(lim, lim, "k--")
    ax[2].set_xlim(lim)
    ax[2].set_xlabel("log10 measured baseline-subtracted power")
    ax[2].set_ylabel("log10 modeled power")
    ax[2].set_title(f"corr = {corr:.3f}, scale = {scale:.3g}")
    for a in ax[:2]:
        a.set_xlabel("Frequency [MHz]")
        a.set_ylabel("Integration")
    fig.suptitle(f"{Path(data_file).name}, input {key}, TX height {height_m:g} m")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"key": key, "n_good": int(good.sum()), "scale": scale, "log_corr": corr}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("beam_file")
    ap.add_argument("-o", "--output", default="tx_beam_model_vs_measured.png")
    ap.add_argument("--key")
    ap.add_argument("--height-m", type=float, default=92.5)
    args = ap.parse_args()
    print(make_diagnostic(args.data_file, args.beam_file, args.output, args.key, args.height_m))
