"""Continuum-subtracted rotation profiles with an empirical per-bin uncertainty.

Uses the flanking median, which beat every DPSS variant tried in 45-47 at
predicting a comb channel (1.6% vs 2.5%, transmitter-off truth). The excess
T = P_tone - C_hat is formed per integration, binned on rotation angle exactly as
the published figure bins raw power, and its uncertainty taken from the scatter
within each bin. A tone is followed only while its excess is significant.
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter

SIDECAR = Path("/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/deployment5/motor_scan_20260717_key4.h5")
KEY0 = Path("/home/christian/Documents/research/eigsep/data-analysis/notebooks/christian/beam_modulation/key0_raster.npz")
COMB_RESIDUE, COMB_SPILL = 8, (7, 8, 9, 15, 0, 1)
FM_BAND, BIN_DEG, FLAG_DB, MAX_ROUGHNESS = (86.0, 110.0), 5.0, 3.0, 0.06
NSIG = 3.0

with h5py.File(SIDECAR, "r") as f:
    d4 = f["data4"][:].astype(np.float64)
    deg = 1.0 / f.attrs["counts_per_deg"]
    az, el = f["az_counts"][:] * deg, f["el_counts"][:] * deg
    freq = f["freqs"][:]
d4[d4 <= 0] = np.nan
d0 = np.load(KEY0)["d0"].astype(np.float64)
d0[d0 <= 0] = np.nan
chan = np.arange(d4.shape[1])
is_fm = (freq > FM_BAND[0]) & (freq < FM_BAND[1])
finite = np.isfinite(d4).all(0)

turns = np.where(np.diff(np.sign(np.diff(el))) != 0)[0] + 1
rot = [(a, b) for a, b in zip(np.r_[0, turns], np.r_[turns, len(el)]) if b - a > 100]
rot_az = np.array([np.nanmedian(az[a:b]) for a, b in rot])
A, B = rot[int(np.argmin(np.abs(rot_az - (-90.0))))]

edges = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
centres = 0.5 * (edges[:-1] + edges[1:])
bin_idx = np.digitize(el, edges) - 1
ZERO = int(np.argmin(np.abs(centres)))
to_db = lambda x: 10.0 * np.log10(x)


def flanking(c):
    return [c + o for o in range(-6, 7) if 3 <= abs(o) <= 6 and 0 <= c + o < len(chan)
            and finite[c + o] and (c + o) % 16 not in COMB_SPILL and not is_fm[c + o]]


def binned(series, a, b, flag=True):
    """Median and uncertainty-on-the-median per rotation-angle bin."""
    s = series[a:b].copy()
    if flag:
        sm = median_filter(np.nan_to_num(s, nan=np.nanmedian(s)), size=5, mode="nearest")
        s[np.abs(s - sm) > FLAG_DB * np.nanstd(s[np.isfinite(s)])] = np.nan
    idx = bin_idx[a:b]
    mu = np.full(len(centres), np.nan)
    sd = np.full(len(centres), np.nan)
    for j in range(len(centres)):
        v = s[(idx == j) & np.isfinite(s)]
        if v.size:
            mu[j] = np.median(v)
            sd[j] = (np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
    sd[~np.isfinite(sd)] = np.nanmedian(sd)
    return mu, sd


tones, rows = [], []
for c in chan[(chan % 16 == COMB_RESIDUE) & (freq > 50) & (freq < 200) & finite & ~is_fm]:
    nb = flanking(c)
    rough = np.nanmedian(np.abs(np.diff(binned(to_db(np.nanmedian(d4[:, nb], 1)), A, B)[0], 2)))
    if rough >= MAX_ROUGHNESS:
        continue
    cont = np.nanmedian(d4[:, nb], axis=1)
    raw_mu, _ = binned(to_db(d4[:, c]), A, B)
    ex_mu, ex_sd = binned(d4[:, c] - cont, A, B, flag=False)
    ref = ex_mu[ZERO]
    ok = ex_mu > NSIG * ex_sd
    if not ok[ZERO] or ok.sum() < 20:
        continue
    prof = np.where(ok, to_db(np.where(ok, ex_mu, np.nan) / ref), np.nan)
    raw_depth = np.nanmax(raw_mu) - np.nanmin(raw_mu)
    tones.append(c)
    rows.append((freq[c], raw_depth, -np.nanmin(prof), int(ok.sum()), len(centres),
                 to_db(np.nanmedian(cont[A:B]) * 0 + 1) if False else
                 to_db(1 + np.nanmedian(d4[A:B][np.abs(el[A:B]) < 30][:, c]
                                        / cont[A:B][np.abs(el[A:B]) < 30] - 1))))

print(f"{len(tones)} tones kept ({freq[tones[0]]:.1f}-{freq[tones[-1]]:.1f} MHz)\n")
print("  freq   raw_depth  sub_depth(3sig)  bins_kept  tone/cont_at_peak")
for f_, rd, sd_, nk, nt, snr in rows:
    print(f" {f_:6.1f}   {rd:7.1f}      {sd_:9.1f}      {nk:3d}/{nt}       {snr:8.2f} dB")
np.savez_compressed(Path(__file__).parent / "subtracted_depths.npz",
                    rows=np.array(rows), tones=np.array(tones))
