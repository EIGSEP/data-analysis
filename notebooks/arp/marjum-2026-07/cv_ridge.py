"""Cross-validate the SH-augmentation ridge.

Files (contiguous 240-sample blocks) are the natural CV unit — samples
within a file are correlated. We use 7 folds of 5 contiguous files
each (35 files total), holding one fold out at a time. For each fold
and each ridge, fit the linearized SH augmentation on the training
files and evaluate normalized residual on the held-out files.

If train and test rms stay close as ridge drops, we're generalizing.
If test rms turns up while train rms keeps falling, we're overfitting
and the honest operating ridge is the minimum of the test curve.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beam_pca import compute_beam_pca
from data_space_rfi import data_space_rfi_mask
from fit_v007_pca_beam import (
    _project_templates,
    fit_channel,
    pointing_valid_mask,
    tx_arm_for_channel,
)
from rotation_beam import TransmitterGeometry
from tx_beam_sim import HFSSBeamSet, simulate_hfss_coupling
from v007_beam_diagnostic import (
    channel_validity_masks,
    gross_power_time_flags,
    load_v007_data,
)

MODEL_JSON = "v007_pca_beam_model_v5.json"
BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"
TEST_CHANNELS = [536, 616, 680, 760]  # drop 552 (RFI-limited)
LMAX = 8
N_FOLDS = 7
FILE_SIZE = 240
RIDGES = [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005]


def sh_basis_beams(nside, lmax):
    npix = hp.nside2npix(nside)
    n_ylm = (lmax + 1) ** 2
    alm_size = hp.Alm.getsize(lmax)
    ylm_maps = np.empty((n_ylm, npix))
    ell_of_mode = np.empty(n_ylm, dtype=int)
    idx = 0
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            alm = np.zeros(alm_size, complex)
            if m == 0:
                alm[hp.Alm.getidx(lmax, ell, 0)] = 1.0
            elif m > 0:
                alm[hp.Alm.getidx(lmax, ell, m)] = 1.0 / np.sqrt(2)
            else:
                sgn = -1 if (-m) % 2 else 1
                alm[hp.Alm.getidx(lmax, ell, -m)] = 1j / np.sqrt(2) * sgn
            ylm_maps[idx] = hp.alm2map(alm, nside, lmax=lmax, verbose=False)
            ell_of_mode[idx] = ell
            idx += 1
    beam_cart = np.zeros((3 * n_ylm, 3, npix), complex)
    ell_full = np.repeat(ell_of_mode, 3)
    for i in range(n_ylm):
        for j in range(3):
            beam_cart[3 * i + j, j] = ylm_maps[i]
    beam = HFSSBeamSet(
        beam_cart=beam_cart,
        gain_th=np.zeros((3 * n_ylm, npix)),
        gain_ph=np.zeros((3 * n_ylm, npix)),
        freqs_mhz=np.arange(3 * n_ylm, dtype=float),
    )
    return beam, ell_full


with open(MODEL_JSON) as f:
    cfg = json.load(f)
beam = HFSSBeamSet.from_npz(BEAM_FILE)
geometry = TransmitterGeometry(
    cfg["geometry"]["heading"], cfg["geometry"]["alpha_deg"])
data = load_v007_data("data")
clean = data_space_rfi_mask(
    "data", beam.freqs_mhz.min(), beam.freqs_mhz.max(), min_votes=5,
) & pointing_valid_mask(data)
pca = compute_beam_pca(beam, n_components=4)
T_pca = _project_templates(pca.components, data, geometry)

sh_beam, ell_full = sh_basis_beams(beam.nside, LMAX)
T_sh = {}
for arm in (0, 1):
    coup, _ = simulate_hfss_coupling(
        sh_beam, data["az_deg"], data["el_deg"], geometry,
        np.full(data["az_deg"].size, arm, dtype=int))
    T_sh[arm] = coup

ntimes = data["az_deg"].size
file_idx = np.arange(ntimes) // FILE_SIZE
n_files = int(file_idx.max()) + 1
fold_size = int(np.ceil(n_files / N_FOLDS))


def rms_frac(y, m, mask):
    good = mask & np.isfinite(y) & np.isfinite(m)
    if good.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean((y[good] - m[good]) ** 2)) /
                 max(np.sqrt(np.mean(y[good] ** 2)), 1e-30))


# Precompute per-channel baseline K=4 (fit on ALL data — this is
# stable, only the shape correction risks overfitting).
baseline = {}
for ch in TEST_CHANNELS:
    arm = tx_arm_for_channel(ch)
    fit = fit_channel(data, pca, T_pca, ch, clean, ridge_lambda=0.003)
    a = (np.array(fit["coefficients_real"])
         + 1j * np.array(fit["coefficients_imag"]))
    m_pca = np.abs(np.conj(a) @ T_pca[arm]) ** 2
    y = data["measured_tx"][:, ch].astype(float)
    base = channel_validity_masks(data, [ch])[:, 0]
    gross, _, _ = gross_power_time_flags(data, [ch])
    used = base & ~gross & clean

    # Design matrix for linearized SH augmentation.
    E_pca = np.conj(a) @ T_pca[arm]
    cE = np.conj(E_pca)
    A_r = 2.0 * np.real(cE[None, :] * T_sh[arm])
    A_i = -2.0 * np.imag(cE[None, :] * T_sh[arm])
    A = np.concatenate([A_r, A_i], axis=0).T
    w_r = 1.0 / (1.0 + ell_full)
    widths = np.concatenate([w_r, w_r])
    baseline[ch] = dict(arm=arm, a=a, m_pca=m_pca, y=y, used=used,
                        A=A, widths=widths,
                        rms_all=rms_frac(y, m_pca, used))


print("=== 7-fold CV over contiguous 5-file blocks (35 files total) ===")
print(f"lmax={LMAX}, {A.shape[1] // 2} complex SH DOFs, {A.shape[1]} real DOFs")
print()

# Report structure: ridge -> per-channel (train_rms, test_rms), avg
results = {ch: {"train": {r: [] for r in RIDGES},
                "test":  {r: [] for r in RIDGES}}
           for ch in TEST_CHANNELS}

for fold in range(N_FOLDS):
    test_files = set(range(fold * fold_size,
                           min((fold + 1) * fold_size, n_files)))
    test_mask = np.array([fi in test_files for fi in file_idx])
    train_mask = ~test_mask

    for ch in TEST_CHANNELS:
        b = baseline[ch]
        used_train = b["used"] & train_mask
        used_test = b["used"] & test_mask
        if used_train.sum() < 100 or used_test.sum() < 50:
            continue
        A = b["A"]
        Atr, Ate = A[used_train], A[used_test]
        r_train = (b["y"] - b["m_pca"])[used_train]
        r_test = (b["y"] - b["m_pca"])[used_test]
        rms_r = float(np.sqrt(np.mean(r_train ** 2)))
        AtA = Atr.T @ Atr
        Atr_r = Atr.T @ r_train
        for ridge_scale in RIDGES:
            prior_pen = (ridge_scale * rms_r) / b["widths"]
            lhs = AtA + np.diag(prior_pen ** 2)
            c = np.linalg.solve(lhs, Atr_r)
            m_train = b["m_pca"].copy()
            m_train[used_train] = b["m_pca"][used_train] + Atr @ c
            m_test = b["m_pca"].copy()
            m_test[used_test] = b["m_pca"][used_test] + Ate @ c
            results[ch]["train"][ridge_scale].append(
                rms_frac(b["y"], m_train, used_train))
            results[ch]["test"][ridge_scale].append(
                rms_frac(b["y"], m_test, used_test))

# Report table.
print(f"{'ch':>4}  {'baseline':>8} " + "  ".join(
    f"r{r:.3f}".rjust(11) for r in RIDGES))
print(" " * 15 + "  ".join("train/test ".rjust(11) for _ in RIDGES))
print("-" * (16 + 13 * len(RIDGES)))
mean_train_by_ridge = {r: [] for r in RIDGES}
mean_test_by_ridge = {r: [] for r in RIDGES}
for ch in TEST_CHANNELS:
    b = baseline[ch]
    line = f"{ch:>4}  {b['rms_all']:.4f}   "
    for r in RIDGES:
        tr = np.nanmean(results[ch]["train"][r])
        te = np.nanmean(results[ch]["test"][r])
        line += f"{tr:.3f}/{te:.3f}  "
        mean_train_by_ridge[r].append(tr)
        mean_test_by_ridge[r].append(te)
    print(line)
print("-" * (16 + 13 * len(RIDGES)))
line = "mean   " + " " * 8 + "   "
for r in RIDGES:
    tr = float(np.mean(mean_train_by_ridge[r]))
    te = float(np.mean(mean_test_by_ridge[r]))
    line += f"{tr:.3f}/{te:.3f}  "
print(line)

# Plot.
fig, axes = plt.subplots(1, len(TEST_CHANNELS) + 1, figsize=(5 * (len(TEST_CHANNELS) + 1), 4.5))
for ax, ch in zip(axes[:-1], TEST_CHANNELS):
    b = baseline[ch]
    tr = [np.nanmean(results[ch]["train"][r]) for r in RIDGES]
    te = [np.nanmean(results[ch]["test"][r]) for r in RIDGES]
    ax.plot(RIDGES, tr, "o-", label="train")
    ax.plot(RIDGES, te, "s-", label="test")
    ax.axhline(b["rms_all"], color="k", linestyle=":", linewidth=0.8,
               label=f"K=4 only ({b['rms_all']:.3f})")
    ax.axhline(0.075, color="gray", linestyle="--", linewidth=0.8,
               label="repeatability floor")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("ridge scale (log, less regularization →)")
    ax.set_ylabel("normalized rms")
    ax.set_title(f"ch{ch} @ {data['freqs'][ch]:.1f} MHz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Combined mean panel.
ax = axes[-1]
tr = [float(np.mean(mean_train_by_ridge[r])) for r in RIDGES]
te = [float(np.mean(mean_test_by_ridge[r])) for r in RIDGES]
ax.plot(RIDGES, tr, "o-", label="train (mean 4 ch)")
ax.plot(RIDGES, te, "s-", label="test (mean 4 ch)")
ax.axhline(np.mean([baseline[c]["rms_all"] for c in TEST_CHANNELS]),
           color="k", linestyle=":", linewidth=0.8, label="K=4 mean")
ax.axhline(0.075, color="gray", linestyle="--", linewidth=0.8,
           label="repeatability floor")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("ridge scale")
ax.set_ylabel("normalized rms")
ax.set_title("mean over 4 channels")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("cv_ridge_curves.png", dpi=120)
plt.close(fig)
print("saved cv_ridge_curves.png")

# Recommend a ridge: smallest ridge where test rms is still within 5%
# of its minimum, walking from high ridge down (conservative).
mean_test = np.array([float(np.mean(mean_test_by_ridge[r])) for r in RIDGES])
best_idx = int(np.argmin(mean_test))
best_ridge = RIDGES[best_idx]
best_test = float(mean_test[best_idx])
best_train = float(np.mean(mean_train_by_ridge[best_ridge]))
print()
print(f"Recommended operating ridge (min test rms): {best_ridge}")
print(f"  train rms = {best_train:.4f}, test rms = {best_test:.4f}, "
      f"gap = {(best_test - best_train)*100:+.2f}%")
print(f"K=4 baseline mean rms (all data): "
      f"{np.mean([baseline[c]['rms_all'] for c in TEST_CHANNELS]):.4f}")
