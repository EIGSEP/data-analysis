"""B3 — HFSS-only beam chromaticity mode budget.

How many spectral eigenmodes describe the EIGSEP bowtie beam's frequency
evolution to ~1 part in 1e4 (the foreground/signal dynamic range)?

Method: uncentered SVD over the frequency axis of the HFSS beam power map,
matching ``eigsep_sim.basis.BeamBasis.from_map`` and the prototype in
``eigsep_data.beam_mapping.basis``. Reconstruct with K modes, report residual
curves vs K against the 1e-4 line.

Geometry-free by construction: no terrain, no horizon, no sky weighting.

Usage:
    python beam_mode_budget.py            # writes b3_mode_budget.json
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import healpy as hp

# data-analysis/scripts/marjum-2026-07/b3/ -> the workspace root.
# Was parents[3] when this lived at marjum-2026-07/analysis/b3/.
REPO = Path(__file__).resolve().parents[4]
PROD_A = REPO / "eigsep_data/hfss_beam_maps/bowtie_beam.npz"
PROD_B = REPO / "eigsep_sim/src/eigsep_sim/data/eigsep_bowtie_v000.npz"

# The 1-in-1e4 target is an *amplitude* fraction. In SVD variance terms that is
# 1e-8 of the total; keeping the two straight is the whole point of this study.
TARGET = 1e-4

BANDS = {
    "cosmology_50_110": (50.0, 110.0),
    "midband_50_130": (50.0, 130.0),
    "trough_60_100": (60.0, 100.0),
    "fullband_50_250": (50.0, 250.0),
}

CHAN_MHZ = 250.0 / 1024  # 0.244140625 MHz, the correlator channel width
KMAX = 24


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_products():
    """Return {name: (freqs_mhz, power_map (nfreq, npix), nside)}."""
    a = np.load(PROD_A, allow_pickle=True)
    b = np.load(PROD_B, allow_pickle=True)
    return {
        # total power = sum of the two polarization gains
        "bowtie_beam": (
            np.asarray(a["freqs"], float),
            np.asarray(a["gain_th"] + a["gain_ph"], float),
            int(a["nside"]),
        ),
        "eigsep_bowtie_v000": (
            np.asarray(b["freqs"], float) / 1e6,
            np.asarray(b["bm"], float),
            int(b["nside"]),
        ),
    }


def residual_curves(freqs, bm, nside, kmax=KMAX):
    """Uncentered frequency-axis SVD; residual metrics vs mode count K.

    ``bm`` is (nfreq, npix). Matches BeamBasis.from_map, which SVDs the
    (npix, nfreq) transpose -- same singular values, same subspace.

    Returns dict of arrays indexed by K = 1..kmax.
    """
    nfreq, npix = bm.shape
    dOmega = 4.0 * np.pi / npix

    # (nfreq, npix): rows are frequency slices, so Vt rows are spatial modes.
    U, s, Vt = np.linalg.svd(bm, full_matrices=False)
    kmax = min(kmax, s.size)

    var = s ** 2
    cumvar = np.cumsum(var) / np.sum(var)

    # Per-frequency normalizations for the amplitude metrics.
    rms_f = np.sqrt(np.mean(bm ** 2, axis=1))       # (nfreq,)
    peak_f = np.max(bm, axis=1)                      # (nfreq,)
    mono_f = np.sum(bm, axis=1) * dOmega             # (nfreq,) = \int B dOmega

    out = {
        "K": [], "resid_var_frac": [], "resid_rms_frac": [],
        "max_resid_over_peak": [], "max_monopole_err": [],
        "worst_freq_mhz": [],
    }
    for K in range(1, kmax + 1):
        recon = (U[:, :K] * s[:K]) @ Vt[:K]
        resid = bm - recon

        # Global amplitude residual: ||resid||_F / ||bm||_F
        rvf = float(np.sum(resid ** 2) / np.sum(bm ** 2))

        # Per-frequency worst case, normalized two ways.
        per_f_rms = np.sqrt(np.mean(resid ** 2, axis=1)) / rms_f
        per_f_max = np.max(np.abs(resid), axis=1) / peak_f
        per_f_mono = np.abs(np.sum(resid, axis=1) * dOmega) / mono_f

        out["K"].append(K)
        out["resid_var_frac"].append(rvf)
        out["resid_rms_frac"].append(float(np.sqrt(rvf)))
        out["max_resid_over_peak"].append(float(per_f_max.max()))
        out["max_monopole_err"].append(float(per_f_mono.max()))
        out["worst_freq_mhz"].append(float(freqs[int(per_f_rms.argmax())]))

    out["singular_values"] = s[:kmax].tolist()
    out["cum_explained_var"] = cumvar[:kmax].tolist()
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out.items()}


def first_k_below(curve, target):
    """Smallest K (1-indexed) whose metric is <= target; None if never."""
    idx = np.where(np.asarray(curve) <= target)[0]
    return int(idx[0] + 1) if idx.size else None


SKY_MODELS = ("GSM08", "GSM16", "LFSM")


def sky_spectral_rank(model, freqs_mhz, nside=32, kmax=10):
    """Spectral-mode residuals of a public sky model over a band.

    NOTE: every pygdsm-family model is a *low-rank interpolation* -- GSM08 ships
    3 spatial PCA components, GSM16 ~5, LFSM ~4 -- so the frequency-axis rank
    recovered here is the rank its authors built in, not a property of the sky.
    This function is therefore a diagnostic of the sky *model*, and N_fg cannot
    be determined from it. See the memo.
    """
    import pygdsm
    cls = {"GSM08": pygdsm.GlobalSkyModel08,
           "GSM16": pygdsm.GlobalSkyModel16,
           "LFSM": pygdsm.LowFrequencySkyModel}[model]
    cache = Path(__file__).parent / f"_sky_cache_{model}_n{nside}.npz"
    store = dict(np.load(cache)) if cache.exists() else {}
    need = [f for f in freqs_mhz if f"{f:.3f}" not in store]
    if need:
        g = cls(freq_unit="MHz")
        for f in need:
            store[f"{f:.3f}"] = hp.ud_grade(g.generate(f), nside)
        np.savez_compressed(cache, **store)
    maps = np.array([store[f"{f:.3f}"] for f in freqs_mhz])
    U, s, Vt = np.linalg.svd(maps, full_matrices=False)
    kmax = min(kmax, s.size)
    var = s ** 2
    res = [float(np.sqrt(max(0.0, 1.0 - var[:K].sum() / var.sum())))
           for K in range(1, kmax + 1)]
    # numerical rank: where the spectrum collapses to round-off
    nrank = int(np.sum(s > s[0] * 1e-12))
    return {"resid_rms_frac": res, "numerical_rank": nrank,
            "singular_values": s[:kmax].tolist(),
            "K_at_target": first_k_below(res, TARGET)}


def main():
    products = load_products()
    results = {"bands": {}, "products": {}}

    for pname, (freqs, bm, nside) in products.items():
        results["products"][pname] = {
            "nfreq": int(freqs.size), "nside": nside,
            "freq_min_mhz": float(freqs.min()), "freq_max_mhz": float(freqs.max()),
            "df_mhz": float(np.median(np.diff(freqs))),
        }

    for bname, (lo, hi) in BANDS.items():
        band = {"range_mhz": [lo, hi],
                "n_corr_channels": int(round((hi - lo) / CHAN_MHZ))}
        for pname, (freqs, bm, nside) in products.items():
            sel = (freqs >= lo) & (freqs <= hi)
            if sel.sum() < 4:
                continue
            cur = residual_curves(freqs[sel], bm[sel], nside)
            band[pname] = {
                "nfreq_in_band": int(sel.sum()),
                "K": cur["K"].tolist(),
                "resid_rms_frac": cur["resid_rms_frac"].tolist(),
                "max_resid_over_peak": cur["max_resid_over_peak"].tolist(),
                "max_monopole_err": cur["max_monopole_err"].tolist(),
                "cum_explained_var": cur["cum_explained_var"],
                "singular_values": cur["singular_values"],
                "N_ant_rms": first_k_below(cur["resid_rms_frac"], TARGET),
                "N_ant_peak": first_k_below(cur["max_resid_over_peak"], TARGET),
                "N_ant_monopole": first_k_below(cur["max_monopole_err"], TARGET),
            }
        results["bands"][bname] = band

    # --- frequency-sampling control ---------------------------------------
    # Does the coarse 3.90625 MHz grid of bowtie_beam.npz under-resolve the
    # chromaticity? Decimate the 1 MHz product onto ~4 MHz and compare.
    freqs_b, bm_b, nside_b = products["eigsep_bowtie_v000"]
    sampling = {}
    for bname, (lo, hi) in BANDS.items():
        sel = (freqs_b >= lo) & (freqs_b <= hi)
        fine = residual_curves(freqs_b[sel], bm_b[sel], nside_b)
        coarse_sel = np.zeros_like(sel)
        coarse_sel[np.where(sel)[0][::4]] = True
        coarse = residual_curves(freqs_b[coarse_sel], bm_b[coarse_sel], nside_b)
        sampling[bname] = {
            "fine_nfreq": int(sel.sum()), "coarse_nfreq": int(coarse_sel.sum()),
            "fine_N_ant_rms": first_k_below(fine["resid_rms_frac"], TARGET),
            "coarse_N_ant_rms": first_k_below(coarse["resid_rms_frac"], TARGET),
            "fine_resid_rms_frac": fine["resid_rms_frac"].tolist(),
            "coarse_resid_rms_frac": coarse["resid_rms_frac"].tolist(),
        }
    results["sampling_control"] = sampling

    # --- product-disagreement floor ---------------------------------------
    # The two HFSS products bound how well any mode budget is actually known.
    freqs_a, bm_a, nside_a = products["bowtie_beam"]
    disagree = {}
    for bname, (lo, hi) in BANDS.items():
        sa = (freqs_a >= lo) & (freqs_a <= hi)
        fr, num, den = [], 0.0, 0.0
        for i in np.where(sa)[0]:
            j = int(np.argmin(np.abs(freqs_b - freqs_a[i])))
            if abs(freqs_b[j] - freqs_a[i]) > 0.6:
                continue
            bd = hp.ud_grade(bm_b[j], nside_a)
            num += float(np.sum((bm_a[i] - bd) ** 2))
            den += float(np.sum(bm_a[i] ** 2))
            fr.append(float(np.sqrt(np.mean((bm_a[i] - bd) ** 2)
                                    / np.mean(bm_a[i] ** 2))))
        disagree[bname] = {
            "n_matched_freqs": len(fr),
            "rms_frac_diff_band": float(np.sqrt(num / den)) if den else None,
            "rms_frac_diff_max": float(np.max(fr)) if fr else None,
        }
    results["product_disagreement"] = disagree

    # --- foreground modes --------------------------------------------------
    fg = {}
    for bname, (lo, hi) in BANDS.items():
        gf = np.arange(lo, hi + 1e-6, 2.0)
        fg[bname] = {}
        for model in SKY_MODELS:
            try:
                fg[bname][model] = sky_spectral_rank(model, gf)
            except Exception as exc:  # pragma: no cover
                fg[bname][model] = {"error": f"{type(exc).__name__}: {exc}"}
    results["sky_spectral_rank"] = fg

    # --- mode budget -------------------------------------------------------
    # N_fg is NOT determinable from the public sky models (all are low-rank by
    # construction), so the budget is tabulated over a range of N_fg with
    # BLOOM's 5 as the working value.
    N_FG_WORKING = 5
    budget = {}
    for bname, (lo, hi) in BANDS.items():
        bd = results["bands"][bname]
        nchan = bd["n_corr_channels"]
        ent = {"n_corr_channels": nchan, "N_fg_working": N_FG_WORKING,
               "per_product": {}, "vs_N_fg": {}}
        for pname in products:
            if pname not in bd:
                continue
            nant = bd[pname]["N_ant_rms"]
            if nant is None:
                continue
            nmodes = nant * N_FG_WORKING
            ent["per_product"][pname] = {
                "N_ant": nant, "N_modes": nmodes,
                "dof_remaining": nchan - nmodes,
                "frac_dof_remaining": (nchan - nmodes) / nchan,
            }
        nant_ref = bd.get("bowtie_beam", {}).get("N_ant_rms")
        if nant_ref:
            for nfg in (3, 4, 5, 6, 8, 10):
                nm = nant_ref * nfg
                ent["vs_N_fg"][str(nfg)] = {
                    "N_modes": nm, "dof_remaining": nchan - nm,
                    "frac_dof_remaining": (nchan - nm) / nchan}
        budget[bname] = ent
    # BLOOM's published comparison point: 5 modes of 45 channels.
    budget["_bloom_reference"] = {
        "n_channels": 45, "N_modes": 5, "dof_remaining": 40,
        "frac_dof_remaining": 40 / 45,
        "note": "4 GSM spectral eigenmodes + 1 flat; PROGRAM.md section 4 move 2",
    }
    results["mode_budget"] = budget

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO / "eigsep_sim"), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"

    results["provenance"] = {
        "product": "beam_mode_budget",
        "campaign": "marjum-2026-07",
        "version": "v0",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "eigsep_sim/analysis/b3/beam_mode_budget.py",
        "generator_commit": commit,
        "inputs": [
            {"path": "eigsep_data/hfss_beam_maps/bowtie_beam.npz",
             "sha256": sha256(PROD_A)},
            {"path": "eigsep_sim/src/eigsep_sim/data/eigsep_bowtie_v000.npz",
             "sha256": sha256(PROD_B)},
        ],
        "params": {"target_amplitude_frac": TARGET, "kmax": KMAX,
                   "chan_mhz": CHAN_MHZ, "bands": {k: list(v) for k, v in BANDS.items()}},
        "notes": "HFSS-only, geometry-free (no terrain/horizon/sky weighting). v0 "
                 "of the budget; repeats with the measured beam under B4.",
    }

    def _jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not JSON serializable: {type(o)}")

    out = Path(__file__).parent / "b3_mode_budget.json"
    out.write_text(json.dumps(results, indent=2, default=_jsonable))
    print(f"wrote {out}")
    return results


if __name__ == "__main__":
    main()
