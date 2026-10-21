"""
Does the packaged solver reproduce the September 2026 results?

Runs eigsep_data.vector_beam on the Marjum 2026-07 inputs (pipeline.py)
and compares, channel by channel, against the two results files the
September scripts wrote:

- ``reference/all_channels_xyz_validation.json``: recovered u against
  the HFSS vector beam (power, x, y, z correlations, pixel counts, g).
- ``reference/loo_forward_check_results.json``: the held-out forward
  check.

Writes ``regression_results.json`` beside this file and prints the
largest differences. Takes ~10 min; most of it is the closure g fits.

    EIGSEP_DATA_PYTHON=... python regression_check.py
"""

import json
import time
from pathlib import Path

import numpy as np

from eigsep_data import vector_beam as vb

import pipeline

HERE = Path(__file__).resolve().parent
REF = HERE / "reference"


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    t0 = time.time()
    inp = pipeline.load()
    pxs, r_hat, E1v, E2v = inp["geometry"]
    print(
        f"loaded {inp['n_integrations']} integrations, "
        f"{len(inp['channels'])} channels in {time.time() - t0:.0f} s"
    )

    ref_xyz = {
        r["d_idx"]: r
        for r in json.loads((REF / "all_channels_xyz_validation.json").read_text())
    }
    ref_loo = {
        r["d_idx"]: r
        for r in json.loads((REF / "loo_forward_check_results.json").read_text())
    }

    rows = []
    for ch in inp["channels"]:
        args = (pxs, r_hat, E1v, E2v, ch["Ex"], ch["Ey"], ch["valid"])
        g = vb.fit_closure_g(*args)
        sol = vb.solve_sampled_beam(*args, g=g)
        truth = inp["beam"][ch["h_idx"]][:, sol["pix"]].T
        xyz = vb.compare_to_truth(sol["u"], truth)
        loo = vb.leave_one_out(*args, g)
        row = dict(
            d_idx=ch["d_idx"],
            freq_mhz=ch["freq_mhz"],
            g_mag=abs(g),
            g_phase_deg=float(np.degrees(np.angle(g))),
            n_valid=int(ch["valid"].sum()),
        )
        if xyz is not None:
            row.update(
                n_pixels=xyz["n"],
                n_absurd=xyz["n_absurd"],
                **{k: xyz[k] for k in ("power", "x", "y", "z")},
            )
        if len(loo["ex_pred"]) >= 5:
            row.update(
                loo_n=len(loo["ex_pred"]),
                loo_ex=corr(np.abs(loo["ex_pred"]) ** 2, np.abs(loo["ex_true"]) ** 2),
                loo_ey=corr(np.abs(loo["ey_pred"]) ** 2, np.abs(loo["ey_true"]) ** 2),
                loo_power=corr(loo["power_pred"], loo["power_true"]),
            )
        rows.append(row)
        rx, rl = ref_xyz.get(ch["d_idx"], {}), ref_loo.get(ch["d_idx"], {})
        print(
            f"d{ch['d_idx']} {ch['freq_mhz']:6.1f} MHz  "
            f"g {abs(g):6.3f}/{row['g_phase_deg']:7.1f} "
            f"(ref {rx.get('g_mag', np.nan):6.3f}/{rx.get('g_phase_deg', np.nan):7.1f})  "
            f"power {row.get('power', np.nan):.3f} (ref {rx.get('power', np.nan):.3f})  "
            f"y {row.get('y', np.nan):.3f} (ref {rx.get('y', np.nan):.3f})  "
            f"npix {row.get('n_pixels', 0)} (ref {rx.get('n_pixels', 0)})  "
            f"loo {row.get('loo_power', np.nan):.3f} (ref {rl.get('power_corr', np.nan):.3f})",
            flush=True,
        )

    (HERE / "regression_results.json").write_text(json.dumps(rows, indent=2))

    def worst(key, refs, refkey):
        d = [
            (abs(r[key] - refs[r["d_idx"]][refkey]), r["d_idx"])
            for r in rows
            if key in r and refkey in refs.get(r["d_idx"], {})
        ]
        return max(d) if d else (np.nan, None)

    print("\nlargest |new - ref| per quantity:")
    for key, refs, refkey in [
        ("power", ref_xyz, "power"),
        ("x", ref_xyz, "x"),
        ("y", ref_xyz, "y"),
        ("z", ref_xyz, "z"),
        ("g_phase_deg", ref_xyz, "g_phase_deg"),
        ("n_pixels", ref_xyz, "n_pixels"),
        ("loo_power", ref_loo, "power_corr"),
        ("loo_ey", ref_loo, "ey_corr"),
    ]:
        diff, d = worst(key, refs, refkey)
        print(f"  {key:12s} {diff:.4g} (d{d})")
    lp = [r["loo_power"] for r in rows if "loo_power" in r]
    print(
        f"\nLOO power median {np.median(lp):.3f} over {len(lp)} channels "
        f"(September: 0.912)"
    )
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
