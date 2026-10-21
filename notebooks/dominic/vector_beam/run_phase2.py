"""
Phase 2: the vector-beam solve on the production beam fit's pointing and
geometry, compared channel by channel with Phase 1.

Phase 1 (regression_results.json) used potentiometer azimuth and the
geometry fitted by the old fit_multi_freq_joint. Here the cross data,
flags and Ex_hat/Ey_hat are Phase 1's exactly; only where each sample
points changes:

- pointing: pointing table v3 (commanded azimuth, IMU elevation), joined
  to each integration by nearest time within 0.35 s, as fit_beam.py does;
- geometry: geometry_v0013 (release v0004 positions, v0013 corrections),
  through eigsep_data.vector_beam.fields_from_model.

Two row selections:

- A: every Phase 1 row with a finite table az/el, so the difference from
  Phase 1 is the geometry and pointing alone;
- B: additionally fit_beam.py's selection (quality 'ok', RFANT, isolated
  elevation reversal glitches dropped).

    POINTING_TABLE=.../pointing_table.parquet python run_phase2.py

DPSS fills without a measured same-arm tone on both sides are NaN
(vb.reduce_cross's default) and results go to
phase2_bracketed_results.json. BRACKETED_FILL=0 keeps those fills, as on
2026-10-20, and writes phase2_results.json.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from eigsep_data import MetadataIndex
from eigsep_data.beam_mapping import vector_beam as vb

import geometry_v0013
import pipeline

HERE = Path(__file__).resolve().parent
POINTING_TABLE = Path(os.environ["POINTING_TABLE"])
JOIN_TOL_S = 0.35
BRACKETED = os.environ.get("BRACKETED_FILL", "1") != "0"


def reversal_glitches(el_deg, tol=15.0):
    """fit_beam.py's isolated elevation-fault test: a sample is good if at
    least 3 of its 4 neighbours (+/-1, +/-2) lie within tol deg."""
    el = np.asarray(el_deg, float)
    close = np.zeros(el.size, int)
    for k in (-2, -1, 1, 2):
        d = np.abs((el - np.roll(el, k) + 180.0) % 360.0 - 180.0)
        d[slice(0, -k) if k < 0 else slice(-k, None)] = 0.0
        close += d <= tol
    return close < 3


def join_pointing(times, data_dir):
    t0, t1 = times.min() - 60, times.max() + 60
    pt = pq.read_table(
        POINTING_TABLE,
        columns=["t_utc_s", "az_deg", "el_deg", "quality"],
        filters=[("t_utc_s", ">=", t0), ("t_utc_s", "<=", t1)],
    ).to_pandas()
    rows = pd.DataFrame({"t": times, "row": np.arange(len(times))})
    j = pd.merge_asof(
        rows.sort_values("t"),
        pt.sort_values("t_utc_s"),
        left_on="t",
        right_on="t_utc_s",
        direction="nearest",
        tolerance=JOIN_TOL_S,
    ).sort_values("row")
    meta = (
        MetadataIndex(data_dir)
        .select(time=(t0, t1))
        .meta.assign(t=lambda x: x.time_best.astype(float))
    )
    j = pd.merge_asof(
        j.sort_values("t"),
        meta[["t", "rfswitch"]].sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=1e-3,
    ).sort_values("row")
    return j.reset_index(drop=True)


def score(inp, geometry, valid_rows):
    pxs, r_hat, E1v, E2v = geometry
    rows = []
    for ch in inp["channels"]:
        valid = ch["valid"] & valid_rows
        if valid.sum() < 50:  # e.g. d89 when bracketed: no anchor above
            rows.append(
                dict(
                    d_idx=ch["d_idx"],
                    freq_mhz=ch["freq_mhz"],
                    n_valid=int(valid.sum()),
                )
            )
            print(f"  d{ch['d_idx']} skipped: {valid.sum()} valid")
            continue
        args = (pxs, r_hat, E1v, E2v, ch["Ex"], ch["Ey"], valid)
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
            n_valid=int(valid.sum()),
        )
        if xyz is not None:
            row.update(
                n_pixels=xyz["n"],
                **{k: xyz[k] for k in ("power", "x", "y", "z")},
            )
        if len(loo["ex_pred"]) >= 5:
            row.update(
                loo_n=len(loo["ex_pred"]),
                loo_ex=float(
                    np.corrcoef(
                        np.abs(loo["ex_pred"]) ** 2,
                        np.abs(loo["ex_true"]) ** 2,
                    )[0, 1]
                ),
                loo_ey=float(
                    np.corrcoef(
                        np.abs(loo["ey_pred"]) ** 2,
                        np.abs(loo["ey_true"]) ** 2,
                    )[0, 1]
                ),
                loo_power=float(
                    np.corrcoef(loo["power_pred"], loo["power_true"])[0, 1]
                ),
            )
        rows.append(row)
        print(
            f"  d{row['d_idx']} power {row.get('power', np.nan):.3f} "
            f"x {row.get('x', np.nan):.3f} y {row.get('y', np.nan):.3f} "
            f"z {row.get('z', np.nan):.3f} npix {row.get('n_pixels', 0)} "
            f"loo {row.get('loo_power', np.nan):.3f} "
            f"g {abs(g):.2f}/{row['g_phase_deg']:.1f}",
            flush=True,
        )
    return rows


def main():
    t0 = time.time()
    inp = pipeline.load(require_bracketed_fill=BRACKETED)
    cache = np.load(pipeline.CACHE, allow_pickle=True)
    j = join_pointing(inp["times"], str(cache["data_dir"]))
    joined = np.isfinite(j.az_deg.to_numpy()) & np.isfinite(
        j.el_deg.to_numpy()
    )
    print(
        f"{len(j)} rows, {joined.sum()} joined to the pointing table, "
        f"{(j.quality == 'ok').sum()} quality ok, "
        f"{(j.rfswitch == 'RFANT').sum()} RFANT"
    )
    az = np.where(joined, j.az_deg.to_numpy(), 0.0)
    el = np.where(joined, j.el_deg.to_numpy(), 0.0)
    nside = int(np.sqrt(inp["beam"].shape[-1] / 12))
    geometry = vb.fields_from_model(
        geometry_v0013.model(), geometry_v0013.PARAMS, az, el, nside
    )

    sel_b = joined & (j.quality == "ok").to_numpy()
    sel_b &= (j.rfswitch == "RFANT").to_numpy()
    glitch = np.zeros(len(j), bool)
    idx = np.flatnonzero(sel_b)
    glitch[idx] = reversal_glitches(el[idx])
    sel_b &= ~glitch
    print(f"selection B keeps {sel_b.sum()} rows ({glitch.sum()} glitches)")

    out = {}
    for name, sel in (("A", joined), ("B", sel_b)):
        print(f"\nselection {name}:")
        out[name] = score(inp, geometry, sel)

    ref = {
        r["d_idx"]: r
        for r in json.loads((HERE / "regression_results.json").read_text())
    }
    name = "phase2_bracketed" if BRACKETED else "phase2"
    (HERE / f"{name}_results.json").write_text(json.dumps(out, indent=2))
    print("\nmedians over 30 channels (Phase 1 -> A -> B):")
    for k in ("power", "x", "y", "z", "loo_power", "loo_ex", "loo_ey"):
        p1 = np.nanmedian([r.get(k, np.nan) for r in ref.values()])
        a = np.nanmedian([r.get(k, np.nan) for r in out["A"]])
        b = np.nanmedian([r.get(k, np.nan) for r in out["B"]])
        print(f"  {k:10s} {p1:.3f} -> {a:.3f} -> {b:.3f}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
