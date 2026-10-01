"""Validate ``derived/tcal/<version>/`` and write ``validation.json``.

Four checks, each of which could come out against the product:

1. **Interpolation, leave one out.** Each interior load / noise-source
   visit, and each interior S11 sweep, is predicted from its neighbours
   by linear interpolation and by the nearer neighbour, and compared
   with what was measured. The limits the product uses are applied, so
   only pairs it would itself interpolate are scored. Dropping one node
   doubles the bracket, so this overstates the product's own error.
2. **Plumbing.** Through ``load_bundle``, ``t_star`` on load rows has
   to return the thermistor temperature and on noise-source rows that
   plus T_NS. Wrong signs, swapped spectra or misjoined rows fail here.
3. **Christian's night.** The 16/17 Jul window of Christian's T12 notebook, run
   through the vendored ``d5_yfactor`` (nearest visit, nearest sweep)
   and through the product (linear in time, nearest thermistor), per
   antenna visit. The two differ by construction; this measures by how
   much.
4. **Coverage.** Fraction of box-air antenna rows in the span that come
   out calibrated.

Run::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 python validate_tcal.py v0000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import eigsep_data
from eigsep_data import MetadataIndex

BAND = (50.0, 200.0)
NOTEBOOK_DIR = Path(__file__).resolve().parents[3] / "notebooks" / "arp" / "marjum-2026-07"


def band_mask(f):
    return (f >= BAND[0]) & (f < BAND[1])


def leave_one_out(t, v, label, max_gap, f):
    """Median over interior nodes of the median-over-band |error|,
    fractional for real spectra and absolute for complex Gamma."""
    b = band_mask(f)
    lin, near = [], []
    for k in range(1, t.size - 1):
        if not (label[k - 1] == label[k] == label[k + 1]):
            continue
        if t[k + 1] - t[k - 1] > max_gap:
            continue
        w = (t[k] - t[k - 1]) / (t[k + 1] - t[k - 1])
        pred_lin = (1 - w) * v[k - 1] + w * v[k + 1]
        pred_near = v[k - 1] if w <= 0.5 else v[k + 1]
        if np.iscomplexobj(v):
            lin.append(np.nanmedian(np.abs(pred_lin - v[k])[b]))
            near.append(np.nanmedian(np.abs(pred_near - v[k])[b]))
        else:
            lin.append(np.nanmedian(np.abs(pred_lin / v[k] - 1)[b]))
            near.append(np.nanmedian(np.abs(pred_near / v[k] - 1)[b]))
    lin, near = np.array(lin), np.array(near)
    return {
        "n_scored": int(lin.size),
        "median_linear": float(np.median(lin)),
        "median_nearest": float(np.median(near)),
        "linear_better_fraction": float(np.mean(lin < near)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version")
    ap.add_argument(
        "--products",
        help="campaign tree to read/write derived/tcal/ in (default: the "
        "campaign root, which also supplies the raw data)",
    )
    args = ap.parse_args(argv)
    root = eigsep_data.get_campaign_root(required=True)
    proot = Path(args.products) if args.products else root
    pdir = proot / "derived" / "tcal" / args.version
    z = dict(np.load(pdir / "solutions.npz"))
    f = z["freqs"]
    out = {"band_mhz": BAND}

    # 1. Leave one out.
    gain = z["non_p"] - z["amb_p"][
        [np.argmin(np.abs(z["amb_t"] - t)) for t in z["non_t"]]
    ]
    loo = {
        "amb_power_frac": leave_one_out(z["amb_t"], z["amb_p"], z["amb_regime"], z["max_cal_gap_s"], f),
        "non_power_frac": leave_one_out(z["non_t"], z["non_p"], z["non_regime"], z["max_cal_gap_s"], f),
        "non_minus_amb_frac": leave_one_out(z["non_t"], gain, z["non_regime"], z["max_cal_gap_s"], f),
    }
    for dut in ("ant", "amb", "rec"):
        loo[f"s11_{dut}_abs"] = leave_one_out(
            z[f"s11_{dut}_t"], z[f"s11_{dut}"], z[f"s11_{dut}_epoch"], z["max_s11_gap_s"], f
        )
    out["leave_one_out"] = loo

    # 2 and 4: through the product.
    index = MetadataIndex(root / "data")
    t0, t1 = float(z["amb_t"].min()), float(z["amb_t"].max())
    sel = index.select(time=(t0, t1), files=("corr_20260715_003217Z.h5", "corr_20991231"))
    b = sel.load_bundle(key="4", root=proot, products=[f"tcal@{args.version}"])
    fb = b.freqs_mhz
    bb = band_mask(fb)
    st = b.meta.rfswitch.to_numpy(str)
    tl = b.products["tcal"]["tcal_t_load_k"]
    ts = b.t_star
    plumb = {}
    for state, extra in (("RFAMB", 0.0), ("RFNON", float(z["t_ns_k"]))):
        rows = (st == state) & np.isfinite(ts[:, bb]).all(1)
        resid = np.nanmedian(ts[rows][:, bb], axis=1) - (tl[rows] + extra)
        # Worst cases and single channels, not just medians: a few bad
        # calibration visits corrupt hundreds of channels for ~25 min while
        # leaving every band-median statistic unchanged (v0000).
        expect = (tl[rows] + extra)[:, None]
        wide = (fb >= 30) & (fb < 245)
        with np.errstate(invalid="ignore"):
            off = np.abs(ts[rows][:, wide] - expect) > 0.05 * (expect if extra == 0 else extra)
        plumb[state] = {
            "rows": int(rows.sum()),
            "median_resid_k": float(np.median(resid)),
            "p95_abs_resid_k": float(np.percentile(np.abs(resid), 95)),
            "max_abs_resid_k": float(np.max(np.abs(resid))),
            "channel_fraction_off_5pct": float(np.nanmean(off)),
            "rows_with_over_50_channels_off_5pct": int((off.sum(1) > 50).sum()),
        }
    out["plumbing"] = plumb
    ant = st == "RFANT"
    cal = b.calibrated
    ok = np.isfinite(cal[:, bb]).all(1)
    out["coverage"] = {
        "span_utc": [datetime.utcfromtimestamp(t0).isoformat() + "Z", datetime.utcfromtimestamp(t1).isoformat() + "Z"],
        "rfant_rows": int(ant.sum()),
        "rfant_rows_calibrated": int((ant & ok).sum()),
        "fraction": float((ant & ok).sum() / max(ant.sum(), 1)),
    }

    # 3. Christian's night.
    sys.path.insert(0, str(NOTEBOOK_DIR))
    import d5_yfactor as yf
    from eigsep_cal import S11
    import eigsep_cal.s11

    tz = ZoneInfo("America/Denver")
    start = datetime(2026, 7, 16, 21, 30, tzinfo=tz).timestamp()
    stop = datetime(2026, 7, 17, 7, 55, tzinfo=tz).timestamp()
    d = yf.load_corr(root / "data", start, stop, "4")
    vis = yf.average_blocks(d, use=~np.all(d["spec"] == 0, axis=1))
    sky, non, amb = (vis[s] for s in (yf.SKY, yf.NOISE, yf.LOAD))
    il, _ = yf.nearest(sky["t"], amb["t"])
    inn, _ = yf.nearest(sky["t"], non["t"])
    tload = amb["t_load"][il][:, None]
    tstar = yf.tant_star(sky["spec"], non["spec"][inn], amb["spec"][il], tload)
    s = S11(root / "data" / "s11").calibrate(
        root / "data" / "s11" / "cal_materials" / "switch_sparams.npz",
        eigsep_cal.s11.PACKAGED_FILES[2026]["osldata"],
    )
    g = {}
    for dut in ("ant", "amb", "rec"):
        tt, gg = s.get_all_s11s(dut, "default")
        i, _ = yf.nearest(sky["t"], np.asarray(tt, float))
        g[dut] = yf.s11_to_channels(np.asarray(gg)[i], s.freqs / 1e6, d["freqs"])
    t_chr = yf.correct_receiver_s11(tstar, tload, g["ant"], g["amb"], g["rec"])
    # The product, per the same sky visits: median over each visit's rows.
    bn = index.select(time=(start, stop)).load_bundle(key="4", root=proot, products=[f"tcal@{args.version}"])
    cn = bn.calibrated
    assert np.allclose(bn.freqs_mhz, d["freqs"])
    t_prod = np.full_like(t_chr, np.nan)
    for k in range(sky["t"].size):
        rows = (bn.t >= sky["t_start"][k]) & (bn.t <= sky["t_stop"][k])
        if rows.any():
            t_prod[k] = np.nanmedian(cn[rows], axis=0)
    bd = band_mask(d["freqs"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = t_prod[:, bd] / t_chr[:, bd]
    per_visit = np.nanmedian(ratio, axis=1)
    fin = np.isfinite(per_visit)
    out["christian_night"] = {
        "sky_visits": int(sky["t"].size),
        "visits_calibrated_by_product": int(fin.sum()),
        "median_ratio_product_over_christian": float(np.median(per_visit[fin])),
        "ratio_p5_p95_over_visits": [float(np.percentile(per_visit[fin], 5)), float(np.percentile(per_visit[fin], 95))],
        "max_abs_channel_deviation_median_over_visits": float(np.nanmedian(np.nanmax(np.abs(ratio - 1), axis=1))),
    }
    (pdir / "validation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
