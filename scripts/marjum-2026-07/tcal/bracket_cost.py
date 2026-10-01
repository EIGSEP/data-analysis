"""What wider tcal limits would cost, in temperature.

Measured on the rx-A stretch of ``derived/tcal/<version>/``, where the
calibrator cycles every ~13 min and S11 is swept hourly:

1. **Calibration brackets.** For every pair of cycles m apart, each
   cycle between them is predicted by linear interpolation and compared
   with what was measured. The errors in P_load and in the gain proxy
   G = P_NS - P_load become an error in T* through
   dT* = -T_NS dP_load / G - (T* - T_load) dG / G, on the median T*
   spectrum of the 16/17 Jul night.
2. **S11 holds.** Each sweep is used in place of the one dt later, and
   the calibrated T (antenna + load + receiver correction) is compared.
3. **The 16:20 -> 19:19 hole.** The antenna was raised to ~91 m inside
   it (field notes, ~18:50 UTC). Normal drift over a span cannot see a
   step, so the two edges of the hole are compared directly with the
   distribution of changes over the same span elsewhere.

Writes ``bracket_cost.json`` beside the product. Run::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 python bracket_cost.py v0000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import eigsep_data
from eigsep_cal.dicke import mismatch, receiver_s11_coefficients

NOTEBOOK_DIR = Path(__file__).resolve().parents[3] / "notebooks" / "arp" / "marjum-2026-07"
BAND = (50.0, 200.0)
#: Raise to ~91 m, field notes (INDEX.md height eras).
RAISE_UTC = "2026-07-17T18:50:00Z"


def unix(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def median_night_tstar(root):
    """Median T* over the sky visits of Christian's night (Christian's chain)."""
    sys.path.insert(0, str(NOTEBOOK_DIR))
    import d5_yfactor as yf

    tz = ZoneInfo("America/Denver")
    np.seterr(divide="ignore", invalid="ignore")
    d = yf.load_corr(
        root / "data",
        datetime(2026, 7, 16, 21, 30, tzinfo=tz).timestamp(),
        datetime(2026, 7, 17, 7, 55, tzinfo=tz).timestamp(),
        "4",
    )
    v = yf.average_blocks(d, use=~np.all(d["spec"] == 0, axis=1))
    sky, non, amb = (v[s] for s in (yf.SKY, yf.NOISE, yf.LOAD))
    il, _ = yf.nearest(sky["t"], amb["t"])
    inn, _ = yf.nearest(sky["t"], non["t"])
    tl = amb["t_load"][il][:, None]
    ts = yf.tant_star(sky["spec"], non["spec"][inn], amb["spec"][il], tl)
    return np.median(ts, axis=0), float(np.nanmedian(tl))


def cycles(z):
    """Pair each rx-A noise-source visit with its load visit."""
    keep_n = z["non_regime"] == "rx-A"
    tn, pn = z["non_t"][keep_n], z["non_p"][keep_n]
    ka = z["amb_regime"] == "rx-A"
    ta, pa = z["amb_t"][ka], z["amb_p"][ka]
    j = np.array([np.argmin(np.abs(ta - t)) for t in tn])
    ok = np.abs(ta[j] - tn) < 300
    return tn[ok], pa[j][ok], pn[ok] - pa[j][ok]


def stats(x):
    x = np.asarray(x)
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
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
    b = (f >= BAND[0]) & (f < BAND[1])
    t_ns = float(z["t_ns_k"])
    tstar, t_load = median_night_tstar(root)
    # The hole is the longest gap between rx-A cycles.
    tc, plc, gc = cycles(z)
    ih = int(np.argmax(np.diff(tc)))
    before, after = tc[ih], tc[ih + 1]
    out = {"band_mhz": BAND, "t_star_reference": "median over the 16/17 Jul sky visits"}

    # 1. Calibration brackets, quiet stretch only (no bracket spans the hole).
    quiet = tc <= before
    t, pl, g = tc[quiet], plc[quiet], gc[quiet]
    cadence = float(np.median(np.diff(t)))
    rows = []
    for m in (2, 3, 4, 5, 7, 9, 11, 14):
        errs = []
        for i in range(t.size - m):
            k_all = range(i + 1, i + m)
            for k in k_all:
                w = (t[k] - t[i]) / (t[i + m] - t[i])
                dpl = (1 - w) * pl[i] + w * pl[i + m] - pl[k]
                dg = (1 - w) * g[i] + w * g[i + m] - g[k]
                dts = -t_ns * dpl / g[k] - (tstar - t_load) * dg / g[k]
                errs.append(np.median(np.abs(dts / tstar)[b]))
        rows.append({"bracket_min": round(m * cadence / 60, 1), **stats(errs)})
    out["cal_bracket_frac_error_in_tstar"] = rows

    # 2. S11 hold, rx-A pre-hole sweeps.
    st = z["s11_ant_t"]
    pre = st <= before
    g_ant, g_amb, g_rec = (z[f"s11_{d}"][pre] for d in ("ant", "amb", "rec"))
    st = st[pre]

    def t_cal(i):
        a, o = 1.0, 0.0  # T* is given, so the T* line is the identity
        sc, of = receiver_s11_coefficients(a, o, t_load, g_ant[i], g_amb[i], g_rec[i])
        return np.real(sc * tstar + of)

    rows = []
    for m in (1, 2, 3, 4):
        errs, dts = [], []
        for i in range(st.size - m):
            ref = t_cal(i + m)
            errs.append(np.nanmedian(np.abs(t_cal(i) / ref - 1)[b]))
            dts.append(st[i + m] - st[i])
        rows.append({"hold_h": round(float(np.median(dts)) / 3600, 2), **stats(errs)})
    out["s11_hold_frac_error_in_t"] = rows

    # 3. Across the hole: edges vs the same span elsewhere.
    ib, ia = ih, ih + 1
    span = after - before
    edge = float(np.median(np.abs(gc[ia] / gc[ib] - 1)[b]))
    ref = []
    for i in np.flatnonzero(tc <= before):
        j = np.argmin(np.abs(tc - (tc[i] + span)))
        if abs(tc[j] - tc[i] - span) < 600 and tc[j] <= before:
            ref.append(np.median(np.abs(gc[j] / gc[i] - 1)[b]))
    s_all = z["s11_ant_t"]
    raise_t = unix(RAISE_UTC)
    sb = np.flatnonzero(s_all < raise_t).max()
    sa = np.flatnonzero(s_all > raise_t).min()
    s_span = s_all[sa] - s_all[sb]
    dgam = float(np.median(np.abs(z["s11_ant"][sa] - z["s11_ant"][sb])[b]))
    sref = []
    for i in range(s_all.size):
        j = np.argmin(np.abs(s_all - (s_all[i] + s_span)))
        if abs(s_all[j] - s_all[i] - s_span) < 1800 and s_all[j] < s_all[sb] + 1:
            sref.append(np.median(np.abs(z["s11_ant"][j] - z["s11_ant"][i])[b]))
    out["hole"] = {
        "cycle_edges_utc": [datetime.utcfromtimestamp(x).strftime("%H:%M") for x in (before, after)],
        "gain_change_across_hole": edge,
        "gain_change_same_span_elsewhere": stats(ref) if ref else None,
        "s11_edges_utc": [datetime.utcfromtimestamp(x).strftime("%H:%M") for x in (s_all[sb], s_all[sa])],
        "s11_ant_abs_change_across_hole": dgam,
        "s11_ant_abs_change_same_span_elsewhere": stats(sref) if sref else None,
    }
    (pdir / "bracket_cost.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
