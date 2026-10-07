"""Blind scan of raw box-air for impulsive broadband events, by Sun visibility.

Builds ``derived/ground_sky/impulsive_vNNNN``. Raw antenna rows of phase C
(no flags) are divided, channel by channel, by a running median over
``--win`` rows. The 30 % most variable channels of each 30-min chunk are
dropped, and the median over channels gives band light curves in 45-65,
65-85, 110-140, 140-170, 170-200 and 200-235 MHz. An event is a run of rows
where any band exceeds ``--thr``. For each event the scan records its start,
length, the peak in each band, and when each band peaks.

Each event and each chunk of data is classed by the Sun: **visible** (above the
DEM horizon at the antenna, ``sun.sun_geometry``), **day, behind ridge** (above
0 deg altitude but not visible) or **night**. If the events were solar, their rate
would follow visibility; if they are local, it follows the time of day.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python impulsive_scan.py v0001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label, median_filter

import eigsep_data
from eigsep_data import MetadataIndex

import sun as sunmod
from common import ERAS, HERE, campaign_root, era_of_files, git_rev

BANDS = [(45, 65), (65, 85), (110, 140), (140, 170), (170, 200), (200, 235)]


def scan_chunk(index, a, b, thr, win):
    sel = index.select(time=(a, b))
    if sel.nrows == 0:
        return [], None
    B = sel.load_bundle(antenna="box-air")
    ok = (B.meta.rfswitch == "RFANT").to_numpy()
    if ok.sum() < win:
        return [], None
    f, t = B.freqs_mhz, B.t[ok]
    D = B.data.astype(float)[ok]
    D[D <= 0] = np.nan
    files = B.meta.file.to_numpy()[ok]
    bg = median_filter(np.nan_to_num(D, nan=np.nanmedian(D)), size=(win, 1), mode="nearest")
    R = D / bg - 1
    mad = np.nanmedian(np.abs(R), axis=0)
    good = mad < np.nanpercentile(mad, 70)
    E = np.array([np.nanmedian(R[:, (f > lo) & (f < hi) & good], axis=1)
                  for lo, hi in BANDS]).T
    lab, n = label(np.nanmax(E, axis=1) > thr)
    ev = []
    for k in range(1, n + 1):
        rows = np.flatnonzero(lab == k)
        Ek = np.nan_to_num(E[rows[0]:rows[-1] + 1], nan=-1.0)
        pk = Ek.argmax(axis=0)
        ev.append({"t": float(t[rows[0]]), "file": str(files[rows[0]]),
                   "n_rows": int(len(rows)), "peak": Ek.max(axis=0).round(4).tolist(),
                   "t_peak_s": (t[rows[0] + pk] - t[rows[0]]).round(3).tolist()})
    return ev, {"t0": float(a), "n_rows": int(ok.sum()),
                "dt_s": float(np.median(np.diff(t))), "file": str(files[0])}


def classify(t, files, campaign):
    eras = era_of_files(np.asarray(files), campaign)
    out = np.full(len(t), "unknown", dtype=object)
    known = np.isin(eras, list(ERAS))
    if known.any():
        g = sunmod.sun_geometry(np.asarray(t)[known],
                                np.array([ERAS[e] for e in eras[known]], dtype=object))
        out[known] = np.where(g["theta_deg"] > 0, "visible",
                              np.where(g["alt"] > 0, "day, behind ridge", "night"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--start", default="2026-07-15T00:30:00Z")
    ap.add_argument("--stop", default="2026-07-18T03:30:00Z")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--win", type=int, default=61)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"impulsive_{args.version}"
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; versions are immutable")
    index = MetadataIndex(campaign / "data")
    events, cov = [], []
    for c in pd.date_range(args.start, args.stop, freq="30min"):
        try:
            e, cv = scan_chunk(index, c.timestamp(), c.timestamp() + 1800, args.thr, args.win)
        except Exception as ex:  # one unreadable chunk should not stop the scan
            print("FAIL", c, ex, flush=True)
            continue
        events += e
        if cv:
            cov.append(cv)
        print(c, len(e), flush=True)
    ev = pd.DataFrame(events)
    cv = pd.DataFrame(cov)
    ev["class"] = classify(ev.t.values, ev.file.values, campaign)
    cv["class"] = classify(cv.t0.values + 900, cv.file.values, campaign)
    cv["hours"] = cv.n_rows * cv.dt_s / 3600
    P = np.stack(ev.peak.values)
    T = np.stack(ev.t_peak_s.values)
    sel = {"all": np.ones(len(ev), bool), "peak > 0.3": P.max(1) > 0.3,
           "3 or more rows": ev.n_rows.values >= 3,
           "3+ rows, 45-65 MHz peaks after 170-200 MHz":
               (ev.n_rows.values >= 3) & (T[:, 0] - T[:, 4] >= 0.5)}
    rates = {}
    for lab, s in sel.items():
        rates[lab] = {c: {"n": int((s & (ev["class"] == c)).sum()),
                          "hours": float(cv[cv["class"] == c].hours.sum()),
                          "per_hour": float((s & (ev["class"] == c)).sum()
                                            / max(cv[cv["class"] == c].hours.sum(), 1e-9))}
                      for c in ("visible", "day, behind ridge", "night")}
    out_dir.mkdir(parents=True)
    ev.to_json(out_dir / "events.jsonl", orient="records", lines=True)
    cv.to_json(out_dir / "coverage.jsonl", orient="records", lines=True)
    manifest = {"provenance": {
        "product": "ground_sky_impulsive", "version": args.version,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "data-analysis/scripts/marjum-2026-07/ground_sky/impulsive_scan.py",
        "argv": sys.argv[1:],
        "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0])}},
        "params": {k: v for k, v in vars(args).items() if k not in ("version", "out_dir")},
        "bands_mhz": BANDS, "rates": rates,
        "files": {"events.jsonl": "one event per line: t, file, n_rows, peak and t_peak_s per band, class",
                  "coverage.jsonl": "one 30-min chunk per line: rows scanned, hours, class"}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
