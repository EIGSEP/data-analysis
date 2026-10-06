"""Look for externally recorded solar radio bursts in raw box-air data.

Builds ``derived/ground_sky/bursts_vNNNN``. For every burst in
``curation/solar_bursts_v0001.json`` (type III bursts recorded by two or more
e-CALLISTO stations while the Sun was above the Marjum ridge), it reads raw
box-air antenna rows (``rfswitch == RFANT``) from 90 s before to 90 s after the
burst. **No flags are applied**: the RFI flags mark positive excesses, which is
what a burst is. Each channel is divided by its median over the cutout, and

- **band light curves** (median over channels) are formed in 45-65, 65-85,
  110-170 and 170-235 MHz. Each band's excess during the burst (from 3 s before
  its start to 8 s after its end) is expressed in robust standard deviations
  (MAD) of the same band outside the burst (more than 20 s before or 30 s after);
- **sub-band peak times**, in 10 MHz sub-bands from 25 to 235 MHz, record when
  each part of the band peaks, so a drift from high to low frequency shows;
- the **cutout** (ratio to median, all channels, every row) is saved for figures.

The null is the same measurement at ``--n-null`` random times inside the same
Sun-up windows and at least 5 min from any listed burst (seeded). A burst seen
by e-CALLISTO at 45-85 MHz should be bright at 45-65 MHz and faint above
170 MHz in our data; the local daytime spikes are broadband. So the test
statistic is the fraction of times with z(45-65) above a threshold and
z(170-235) below one, in bursts against the null.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python sun_bursts.py v0001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import eigsep_data
from eigsep_data import MetadataIndex

from common import HERE, campaign_root, git_rev, sha256

BURSTS = "curation/solar_bursts_v0001.json"
BANDS = [(45, 65), (65, 85), (110, 170), (170, 235)]
SUBBANDS = [(lo, lo + 10) for lo in range(25, 235, 10)]


def ts(iso):
    return pd.Timestamp(iso).timestamp()


def load(index, a, b):
    sel = index.select(time=(a, b))
    if sel.nrows == 0:
        return None
    B = sel.load_bundle(antenna="box-air")
    ok = (B.meta.rfswitch == "RFANT").to_numpy()
    D = B.data.astype(float)[ok]
    D[D <= 0] = np.nan
    if ok.sum() < 50:
        return None
    return B.t[ok], B.freqs_mhz, D


def measure(index, t0, t1, pad=90.0, keep=False):
    got = load(index, t0 - pad, t1 + pad)
    if got is None:
        return None
    t, f, D = got
    R = D / np.nanmedian(D, axis=0) - 1
    inw = (t >= t0 - 3) & (t <= t1 + 8)
    outw = ~((t >= t0 - 20) & (t <= t1 + 30))
    if not inw.any() or outw.sum() < 20:
        return None
    out = {"n_rows": int(len(t)), "bands": {}, "subbands": {}}

    def stat(lc):
        base = lc[outw]
        mad = 1.4826 * np.nanmedian(np.abs(base - np.nanmedian(base)))
        k = int(np.nanargmax(np.where(inw, lc, -np.inf)))
        return {"peak": float(lc[k]), "t_peak_s": float(t[k] - t0),
                "z": float((lc[k] - np.nanmedian(base)) / mad) if mad > 0 else np.nan}

    for lo, hi in BANDS:
        out["bands"][f"{lo}-{hi}"] = stat(np.nanmedian(R[:, (f > lo) & (f < hi)], axis=1))
    for lo, hi in SUBBANDS:
        out["subbands"][f"{lo}-{hi}"] = stat(np.nanmedian(R[:, (f > lo) & (f < hi)], axis=1))
    if keep:
        out["cutout"] = (t - t0, f, R.astype(np.float32))
    return out


def pattern(m, z_lo, z_hi):
    b = m["bands"]
    return b["45-65"]["z"] > z_lo and b["170-235"]["z"] < z_hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--n-null", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--z-lo", type=float, default=10.0)
    ap.add_argument("--z-hi", type=float, default=5.0)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"bursts_{args.version}"
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; versions are immutable")
    ref = json.loads((campaign / BURSTS).read_text())
    index = MetadataIndex(campaign / "data")

    events, cut = [], {}
    for k, e in enumerate(ref["events"]):
        t0, t1 = ts(e["start_utc"]), ts(e["end_utc"])
        m = measure(index, t0, t1, keep=True)
        rec = {**e, "has_data": m is not None}
        if m is not None:
            tt, f, R = m.pop("cutout")
            cut[f"t_{k}"], cut[f"R_{k}"], cut["freqs_mhz"] = tt, R, f
            rec.update(m)
            rec["pattern"] = bool(pattern(m, args.z_lo, args.z_hi))
        events.append(rec)
        print(e["start_utc"], "no data" if m is None else
              " ".join(f"{b} z {v['z']:.1f}" for b, v in m["bands"].items()), flush=True)

    rng = np.random.default_rng(args.seed)
    starts = np.array([ts(e["start_utc"]) for e in ref["events"]])
    wins = [(ts(a), ts(b)) for a, b in ref["provenance"]["windows_utc"]]
    null, tries = [], 0
    while len(null) < args.n_null and tries < 20 * args.n_null:
        tries += 1
        a, b = wins[rng.integers(len(wins))]
        t0 = rng.uniform(a, b - 60)
        if np.min(np.abs(starts - t0)) < 300:
            continue
        m = measure(index, t0, t0 + 15)
        if m is not None:
            null.append({"t": t0, **m, "pattern": bool(pattern(m, args.z_lo, args.z_hi))})

    from scipy.stats import binom
    have = [e for e in events if e["has_data"]]
    p0 = float(np.mean([n["pattern"] for n in null]))
    k = int(sum(e["pattern"] for e in have))
    summary = {"n_bursts": len(events), "n_with_data": len(have), "n_null": len(null),
               "rule": f"z(45-65) > {args.z_lo} and z(170-235) < {args.z_hi}",
               "null_rate": p0, "bursts_matching": k,
               "p_value": float(binom.sf(k - 1, len(have), p0)) if have else None}

    out_dir.mkdir(parents=True)
    (out_dir / "bursts.json").write_text(json.dumps(
        {"summary": summary, "events": events, "null": null}, indent=1))
    np.savez_compressed(out_dir / "cutouts.npz", **cut)
    manifest = {"provenance": {
        "product": "ground_sky_bursts", "version": args.version,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "data-analysis/scripts/marjum-2026-07/ground_sky/sun_bursts.py",
        "argv": sys.argv[1:],
        "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0])},
        "inputs": {"solar_bursts": {"path": f"marjum-2026-07/{BURSTS}",
                                    "sha256": sha256(campaign / BURSTS)}}},
        "params": {k_: v for k_, v in vars(args).items() if k_ not in ("version", "out_dir")},
        "summary": summary,
        "files": {"bursts.json": "per-burst band and sub-band statistics, the null draws, the summary",
                  "cutouts.npz": "per burst k with data: t_k (s from burst start), R_k (rows x channels, raw/median - 1), freqs_mhz"}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
