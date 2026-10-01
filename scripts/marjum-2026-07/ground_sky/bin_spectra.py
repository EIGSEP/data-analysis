"""Build derived/ground_sky/binned_vNNNN: calibrated box-air spectra in static bins.

The input to the ground-temperature and sky fit. For each beam frequency
(the HFSS slices, which fall on correlator channels), every calibrated row
gets the mean of the unflagged channels within +-``--half-window`` of it.
Rows are then averaged in ``--bin-s`` bins of constant pointing, height era
and receiver regime.

Row selection:

- ``tcal`` (``--tcal``) calibrated temperatures at plane P, so RFANT rows
  inside its coverage only.
- ``flags@v2`` bits 0-7. Bit 8 is the known-defective
  ``dpss_residual_outlier`` and is ignored (``flags/v2/README.md``).
- Transmitter comb: channels = 0 mod 8 are dropped from 07-17 15:36 to
  07-18 03:00 whatever the flags say (memo 001, Combs). Every beam
  frequency is such a channel, so in that span the window loses its centre.
- ``pointing@v2.0`` with ``quality == "ok"``; a bin is kept only if az and
  el each span less than ``--max-spread-deg`` within it.
- Height era from ``curation/mode_table.jsonl`` and receiver regime from
  ``curation/cal_windows.jsonl``, per file; a bin never mixes either.

Each bin's noise is the standard error of its rows' values. The manifest
compares it with the radiometer expectation.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python bin_spectra.py v0001
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

from common import (
    CHANNEL_MHZ,
    HERE,
    campaign_root,
    era_of_files,
    git_rev,
    regime_of_files,
    sha256,
    workspace_root,
)

# tcal v0002 coverage (derived/tcal/v0002/README.md, Coverage), with margin;
# rows outside it are NaN in Bundle.calibrated anyway.
DEFAULT_SPANS = [
    ("2026-07-17T04:00:00Z", "2026-07-17T16:30:00Z"),
    ("2026-07-18T01:20:00Z", "2026-07-18T03:00:00Z"),
]
TX_SPAN = ("2026-07-17T15:36:00Z", "2026-07-18T03:00:00Z")
CLEAN_BITS = 0xFF


def beam_channels():
    """Correlator channels of the HFSS beam slices (250 MHz dropped)."""
    with np.load(workspace_root() / "data-analysis/hfss_beam_maps/bowtie_beam.npz") as z:
        freqs = z["freqs"][:-1]
    chans = np.rint(freqs / CHANNEL_MHZ).astype(int)
    assert np.allclose(chans * CHANNEL_MHZ, freqs), "beam slices off-grid"
    return freqs, chans


def unix(iso):
    return pd.Timestamp(iso).timestamp()


def reduce_rows(b, chans, half_window, min_channels):
    """Per-row mean over each beam channel's window, NaN where too few."""
    T = b.calibrated
    flags = b.flags
    # A file with no flag payload comes back NaN; its rows are dropped.
    unflagged = np.isnan(flags) if flags.dtype.kind == "f" else np.zeros(flags.shape, bool)
    codes = np.where(unflagged, 0, flags).astype(np.uint16)
    bad = ((codes & CLEAN_BITS) != 0) | unflagged | ~np.isfinite(T)
    in_tx = (b.t >= unix(TX_SPAN[0])) & (b.t <= unix(TX_SPAN[1]))
    comb = np.arange(T.shape[1]) % 8 == 0
    bad[np.ix_(in_tx, comb)] = True
    T = np.where(bad, 0.0, T)
    good = (~bad).astype(float)
    out = np.full((T.shape[0], len(chans)), np.nan)
    nch = np.zeros((T.shape[0], len(chans)), dtype=int)
    for k, c in enumerate(chans):
        sl = slice(c - half_window, c + half_window + 1)
        n = good[:, sl].sum(axis=1)
        ok = n >= min_channels
        out[ok, k] = T[ok, sl].sum(axis=1) / n[ok]
        nch[:, k] = n
    return out, nch, int(unflagged.all(axis=1).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version", help="output version, e.g. v0001")
    ap.add_argument("--tcal", default="v0002")
    ap.add_argument("--flags", default="v2")
    ap.add_argument("--pointing", default="v2.0")
    ap.add_argument("--bin-s", type=float, default=120.0)
    ap.add_argument("--half-window", type=int, default=2)
    ap.add_argument("--min-channels", type=int, default=3)
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--max-spread-deg", type=float, default=1.0)
    ap.add_argument("--chunk-s", type=float, default=3600.0)
    ap.add_argument("--span", nargs=2, action="append", metavar=("START", "STOP"),
                    help="UTC span to process (repeatable); default: tcal coverage")
    ap.add_argument("--out-dir", type=Path,
                    help="write here instead of derived/ground_sky (for trials)")
    args = ap.parse_args()
    spans = args.span or DEFAULT_SPANS

    campaign = campaign_root()
    out_dir = args.out_dir or (
        campaign / "derived" / "ground_sky" / f"binned_{args.version}")
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; versions are immutable")

    freqs, chans = beam_channels()
    index = MetadataIndex(campaign / "data")
    products = [f"tcal@{args.tcal}", f"flags@{args.flags}",
                f"pointing@{args.pointing}"]

    rows = []
    no_flags = {"rows": 0, "files": []}
    for start, stop in spans:
        t0, t1 = unix(start), unix(stop)
        for c0 in np.arange(t0, t1, args.chunk_s):
            sel = index.select(time=(c0, min(c0 + args.chunk_s, t1)))
            if sel.nrows == 0:
                continue
            b = sel.load_bundle(key="4", products=products)
            T, nch, n_unflagged = reduce_rows(
                b, chans, args.half_window, args.min_channels)
            no_flags["rows"] += n_unflagged
            no_flags["files"] += b.provenance["products"]["flags"].get("skipped") or []
            p = b.pointing
            keep = (
                np.isfinite(T).any(axis=1)
                & (p.quality == "ok").to_numpy()
                & np.isfinite(p.az_deg.to_numpy())
                & np.isfinite(p.el_deg.to_numpy())
            )
            files = b.meta.file.to_numpy()[keep]
            rows.append(pd.DataFrame({
                "t": b.t[keep],
                "file": files,
                "az": p.az_deg.to_numpy()[keep],
                "el": p.el_deg.to_numpy()[keep],
                "itime": b.meta.integration_time.to_numpy(float)[keep],
            }).assign(T=list(T[keep]), nch=list(nch[keep])))
            print(f"{pd.Timestamp(c0, unit='s')}: {sel.nrows} rows, "
                  f"{keep.sum()} calibrated and pointed", flush=True)
    rows = pd.concat(rows, ignore_index=True)
    rows["era"] = era_of_files(rows.file.to_numpy(), campaign)
    rows["regime"] = regime_of_files(rows.file.to_numpy(), campaign)
    rows["bin"] = (rows.t // args.bin_s).astype(np.int64)

    out = {k: [] for k in ("t", "az", "el", "era", "regime", "n_rows",
                           "file_first", "T", "sigma", "n_good", "itime")}
    dropped = {"mixed": 0, "moving": 0, "few_rows": 0}
    for _, g in rows.groupby("bin"):
        if g.era.nunique() > 1 or g.regime.nunique() > 1:
            dropped["mixed"] += 1
            continue
        if np.ptp(g.az) >= args.max_spread_deg or np.ptp(g.el) >= args.max_spread_deg:
            dropped["moving"] += 1
            continue
        if len(g) < args.min_rows:
            dropped["few_rows"] += 1
            continue
        T = np.stack(g["T"].to_numpy())
        n = np.isfinite(T).sum(axis=0)
        mean = np.nanmean(T, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma = np.nanstd(T, axis=0, ddof=1) / np.sqrt(n)
        few = n < args.min_rows
        mean[few], sigma[few] = np.nan, np.nan
        out["t"].append(g.t.mean())
        out["az"].append(g.az.median())
        out["el"].append(g.el.median())
        out["era"].append(g.era.iloc[0])
        out["regime"].append(g.regime.iloc[0])
        out["n_rows"].append(len(g))
        out["file_first"].append(g.file.iloc[0])
        out["T"].append(mean)
        out["sigma"].append(sigma)
        out["n_good"].append(n)
        out["itime"].append(g.itime.median())

    arrays = {k: np.asarray(v) for k, v in out.items()}
    for k in ("era", "regime", "file_first"):
        arrays[k] = arrays[k].astype(str)

    # Radiometer check: rows are integration_time long, each averaging the
    # window's good channels.
    nch_mean = np.mean([np.mean(x) for x in rows.nch])
    expected = arrays["T"] / np.sqrt(
        CHANNEL_MHZ * 1e6 * nch_mean * arrays["itime"][:, None]
        * arrays["n_good"]
    )
    ratio = np.nanmedian(arrays["sigma"] / expected, axis=0)

    out_dir.mkdir(parents=True)
    np.savez_compressed(
        out_dir / "binned.npz", freqs_mhz=freqs, channels=chans, **arrays
    )
    tcal_dir = campaign / "derived" / "tcal" / args.tcal
    manifest = {
        "provenance": {
            "product": "ground_sky_binned",
            "version": args.version,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/bin_spectra.py",
            "argv": sys.argv[1:],
            "code": {
                "data-analysis": git_rev(HERE),
                "eigsep_data": git_rev(eigsep_data.__path__[0]),
            },
            "inputs": {
                "tcal": {"spec": f"tcal@{args.tcal}",
                         "manifest_sha256": sha256(tcal_dir / "manifest.json")},
                "flags": {"spec": f"flags@{args.flags}", "bits": "0-7 (0xFF)",
                          "manifest_sha256": sha256(
                              campaign / "flags" / args.flags / "manifest.json")},
                "pointing": {"spec": f"pointing@{args.pointing}",
                             "sha256": sha256(
                                 campaign / "curation" / "pointing_table.parquet")},
                "mode_table_sha256": sha256(campaign / "curation" / "mode_table.jsonl"),
                "cal_windows_sha256": sha256(campaign / "curation" / "cal_windows.jsonl"),
            },
        },
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()
                   if k not in ("version", "span", "out_dir")},
        "spans_utc": spans,
        "tx_comb_span_utc": TX_SPAN,
        "counts": {
            "rows_calibrated_and_pointed": int(len(rows)),
            "bins": int(len(arrays["t"])),
            "rows_dropped_no_flag_payload": no_flags,
            "bins_dropped": dropped,
            "bins_by_era_regime": {
                f"{e}/{r}": int(n) for (e, r), n in pd.Series(
                    list(zip(arrays["era"], arrays["regime"]))
                ).value_counts().items()
            },
        },
        "noise_over_radiometer_median": dict(
            zip([f"{f:.3f}" for f in freqs], np.round(ratio, 3).tolist())
        ),
        "arrays": {
            "freqs_mhz, channels": "beam frequencies and their correlator channels",
            "t": "bin mean time, Unix s",
            "az, el": "median pointing-table az and el, deg (table convention; add the beam fit's az offset)",
            "era, regime": "height era (mode table) and receiver regime (cal windows)",
            "T": "(nbin, nfreq) mean calibrated temperature at plane P, K",
            "sigma": "(nbin, nfreq) standard error of T from the rows' scatter, K",
            "n_good": "(nbin, nfreq) rows contributing",
            "n_rows, itime, file_first": "rows in the bin, median row integration time (s), first raw file",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["counts"], indent=1))
    print("noise / radiometer, median by frequency:",
          np.round(ratio, 2).tolist())


if __name__ == "__main__":
    main()
