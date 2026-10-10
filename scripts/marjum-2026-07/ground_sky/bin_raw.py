"""Build derived/ground_sky/raw_binned_vNNNN: raw box-air power over all of phase C.

The input to ``global_fit.py``, the per-frequency model of every phase-C
spectrum. Unlike ``bin_spectra.py`` it uses raw correlator power, not
``tcal`` temperatures, so it covers phase C wherever the switch rests on the
antenna, not only where a kelvin scale exists. The receiver gain is left to
the fit.

Rows:

- box-air (input 4) antenna rows: ``rfswitch == RFANT``, or any row of a file
  recorded with the switch daemon off (every row ``MISSING``; the switch then
  rests on the antenna, as ``flags@v3`` assumes);
- phase C, ``--start`` to ``--stop``.

Channels and masks, per beam frequency (the HFSS slices, which fall on the
transmitter's 8-channel grid):

- the six channels 3-5 either side of the slice, midway between transmitter
  teeth (as ``raster_sky.py``);
- ``flags@v3``: any bit except 9 (advisory ``high_scatter``) and 10;
  non-positive samples (int32 wraps, dropped integrations);
- box-air's self-EMI (integer MHz +- 0.15 MHz, memo 006) in the files
  ``curation/boxair_emi_presence.jsonl`` marks on.

Bins: consecutive antenna rows of one file group (height era, receiver
regime, accumulation length), closed after ``--bin-s`` seconds, at a row gap
over 5 s, or when the pointing moves more than ``--max-move-deg`` from the
bin's first pointed row. A bin's value at each frequency is the mean of its
unmasked samples. Its noise is given two ways: the radiometer prediction
y / sqrt(dnu tau) for the unmasked samples, and the standard error of its
rows' values.

Pointing: ``pointing@v2.0`` rows of quality ``ok`` or ``suspect``. A bin with
none takes the pointing of the bins either side if those agree within
``--fill-tol-deg`` in az and el (the drives are locked when idle), and is
marked ``pointing_filled``; otherwise its pointing stays NaN.

Also stored per bin: the transmitter state (``curation/transmitter_transitions_boxgnd.jsonl``,
on if any row is inside an episode, with the on fraction), whether box-air's
self-EMI file flag is set, and where ``tcal@v0003`` covers the bin, its
mean scale and offset at each frequency (T = scale P + offset at plane P),
so the fit's gains can be tied to kelvin.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python bin_raw.py v0001 --out-dir /some/trial/dir
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

from delay_filter import BANDS, delay_lowpass
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

#: Flag bits that do not reject a sample: 9 (advisory high_scatter) and 10.
KEEP_BITS = (1 << 9) | (1 << 10)
#: Channel offsets from each beam slice: midway between transmitter teeth.
OFFSETS = np.array([-5, -4, -3, 3, 4, 5])
MAX_ROW_GAP_S = 5.0
#: Per-process delay-filter counts, collected per chunk.
FILTER_STATS = []


def ts(iso):
    return pd.Timestamp(iso).timestamp()


def beam_freqs():
    with np.load(workspace_root() / "data-analysis/hfss_beam_maps/bowtie_beam.npz") as z:
        freqs = z["freqs"][:-1]
    chans = np.rint(freqs / CHANNEL_MHZ).astype(int)
    assert np.allclose(chans * CHANNEL_MHZ, freqs), "beam slices off-grid"
    return freqs, chans


def tx_episodes(campaign):
    rows = [json.loads(x) for x in open(campaign / "curation/transmitter_transitions_boxgnd.jsonl")]
    return np.array([(ts(r["t_start_utc"]), ts(r["t_end_utc"])) for r in rows if "episode" in r])


def emi_files(campaign):
    rows = [json.loads(x) for x in open(campaign / "curation/boxair_emi_presence.jsonl")]
    return {r["file"] for r in rows if r.get("boxair_emi") is True}


def reduce_chunk(B, chans, emi, tcal, t=None, ant=None, delay_filter=None):
    """Per-row values at each beam frequency (mean over unmasked offset channels),
    the unmasked sample count, and the tcal scale/offset there.

    ``delay_filter``: None, or a dict of ``delay_filter.delay_lowpass`` keywords
    (``tau_ns`` None for its union-mask-only control), applied to the antenna
    rows ``ant`` after masking."""
    f = B.freqs_mhz
    ch = np.rint(f / CHANNEL_MHZ).astype(int)
    D = B.data.astype(float)
    fl = B.flags
    codes = (np.where(np.isnan(fl), 0, fl) if fl.dtype.kind == "f" else fl).astype(np.uint16)
    bad = ((codes & ~np.uint16(KEEP_BITS)) != 0) | (D <= 0)
    row_emi = B.meta.file.isin(emi).to_numpy()
    bad |= row_emi[:, None] & (np.abs(f - np.round(f)) < 0.15)[None, :]
    D[bad] = np.nan
    if delay_filter is not None:
        k = np.flatnonzero(ant)
        if k.size:
            D[k], st = delay_lowpass(t[k], f, D[k], **delay_filter)
            FILTER_STATS.append(st)
        D[np.flatnonzero(~ant)] = np.nan
    pos = {c: i for i, c in enumerate(ch)}
    idx = np.array([[pos.get(c + o, -1) for o in OFFSETS] for c in chans])
    ok = idx >= 0
    Y = np.full((len(D), len(chans)), np.nan)
    N = np.zeros((len(D), len(chans)), int)
    S = np.full((len(D), len(chans)), np.nan)
    O = np.full((len(D), len(chans)), np.nan)
    for j in range(len(chans)):
        cols = idx[j][ok[j]]
        if not cols.size:
            continue
        sub = D[:, cols]
        N[:, j] = np.isfinite(sub).sum(1)
        with np.errstate(invalid="ignore"):
            Y[:, j] = np.nanmean(sub, 1)
        if tcal is not None:
            S[:, j] = np.nanmean(tcal["scale"][:, cols], 1)
            O[:, j] = np.nanmean(tcal["offset"][:, cols], 1)
    return Y, N, S, O, row_emi


def process_chunk(c, args, chans, episodes, emi):
    """Bins of one chunk of ``args.chunk_s`` seconds starting at ``c``."""
    campaign = campaign_root()
    index = MetadataIndex(campaign / "data")
    a1 = ts(args.stop)
    bins = []
    sel = index.select(time=(c, min(c + args.chunk_s, a1)))
    if sel.nrows == 0:
        return []
    B = sel.load_bundle(antenna="box-air", products=["flags@v3", "pointing@v2.0", "tcal@v0003"])
    meta = B.meta.reset_index(drop=True)
    o = np.argsort(B.t, kind="stable")
    t = B.t[o]
    rf = meta.rfswitch.to_numpy()[o]
    files = meta.file.to_numpy()[o]
    daemon_off = meta.groupby("file").rfswitch.transform(
        lambda x: (x == "MISSING").all()).to_numpy()[o]
    ant = (rf == "RFANT") | daemon_off
    tc = B.products.get("tcal")
    tcal = {"scale": np.asarray(tc["scale"])[o], "offset": np.asarray(tc["offset"])[o]} if tc else None
    dfk = None
    if args.delay_filter_ns is not None or args.union_mask_only:
        dfk = {"tau_ns": None if args.union_mask_only else args.delay_filter_ns,
               "block_s": args.filter_block_s}
    Y, N, S, O, row_emi = reduce_chunk(_Ordered(B, o), chans, emi, tcal, t, ant, dfk)
    p = B.pointing.iloc[o].reset_index(drop=True)
    good_p = (p.quality.isin(["ok", "suspect"]).to_numpy()
              & np.isfinite(p.az_deg.to_numpy()) & np.isfinite(p.el_deg.to_numpy()))
    az = p.az_deg.to_numpy()
    el = p.el_deg.to_numpy()
    tau = meta.integration_time.to_numpy(float)[o]
    # The accumulation length, from the integration time (0.268 s or 0.537 s;
    # corr_acc_len is not in the bundle metadata).
    acc = np.rint(tau * 1e3).astype(int)
    era = era_of_files(files, campaign)
    reg = regime_of_files(files, campaign)
    tx = np.zeros(len(t), bool)
    for s_, e_ in episodes:
        tx |= (t >= s_) & (t < e_)
    group = pd.Series(list(zip(era, reg, acc))).astype(str).to_numpy()

    rows = np.flatnonzero(ant)
    cur = []

    def close(cur):
        if len(cur) < args.min_rows:
            return
        k = np.array(cur)
        gp = k[good_p[k]]
        w = N[k]
        yv = Y[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.nansum(yv * w, 0) / w.sum(0)
            nrow = np.isfinite(yv).sum(0)
            sem = np.nanstd(yv, 0, ddof=1) / np.sqrt(nrow)
        nsamp = w.sum(0)
        bins.append({
            "t": float(t[k].mean()), "t0": float(t[k[0]]), "t1": float(t[k[-1]]),
            "n": len(k), "tau": float(tau[k].mean()), "first_file": files[k[0]],
            "era": era[k[0]], "regime": reg[k[0]], "acc": int(acc[k[0]]),
            "az": float(np.mean(az[gp])) if gp.size else np.nan,
            "el": float(np.mean(el[gp])) if gp.size else np.nan,
            "az_span": float(np.ptp(az[gp])) if gp.size else np.nan,
            "n_pointed": int(gp.size),
            "tx_frac": float(tx[k].mean()), "emi": bool(row_emi[k].any()),
            "daemon_off": bool(daemon_off[k].all()),
            "y": y, "nsamp": nsamp, "sem": sem,
            "scale": np.nanmean(S[k], 0) if tcal is not None else np.full(len(chans), np.nan),
            "offset": np.nanmean(O[k], 0) if tcal is not None else np.full(len(chans), np.nan),
        })

    for i in rows:
        if cur:
            j0 = cur[0]
            gap = t[i] - t[cur[-1]] > MAX_ROW_GAP_S
            long = t[i] - t[j0] >= args.bin_s
            newgrp = group[i] != group[j0]
            pts = [j for j in cur if good_p[j]]
            moved = bool(pts) and good_p[i] and (
                abs(az[i] - az[pts[0]]) > args.max_move_deg or abs(el[i] - el[pts[0]]) > args.max_move_deg)
            if gap or long or newgrp or moved:
                close(cur)
                cur = []
        cur.append(i)
    close(cur)
    print(pd.Timestamp(c, unit="s"), len(bins), flush=True)
    if FILTER_STATS:
        bins.append({"_filter_stats": {k: sum(x[k] for x in FILTER_STATS) for k in FILTER_STATS[0]}})
        FILTER_STATS.clear()
    return bins


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--start", default="2026-07-15T00:30:00Z")
    ap.add_argument("--stop", default="2026-07-18T03:30:00Z")
    ap.add_argument("--bin-s", type=float, default=60.0)
    ap.add_argument("--max-move-deg", type=float, default=1.0)
    ap.add_argument("--fill-tol-deg", type=float, default=1.0)
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--chunk-s", type=float, default=3600.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--delay-filter-ns", type=float, default=None,
                    help="delay low-pass (delay_filter.py) of each integration before binning")
    ap.add_argument("--union-mask-only", action="store_true",
                    help="the filter's control: its block-union mask and drops, no filter")
    ap.add_argument("--filter-block-s", type=float, default=5.0)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"raw_binned_{args.version}"
    if (out_dir / "binned.npz").exists():
        sys.exit("exists; versions are immutable")
    freqs, chans = beam_freqs()
    episodes = tx_episodes(campaign)
    emi = emi_files(campaign)
    a0, a1 = ts(args.start), ts(args.stop)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.jobs) as ex:
        parts = list(ex.map(process_chunk, np.arange(a0, a1, args.chunk_s),
                            *zip(*[(args, chans, episodes, emi)] * len(np.arange(a0, a1, args.chunk_s)))))
    bins = [b_ for p_ in parts for b_ in p_ if "_filter_stats" not in b_]
    fst = [b_["_filter_stats"] for p_ in parts for b_ in p_ if "_filter_stats" in b_]
    filter_stats = {k: int(sum(x[k] for x in fst)) for k in fst[0]} if fst else None

    df = pd.DataFrame([{k: v for k, v in b.items() if np.ndim(v) == 0} for b in bins])
    arr = {k: np.stack([b[k] for b in bins]) for k in ("y", "nsamp", "sem", "scale", "offset")}
    # Fill pointing across gaps where the pointing either side agrees.
    df["pointing_filled"] = False
    have = np.isfinite(df.az.to_numpy()) & np.isfinite(df.el.to_numpy())
    idx = np.flatnonzero(have)
    for i in np.flatnonzero(~have):
        lo, hi = idx[idx < i], idx[idx > i]
        if not lo.size or not hi.size:
            continue
        a, b = lo[-1], hi[0]
        if df.era[a] != df.era[i] or df.era[b] != df.era[i]:
            continue
        if (abs(df.az[a] - df.az[b]) <= args.fill_tol_deg and abs(df.el[a] - df.el[b]) <= args.fill_tol_deg):
            df.loc[i, ["az", "el"]] = [df.az[a], df.el[a]]
            df.loc[i, "pointing_filled"] = True

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "binned.npz", freqs_mhz=freqs, chans=chans,
                        **{c: df[c].to_numpy() for c in df.columns if df[c].dtype != object},
                        **{c: df[c].to_numpy().astype(str) for c in df.columns if df[c].dtype == object},
                        **arr)
    manifest = {
        "provenance": {
            "product": "ground_sky_raw_binned", "version": args.version,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/bin_raw.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0])},
            "inputs": {k: sha256(campaign / "curation" / k) for k in (
                "transmitter_transitions_boxgnd.jsonl", "boxair_emi_presence.jsonl",
                "mode_table.jsonl", "cal_windows.jsonl")},
            "products": ["flags@v3", "pointing@v2.0", "tcal@v0003"],
        },
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "version"},
        "channel_offsets": OFFSETS.tolist(), "keep_flag_bits": [9, 10],
        "delay_filter": filter_stats and {
            **filter_stats, "bands_mhz": [list(b) for b in BANDS],
            "flag_frac_in": filter_stats["flag_in"] / filter_stats["cells"],
            "flag_frac_union": filter_stats["flag_union"] / filter_stats["cells"]},
        "counts": {
            "bins": int(len(df)), "rows": int(df.n.sum()),
            "bins_pointed": int(np.isfinite(df.az).sum()),
            "bins_pointing_filled": int(df.pointing_filled.sum()),
            "bins_by_era": df.era.value_counts().to_dict(),
            "bins_by_regime": df.regime.value_counts().to_dict(),
            "bins_daemon_off": int(df.daemon_off.sum()),
            "bins_tx_on": int((df.tx_frac > 0).sum()),
            "bins_with_tcal": int(np.isfinite(arr["scale"]).all(1).sum()),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["counts"], indent=1))


class _Ordered:
    """A bundle view with rows reordered (for reduce_chunk)."""

    def __init__(self, B, o):
        self.freqs_mhz = B.freqs_mhz
        self.data = B.data[o]
        self.flags = B.flags[o]
        self.meta = B.meta.iloc[o].reset_index(drop=True)


if __name__ == "__main__":
    main()
