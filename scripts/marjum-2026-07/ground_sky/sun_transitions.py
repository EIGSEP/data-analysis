"""Raw box-air data around every terrain sunrise and sunset of phase C.

Builds ``derived/ground_sky/transitions_vNNNN``. For each time the Sun's centre
crosses the DEM horizon at the antenna (from ``sun_events_v0001``), it keeps
the raw context ±``--half-min`` minutes around it, minute by minute:

- the mean raw antenna spectrum (``RFANT`` rows, ``flags@v3`` any bit but 9;
  the transmitter's comb channels masked while it is on);
- how many rows were antenna rows, and the switch states of the others;
- whether the transmitter was on (box-gnd transitions);
- the pointing (median and spread of az and el).

It then fits raw power at each HFSS beam frequency, over the contiguous parked
stretch of data containing the crossing (gaps ≤ 3 min, pointing within 1.5° of
the crossing's; azimuth within 3°, for potentiometer noise), with
transmitter-on minutes dropped unless it is on for over 80% of them, as a line plus the Sun column (K per SFU; beam gain toward
the Sun × knife-edge visibility at the ridge, at that pointing) shifted by τ.
Channels within ±2 of the beam frequency are used, or, when the transmitter is
on anywhere in the window, channels 3-5 away (midway between its comb teeth).

Raw power is uncalibrated. The Sun coefficient over the mean power is S/T_sys
(per SFU), and is turned into a flux with the 5th-95th percentile range of the
calibrated temperatures in ``fit_v0003`` as a stand-in for T_sys. That
stand-in comes from other hours and heights.

It also records when the ground under the antenna (2 and 10 m above it) goes
into or comes out of shadow: a signal from the lit ground, or from heating
there, would follow those times instead of the antenna's.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python sun_transitions.py v0001
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
import eigsep_sim
from eigsep_base.rotations import mount_rotation
from eigsep_data import MetadataIndex
from eigsep_sim.design_matrix import HealpixBeam

import sun as sunmod
from common import CHANNEL_MHZ, HERE, campaign_root, git_rev, sha256, workspace_root

SUN_EVENTS = "derived/ground_sky/sun_events_v0001/sun_events.json"
FIT_T = "derived/ground_sky/fit_v0003/fit.npz"
TX = "curation/transmitter_transitions_boxgnd.jsonl"
MODES = "curation/mode_table.jsonl"


def ts(iso):
    return pd.Timestamp(iso).timestamp()


def tx_episodes(campaign):
    out = []
    for line in open(campaign / TX):
        e = json.loads(line)
        if "t_start_utc" in e:
            out.append((ts(e["t_start_utc"]), ts(e["t_end_utc"])))
    return out


def minute_context(B, a, n_min, tx):
    """Per-minute raw spectra and metadata."""
    f = B.freqs_mhz
    ch = np.rint(f / CHANNEL_MHZ).astype(int)
    D = B.data.astype(float)
    fl = B.flags
    codes = np.where(np.isnan(fl), 0, fl).astype(np.uint16) if fl.dtype.kind == "f" else fl.astype(np.uint16)
    D[((codes & ~np.uint16(1 << 9)) != 0) | (D <= 0)] = np.nan
    k = ((B.t - a) // 60).astype(int)
    rf = B.meta.rfswitch.to_numpy()
    p = B.pointing
    az = np.mod(p.az_deg.to_numpy() + 180, 360) - 180
    el = p.el_deg.to_numpy()
    tx_row = np.zeros(len(B.t), bool)
    for s, e in tx:
        tx_row |= (B.t >= s) & (B.t <= e)
    comb = ch % 8 == 0
    spec = np.full((n_min, len(f)), np.nan, np.float32)
    meta = {k_: np.full(n_min, np.nan) for k_ in
            ("n_rows", "n_ant", "tx_frac", "az", "el", "az_ptp", "el_ptp")}
    states = []
    for m in range(n_min):
        s = k == m
        meta["n_rows"][m] = s.sum()
        a_ = s & (rf == "RFANT")
        meta["n_ant"][m] = a_.sum()
        states.append(pd.Series(rf[s]).value_counts().to_dict() if s.any() else {})
        if a_.sum() >= 10:
            X = D[a_].copy()
            if tx_row[a_].any():
                X[:, comb] = np.nan
            spec[m] = np.nanmean(X, axis=0)
            meta["tx_frac"][m] = tx_row[a_].mean()
            for nm, v in (("az", az), ("el", el)):
                vv = v[a_]
                if np.isfinite(vv).sum() >= 5:
                    meta[nm][m] = np.nanmedian(vv)
                    meta[nm + "_ptp"][m] = np.nanpercentile(vv, 95) - np.nanpercentile(vv, 5)
    return spec, meta, states


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--half-min", type=int, default=60)
    ap.add_argument("--fit-half-min", type=int, default=45)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"transitions_{args.version}"
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; versions are immutable")
    index = MetadataIndex(campaign / "data")
    crossings = [e for e in json.loads((campaign / SUN_EVENTS).read_text())["events"]
                 if e["horizon"] == "terrain"]
    tx = tx_episodes(campaign)
    hb = HealpixBeam.from_npz(workspace_root() / "data-analysis/hfss_beam_maps/bowtie_beam.npz",
                              drop_last=True)
    bf = hb.freqs_hz / 1e6
    freqs = np.array([x for x in bf if 45 < x < 235 and not 85 < x < 110])
    beam = hb.select([int(np.argmin(np.abs(bf - x))) for x in freqs])
    with np.load(campaign / "derived/beam/empirical_raster_v0012/dpss/diagnostics.npz") as d:
        az_off, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    with np.load(campaign / FIT_T, allow_pickle=True) as z:
        fT = z["freqs_mhz"]
        Tlo, Thi = np.nanpercentile(z["data_k"], [5, 95], axis=1)
    taus = np.arange(-60, 61, 2) * 60.0

    arrays, summary = {}, []
    for i, cr in enumerate(crossings):
        tev, era = cr["t_unix"], cr["era"]
        a = tev - args.half_min * 60
        n_min = 2 * args.half_min
        sel = index.select(time=(a, a + n_min * 60))
        rec = {"kind": cr["kind"], "era": era, "utc": cr["utc"], "t_unix": tev}
        print(cr["utc"], cr["kind"], era, flush=True)
        if sel.nrows == 0:
            rec["status"] = "no data"
            summary.append(rec)
            continue
        B = sel.load_bundle(antenna="box-air", products=["flags@v3", "pointing@v2.0"])
        spec, meta, states = minute_context(B, a, n_min, tx)
        tm = a + 60 * np.arange(n_min) + 30
        f = B.freqs_mhz
        ch = np.rint(f / CHANNEL_MHZ).astype(int)
        mid = bool(np.nanmax(np.nan_to_num(meta["tx_frac"])) > 0)
        cols = [np.isin(np.abs(ch - int(round(x / CHANNEL_MHZ))), [3, 4, 5]) if mid
                else np.abs(ch - int(round(x / CHANNEL_MHZ))) <= 2 for x in freqs]
        Y = np.array([np.nanmean(spec[:, c], axis=1) for c in cols])

        # Pointing at the crossing and the parked stretch around it.
        near = np.argsort(np.abs(tm - tev))
        have = [j for j in near if np.isfinite(meta["az"][j]) and meta["el_ptp"][j] < 1.5]
        mt = pd.DataFrame([json.loads(x) for x in open(campaign / MODES)])
        mt = mt[mt.get("file_first").notna()]
        ma = pd.to_datetime(mt.t_start_utc).astype("int64") / 1e9
        mb = pd.to_datetime(mt.t_end_utc).astype("int64") / 1e9
        over = mt[(mb >= a) & (ma <= a + n_min * 60)]
        rec["boxair_emi"] = sorted(set(over.boxair_emi.astype(str)))
        rec["rot_states"] = sorted(set(over.rot_state.astype(str)))
        rec.update({"mid_channels": mid, "states": states and {
            k_: int(sum(s.get(k_, 0) for s in states)) for k_ in set().union(*states)}})
        if not have:
            rec["status"] = "no parked pointing near the crossing"
        else:
            az0, el0 = meta["az"][have[0]], meta["el"][have[0]]
            parked = (np.abs(meta["az"] - az0) < 3.0) & (np.abs(meta["el"] - el0) < 1.5) \
                & (meta["n_ant"] >= 10)
            # Transmitter-on minutes leak power even between its comb teeth;
            # drop them unless (nearly) the whole window has it on.
            txon = np.nan_to_num(meta["tx_frac"]) > 0
            near_ = np.abs(tm - tev) <= args.fit_half_min * 60
            tx_all = txon[near_ & parked].mean() > 0.8 if (near_ & parked).any() else False

            # contiguous stretch containing the crossing, gaps <= 3 min
            j0 = int(np.argmin(np.abs(tm - tev)))
            idx_p = np.flatnonzero(parked & (np.abs(tm - tev) <= args.fit_half_min * 60))
            seg = []
            if len(idx_p):
                c0 = idx_p[np.argmin(np.abs(idx_p - j0))]
                seg = [c0]
                for step in (-1, 1):
                    last = c0
                    for j in idx_p[::step]:
                        if (j - last) * step > 0 and abs(j - last) <= 3:
                            seg.append(j)
                            last = j
                seg = np.array(sorted(set(seg)))
                if not tx_all:
                    seg = seg[~txon[seg]]
            grid = np.arange(tev - 3 * 3600, tev + 3 * 3600, 30.0)
            R = np.broadcast_to(mount_rotation(az0 + az_off, el0, psi), (len(grid), 3, 3))
            geom = sunmod.sun_geometry(grid, era)
            col = sunmod.sun_column(beam, R, geom, "knife")
            arrays[f"grid_{i}"], arrays[f"col_{i}"] = grid, col.astype(np.float32)
            fits, chi = [], np.full((len(freqs), len(taus)), np.nan)
            seg_ok = len(seg) >= 15 and tm[seg].min() < tev - 300 and tm[seg].max() > tev - 120
            for fi, x in enumerate(freqs):
                y = Y[fi, seg] if seg_ok else np.array([])
                m = np.isfinite(y)
                if m.sum() < 15:
                    fits.append(None)
                    continue
                tt = tm[seg][m]

                def fit(tau):
                    A = np.stack([np.ones(m.sum()), (tt - tev) / 3600,
                                  np.interp(tt - tau, grid, col[fi])], 1)
                    co, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                    r = y[m] - A @ co
                    return co, float(r @ r), A

                co, c2, A = fit(0.0)
                s2 = c2 / (m.sum() - 3)
                cov = np.linalg.inv(A.T @ A) * s2
                for j, tau in enumerate(taus):
                    chi[fi, j] = fit(tau)[1] / s2
                frac = co[2] / np.nanmean(y[m])
                efrac = np.sqrt(cov[2, 2]) / np.nanmean(y[m])
                k_ = int(np.argmin(np.abs(fT - x)))
                fits.append({"freq_mhz": float(x), "s_over_tsys": float(frac),
                             "sigma": float(efrac), "S_lo": float(frac * Tlo[k_]),
                             "S_hi": float(frac * Thi[k_]), "T_lo": float(Tlo[k_]),
                             "T_hi": float(Thi[k_]), "n_min": int(m.sum())})
            dchi = np.nanmean(chi - np.nanmin(chi, axis=1, keepdims=True), axis=0) \
                if np.isfinite(chi).any() else np.full(len(taus), np.nan)
            arrays[f"dchi_{i}"] = dchi
            shadow = {}
            for name in ("ground+2", "ground+10"):
                evs = sunmod.terrain_events(tev - 3 * 3600, tev + 3 * 3600, name)
                ts_ = [e["t_unix"] for e in evs if e["horizon"] == "terrain" and e["kind"] == cr["kind"]]
                shadow[name] = (float(min(ts_, key=lambda x: abs(x - tev))) - tev) / 60 if ts_ else None
            rec.update({
                "status": "fitted" if seg_ok and any(fits) else "no usable parked stretch",
                "transmitter_minutes_kept": bool(tx_all),
                "pointing_az_el": [float(az0), float(el0)],
                "fit_minutes": [float((tm[seg].min() - tev) / 60), float((tm[seg].max() - tev) / 60)]
                if len(seg) else None,
                "tau_best_min": float(taus[np.nanargmin(dchi)] / 60) if np.isfinite(dchi).any() else None,
                "fits": fits, "ground_shadow_min": shadow})
        arrays[f"spec_{i}"] = spec
        arrays[f"Y_{i}"] = Y.astype(np.float32)
        arrays[f"tm_{i}"] = tm
        for k_, v in meta.items():
            arrays[f"{k_}_{i}"] = v
        arrays[f"freqs_mhz_{i}"] = f
        summary.append(rec)
        print("  ", rec.get("status"), rec.get("tau_best_min"), flush=True)

    out_dir.mkdir(parents=True)
    np.savez_compressed(out_dir / "transitions.npz", beam_freqs_mhz=freqs, taus_s=taus, **arrays)
    (out_dir / "transitions.json").write_text(json.dumps({"crossings": summary}, indent=1))
    manifest = {"provenance": {
        "product": "ground_sky_transitions", "version": args.version,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "data-analysis/scripts/marjum-2026-07/ground_sky/sun_transitions.py",
        "argv": sys.argv[1:],
        "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0]),
                 "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
        "inputs": {k: {"path": f"marjum-2026-07/{v}", "sha256": sha256(campaign / v)}
                   for k, v in (("sun_events", SUN_EVENTS), ("fit_v0003", FIT_T), ("tx", TX),
                                ("dem", sunmod.DEM_PATH))}},
        "params": {k: v for k, v in vars(args).items() if k not in ("version", "out_dir")},
        "files": {"transitions.json": "per crossing: status, pointing, fit stretch, per-frequency fits, timing scan best, ground-shadow times",
                  "transitions.npz": "per crossing i: spec_i (minute x channel raw mean), Y_i (beam freq x minute), tm_i, n_rows/n_ant/tx_frac/az/el/az_ptp/el_ptp_i, grid_i/col_i (Sun column), dchi_i"}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
