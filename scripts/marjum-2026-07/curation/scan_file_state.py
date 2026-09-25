#!/usr/bin/env python3
"""Scan every filtered corr_*.h5 file and record a per-file state fingerprint.

Output: file_state.csv (one row per file, chronological) + boundaries.jsonl
(one entry per detected state change).

The fingerprint has two layers:

  A. Deterministic metadata:
       - filter_phase, data keys, mux copy flags (root attrs)
       - header/wiring (hash + parsed snap-input map)
       - header/obs_config (hash)
       - header/input_to_ant, header/pol_delay (hash)
       - metadata daemon presence: rfswitch / motor / lidar / imu_az / imu_el / adc_stats
       - file open/close time (from header/times) + inter-file gap
       - file cadence (span of header/times)
       - dominant rfswitch state during the file

  B. Numeric signal (per auto input present in the file):
       - total power (mean of the whole 240x1024 auto array)
       - live-channel count (median-nonzero channels)
       - argmax channel (the strongest tone if any)
       - tx-comb detection score: harmonic-sum of the median spectrum at
         candidate spacings (~4 MHz nominal, i.e. 16 chan; scan +/- a few chans)
       - 1.25 MHz comb detection score (~5 chan spacing) - the 7/15 storm event
       - median az/el from motor when populated (real samples only)

The two-layer split matters because deterministic-metadata boundaries are
cheap and exact (any change in a hash = a boundary), while numeric-signal
boundaries need a threshold and are inherently noisier but pick up things
that never touched the config (RFI, battery drop, cable warm-up).

Boundaries are the union of:
  - any change in the fingerprint tuple between consecutive files
  - any large jump (>=5x MAD from local baseline) in a numeric feature
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from eigsep_data.paths import get_campaign_root


def _campaign_root():
    """Campaign root; ``MARJUM_DATA_ROOT`` wins, else the package setting.

    This script anchored on its own ``__file__`` until it moved out of
    the campaign tree on 2026-09-19.
    """
    env = os.environ.get("MARJUM_DATA_ROOT")
    if env:
        return Path(env)
    return get_campaign_root(required=True)


CAMPAIGN_ROOT = _campaign_root()
DATA_DIR = CAMPAIGN_ROOT / "data"
OUT_CSV = CAMPAIGN_ROOT / "curation" / "file_state.csv"
OUT_JSONL = CAMPAIGN_ROOT / "boundaries.jsonl"

DFREQ_MHZ = 0.244140625  # header/dfreq
TX_SPACING_MHZ_NOMINAL = 4.0  # notes say every ~4 MHz
COMB_1P25_MHZ = 1.25         # 7/15 storm-event comb spacing
AUTO_INPUTS = ("0", "1", "2", "3", "4", "5")


def sha1(obj) -> str:
    if isinstance(obj, bytes):
        return hashlib.sha1(obj).hexdigest()[:10]
    if isinstance(obj, str):
        return hashlib.sha1(obj.encode()).hexdigest()[:10]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:10]


def _read_obj(f: h5py.File, name: str):
    """Read a scalar object dataset (JSON-in-bytes) or return None."""
    try:
        v = f[name][()]
    except (KeyError, OSError):
        return None
    if isinstance(v, bytes):
        v = v.decode(errors="replace")
    return v


def comb_score(spectrum: np.ndarray, spacing_chan: float, offset_range=(0, None), max_tones=None) -> tuple[float, int]:
    """Return (score, best_offset) for a peaks-every-`spacing_chan`-channels comb.

    Score = sum of median-subtracted spectrum at comb positions divided by MAD of
    non-comb positions; best_offset is the phase offset (in channels) that
    maximised it. Only channels within (offset_range) are considered live.
    """
    lo, hi = offset_range
    if hi is None:
        hi = int(spacing_chan)
    med = np.median(spectrum)
    mad = np.median(np.abs(spectrum - med)) + 1e-9
    n = len(spectrum)
    best = (-np.inf, 0)
    for off in range(int(lo), int(hi)):
        idx = np.arange(off, n, spacing_chan).round().astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        if max_tones is not None:
            idx = idx[:max_tones]
        if len(idx) < 4:
            continue
        vals = spectrum[idx] - med
        # non-comb positions for reference
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        noncomb_mad = np.median(np.abs(spectrum[mask] - med)) + 1e-9
        score = float(np.median(vals) / noncomb_mad)
        if score > best[0]:
            best = (score, off)
    return best


def summarise_metadata_series(json_text: str | None, keys: tuple[str, ...]):
    """Return (n_real_samples, {key: median_of_non_null}) for a JSON-list metadata sensor."""
    if not json_text:
        return 0, {k: None for k in keys}
    try:
        arr = json.loads(json_text)
    except Exception:
        return 0, {k: None for k in keys}
    reals = 0
    accum = {k: [] for k in keys}
    for row in arr:
        if not isinstance(row, dict):
            continue
        if row.get("status") == "error":
            continue
        good = False
        for k in keys:
            v = row.get(k)
            if v is None:
                continue
            try:
                accum[k].append(float(v))
                good = True
            except (TypeError, ValueError):
                pass
        if good:
            reals += 1
    med = {k: (float(np.median(vs)) if vs else None) for k, vs in accum.items()}
    return reals, med


def per_file(path: Path) -> dict:
    """Compute one row of file_state.csv."""
    row: dict = {"file": path.name}
    try:
        with h5py.File(path, "r") as f:
            # --- Layer A: deterministic ---
            root_attrs = {k: f.attrs[k] for k in f.attrs}
            row["filter_phase"] = str(root_attrs.get("filter_phase", ""))
            row["mux_0to1"] = bool(root_attrs.get("mux_copy_0to1", False))
            row["mux_4to5"] = bool(root_attrs.get("mux_copy_4to5", False))
            data_keys = sorted(f["data"].keys()) if "data" in f else []
            row["data_keys"] = ",".join(data_keys)
            # header
            times = f["header/times"][:]
            row["t_open_unix"] = float(times[0])
            row["t_close_unix"] = float(times[-1])
            row["n_int"] = int(len(times))
            row["file_span_s"] = float(times[-1] - times[0])
            row["dt_int_median_s"] = float(np.median(np.diff(times))) if len(times) > 1 else float("nan")
            hdr_attrs = {k: f["header"].attrs[k] for k in f["header"].attrs}
            row["adc_gain"] = int(hdr_attrs.get("adc_gain", -1))
            row["adc_mux_sel"] = int(hdr_attrs.get("adc_mux_sel", -1))
            row["fft_shift"] = int(hdr_attrs.get("fft_shift", -1))
            row["corr_scalar"] = int(hdr_attrs.get("corr_scalar", -1))
            row["corr_acc_len"] = int(hdr_attrs.get("corr_acc_len", -1))
            row["integration_time_s"] = float(hdr_attrs.get("integration_time", float("nan")))
            row["use_noise"] = bool(hdr_attrs.get("use_noise", False))
            row["use_ref"] = bool(hdr_attrs.get("use_ref", False))
            row["fpg_file"] = str(hdr_attrs.get("fpg_file", ""))
            row["sync_time"] = float(hdr_attrs.get("sync_time", 0.0))
            row["header_upload_unix"] = float(hdr_attrs.get("header_upload_unix", 0.0))
            row["run_tag"] = str(hdr_attrs.get("run_tag", ""))

            # object datasets (as strings)
            wiring = _read_obj(f, "header/wiring")
            obs_cfg = _read_obj(f, "header/obs_config")
            input_to_ant = _read_obj(f, "header/input_to_ant")
            pairs = _read_obj(f, "header/pairs")
            pol_delay = _read_obj(f, "header/pol_delay")

            row["wiring_hash"] = sha1(wiring or "")
            row["obs_config_hash"] = sha1(obs_cfg or "")
            row["input_to_ant_hash"] = sha1(input_to_ant or "")
            row["pairs_hash"] = sha1(pairs or "")
            row["pol_delay_hash"] = sha1(pol_delay or "")

            # extract snap_id and per-antenna Rx IDs from wiring (compact)
            try:
                w = json.loads(wiring) if wiring else {}
                row["snap_id"] = str(w.get("snap_id", ""))
                ants = w.get("ants", {})
                names = sorted(ants.keys())
                row["ants"] = ",".join(names)
                row["ant_rx_ids"] = ",".join(f"{n}:{ants[n].get('rx', {}).get('id')}" for n in names)
                row["ant_snap_inputs"] = ",".join(f"{n}:{ants[n].get('snap', {}).get('input')}" for n in names)
            except Exception:
                row["snap_id"] = ""
                row["ants"] = ""
                row["ant_rx_ids"] = ""
                row["ant_snap_inputs"] = ""

            # metadata daemons: presence + dominant state
            rfswitch_txt = _read_obj(f, "metadata/rfswitch")
            row["has_meta_rfswitch"] = rfswitch_txt is not None
            if rfswitch_txt:
                try:
                    sw = json.loads(rfswitch_txt)
                    c = Counter(str(s) for s in sw if s is not None)
                    row["rfswitch_dominant"] = c.most_common(1)[0][0] if c else ""
                    row["rfswitch_n_none"] = int(sum(1 for s in sw if s is None))
                    row["rfswitch_n_states"] = int(len(c))
                    row["rfswitch_states_json"] = json.dumps(dict(c))
                except Exception:
                    row["rfswitch_dominant"] = "PARSE_ERR"
                    row["rfswitch_n_none"] = -1
                    row["rfswitch_n_states"] = -1
                    row["rfswitch_states_json"] = ""
            else:
                row["rfswitch_dominant"] = ""
                row["rfswitch_n_none"] = -1
                row["rfswitch_n_states"] = -1
                row["rfswitch_states_json"] = ""

            row["has_meta_motor"] = "metadata/motor" in f
            row["has_meta_lidar"] = "metadata/lidar" in f
            row["has_meta_imu_az"] = "metadata/imu_az" in f
            row["has_meta_imu_el"] = "metadata/imu_el" in f
            row["has_meta_adc_stats"] = "metadata/adc_stats" in f
            row["has_meta_potmon"] = "metadata/potmon" in f
            row["has_meta_system_current"] = "metadata/system_current" in f
            row["has_meta_tempctrl_lna"] = "metadata/tempctrl_lna" in f
            row["has_meta_tempctrl_load"] = "metadata/tempctrl_load" in f

            # motor: real az/el
            motor_txt = _read_obj(f, "metadata/motor")
            if motor_txt:
                try:
                    m = json.loads(motor_txt)
                    az = [row_.get("az_pos") for row_ in m
                          if isinstance(row_, dict) and row_.get("az_pos") is not None and row_.get("az_pos") != -11111.0]
                    el = [row_.get("el_pos") for row_ in m
                          if isinstance(row_, dict) and row_.get("el_pos") is not None and row_.get("el_pos") != -11111.0]
                    row["motor_n_real"] = int(min(len(az), len(el)))
                    row["motor_az_med"] = float(np.median(az)) if az else float("nan")
                    row["motor_el_med"] = float(np.median(el)) if el else float("nan")
                    row["motor_az_std"] = float(np.std(az)) if az else float("nan")
                    row["motor_el_std"] = float(np.std(el)) if el else float("nan")
                except Exception:
                    row["motor_n_real"] = 0
                    row["motor_az_med"] = row["motor_el_med"] = row["motor_az_std"] = row["motor_el_std"] = float("nan")
            else:
                row["motor_n_real"] = 0
                row["motor_az_med"] = row["motor_el_med"] = row["motor_az_std"] = row["motor_el_std"] = float("nan")

            # imu_el pitch/roll/el_deg
            imu_el_txt = _read_obj(f, "metadata/imu_el")
            n_imu_el, imu_el_stats = summarise_metadata_series(imu_el_txt, ("el_deg", "pitch", "roll"))
            row["imu_el_n_real"] = n_imu_el
            row["imu_el_deg_med"] = imu_el_stats["el_deg"]
            row["imu_el_pitch_med"] = imu_el_stats["pitch"]
            row["imu_el_roll_med"] = imu_el_stats["roll"]

            imu_az_txt = _read_obj(f, "metadata/imu_az")
            n_imu_az, imu_az_stats = summarise_metadata_series(imu_az_txt, ("yaw", "pitch", "roll"))
            row["imu_az_n_real"] = n_imu_az
            row["imu_az_yaw_med"] = imu_az_stats["yaw"]

            # system current
            sc_txt = _read_obj(f, "metadata/system_current")
            n_sc, sc_stats = summarise_metadata_series(sc_txt, ("current_a", "current_voltage"))
            row["sys_current_n_real"] = n_sc
            row["sys_current_a_med"] = sc_stats["current_a"]

            # potmon
            pm_txt = _read_obj(f, "metadata/potmon")
            n_pm, pm_stats = summarise_metadata_series(pm_txt, ("pot_az_angle",))
            row["potmon_n_real"] = n_pm
            row["potmon_az_med"] = pm_stats["pot_az_angle"]

            # lidar
            lid_txt = _read_obj(f, "metadata/lidar")
            n_lid, lid_stats = summarise_metadata_series(lid_txt, ("distance_m",))
            row["lidar_n_real"] = n_lid
            row["lidar_distance_med"] = lid_stats["distance_m"]

            # tempctrl_lna / _load: fraction "active"
            for name, out_prefix in (("metadata/tempctrl_lna", "tc_lna"), ("metadata/tempctrl_load", "tc_load")):
                txt = _read_obj(f, name)
                n_active = n_ok = 0
                t_now_samples: list[float] = []
                if txt:
                    try:
                        arr = json.loads(txt)
                        for r in arr:
                            if not isinstance(r, dict) or r.get("status") == "error":
                                continue
                            n_ok += 1
                            if r.get("active"):
                                n_active += 1
                            v = r.get("T_now")
                            if v is not None:
                                try:
                                    t_now_samples.append(float(v))
                                except (TypeError, ValueError):
                                    pass
                    except Exception:
                        pass
                row[f"{out_prefix}_n_ok"] = int(n_ok)
                row[f"{out_prefix}_n_active"] = int(n_active)
                row[f"{out_prefix}_T_med"] = float(np.median(t_now_samples)) if t_now_samples else None

            # --- Layer B: signal per auto ---
            for inp in AUTO_INPUTS:
                key = f"data/{inp}"
                if key not in f:
                    for suf in ("power", "live_chans", "argmax_ch", "argmax_val",
                                "comb4mhz_score", "comb4mhz_offset",
                                "comb1p25mhz_score", "comb1p25mhz_offset"):
                        row[f"a{inp}_{suf}"] = None
                    continue
                a = f[key][:]  # (n_int, 1024) int32
                med = np.median(a, axis=0).astype(np.float64)
                row[f"a{inp}_power"] = float(a.astype(np.float64).mean())
                row[f"a{inp}_live_chans"] = int(np.sum(med > 0))
                row[f"a{inp}_argmax_ch"] = int(np.argmax(med))
                row[f"a{inp}_argmax_val"] = float(med.max())

                # tone-comb detection: only run on non-zero band
                nz = med > 0
                # limit tone search to the passband: from first to last nonzero channel
                if nz.sum() > 100:
                    lo = int(np.argmax(nz))
                    hi = int(len(nz) - 1 - np.argmax(nz[::-1]))
                    band = med[lo:hi+1]
                    # 4 MHz spacing -> ~16.384 chan
                    spacing_4 = TX_SPACING_MHZ_NOMINAL / DFREQ_MHZ
                    s4, o4 = comb_score(band, spacing_4, (0, int(round(spacing_4))))
                    row[f"a{inp}_comb4mhz_score"] = s4
                    row[f"a{inp}_comb4mhz_offset"] = int(o4 + lo)
                    # 1.25 MHz spacing -> ~5.12 chan
                    spacing_125 = COMB_1P25_MHZ / DFREQ_MHZ
                    s125, o125 = comb_score(band, spacing_125, (0, int(round(spacing_125))))
                    row[f"a{inp}_comb1p25mhz_score"] = s125
                    row[f"a{inp}_comb1p25mhz_offset"] = int(o125 + lo)
                else:
                    row[f"a{inp}_comb4mhz_score"] = None
                    row[f"a{inp}_comb4mhz_offset"] = None
                    row[f"a{inp}_comb1p25mhz_score"] = None
                    row[f"a{inp}_comb1p25mhz_offset"] = None

            row["error"] = ""
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


# stable column order so the CSV is human-friendly
COLUMNS = [
    "file", "t_open_unix", "t_close_unix", "n_int", "file_span_s", "dt_int_median_s",
    "filter_phase", "mux_0to1", "mux_4to5", "data_keys",
    "adc_gain", "adc_mux_sel", "fft_shift", "corr_scalar", "corr_acc_len",
    "integration_time_s", "use_noise", "use_ref", "sync_time", "header_upload_unix",
    "run_tag", "fpg_file",
    "snap_id", "ants", "ant_rx_ids", "ant_snap_inputs",
    "wiring_hash", "obs_config_hash", "input_to_ant_hash", "pairs_hash", "pol_delay_hash",
    "has_meta_rfswitch", "rfswitch_dominant", "rfswitch_n_none", "rfswitch_n_states", "rfswitch_states_json",
    "has_meta_motor", "has_meta_lidar", "has_meta_imu_az", "has_meta_imu_el",
    "has_meta_adc_stats", "has_meta_potmon", "has_meta_system_current",
    "has_meta_tempctrl_lna", "has_meta_tempctrl_load",
    "motor_n_real", "motor_az_med", "motor_el_med", "motor_az_std", "motor_el_std",
    "imu_el_n_real", "imu_el_deg_med", "imu_el_pitch_med", "imu_el_roll_med",
    "imu_az_n_real", "imu_az_yaw_med",
    "sys_current_n_real", "sys_current_a_med",
    "potmon_n_real", "potmon_az_med",
    "lidar_n_real", "lidar_distance_med",
    "tc_lna_n_ok", "tc_lna_n_active", "tc_lna_T_med",
    "tc_load_n_ok", "tc_load_n_active", "tc_load_T_med",
]
for inp in AUTO_INPUTS:
    for suf in ("power", "live_chans", "argmax_ch", "argmax_val",
                "comb4mhz_score", "comb4mhz_offset",
                "comb1p25mhz_score", "comb1p25mhz_offset"):
        COLUMNS.append(f"a{inp}_{suf}")
COLUMNS.append("error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Debug: limit files")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("corr_*.h5"))
    if args.limit:
        files = files[: args.limit]
    print(f"scanning {len(files)} files with {args.jobs} workers -> {args.out_csv}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(per_file, p): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)} done")

    rows.sort(key=lambda r: (r["file"],))  # filenames are lexicographically time-sorted

    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
