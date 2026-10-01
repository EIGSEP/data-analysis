"""Build ``derived/tcal/<version>/``: the box-air field temperature calibration.

Collects, for the stretches of the campaign where the RF switch was
cycling through its calibrators, everything ``eigsep_data.products.tcal``
needs to put a box-air integration on a kelvin scale at reference plane P:

- one masked, integration-time-weighted mean spectrum per ambient-load
  (RFAMB) and noise-source (RFNON) visit, labelled with its receiver regime from ``cal_windows.jsonl``;
- every field S11 sweep of the antenna, the ambient load and the
  receiver, calibrated to plane P with ``eigsep_cal.S11`` and put on the
  correlator channels;
- every load-thermistor reading in the same span.

It writes ``solutions.npz`` and ``manifest.json``; the hand-written
``README.md`` beside them says what the product is for. Run::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 python build_tcal.py v0000

The calibration itself (interpolation, limits, equations) is applied
when a row is loaded, by ``eigsep_data.products.tcal`` and
``eigsep_cal.dicke``; nothing here computes a temperature.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import eigsep_cal
import eigsep_cal.s11
import eigsep_data
from eigsep_cal import S11
from eigsep_data import MetadataIndex

#: The switched input: box-air in phase C (INDEX.md, calibration
#: constraints). The only one the calibration applies to.
KEY = "4"
#: Noise-source excess over the load, K: nameplate ENR 35 dB behind a
#: 30 dB pad, 290 K * 10**0.5. Not measured.
T_NS_K = 290.0 * 10 ** ((35.0 - 30.0) / 10)
#: Span searched for calibrator visits. Phase C, and after the first
#: regular cycle; the isolated earlier cycles bracket nothing.
SPAN_UTC = ("2026-07-17T00:00:00Z", "2026-07-18T04:00:00Z")
#: A new visit starts where consecutive rows of a state are further
#: apart than this. Integrations are 0.27 or 0.54 s apart.
MAX_ROW_GAP_S = 5.0
#: Visits with fewer usable rows than this are partial and dropped.
MIN_VISIT_ROWS = 50
#: A visit is contaminated, and dropped, when more than MAX_BAD_CHANNELS
#: channels in OUTLIER_BAND_MHZ sit more than OUTLIER_FRAC from the median
#: of its neighbouring visits (up to two either side, same regime). In v0000
#: three visits that shared a window with an S11 sweep (07-17 08:53 and
#: 14:08 RFNON, 14:09 RFAMB) had 278-296 such channels, up to 900x bright.
#: Clean rx-A visits have at most 4; rx-B's edge visits reach 48, because
#: rx-B drifts fast and an edge visit's neighbours all lie on one side.
OUTLIER_BAND_MHZ = (30.0, 245.0)
OUTLIER_FRAC = 0.10
MAX_BAD_CHANNELS = 100
#: Per-sample mask inside a visit, built here because flags@v2 runs no
#: detector on calibration rows (it sets only their ``cal`` bit). A sample
#: is masked if it is wrapped (int32 overflow, negative counts), in an
#: all-zero (dropped) integration, or more than SAMPLE_NSIG robust sigmas
#: (1.4826 MAD) from the visit's own per-channel median. The median is used
#: only to flag; the visit spectrum is the weighted mean of what remains.
SAMPLE_NSIG = 5.0
#: Regime labels that may bracket a row. The rx-transition cycle at
#: 07-17 19:43 sits inside the bounded gap and is not used.
REGIMES = ("rx-A", "rx-B")
#: S11 epochs: either side of the 07-17 19:11 -> 07-18 01:17 gap in
#: sweeps, which spans the receiver-regime boundary (no calibration of
#: any kind ran in it; the antenna data did). Sweeps are never
#: interpolated or held across it.
EPOCH_SPLIT_UTC = "2026-07-17T22:30:00Z"

#: Chosen from bracket_cost.py (2026-10-01): on rx-A, widening the
#: calibration bracket from 60 to 90 min costs ~0.1 % in T* (median
#: 0.33 -> ~0.42 %), and holding S11 for 2 h costs 0.04 % in T. The
#: 177 min rx-A hole is still excluded: the gain changes 8.5 % across
#: it, against at most 5.2 % over the same span elsewhere.
LIMITS = {
    # 13 min cadence; bridges rx-B's 65 min gap, not the 177 min hole.
    "max_cal_gap_s": 90 * 60.0,
    # Sweeps are hourly with one 162 min gap (16:29 -> 19:10).
    "max_s11_gap_s": 3 * 3600.0,
    # Must stay under half the 367 min gap between S11 epochs (checked).
    "s11_hold_s": 2 * 3600.0,
    # Thermistor readings are per integration.
    "therm_tol_s": 60.0,
}


def unix(iso):
    return pd.Timestamp(iso).timestamp()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_rev(path):
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain", "--", ".")),
    }


def regime_of(files, windows):
    """Receiver regime of each visit, from the curated cal windows.

    Matched on file, not time: the windows are bounded by filenames,
    which are file close times and trail the rows by a minute or two.
    A visit takes the label of the window whose ``file_first`` ..
    ``file_last`` range holds its first file; a visit in no window gets
    none and is dropped rather than guessed.
    """
    out = []
    for f in files:
        hit = windows[(windows.file_first <= f) & (windows.file_last >= f)]
        out.append(hit.receiver_regime.iloc[0] if len(hit) else "")
    return np.array(out, dtype=str)


def visits(index, state, windows):
    sel = index.select(time=SPAN_UTC, rfswitch=state)
    sel = sel.select(files=("corr_20260715_003217Z.h5", "corr_20991231"))
    loaded = sel.load(keys=[KEY])
    p = np.asarray(loaded.data[KEY], dtype=float)
    t = loaded.meta.time_best.to_numpy(dtype=float)
    tau = loaded.meta.integration_time.to_numpy(dtype=float)
    files = loaded.meta.file.to_numpy(str)
    order = np.argsort(t, kind="stable")
    p, t, tau, files = p[order], t[order], tau[order], files[order]
    masked_frac = []
    usable = ~np.all(p == 0, axis=1)
    starts = np.flatnonzero(np.r_[True, np.diff(t) > MAX_ROW_GAP_S])
    stops = np.r_[starts[1:], t.size]
    rows = []
    for a, b in zip(starts, stops):
        use = a + np.flatnonzero(usable[a:b])
        spec = None
        if use.size:
            spec, frac = masked_mean(p[use], tau[use])
            masked_frac.append(frac)
        rows.append(
            {
                "t": np.average(t[use], weights=tau[use]) if use.size else np.nan,
                "file": files[a],
                "n": use.size,
                "p": spec,
            }
        )
    v = pd.DataFrame(rows)
    v["regime"] = regime_of(v.file, windows)
    keep = (v.n >= MIN_VISIT_ROWS) & v.regime.isin(REGIMES)
    log = {
        "visits_found": int(len(v)),
        "dropped_partial": int((v.n < MIN_VISIT_ROWS).sum()),
        "dropped_regime": int(
            ((v.n >= MIN_VISIT_ROWS) & ~v.regime.isin(REGIMES)).sum()
        ),
        "kept": int(keep.sum()),
        "rows_per_visit": [int(v.n[keep].min()), int(v.n[keep].max())],
        "masked_sample_fraction": {
            "median_over_visits": float(np.median(masked_frac)),
            "max_over_visits": float(np.max(masked_frac)),
        },
    }
    v = v[keep].reset_index(drop=True)
    freqs = np.asarray(loaded.freq, dtype=float)
    bad, v = contaminated(v, freqs)
    log["dropped_contaminated"] = bad
    log["kept"] = int(len(v))
    return v, freqs, log


def masked_mean(p, tau):
    """Integration-time-weighted mean of one visit, masked per sample.

    Returns the spectrum (NaN in a channel with no unmasked sample) and
    the fraction of samples masked.
    """
    bad = p < 0
    med = np.median(p, axis=0)
    mad = 1.4826 * np.median(np.abs(p - med), axis=0)
    with np.errstate(invalid="ignore"):
        bad |= (mad > 0) & (np.abs(p - med) > SAMPLE_NSIG * mad)
    w = np.where(bad, 0.0, tau[:, None])
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        spec = np.where(wsum > 0, (w * p).sum(axis=0) / wsum, np.nan)
    return spec, float(bad.mean())


def contaminated(v, freqs):
    """Drop visits that disagree with their neighbours in many channels.

    Returns the dropped visits (UTC and channel count) and the rest.
    """
    band = (freqs >= OUTLIER_BAND_MHZ[0]) & (freqs < OUTLIER_BAND_MHZ[1])
    p = np.stack(v.p.to_list())
    n_bad = np.zeros(len(v), dtype=int)
    for i in range(len(v)):
        nb = [
            j for j in range(i - 2, i + 3)
            if j != i and 0 <= j < len(v) and v.regime[j] == v.regime[i]
        ]
        if len(nb) < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = p[i][band] / np.median(p[nb][:, band], axis=0)
        n_bad[i] = int(np.sum(np.abs(r - 1) > OUTLIER_FRAC))
    drop = n_bad > MAX_BAD_CHANNELS
    dropped = [
        {
            "utc": datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M"),
            "channels_off": int(n),
        }
        for t, n in zip(v.t[drop], n_bad[drop])
    ]
    return dropped, v[~drop].reset_index(drop=True)


def s11_on_channels(s11_dir, switch_path, osl_path, freqs_mhz):
    s = S11(s11_dir).calibrate(switch_path, osl_path)
    fs = s.freqs / 1e6
    split = unix(EPOCH_SPLIT_UTC)
    out, log = {}, {"skipped_files": [list(x) for x in s.skipped]}
    for dut in ("ant", "amb", "rec"):
        t, g = s.get_all_s11s(dut, "default")
        t = np.asarray(t, dtype=float)
        keep = (t >= unix(SPAN_UTC[0])) & (t < unix(SPAN_UTC[1]))
        t, g = t[keep], np.asarray(g)[keep]
        on = np.array(
            [
                np.interp(freqs_mhz, fs, x.real, left=np.nan, right=np.nan)
                + 1j * np.interp(freqs_mhz, fs, x.imag, left=np.nan, right=np.nan)
                for x in g
            ]
        )
        epoch = np.where(t < split, "pre-gap", "post-gap")
        out[dut] = (t, on, epoch)
        log[dut] = {
            "sweeps": int(t.size),
            "plane": "lna (P)" if dut != "rec" else "dut (P, receiver)",
            "utc": [
                datetime.fromtimestamp(x, timezone.utc).strftime("%m-%d %H:%M")
                for x in t
            ],
        }
    # The hold limit must keep a held sweep inside its own epoch.
    for dut, (t, _, epoch) in out.items():
        pre, post = t[epoch == "pre-gap"], t[epoch == "post-gap"]
        if pre.size and post.size:
            assert LIMITS["s11_hold_s"] < (post.min() - pre.max()) / 2, dut
    return out, log


def thermistor(index):
    sel = index.select(time=SPAN_UTC)
    m = sel.meta
    t = m.time_best.to_numpy(dtype=float)
    c = pd.to_numeric(m.tempctrl_load_T_now, errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(c)
    order = np.argsort(t[ok], kind="stable")
    tt, kk = t[ok][order], c[ok][order] + 273.15
    # Collapse duplicate timestamps (one row per file per integration).
    tt, first = np.unique(tt, return_index=True)
    return tt, kk[first].astype(np.float32), {
        "readings": int(tt.size),
        "rows_without_reading": int((~ok).sum()),
        "range_k": [float(kk.min()), float(kk.max())],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version", help="product version, e.g. v0000")
    ap.add_argument(
        "--products",
        help="campaign tree to read/write derived/tcal/ in (default: the "
        "campaign root, which also supplies the raw data)",
    )
    ap.add_argument(
        "--switch",
        default="data/s11/cal_materials/switch_sparams.npz",
        help="switch-path S-parameters, relative to the campaign root",
    )
    args = ap.parse_args(argv)

    root = eigsep_data.get_campaign_root(required=True)
    proot = Path(args.products) if args.products else root
    out_dir = proot / "derived" / "tcal" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    switch = root / args.switch
    osl = eigsep_cal.s11.PACKAGED_FILES[2026]["osldata"]
    windows = pd.DataFrame(
        [json.loads(x) for x in open(root / "curation" / "cal_windows.jsonl")]
    )

    index = MetadataIndex(root / "data")
    amb, freqs, amb_log = visits(index, "RFAMB", windows)
    non, freqs_n, non_log = visits(index, "RFNON", windows)
    assert np.array_equal(freqs, freqs_n)
    s11, s11_log = s11_on_channels(root / "data" / "s11", switch, osl, freqs)
    therm_t, therm_k, therm_log = thermistor(index)

    sols = dict(
        freqs=freqs,
        keys=np.array([KEY]),
        t_ns_k=T_NS_K,
        **LIMITS,
        amb_t=amb.t.to_numpy(float),
        amb_p=np.stack(amb.p.to_list()),
        amb_regime=amb.regime.to_numpy(str),
        non_t=non.t.to_numpy(float),
        non_p=np.stack(non.p.to_list()),
        non_regime=non.regime.to_numpy(str),
        therm_t=therm_t,
        therm_k=therm_k,
    )
    for dut, (t, g, epoch) in s11.items():
        sols[f"s11_{dut}_t"] = t
        sols[f"s11_{dut}"] = g.astype(np.complex64)
        sols[f"s11_{dut}_epoch"] = epoch
    np.savez_compressed(out_dir / "solutions.npz", **sols)

    here = Path(__file__).resolve()
    manifest = {
        "provenance": {
            "product": "tcal",
            "version": args.version,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/tcal/build_tcal.py",
            "argv": sys.argv[1:],
            "code": {
                "data-analysis": git_rev(here.parent),
                "eigsep_cal": git_rev(Path(eigsep_cal.__file__).parent),
                "eigsep_data": git_rev(Path(eigsep_data.__file__).parent),
            },
        },
        "applies_to": {"keys": [KEY], "antenna": "box-air", "plane": "P"},
        "model": {
            "equations": "eigsep_cal.dicke.tstar_coefficients, receiver_s11_coefficients",
            "t_ns_k": T_NS_K,
            "t_ns_source": "nameplate ENR 35 dB behind 30 dB pad; not measured",
            "visit_average": "integration-time-weighted mean, per-sample mask (wrap, dropped, > SAMPLE_NSIG robust sigma)",
            "sample_nsig": SAMPLE_NSIG,
            "visit_rejection": {
                "band_mhz": OUTLIER_BAND_MHZ,
                "frac": OUTLIER_FRAC,
                "max_bad_channels": MAX_BAD_CHANNELS,
            },
            "omits": [
                "receiver noise waves",
                "noise-source port mismatch (VNANON/RFNON paths uncharacterized)",
                "loss in balun, coax and RFANT switch path (temperatures stay at P)",
            ],
        },
        "limits": LIMITS,
        "inputs": {
            "switch_sparams": {"path": args.switch, "sha256": sha256(switch)},
            "internal_osl": {"path": "eigsep_cal package data/" + Path(osl).name, "sha256": sha256(osl)},
            "s11_manifest_sha256": sha256(root / "data" / "s11" / "manifest.csv"),
            "cal_windows_sha256": sha256(root / "curation" / "cal_windows.jsonl"),
            "span_utc": SPAN_UTC,
            "regimes": REGIMES,
            "s11_epoch_split_utc": EPOCH_SPLIT_UTC,
        },
        "contents": {
            "amb_visits": amb_log,
            "non_visits": non_log,
            "s11": s11_log,
            "thermistor": therm_log,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["contents"], indent=1)[:3000])
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
