"""Per-file digital-self-comb presence across the 07-17/18 beam-scan window.

Why this product exists
-----------------------
INDEX.md records the digital self-comb (1.953125 MHz = 8.000 ch, channel-locked,
residue 0 — generated inside our own signal chain) as running continuously from
07-17 ~15:37 to 07-18 ~03:00. That continuity claim is **wrong**: the comb is
absent from 65 of the 227 files in the 07-17 18:51 -> 07-18 03:22 beam-scan
window, including one contiguous ~2 h block.

The underlying per-file measurement is `beam-analyst`'s
(`eigsep_data/notebooks/arp/marjum-2026-07/check_comb_presence_scan.py`), which
used a deliberately spacing-agnostic detector: count channels in 480-800 whose
adjacent-channel second difference exceeds 20 MADs. That detector cannot name
which comb it sees. `curation/verify_beamscan_comb_identity.py` settled the
identity with the archive's own periodogram + channel-lock test, sampling every
contiguous ON/OFF run: **every ON file is the digital self-comb** (period
7.996-8.007 ch, lock_frac 0.90-1.00, residue 0) and the walking TX comb at
4.096 ch scores 0.07-0.15 everywhere, i.e. noise. So this is a *self-comb*
presence map. It says nothing about the transmitter, and `mode_table.jsonl`'s
`tx_comb = off` over this window is correct and independently confirmed.

This builder therefore ingests the measurement under its true name and, for
every OFF file, attributes the cause from state the archive already holds, so
no consumer reads "comb off" as "transmitter toggled".

Cause attribution (in priority order):
  ``daemon_outage``    ``has_meta_motor`` false — the observing daemon was down
                       or in a degraded config; no metadata streams, so no
                       pointing solution exists for these files either.
  ``rfswitch_off_ant`` ``rfswitch_dominant`` not RFANT — antenna disconnected
                       (noise-source / VNA calibration cadence), so nothing
                       radiated can be received.
  ``daemon_transition`` ``run_tag`` is UNKNOWN — the daemon did not label the
                       run. Both such files sit exactly on catalogued
                       boundaries in ``boundaries.jsonl`` (20:21:57
                       software-config, 23:42:26 daemon), i.e. they are the
                       edge files of a reconfiguration.
  ``after_self_comb_era`` close time >= 07-18 03:00, past the self-comb's
                       documented end.
  ``unattributed``     none of the above. One file, 07-18 02:53:47, is a
                       genuine single-file dropout with the daemon healthy,
                       the antenna connected and the era still current. It is
                       not explained by anything the archive holds.

Units: ``comb_snr`` is beam-analyst's detector score (dimensionless, count of
qualifying channels scaled by MAD); it is a presence statistic, not a
calibrated amplitude. Times are file CLOSE times (see "Which clock to trust").

Output: ``curation/self_comb_presence.jsonl`` via --json.
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SELF_COMB_ERA_END = "20260718_030000"


def short_sha(repo):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def attribute(row):
    if not row.get("self_comb_on"):
        if not bool(row.get("has_meta_motor")):
            return "daemon_outage"
        if row.get("rfswitch_dominant") != "RFANT":
            return "rfswitch_off_ant"
        if row.get("run_tag") == "UNKNOWN":
            return "daemon_transition"
        if str(row.get("t_close_utc")) >= SELF_COMB_ERA_END:
            return "after_self_comb_era"
        return "unattributed"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presence-json", required=True)
    ap.add_argument("--file-state", default=str(HERE / "file_state.csv"))
    ap.add_argument("--json", default=str(HERE / "self_comb_presence.jsonl"))
    args = ap.parse_args()

    src = json.load(open(args.presence_json))
    p = pd.DataFrame(src["files"])
    fs = pd.read_csv(args.file_state)
    keep = ["file", "rfswitch_dominant", "run_tag", "has_meta_motor"]
    m = p.merge(fs[keep], on="file", how="left")
    m = m.rename(columns={"comb_on": "self_comb_on"})

    rows = []
    for _, r in m.iterrows():
        d = r.to_dict()
        rows.append({
            "file": d["file"],
            "t_close_utc": d["t_close_utc"],
            "self_comb_on": bool(d["self_comb_on"]),
            "comb_snr": round(float(d["comb_snr"]), 4),
            "n_spectra": int(d["n_spectra"]),
            "off_cause": attribute(d),
            "rfswitch_dominant": (None if pd.isna(d["rfswitch_dominant"])
                                  else d["rfswitch_dominant"]),
            "run_tag": (None if pd.isna(d["run_tag"]) else d["run_tag"]),
            "has_meta_motor": bool(d["has_meta_motor"]),
        })

    prov = {
        "product": "self_comb_presence",
        "campaign": "marjum-2026-07",
        "version": "v1",
        "generator": "curation/build_self_comb_presence.py",
        "generator_commit": short_sha(HERE.parent.parent),
        "measurement_source": {
            "agent": "beam-analyst",
            "script": "eigsep_data/notebooks/arp/marjum-2026-07/"
                      "check_comb_presence_scan.py",
            "artifact": Path(args.presence_json).name,
            "method": "channels in 480-800 whose second difference exceeds "
                      "20 MADs; spacing-agnostic",
            "snr_threshold": src.get("snr_threshold"),
        },
        "identity_verification": {
            "script": "curation/verify_beamscan_comb_identity.py",
            "artifact": "curation/beamscan_comb_identity.jsonl",
            "verdict": "every ON file is the digital self-comb (8.000 ch, "
                       "locked, residue 0); TX comb at 4.096 ch scores "
                       "0.07-0.15 (noise) in every file sampled",
        },
        "granularity": "per file, input 4 (box-air)",
        "window": ["corr_20260717_185100Z.h5", "corr_20260718_032248Z.h5"],
        "warning": "This is SELF-comb presence. It is not a transmitter flag. "
                   "mode_table.jsonl tx_comb='off' over this whole window is "
                   "correct; do not override it with this product.",
    }
    with open(args.json, "w") as f:
        f.write(json.dumps({"provenance": prov}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

    on = sum(r["self_comb_on"] for r in rows)
    print(f"{len(rows)} files: {on} self-comb on, {len(rows) - on} off")
    causes = pd.Series([r["off_cause"] for r in rows if r["off_cause"]])
    print(causes.value_counts().to_string())


if __name__ == "__main__":
    main()
