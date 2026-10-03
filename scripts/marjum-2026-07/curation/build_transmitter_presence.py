"""Per-file transmitter-comb presence across the 07-17/18 beam-scan window.

What this measures (memo 001, section "Combs")
----------------------------------------------
The beam-mapping transmitter radiates a comb on correlator channels
== 0 mod 8 (1.953125 MHz = 8.000 ch), seen by both antennas from 07-17
15:36:22 UTC to the end of the data. Before memo 001 this comb was mistaken
for a "digital self-comb" generated in our own signal chain, and this product
was `curation/self_comb_presence.jsonl` with field `self_comb_on`. The values
are unchanged; only the names are.

The comb is absent on box-air from 65 of the 227 files in the 07-17 18:51 ->
07-18 03:22 beam-scan window, including one contiguous ~2 h block (box-air's
outage, 21:44-23:44). Box-gnd's per-integration episodes in
`curation/transmitter_transitions_boxgnd.jsonl` are authoritative for the
transmitter's on/off timing; this product is the per-file box-air view.

The underlying per-file measurement is `beam-analyst`'s
(`data-analysis/notebooks/arp/marjum-2026-07/check_comb_presence_scan.py`),
which used a deliberately spacing-agnostic detector: count channels in 480-800
whose adjacent-channel second difference exceeds 20 MADs. That detector cannot
name which comb it sees. `curation/verify_beamscan_comb_identity.py` settled
the identity with a periodogram + channel-lock test, sampling every contiguous
ON/OFF run: every ON file is the 8-channel transmitter comb (period
7.996-8.007 ch, lock_frac 0.90-1.00, residue 0), and the 1.000 MHz box-air
self-EMI comb at 4.096 ch scores 0.07-0.15 everywhere, i.e. noise.

For every OFF file the cause is attributed from state the archive already
holds (in priority order):
  ``daemon_outage``    ``has_meta_motor`` false: the observing daemon was down
                       or in a degraded config; no metadata streams, so no
                       pointing solution exists for these files either.
  ``rfswitch_off_ant`` ``rfswitch_dominant`` not RFANT: antenna disconnected
                       (noise-source / VNA calibration cadence), so nothing
                       radiated can be received on box-air.
  ``daemon_transition`` ``run_tag`` is UNKNOWN: the daemon did not label the
                       run. Both such files sit exactly on catalogued
                       boundaries in ``boundaries.jsonl`` (20:21:57
                       software-config, 23:42:26 daemon).
  ``after_transmitter_era`` close time >= 07-18 03:00, past the last
                       transmitter tooth box-gnd records (03:00:08).
  ``unattributed``     none of the above. One file, 07-18 02:53:47, with the
                       daemon healthy and the antenna connected; box-gnd's
                       trace shows the transmitter toggling inside it
                       (transitions episodes 18-21).

Units: ``comb_snr`` is beam-analyst's detector score (dimensionless, count of
qualifying channels scaled by MAD); it is a presence statistic, not a
calibrated amplitude. Times are file CLOSE times.

Output: ``curation/transmitter_presence.jsonl`` via --json: one provenance
line, then one row per file
    {file, t_close_utc, transmitter, comb_snr, n_spectra, off_cause,
     rfswitch_dominant, run_tag, has_meta_motor}
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TRANSMITTER_ERA_END = "20260718_030000"


def short_sha(repo):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                            "--", "."], capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def attribute(row):
    if not row.get("transmitter"):
        if not bool(row.get("has_meta_motor")):
            return "daemon_outage"
        if row.get("rfswitch_dominant") != "RFANT":
            return "rfswitch_off_ant"
        if row.get("run_tag") == "UNKNOWN":
            return "daemon_transition"
        if str(row.get("t_close_utc")) >= TRANSMITTER_ERA_END:
            return "after_transmitter_era"
        return "unattributed"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presence-json", required=True,
                    help="beam-analyst's comb_presence_beam_scan.json")
    ap.add_argument("--file-state", required=True,
                    help="curation/file_state.csv")
    ap.add_argument("--json", required=True,
                    help="output, normally curation/transmitter_presence.jsonl")
    args = ap.parse_args()

    src = json.load(open(args.presence_json))
    p = pd.DataFrame(src["files"])
    fs = pd.read_csv(args.file_state)
    keep = ["file", "rfswitch_dominant", "run_tag", "has_meta_motor"]
    m = p.merge(fs[keep], on="file", how="left")
    m = m.rename(columns={"comb_on": "transmitter"})

    rows = []
    for _, r in m.iterrows():
        d = r.to_dict()
        rows.append({
            "file": d["file"],
            "t_close_utc": d["t_close_utc"],
            "transmitter": bool(d["transmitter"]),
            "comb_snr": round(float(d["comb_snr"]), 4),
            "n_spectra": int(d["n_spectra"]),
            "off_cause": attribute(d),
            "rfswitch_dominant": (None if pd.isna(d["rfswitch_dominant"])
                                  else d["rfswitch_dominant"]),
            "run_tag": (None if pd.isna(d["run_tag"]) else d["run_tag"]),
            "has_meta_motor": bool(d["has_meta_motor"]),
        })

    prov = {
        "product": "transmitter_presence",
        "campaign": "marjum-2026-07",
        "version": "v2",
        "generator": "data-analysis/scripts/marjum-2026-07/curation/"
                     "build_transmitter_presence.py",
        "generator_commit": short_sha(HERE),
        "renamed_from": {
            "file": "curation/self_comb_presence.jsonl (v1)",
            "fields": {"self_comb_on": "transmitter"},
            "off_cause_values": {"after_self_comb_era": "after_transmitter_era"},
            "reason": "the 8-channel comb is the beam-mapping transmitter, "
                      "not a digital self-comb (memo 001)",
        },
        "measurement_source": {
            "agent": "beam-analyst",
            "script": "data-analysis/notebooks/arp/marjum-2026-07/"
                      "check_comb_presence_scan.py",
            "artifact": Path(args.presence_json).name,
            "method": "channels in 480-800 whose second difference exceeds "
                      "20 MADs; spacing-agnostic",
            "snr_threshold": src.get("snr_threshold"),
        },
        "identity_verification": {
            "script": "curation/verify_beamscan_comb_identity.py",
            "artifact": "curation/beamscan_comb_identity.jsonl",
            "verdict": "every ON file is the 8-channel transmitter comb "
                       "(8.000 ch, locked, residue 0); the 1.000 MHz box-air "
                       "self-EMI comb at 4.096 ch scores 0.07-0.15 (noise) in "
                       "every file sampled",
        },
        "granularity": "per file, input 4 (box-air)",
        "window": ["corr_20260717_185100Z.h5", "corr_20260718_032248Z.h5"],
        "note": "Box-air view. For transmitter on/off timing use box-gnd's "
                "curation/transmitter_transitions_boxgnd.jsonl; box-air also "
                "reads 'off' whenever its antenna is switched out.",
    }
    with open(args.json, "w") as f:
        f.write(json.dumps({"provenance": prov}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

    on = sum(r["transmitter"] for r in rows)
    print(f"{len(rows)} files: {on} transmitter on, {len(rows) - on} off")
    causes = pd.Series([r["off_cause"] for r in rows if r["off_cause"]])
    print(causes.value_counts().to_string())


if __name__ == "__main__":
    main()
