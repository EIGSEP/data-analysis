"""Per-file matched-detector state of box-air's 1.000 MHz self-EMI comb,
beam-scan window + 07-16 control.

What this measures (memo 001, section "Combs")
----------------------------------------------
Before memo 001 this product was `curation/tx_state_matched.jsonl` with field
`tx_state`, from `build_tx_state_matched.py`, and was read as "transmitter
state". What it detects is the **1.000 MHz comb** (spacing searched
0.990-1.010 MHz), which memo 001 identifies as box-air self-EMI (07-16
01:16:57-16:49:03, box-air only, gone when the Panda was power-cycled), not
the transmitter. Its second detector, fixed at 8.000 ch = 1.953125 MHz, scores
the actual **transmitter** comb. The values are unchanged; only the names are:

    tx_state          -> boxair_emi_state
    tx_S, tx_spacing_mhz, tx_offset_ch -> emi_S, emi_spacing_mhz, emi_offset_ch
    self_S, self_offset_ch -> transmitter_S, transmitter_offset_ch
    tx_presence_tx_on -> boxair_emi_presence (cross-check value)

It is kept, not retired: `curation/transmitter_transitions_boxgnd.jsonl`
dates the transmitter, a different comb, so it supersedes nothing here.
`curation/boxair_emi_presence.jsonl` is the campaign-wide axis for the same
1 MHz comb; this product covers 282 files but publishes `n_eff` and both
coherences, so a consumer can see how well determined each verdict is.

`beam-analyst`'s `tx_state_detector.py`
(`data-analysis/notebooks/arp/marjum-2026-07/`) runs the two matched
detectors on each file's tone-excess spectrum (input 4, channels 240-960)
using a continuous periodogram in frequency space:

  1 MHz EMI comb      spacing searched 0.990-1.010 MHz, free phase
  transmitter comb    spacing fixed at 250/128 MHz = 1.953125 MHz = 8.000 ch,
                      phase reported as offset from DC

Agreement with `boxair_emi_presence.jsonl`, asserted below: 282 of 282 files.

The N_eff gate
--------------
`n_eff = (sum e)^2 / sum(e^2)` over the tone-excess spectrum is the effective
number of channels carrying the excess. Every verdict is gated on it:

  n_eff >= 10 (165 files)   emi_S median 0.289, max 0.470   -> EMI absent
  n_eff <  10  (62 files)   emi_S median 0.925, max 1.000   -> undeterminable

When the excess sits in two or three channels a periodogram is near-perfectly
coherent at almost any trial spacing, so the 62 low-N_eff files are reported
`uncertain`, never `off`. All 62 are a strict subset of the 65
transmitter-off files in `curation/transmitter_presence.jsonl`: with the
transmitter comb off there is no tone excess to measure.

Reading the result
------------------
Beam scan, 07-17 18:51 -> 07-18 03:22, 227 files:
    on 0 | off 165 | uncertain 62
Control, 07-16 10:00-12:00, inside the 1 MHz EMI window, 55 files:
    on 55 | off 0 | uncertain 0
      D = 0.99988 MHz, emi_S median 0.916, offset 4.03 ch from DC, n_eff ~ 50

Scope is input 4 (box-air), channels 240-960, these two windows only.
`boxair_emi_state = off` in the beam scan means "no 1 MHz EMI comb"; the
transmitter comb is usually present and strong in exactly these files
(`transmitter_S`).

Usage
-----
    python build_boxair_emi_matched.py \\
        --producer-dir data-analysis/notebooks/arp/marjum-2026-07 \\
        --presence <campaign>/curation/boxair_emi_presence.jsonl \\
        --out <campaign>/curation/boxair_emi_matched.jsonl
"""

import argparse
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent

SOURCES = [
    ("beamscan", "tx_state_beam_scan.json"),
    ("control_0716", "tx_state_0716_control.json"),
]


def close_time(fname):
    """corr_YYYYMMDD_HHMMSSZ.h5 -> ISO close time. Filenames are CLOSE times."""
    stem = fname.replace("corr_", "").replace("Z.h5", "")
    d, t = stem.split("_")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"


def short_sha(repo):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                            "--", "."], capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--producer-dir", required=True, type=pathlib.Path,
                    help="directory holding beam-analyst's tx_state_*.json")
    ap.add_argument("--presence", required=True, type=pathlib.Path,
                    help="curation/boxair_emi_presence.jsonl (cross-check)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    archive = {}
    for line in args.presence.read_text().splitlines():
        r = json.loads(line) if line.strip() else None
        if r and "file" in r:
            archive[r["file"]] = r

    rows, agree, disagree = [], 0, 0
    for window, fname in SOURCES:
        blob = json.loads((args.producer_dir / fname).read_text())
        for r in blob["files"]:
            state = r["tx_state"]
            gated = r["n_eff"] >= blob["n_eff_min"]

            ref = archive.get(r["file"])
            ref_on = None if ref is None else ref["boxair_emi"]
            if ref_on is not None:
                if (state == "on") == ref_on:
                    agree += 1
                else:
                    disagree += 1

            rows.append(
                {
                    "file": r["file"],
                    "t_close_utc": close_time(r["file"]),
                    "window": window,
                    "input": blob["input"],
                    "boxair_emi_state": state,
                    "n_eff": round(r["n_eff"], 3),
                    "n_eff_gated": gated,
                    "emi_S": round(r["tx_S"], 4),
                    "emi_spacing_mhz": round(r["tx_spacing_mhz"], 6),
                    "emi_offset_ch": round(r["tx_offset_ch"], 3),
                    "transmitter_S": round(r["self_S"], 4),
                    "transmitter_offset_ch": round(r["self_offset_ch"], 3),
                    "boxair_emi_presence": ref_on,
                }
            )

    # The value of this product is the independent corroboration. If a future
    # rebuild breaks it, the headline claim needs re-examining, not patching.
    assert disagree == 0, f"{disagree} files disagree with {args.presence.name}"
    assert agree == 282, f"expected 282 cross-checked files, got {agree}"

    rows.sort(key=lambda r: (r["window"], r["t_close_utc"]))
    prov = {
        "product": "boxair_emi_matched",
        "campaign": "marjum-2026-07",
        "generator": "data-analysis/scripts/marjum-2026-07/curation/"
                     "build_boxair_emi_matched.py",
        "generator_commit": short_sha(HERE),
        "measurement_source": {
            "agent": "beam-analyst",
            "script": "data-analysis/notebooks/arp/marjum-2026-07/"
                      "tx_state_detector.py",
            "artifacts": [f for _, f in SOURCES],
        },
        "renamed_from": {
            "file": "curation/tx_state_matched.jsonl",
            "generator": "build_tx_state_matched.py",
            "fields": {"tx_state": "boxair_emi_state", "tx_S": "emi_S",
                       "tx_spacing_mhz": "emi_spacing_mhz",
                       "tx_offset_ch": "emi_offset_ch",
                       "self_S": "transmitter_S",
                       "self_offset_ch": "transmitter_offset_ch",
                       "tx_presence_tx_on": "boxair_emi_presence"},
            "reason": "the 1.000 MHz comb is box-air self-EMI and the "
                      "8-channel comb is the transmitter (memo 001)",
        },
        "cross_check": {"against": args.presence.name, "agree": agree,
                        "disagree": disagree},
    }
    with args.out.open("w") as fh:
        fh.write(json.dumps({"provenance": prov}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    for window, _ in SOURCES:
        sub = [r for r in rows if r["window"] == window]
        counts = {s: sum(1 for r in sub if r["boxair_emi_state"] == s) for s in
                  ("on", "off", "uncertain")}
        print(f"{window:13s} n={len(sub):3d}  {counts}")
    print(f"cross-check vs {args.presence.name}: {agree} agree, {disagree} disagree")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
