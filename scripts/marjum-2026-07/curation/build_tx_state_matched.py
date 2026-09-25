"""Per-file matched-detector transmitter state, beam-scan window + 07-16 control.

Why this product exists
-----------------------
The archive already carries a campaign-wide TX-presence axis
(`curation/tx_presence.jsonl`, from `scan_tx_presence.py`, 5,120 files, a
tone-counting detector). This product ingests a *second, independent*
measurement of the same quantity over the two windows where the answer was
disputed, and it carries the diagnostics the first one does not.

`beam-analyst`'s `tx_state_detector.py`
(`eigsep_data/notebooks/arp/marjum-2026-07/`) runs two matched detectors on
each file's tone-excess spectrum (input 4, channels 240-960) using a
*continuous* periodogram in frequency space. It never references the channel
grid, so - unlike the modal-integer-gap estimator that produced the retracted
"channel-locked per era" table - it cannot manufacture channel-locked answers:

  TX comb    spacing searched 0.990-1.010 MHz, free phase
  self comb  spacing fixed at 250/128 MHz = 1.953125 MHz = 8.000 ch,
             phase reported as offset from DC

Why both products are kept
--------------------------
They answer at different scope and cost. `tx_presence` covers the whole
campaign and is the axis to join against for ordinary file selection.
`tx_state_matched` covers 282 files but publishes `n_eff` and both coherences,
so a consumer can see *how well determined* each verdict is. Neither
supersedes the other.

Agreement, checked at ingest and asserted below: **282 of 282 files, zero
disagreements.** Two detectors with different failure modes, built by
different agents, returning the same verdict on every file, is the reason the
no-TX-in-the-beam-scan result is treated as settled rather than as one team's
claim.

The N_eff gate is the point
---------------------------
`n_eff = (sum e)^2 / sum(e^2)` over the tone-excess spectrum is the effective
number of channels carrying the excess. Every verdict is gated on it, and the
gate is not a formality - it is what separates this measurement from the
several wrong comb numbers produced during the same week:

  n_eff >= 10 (165 files)   tx_S median 0.289, max 0.470   -> TX absent
  n_eff <  10  (62 files)   tx_S median 0.925, max 1.000   -> undeterminable

When the excess sits in two or three channels, a periodogram is near-perfectly
coherent at *almost any* trial spacing. The 62 low-N_eff files return
`tx_S = 1.000` at `n_eff ~ 1.9`. Those are not transmitter detections; they are
the detector correctly having nothing to work with. They are reported as
`uncertain`, never as `off`.

That population is not arbitrary: **all 62 are a strict subset of the 65
self-comb-off files** in `curation/self_comb_presence.jsonl`. With the
self-comb off there is no tone excess to measure, so no statement about the
transmitter is possible from those files either way. The three comb-off files
that do stay above the gate are `corr_20260717_191105Z.h5`,
`corr_20260718_021925Z.h5`, `corr_20260718_030013Z.h5`.

Reading the result
------------------
Beam scan, 07-17 18:51 -> 07-18 03:22, 227 files:
    on 0 | off 165 | uncertain 62
Control, 07-16 10:00-12:00, TX independently known on, 55 files:
    on 55 | off 0 | uncertain 0
      D = 0.99988 MHz, tx_S median 0.916, offset 4.03 ch from DC, n_eff ~ 50

A broad 0.4-5 MHz search on the high-N_eff scan files (reported by the
producer, not reproduced here) found no coherent comb outside the 8-channel
self-comb family, so the transmitter was not running at some other spacing.

**Quote the gated numbers with their scope.** "tx_S max 0.470" is true over
`n_eff >= 10`; over all 227 files the max is 1.000, for the reason above.
Recomputing the headline stat without the gate and finding 1.000 is the
expected result of dropping the gate, not a contradiction.

Consumer warning
----------------
Scope is input 4 (box-air), channels 240-960, these two windows only. Input 0
across the scan, other bands, and other campaign eras are **not** covered
here; use `tx_presence.jsonl` for campaign-wide questions. `tx_state = off`
means "no transmitter comb", not "no signal" - the self-comb is usually
present and strong in exactly these files, which is precisely what the v007
beam fit mistook for the transmitter.
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PRODUCER = (
    HERE.parent.parent / "eigsep_data" / "notebooks" / "arp" / "marjum-2026-07"
)
OUT = HERE / "tx_state_matched.jsonl"

SOURCES = [
    ("beamscan", "tx_state_beam_scan.json"),
    ("control_0716", "tx_state_0716_control.json"),
]


def close_time(fname):
    """corr_YYYYMMDD_HHMMSSZ.h5 -> ISO close time. Filenames are CLOSE times."""
    stem = fname.replace("corr_", "").replace("Z.h5", "")
    d, t = stem.split("_")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"


def main():
    archive = {
        r["file"]: r
        for r in (
            json.loads(line)
            for line in (HERE / "tx_presence.jsonl").read_text().splitlines()
            if line.strip()
        )
    }

    rows, agree, disagree = [], 0, 0
    for window, fname in SOURCES:
        blob = json.loads((PRODUCER / fname).read_text())
        for r in blob["files"]:
            state = r["tx_state"]
            gated = r["n_eff"] >= blob["n_eff_min"]

            # Cross-check against the archive's own independent scanner.
            ref = archive.get(r["file"])
            ref_on = None if ref is None else ref["tx_on"]
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
                    "tx_state": state,
                    "n_eff": round(r["n_eff"], 3),
                    "n_eff_gated": gated,
                    "tx_S": round(r["tx_S"], 4),
                    "tx_spacing_mhz": round(r["tx_spacing_mhz"], 6),
                    "tx_offset_ch": round(r["tx_offset_ch"], 3),
                    "self_S": round(r["self_S"], 4),
                    "self_offset_ch": round(r["self_offset_ch"], 3),
                    "tx_presence_tx_on": ref_on,
                }
            )

    # The value of this product is the independent corroboration. If a future
    # rebuild breaks it, the headline claim needs re-examining, not patching.
    assert disagree == 0, f"{disagree} files disagree with tx_presence.jsonl"
    assert agree == 282, f"expected 282 cross-checked files, got {agree}"

    rows.sort(key=lambda r: (r["window"], r["t_close_utc"]))
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    for window, _ in SOURCES:
        sub = [r for r in rows if r["window"] == window]
        counts = {s: sum(1 for r in sub if r["tx_state"] == s) for s in
                  ("on", "off", "uncertain")}
        print(f"{window:13s} n={len(sub):3d}  {counts}")
    print(f"cross-check vs tx_presence.jsonl: {agree} agree, {disagree} disagree")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
