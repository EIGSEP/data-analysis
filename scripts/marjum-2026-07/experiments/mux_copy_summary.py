#!/usr/bin/env python
"""mux_copy_summary.py — reduce mux_copy_scan.jsonl to the memo's numbers.

Separates the three reasons a mux-equality flag can read False:
  - mux genuinely off  (both inputs live, data differ)
  - source input dead  (all-zero data; the filter's `.any()` guard trips)
  - copy present       (bitwise identical)

and derives contiguous mux-state windows, which the observing-mode table does
not carry (it has no mux axis). Emits `mux_windows.jsonl` for data-archivist.
"""
import datetime as dt
import json
import os

HERE = os.path.dirname(__file__)


def load(path):
    recs, prov = [], None
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if "provenance" in d:
                prov = d["provenance"]
            else:
                recs.append(d)
    return prov, recs


def classify(r, src, attr):
    """mux state for one copy (src -> src+1), with dead-input disambiguation.

    The filter's equality attr is `array_equal(dX, dY) and dX.any()`, so an
    all-zero source reads False whatever the mux register was doing. Where the
    source auto survived the filter we can tell the two apart; where it did not
    (phase A/B input 0, phase C input 3) the state is simply unknown.
    """
    if r.get("error"):
        return "error"
    if r[attr]:
        return "copy-on"
    live = r.get(f"live_{src}")
    if live is False:
        return "src-dead"
    if live is None:
        return "unknown"
    return "copy-off"


def main():
    prov, recs = load(os.path.join(HERE, "mux_copy_scan.jsonl"))
    recs.sort(key=lambda r: r.get("t_utc", ""))

    print(f"records: {len(recs)}   (from {prov['n_files']} files)")
    errs = [r for r in recs if r.get("error")]
    if errs:
        print(f"  read errors: {len(errs)}  e.g. {errs[0]}")

    # ---- headline: bitwise identity where both copies survive (phase B) ----
    deep = [r for r in recs if "n_mismatch" in r]
    on = [r for r in deep if r["attr_eq45"]]
    off = [r for r in deep if not r["attr_eq45"]]
    print(f"\nphase B files with both 4 and 5 retained: {len(deep)}")
    print(f"  mux ON  (attr eq45 True):  {len(on)}")
    print(f"  mux OFF (attr eq45 False): {len(off)}")

    if on:
        tot = sum(r["n_samp"] for r in on)
        mis = sum(r["n_mismatch"] for r in on)
        mx = max(r["max_absdiff_lsb"] for r in on)
        meds = sorted(r["med_power_4"] for r in on if r["med_power_4"] > 0)
        med = meds[len(meds) // 2] if meds else float("nan")
        print(f"  ON: {tot:,} accumulator samples compared")
        print(f"      mismatching samples: {mis}")
        print(f"      max |diff|: {mx} LSB")
        print(f"      median auto power: {med:,.0f} LSB")
        if mis == 0 and med > 0:
            print(f"      => differential bound < 1 LSB / {med:,.0f} "
                  f"= {1.0/med:.2e} (exact agreement)")
    if off:
        fr = [r["n_mismatch"] / r["n_samp"] for r in off]
        mx = max(r["max_absdiff_lsb"] for r in off)
        print(f"  OFF (control): mismatch fraction "
              f"{min(fr):.4f}-{max(fr):.4f}, max |diff| {mx:,} LSB")

    # ---- Cauchy-Schwarz on the mux-derived 35 cross ----
    coh = [r for r in deep if "coh_max" in r and r["attr_eq45"]]
    if coh:
        viol = sum(r["n_coh_gt1"] for r in coh)
        nev = sum(r["n_coh_eval"] for r in coh)
        print(f"\nmux-derived cross 35, Cauchy-Schwarz (mux ON, power-gated):")
        print(f"  {nev:,} channel-samples evaluated, {viol} with |V|>sqrt(P3 P5)")
        print(f"  max coherence: {max(r['coh_max'] for r in coh):.6f}")

    # ---- mux state windows over the whole campaign ----
    windows = []
    for r in recs:
        s45 = classify(r, "4", "attr_eq45")
        s01 = classify(r, "0", "attr_eq01")
        key = (r.get("phase"), s45, s01)
        if windows and windows[-1]["_key"] == key:
            w = windows[-1]
            w["t_end_utc"] = r["t_utc"]
            w["file_last"] = r["file"]
            w["n_files"] += 1
        else:
            windows.append({
                "_key": key, "phase": r.get("phase"),
                "mux_4to5": s45, "mux_0to1": s01,
                "t_start_utc": r["t_utc"], "t_end_utc": r["t_utc"],
                "file_first": r["file"], "file_last": r["file"], "n_files": 1,
            })
    for w in windows:
        del w["_key"]

    print(f"\ncontiguous mux-state windows: {len(windows)}")
    from collections import Counter
    cnt = Counter((w["phase"], w["mux_4to5"], w["mux_0to1"]) for w in windows)
    print("  phase  4->5       0->1       windows  files")
    for k in sorted(cnt):
        nf = sum(w["n_files"] for w in windows
                 if (w["phase"], w["mux_4to5"], w["mux_0to1"]) == k)
        print(f"  {k[0]:<6} {k[1]:<10} {k[2]:<10} {cnt[k]:5d} {nf:6d}")

    print("\nwindows where a copy that should be on is not (>=2 files):")
    for w in windows:
        interesting = (w["phase"] == "B" and w["mux_4to5"] != "copy-on") or (
            w["phase"] == "C"
            and (w["mux_4to5"] != "copy-on" or w["mux_0to1"] != "copy-on")
        )
        if interesting and w["n_files"] >= 2:
            print(f"  {w['t_start_utc']} -> {w['t_end_utc']}  "
                  f"phase {w['phase']}  4->5={w['mux_4to5']:<9} "
                  f"0->1={w['mux_0to1']:<9} {w['n_files']:4d} files")

    out = os.path.join(HERE, "mux_windows.jsonl")
    with open(out, "w") as fh:
        fh.write(json.dumps({"provenance": {
            "product": "mux_windows",
            "campaign": "marjum-2026-07",
            "version": "v1",
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "generator": "experiments/mux_copy_summary.py",
            "inputs": ["experiments/mux_copy_scan.jsonl"],
            "notes": ("mux_4to5 state per contiguous window; 'src-dead' means "
                      "input 4 all-zero so the equality flag is uninformative. "
                      "The adc_mux_sel register is not in any file header; "
                      "state is recoverable only by byte-comparison."),
        }}) + "\n")
        for w in windows:
            fh.write(json.dumps(w) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
