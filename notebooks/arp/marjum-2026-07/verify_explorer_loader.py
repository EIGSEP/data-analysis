"""Bit-equality proof: explorer_loader (eigsep_data path) vs the legacy path.

Runs both loaders over the full 226-file beam-scan window and compares
every field the live pipeline consumes.  Exits non-zero on any mismatch.

    python3 verify_explorer_loader.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, HERE)

import fit_beam_v2 as v2                      # noqa: E402
from eigsep_data.beam_mapping.diagnostics import load_v007_data  # noqa: E402
import explorer_loader                        # noqa: E402

FIRST = "corr_20260717_185100Z.h5"
LAST = "corr_20260718_032126Z.h5"


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<24} {detail}", flush=True)
    return ok


def main():
    print("legacy path ...", flush=True)
    old = load_v007_data(v2.DATA_DIR, start=v2.FILES_SLICE[0],
                         stop=v2.FILES_SLICE[1])
    old = v2.attach_pointing_v1(old, old["files"])

    print("eigsep_data path ...", flush=True)
    new = explorer_loader.load_window(v2.DATA_DIR, FIRST, LAST, key="4")

    print(f"\nwindow: {len(old['files'])} vs {len(new['files'])} files, "
          f"n = {old['times'].size} vs {new['times'].size}\n")

    ok = True
    ok &= report("file list", [Path(f).name for f in old["files"]]
                 == [Path(f).name for f in new["files"]])
    ok &= report("comb_off_files",
                 sorted(old["comb_off_files"]) == sorted(new["comb_off_files"]),
                 f"{len(old['comb_off_files'])} files")

    for key in ("times", "freqs", "measured_tx", "measured_sigma",
                "az_deg", "el_deg", "flags_v1", "pointing_v1_finite"):
        a = np.asarray(old[key], dtype=float)
        b = np.asarray(new[key], dtype=float)
        if a.shape != b.shape:
            ok &= report(key, False, f"shape {a.shape} vs {b.shape}")
            continue
        na, nb = np.isnan(a), np.isnan(b)
        nan_diff = int((na != nb).sum())
        both = ~na & ~nb
        worst = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        same = nan_diff == 0 and worst == 0.0
        ok &= report(key, same,
                     f"max|d| = {worst:.6g}, NaN-pattern diffs = {nan_diff}")

    ok &= report("quality_v1",
                 np.array_equal(np.asarray(old["quality_v1"], dtype=object),
                                np.asarray(new["quality_v1"], dtype=object)))

    print("\n" + ("ALL FIELDS BIT-IDENTICAL" if ok else "MISMATCH — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
