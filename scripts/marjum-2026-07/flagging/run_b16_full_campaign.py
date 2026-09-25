"""Full-campaign driver for B16 rev4 (linear amplitude), overnight run.

Not part of the reviewed/merged b16_dpss_model.py methodology -- this is
just a batch harness that calls fit_file()/write_companion() (already
approved) across all of data/*.h5 instead of the pilot window, with
per-file error isolation, resumability (skips files that already have a
companion output), and progress logging suitable for an unattended run
with nobody watching live.

Usage:
    MARJUM_DATA_ROOT=/mnt/data02/eigsep/marjum-2026-07 \
    OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    python3 run_b16_full_campaign.py
"""
import glob
import json
import os
import sys
import time
import traceback

from eigsep_data.flagging import detectors as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b15_event_survey as EV  # noqa: E402
import b16_dpss_model as M16  # noqa: E402

RUN_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "flags", "b16", "_run_logs")
os.makedirs(RUN_LOG_DIR, exist_ok=True)
PROGRESS_LOG = os.path.join(RUN_LOG_DIR, "full_campaign_progress.log")
SUMMARY_JSON = os.path.join(RUN_LOG_DIR, "full_campaign_summary.json")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def main():
    t_start = time.time()
    log("=== B16 full-campaign run starting ===")
    log(f"DATA_ROOT={M16.DATA_ROOT}")

    self_freqs, self_mask = EV.self_comb_channel_mask()
    band = (self_freqs >= D.BAND_ANALYSIS[0]) & (self_freqs <= D.BAND_ANALYSIS[1])
    log(f"analysis-band channels: {int(band.sum())}")

    pattern = os.environ.get("B16_GLOB", "*.h5")
    files = sorted(glob.glob(os.path.join(M16.DATA_ROOT, "data", pattern)))
    limit = os.environ.get("B16_LIMIT")
    if limit:
        files = files[: int(limit)]
        log(f"B16_LIMIT set -- restricting to first {limit} files (smoke test)")
    log(f"total files in data/: {len(files)}")

    already_done = set(os.listdir(M16.DERIVED_DIR)) if os.path.isdir(M16.DERIVED_DIR) else set()
    log(f"already-written companion files found (resuming, skipping these): {len(already_done)}")

    n_written = 0
    n_skipped = 0
    n_errors = 0
    n_refined_total = {k: 0 for k in M16.INPUTS}
    n_pixels_total = 0
    errors = []

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        if fname in already_done:
            n_skipped += 1
            continue
        try:
            results_by_input = {}
            for inp in M16.INPUTS:
                r = M16.fit_file(path, fname, inp, self_mask, band, refine=True)
                if r is None:
                    continue
                results_by_input[inp] = r
                n_refined_total[inp] += r["n_refined"]
                n_pixels_total += r["flagged"].size
            if results_by_input:
                M16.write_companion(path, fname, results_by_input, band, self_freqs)
                n_written += 1
        except Exception as e:
            n_errors += 1
            errors.append({"file": fname, "error": repr(e), "traceback": traceback.format_exc()})
            log(f"ERROR on {fname}: {e!r} -- skipping, continuing")
            continue

        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t_start
            done = n_written + n_skipped
            rate = elapsed / max(n_written, 1)
            remaining = len(files) - (i + 1)
            eta_s = remaining * rate if n_written > 0 else float("nan")
            log(f"progress: {i+1}/{len(files)} files scanned, "
                f"{n_written} newly written, {n_skipped} skipped (resumed), "
                f"{n_errors} errors, elapsed={elapsed/60:.1f} min, "
                f"eta={eta_s/60:.1f} min")

    elapsed = time.time() - t_start
    summary = {
        "n_files_total": len(files),
        "n_written": n_written,
        "n_skipped_resumed": n_skipped,
        "n_errors": n_errors,
        "errors": errors,
        "n_refined_total": n_refined_total,
        "n_pixels_total": n_pixels_total,
        "elapsed_s": elapsed,
        "elapsed_hours": elapsed / 3600,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"=== DONE: {n_written} written, {n_skipped} skipped, {n_errors} errors, "
        f"{elapsed/3600:.2f} h total. Summary: {SUMMARY_JSON} ===")


if __name__ == "__main__":
    main()
