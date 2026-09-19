"""Package B16's DPSS-fit + residual-outlier flagging into a flags/vN-
style product, staged next to (not overwriting) the live flags/v0/.

Experimental first attempt, per Aaron 2026-09-17: "I think we should
take a shot at packaging the diagnostic HDF5 into the RFI flags
tonight... It's non-destructive, so there's really no danger in
trying." Not a reviewed product -- writes only to a new staging
directory; promoting it over flags/v0/ is a separate, later, reviewed
decision.

What this adds, precisely: for each (file, input, time, channel) pixel
inside the analysis band (45-235 MHz), a NEW bit is set if and only if
B16's DPSS-fit residual-outlier refinement flagged that pixel AND it
was not already flagged by v0's own per-pixel category bitfield AND
its channel is not already in the campaign-wide static self-comb
exclusion. That isolates exactly what "fitting DPSS modes and removing
outliers" contributes beyond what's already known -- not a wholesale
replacement of v0's categories, which stay untouched and copied
through verbatim.

**Format-compatibility note, not silently absorbed:** v0's bitfield is
uint8 with all 8 bits already assigned (cal, tx_comb, self-RFI,
FM-scatter, airplane, orbcomm, unknown, overflow) -- there is no spare
bit in a uint8 field. This product's mask arrays are therefore uint16
(bit 8, value 256, "dpss_residual_outlier"), NOT byte-compatible with
v0's uint8 arrays. Any consumer reading this must know the dtype
changed; documented in flag_bits.json and manifest.json.

Only processes files that have a completed B16 companion output in
derived/smooth_model/v0/; files without one get v0's bits copied
through unchanged (no new bit, not an error).
"""
import glob
import json
import os
import sys
import time
import traceback

import h5py
import numpy as np

from eigsep_data.flagging import detectors as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b15_event_survey as EV  # noqa: E402
import b16_dpss_model as M16  # noqa: E402

DATA_ROOT = EV.DATA_ROOT
V0_DIR = os.path.join(DATA_ROOT, "flags", "v0")
# NOTE: "flags/v1" is already occupied by an unrelated, unfinished B8
# coincidence-check investigation (not a mask product) -- this product's
# promoted name is "v2", not "v1". Each re-run writes a fresh dated
# staging dir; promoting it to flags/v2 (replacing the prior promoted
# build) is a deliberate, separate step, not automatic here.
OUT_DIR = os.path.join(DATA_ROOT, "flags",
                        "v2_staging_" + time.strftime("%Y%m%d"))
DPSS_OUTLIER_BIT = 8
DPSS_OUTLIER_VALUE = 1 << DPSS_OUTLIER_BIT  # 256, needs uint16

os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "package_run.log")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    t_start = time.time()
    log("=== packaging DPSS-outlier bit into staged flags/v2 build ===")
    log(f"V0_DIR={V0_DIR}")
    log(f"OUT_DIR={OUT_DIR}")

    self_freqs, self_mask = EV.self_comb_channel_mask()  # (1024,), (1024,) bool
    band = (self_freqs >= D.BAND_ANALYSIS[0]) & (self_freqs <= D.BAND_ANALYSIS[1])
    band_idx = np.nonzero(band)[0]
    log(f"analysis-band channels: {band.sum()} of {D.N_CHAN}")

    day_files = sorted(glob.glob(os.path.join(V0_DIR, "flags_*.h5")))
    log(f"v0 day-files found: {len(day_files)}")

    n_files_with_b16 = 0
    n_files_without_b16 = 0
    n_errors = 0
    errors = []
    per_day_stats = {}

    for day_path in day_files:
        day = os.path.basename(day_path).replace("flags_", "").replace(".h5", "")
        out_path = os.path.join(OUT_DIR, os.path.basename(day_path))
        day_new_bit_pixels = 0
        day_band_pixels = 0

        with h5py.File(day_path, "r") as h_in, h5py.File(out_path, "w") as h_out:
            for k in h_in.attrs:
                h_out.attrs[k] = h_in.attrs[k]
            h_out.attrs["note"] = (
                "uint16 category bitfield (WIDENED from v0's uint8 -- "
                "see flag_bits.json), axes (time, channel). v0's 8 bits "
                "copied through unchanged; bit 8 (value 256) is new: "
                "DPSS-fit residual-outlier flagging beyond v0+self-comb.")
            h_out.attrs["version"] = "v2"
            if "freqs_mhz" in h_in:
                h_out.create_dataset("freqs_mhz", data=h_in["freqs_mhz"][:])
            g_out = h_out.create_group("mask")

            for fname in h_in["mask"].keys():
                g_in = h_in["mask"][fname]
                companion_path = os.path.join(M16.DERIVED_DIR, fname)
                has_b16 = os.path.isfile(companion_path)
                b16_h = None
                if has_b16:
                    try:
                        b16_h = h5py.File(companion_path, "r")
                    except Exception as e:
                        log(f"WARN: could not open companion for {fname}: {e!r}")
                        has_b16 = False

                for inp in g_in.keys():
                    v0_bits = g_in[inp][:]  # (n_time, 1024) uint8
                    out_bits = v0_bits.astype(np.uint16)
                    day_band_pixels += v0_bits.shape[0] * band.sum()

                    if has_b16 and b16_h is not None and f"input_{inp}" in b16_h:
                        try:
                            b16_flagged = b16_h[f"input_{inp}"]["flagged"][:]  # (n_time, 778) uint8
                            if b16_flagged.shape[0] != v0_bits.shape[0]:
                                raise ValueError(
                                    f"time-axis mismatch: v0 has {v0_bits.shape[0]} rows, "
                                    f"b16 has {b16_flagged.shape[0]} for {fname}/{inp}")
                            v0_cat_flagged_band = (v0_bits[:, band] != D.CLEAN)
                            self_comb_band = self_mask[band][None, :]
                            new_outlier = (b16_flagged.astype(bool)
                                           & ~v0_cat_flagged_band
                                           & ~self_comb_band)
                            out_bits[:, band_idx] = np.where(
                                new_outlier, out_bits[:, band_idx] | DPSS_OUTLIER_VALUE,
                                out_bits[:, band_idx])
                            day_new_bit_pixels += int(new_outlier.sum())
                        except Exception as e:
                            n_errors += 1
                            errors.append({"file": fname, "input": inp, "error": repr(e),
                                            "traceback": traceback.format_exc()})
                            log(f"ERROR packaging {fname}/{inp}: {e!r} -- "
                                f"copying v0 bits through unchanged for this input")

                    g_out.create_dataset(f"{fname}/{inp}", data=out_bits,
                                          compression="gzip", compression_opts=6,
                                          shuffle=True)

                if b16_h is not None:
                    b16_h.close()

                if has_b16:
                    n_files_with_b16 += 1
                else:
                    n_files_without_b16 += 1

        frac_new = day_new_bit_pixels / max(day_band_pixels, 1)
        per_day_stats[day] = {
            "band_pixels": int(day_band_pixels),
            "new_dpss_outlier_pixels": int(day_new_bit_pixels),
            "frac_new_dpss_outlier": round(frac_new, 6),
        }
        log(f"day {day}: {per_day_stats[day]}")

    with open(os.path.join(OUT_DIR, "flag_bits.json"), "w") as f:
        json.dump({
            "encoding": "uint16 bitfield per (time, channel) sample "
                        "(WIDENED from v0's uint8 -- bit 8 needs 2 bytes)",
            "axes": ["time", "channel"],
            "n_channels": D.N_CHAN,
            "channel_width_mhz": D.CHAN_WIDTH_MHZ,
            "clean_value": 0,
            "bits": [
                {"bit": 0, "value": int(D.CAL), "name": "cal", "rfi": False,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": 1, "value": int(D.TX_COMB), "name": "tx_comb", "rfi": False,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": 2, "value": int(D.SELF_RFI), "name": "self-RFI", "rfi": True,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": 3, "value": int(D.FM_DTV_MS), "name": "FM-scatter", "rfi": True,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": 4, "value": int(D.AIRPLANE), "name": "airplane", "rfi": True,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": 5, "value": int(D.ORBCOMM), "name": "orbcomm", "rfi": True,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": 6, "value": int(D.UNKNOWN), "name": "unknown", "rfi": True,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": 7, "value": int(D.OVERFLOW), "name": "overflow", "rfi": False,
                 "meaning": "copied from flags/v0, unchanged"},
                {"bit": DPSS_OUTLIER_BIT, "value": DPSS_OUTLIER_VALUE,
                 "name": "dpss_residual_outlier", "rfi": True,
                 "meaning": "NEW (2026-09-17): flagged by B16's DPSS "
                            "smooth-band fit + per-channel-MAD residual "
                            "outlier detection (REFINE_NSIG=6), analysis "
                            "band only (45-235 MHz), AND NOT already "
                            "flagged by v0's category bits or the "
                            "campaign-wide static self-comb channel "
                            "exclusion -- i.e. exactly what DPSS-fit + "
                            "outlier-removal added beyond what was "
                            "already known. REFINE_NSIG was left "
                            "unchanged (known to over-mask relative to "
                            "log-domain expectations, per Aaron's "
                            "explicit instruction not to retune it "
                            "for this pass -- see MEMORY.md 'B16 rev4'; "
                            "Aaron has since reviewed the masks and "
                            "considers them reasonable)."},
            ],
        }, f, indent=2)

    elapsed = time.time() - t_start
    manifest = {
        "provenance": {
            "product": "flags",
            "campaign": "marjum-2026-07",
            "version": "v2",
            "status": "EXPERIMENTAL FIRST ATTEMPT, NOT REVIEWED -- staged "
                      "next to flags/v0, does not overwrite it. Promoting "
                      "this over flags/v0 is a separate, later, reviewed "
                      "decision, per Aaron's explicit instruction "
                      "(2026-09-17) that this is 'take a shot at it', "
                      "fine to fail or be left half-done for morning "
                      "review.",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator": "flagging/package_dpss_into_mask.py "
                         "(branch rfi-b16-linear-refit worktree, ops "
                         "script, not part of the reviewed B16 "
                         "methodology commit)",
            "inputs": [
                {"path": "flags/v0/*.h5", "note": "unchanged bits copied through"},
                {"path": "derived/smooth_model/v0/*.h5",
                 "note": "B16 rev4 (linear amplitude) full-campaign DPSS "
                         "fit output; files without a companion here "
                         "get v0 bits unchanged, not an error"},
            ],
            "format_change": "uint16, not v0's uint8 -- see flag_bits.json",
        },
        "n_files_with_b16_companion": n_files_with_b16,
        "n_files_without_b16_companion": n_files_without_b16,
        "n_errors": n_errors,
        "errors": errors[:50],
        "per_day_stats": per_day_stats,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"=== DONE: {n_files_with_b16} files with B16 data, "
        f"{n_files_without_b16} without, {n_errors} errors, "
        f"{elapsed/60:.1f} min. Manifest: {OUT_DIR}/manifest.json ===")


if __name__ == "__main__":
    main()
