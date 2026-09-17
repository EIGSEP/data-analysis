"""Does the locked 8-channel comb arrive through the rotating antenna?

rfi-analyst's discriminator: box-gnd is fixed, box-air rotates.  A source
received through the antenna pattern must modulate as box-air turns; a
conducted signal must not.  Verifying it against our own azimuth measurements,
independently of anyone's comb detector.

Two parts:

  A. absolute amplitude on antenna vs terminated load, per input.  Redone in
     absolute units: an earlier claim of mine that the locked comb is "present
     on box-gnd's load" was made in SNR, and a load's noise floor collapses,
     so SNR there is not comparable to SNR on sky.

  B. locked-comb amplitude vs azimuth through the beam scan, on the rotating
     input and the fixed one.  The fixed input is the control: whatever it
     does with azimuth is what a non-antenna path looks like.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_observing import io

from eigsep_data.beam_mapping.diagnostics import MOTOR_CAL, _records_to_array

LOCKED = np.arange(480, 800, 8)
OFF = np.setdiff1d(np.arange(480, 800),
                   np.concatenate([LOCKED + o for o in (-1, 0, 1)]))
WALKING = [283, 291, 299, 307, 315, 324, 332, 340, 348, 356, 365]
INPUTS = {"4": "box-air (ROTATES)", "0": "box-gnd (fixed)"}


def second_diff(a):
    out = np.zeros_like(a)
    out[..., 1:-1] = a[..., 1:-1] - 0.5 * (a[..., :-2] + a[..., 2:])
    return out


def part_a(files):
    print("=== A. absolute 2nd-difference amplitude, antenna vs load ===")
    print(f"{'file':<26} {'input':<20} {'state':<7} {'n':>4} "
          f"{'continuum':>11} {'locked':>12} {'walking':>12}")
    shown = 0
    for filename in files:
        try:
            dat, header, metadata = io.read_hdf5(filename)
        except Exception:
            continue
        rs = np.array([str(r) for r in (metadata.get("rfswitch") or []) if r])
        if rs.size == 0 or "RFANT" not in set(rs) or not {"RFAMB"} & set(rs):
            continue
        for key, label in INPUTS.items():
            if key not in dat:
                continue
            a = np.asarray(dat[key]).astype(np.float64)
            a[a < 0] += 2 ** 32
            n = min(rs.size, a.shape[0])
            for state in ("RFANT", "RFAMB"):
                m = rs[:n] == state
                if m.sum() < 5:
                    continue
                s = np.median(a[:n][m], axis=0)
                d = second_diff(s)
                print(f"{Path(filename).name:<26} {label:<20} {state:<7} "
                      f"{int(m.sum()):>4} {np.median(s[OFF]):>11.4g} "
                      f"{np.median(d[LOCKED]):>12.4g} "
                      f"{np.median(d[WALKING]):>12.4g}")
        print()
        shown += 1
        if shown >= 2:
            break


def part_b(files):
    print("=== B. locked-comb amplitude vs azimuth through the beam scan ===")
    per = {k: [] for k in INPUTS}
    az_all = []
    for filename in files:
        try:
            dat, header, metadata = io.read_hdf5(filename)
        except Exception:
            continue
        nt = len(header["times"])
        motor = _records_to_array(metadata.get("motor"),
                                  ["az_pos", "el_pos"], nt)
        if motor.ndim != 2 or motor.shape[0] == 0:
            continue
        az = motor[:, 0] * MOTOR_CAL
        ok = True
        vals = {}
        for key in INPUTS:
            if key not in dat:
                ok = False
                break
            a = np.asarray(dat[key]).astype(np.float64)
            a[a < 0] += 2 ** 32
            d = second_diff(a)[:len(az)]
            vals[key] = np.median(d[:, LOCKED], axis=1)
        if not ok:
            continue
        az_all.append(az[:len(vals["4"])])
        for key in INPUTS:
            per[key].append(vals[key])
    az = np.concatenate(az_all)
    print(f"{az.size} integrations with pointing\n")
    for key, label in INPUTS.items():
        y = np.concatenate(per[key])
        good = y > 0
        if good.sum() < 100:
            print(f"{label}: comb not positive often enough to measure "
                  f"({int(good.sum())} of {y.size})")
            continue
        db = 10 * np.log10(y[good])
        a = az[good]
        # spread across azimuth, in dB, using binned medians
        bins = np.round(a / 15.0).astype(int)
        meds = np.array([np.median(db[bins == b]) for b in np.unique(bins)
                         if (bins == b).sum() >= 20])
        r = np.corrcoef(a, db)[0, 1]
        print(f"{label}")
        print(f"   samples with comb > 0        : {int(good.sum())}")
        print(f"   spread across azimuth bins   : "
              f"{meds.max() - meds.min():5.2f} dB "
              f"(10-90%: {np.percentile(meds, 90) - np.percentile(meds, 10):.2f} dB)")
        print(f"   corr(amplitude_dB, azimuth)  : r = {r:+.3f}")
        print(f"   per-sample scatter           : {db.std():5.2f} dB\n")
    print("A source received through the antenna pattern modulates on the")
    print("rotating input and not on the fixed one. Note this separates")
    print("'arrives via the antenna' from 'conducted' -- it does NOT by itself")
    print("separate a far-field transmitter from a fixed near-field radiator,")
    print("since both couple through the rotating pattern.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tx-presence", required=True)
    ap.add_argument("--scan-files", type=int, default=40)
    args = ap.parse_args()
    allf = sorted(glob.glob(str(Path(args.data) / "*.h5")))
    rows = [json.loads(line) for line in open(args.tx_presence)]
    on = {Path(r["file"]).name for r in rows if r["tx_on"]}

    # load-switch files during the 07-17/18 locked-comb era
    era = [f for f in allf if "corr_20260717_1537" <= Path(f).name[:22]
           <= "corr_20260718_0300"]
    part_a(era)
    scan = [f for f in allf if Path(f).name >= "corr_20260717_185000Z"]
    part_b(scan[:args.scan_files])


if __name__ == "__main__":
    main()
