"""Which comb survives a terminated load? The only test that settles identity.

A transmitter signal reaches the receiver through the antenna.  Throw the RF
switch to an internal load and a real sky signal must vanish; anything that
survives is conducted from inside the instrument.  Compare *absolute*
second-difference amplitude, not SNR: a load is quiet, so its noise floor
collapses and any residual structure looks significant when normalised.

Run on files that contain both RFANT and load segments, so the comparison is
within a file -- same electronics, same moment, only the switch differs.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_observing import io

# the walking 1.000 MHz comb's own strong lines, off the 8-channel lattice
WALKING = [283, 291, 299, 307, 315, 324, 332, 340, 348, 356, 365]
LOCKED = np.arange(480, 800, 8)
BASE = np.array([c for c in range(280, 370) if c % 8 and c not in WALKING])


def second_diff(s):
    out = np.zeros_like(s)
    out[1:-1] = s[1:-1] - 0.5 * (s[:-2] + s[2:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tx-presence", required=True)
    ap.add_argument("--max-files", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.tx_presence)]
    on = {Path(r["file"]).name for r in rows if r["tx_on"]}
    files = [f for f in sorted(glob.glob(str(Path(args.data) / "*.h5")))
             if Path(f).name in on]

    print("absolute 2nd-difference amplitude, box-air (input 4), TX-on era")
    print(f"{'file':<26} {'state':<7} {'n':>4} {'continuum':>11} "
          f"{'walking':>12} {'locked':>11}")
    shown = 0
    for filename in files:
        try:
            dat, header, metadata = io.read_hdf5(filename)
        except Exception:
            continue
        rs = np.array([str(r) for r in (metadata.get("rfswitch") or []) if r])
        if rs.size == 0 or "RFANT" not in set(rs) or not {"RFAMB"} & set(rs):
            continue
        a = np.asarray(dat["4"]).astype(np.float64)
        a[a < 0] += 2 ** 32
        n = min(rs.size, a.shape[0])
        for state in ("RFANT", "RFAMB"):
            m = rs[:n] == state
            if m.sum() < 5:
                continue
            s = np.median(a[:n][m], axis=0)
            d = second_diff(s)
            print(f"{Path(filename).name:<26} {state:<7} {int(m.sum()):>4} "
                  f"{np.median(s[BASE]):>11.4g} {np.median(d[WALKING]):>12.4g} "
                  f"{np.median(d[LOCKED]):>11.4g}")
        print()
        shown += 1
        if shown >= args.max_files:
            break
    print("A comb whose amplitude does not fall when the antenna is switched "
          "out is conducted from inside the instrument, not received.")


if __name__ == "__main__":
    main()
