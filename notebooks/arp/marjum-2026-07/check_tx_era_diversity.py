"""Could the TX-on era substitute for the beam scan?

The beam scan has no transmitter (see tx_state_detector.py), so the obvious
fallback is the era where the transmitter *was* on.  A beam map needs both a
signal and pointing diversity; this checks whether the TX-on era has the
second.

Also cross-checks our matched detector against data-archivist's independently
built curation/tx_presence.jsonl, since a campaign-wide null should not rest
on one estimator.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from eigsep_observing import io

from eigsep_data.beam_mapping.diagnostics import MOTOR_CAL, _records_to_array

# notebooks/arp/marjum-2026-07/ -> notebooks/arp/ -> notebooks/ -> eigsep_data/
# -> the parent eigsep checkout, sibling to marjum-2026-07/.
EIGSEP_ROOT = Path(__file__).resolve().parents[4]
TX_PRESENCE = EIGSEP_ROOT / "marjum-2026-07" / "curation" / "tx_presence.jsonl"


def load_tx_on(path):
    rows = [json.loads(line) for line in open(path)]
    return {Path(r["file"]).name for r in rows if r["tx_on"]}, rows


def pointing(files):
    az, el = [], []
    for filename in files:
        try:
            _, header, metadata = io.read_hdf5(filename)
        except Exception:
            continue
        nt = len(header["times"])
        motor = _records_to_array(metadata.get("motor"),
                                  ["az_pos", "el_pos"], nt)
        if motor.ndim != 2 or motor.shape[0] == 0:
            continue
        az.append(motor[:, 0] * MOTOR_CAL)
        el.append(motor[:, 1] * MOTOR_CAL)
    return np.concatenate(az), np.concatenate(el)


def diversity(az, el, cell):
    idx = ((np.round(az / cell).astype(int) + 1000) * 10000
           + np.round(el / cell).astype(int) + 1000)
    uniq, counts = np.unique(idx, return_counts=True)
    total = counts.sum()
    n_eff = total ** 2 / np.sum(counts.astype(float) ** 2)
    order = np.argsort(counts)[::-1]
    top = [((uniq[i] // 10000 - 1000) * cell, (uniq[i] % 10000 - 1000) * cell,
            int(counts[i]), 100.0 * counts[i] / total) for i in order[:6]]
    return uniq.size, n_eff, top, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tx-presence", default=TX_PRESENCE)
    ap.add_argument("--cell-deg", type=float, default=4.435)
    ap.add_argument("--mine", default="tx_state_beam_scan.json")
    ap.add_argument("--control", default="tx_state_0716_control.json")
    args = ap.parse_args()

    tx_on, rows = load_tx_on(args.tx_presence)
    theirs = {Path(r["file"]).name: r for r in rows}
    print(f"tx_presence.jsonl: {len(rows)} files, {len(tx_on)} with tx_on")

    for label, path in [("beam scan", args.mine), ("07-16 control", args.control)]:
        try:
            mine = json.load(open(path))["files"]
        except FileNotFoundError:
            print(f"\n{label}: {path} not found, skipping cross-check")
            continue
        pairs = [(m, theirs[m["file"]]) for m in mine if m["file"] in theirs]
        if not pairs:
            continue
        mt = np.array([m["tx_S"] for m, _ in pairs])
        on = sum(1 for _, t in pairs if t["tx_on"])
        r = np.corrcoef(mt, [1.0 * t["tx_on"] for _, t in pairs])[0, 1] \
            if len({t["tx_on"] for _, t in pairs}) > 1 else np.nan
        print(f"\n{label}: {len(pairs)} files matched")
        print(f"  ours   tx_S median {np.median(mt):.3f}")
        print(f"  theirs tx_on true  {on} / {len(pairs)}")
        if np.isfinite(r):
            print(f"  agreement r = {r:+.3f}")

    files = [f for f in sorted(glob.glob(str(Path(args.data) / "*.h5")))
             if Path(f).name in tx_on]
    az, el = pointing(files)
    print(f"\n=== pointing diversity of the TX-on era "
          f"({len(files)} files, {az.size} spectra) ===")
    for cell in (args.cell_deg, 5.0, 10.0):
        n_cells, n_eff, top, total = diversity(az, el, cell)
        print(f"  cell {cell:6.3f} deg: {n_cells:5d} occupied cells, "
              f"dwell-weighted N_eff = {n_eff:6.2f}")
    n_cells, n_eff, top, total = diversity(az, el, args.cell_deg)
    print(f"\n  top dwell cells at {args.cell_deg} deg:")
    for a, e, n, pct in top:
        print(f"    az {a:7.1f} el {e:6.1f} : {n:6d} spectra ({pct:5.2f}%)")
    print(f"  top cell {top[0][3]:.2f}%, top 3 "
          f"{sum(t[3] for t in top[:3]):.2f}%")
    print("\n  For contrast the beam scan carries N_eff ~ 11 over 3,139 cells "
          "-- and no transmitter.")


if __name__ == "__main__":
    main()
