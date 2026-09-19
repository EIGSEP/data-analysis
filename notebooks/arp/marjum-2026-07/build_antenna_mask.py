"""Per-sample "receiver is on the antenna" mask for the beam-scan window.

Aaron's instruction was to exclude calibration data using `metadata/rfswitch`
(`rfswitch_off_ant`). The `rfswitch_dominant` column in `curation/file_state.csv`
implements that test at *file* granularity, and at that granularity it is wrong
for this window: all 23 files that `curation/cal_windows.jsonl` identifies as
calibration contain non-RFANT samples, but 14 of them are *majority* RFANT, so
a dominant-state test keeps them. Example --
`corr_20260717_191940Z.h5` is {RFANT: 124, RFNON: 111, RFAMB: 3}: dominant
RFANT, yet 114 of 240 samples are off-antenna.

So this reads the `metadata/rfswitch` stream per sample, via the canonical
`flagging/detectors.antenna_mask()` (which resamples the metadata cadence onto
the spectrum axis by nearest neighbour). That is strictly better than both
file-level tests: it catches the mixed files, and it keeps the genuine
on-antenna samples inside a calibration file instead of discarding the whole
file.

Composition. `antenna_mask()` returns all-True when a file has no rfswitch
stream, deliberately -- 56 of the 226 files in this window are in that state --
and its docstring defers those to the coarser cal-window gate. This script
therefore also emits the file-level cal-window mask, and the recommended
exclusion is the OR of the two:

    exclude = ~on_antenna | (cal_window_file & no_rfswitch_stream)

Run: LD_LIBRARY_PATH=$MAMBA/envs/arp/lib $MAMBA/envs/arp/bin/python3 build_antenna_mask.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, "/mnt/data02/eigsep/marjum-2026-07/flagging")
from detectors import antenna_mask  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/mnt/data02/eigsep/marjum-2026-07/data")
CUR = Path("/mnt/data02/eigsep/marjum-2026-07/curation")
FILES_SLICE = (-227, -1)
OUT = HERE / "antenna_mask.npz"

_PRIMARY = re.compile(r"^corr_\d{8}_\d{6}Z\.h5$")
files = sorted(p for p in DATA_DIR.glob("*.h5") if _PRIMARY.match(p.name))
files = files[slice(*FILES_SLICE)]
print(f"window: {len(files)} files, {files[0].name} .. {files[-1].name}")

n = len(files) * 240
on_ant = np.ones(n, dtype=bool)        # True = receiver on the antenna
has_stream = np.zeros(n, dtype=bool)
nt_per_file = np.zeros(len(files), dtype=np.int32)

for i, path in enumerate(files):
    with h5py.File(path, "r") as f:
        nt = len(f["header/times"]) if "header/times" in f else 240
        raw = None
        if "metadata" in f and "rfswitch" in f["metadata"]:
            raw = f["metadata/rfswitch"][()]
    nt_per_file[i] = nt
    sl = slice(240 * i, 240 * i + nt)
    on_ant[sl] = antenna_mask(raw, nt)
    has_stream[sl] = raw is not None
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)

# file-level cal-window gate, for the files with no rfswitch stream
names = [p.name for p in files]
rows = [json.loads(l) for l in open(CUR / "cal_windows.jsonl")]
cal_names = {nm for w in rows for nm in names
             if w["file_first"] <= nm <= w["file_last"]}
cal_file = np.zeros(n, dtype=bool)
for i, nm in enumerate(names):
    if nm in cal_names:
        cal_file[240 * i:240 * i + 240] = True

no_stream_file = np.zeros(n, dtype=bool)
for i in range(len(files)):
    sl = slice(240 * i, 240 * i + int(nt_per_file[i]))
    if not has_stream[sl].any():
        no_stream_file[sl] = True

exclude = (~on_ant) | (cal_file & no_stream_file)

print()
print(f"samples off-antenna per rfswitch      : {int((~on_ant).sum())}")
print(f"files with no rfswitch stream         : "
      f"{int(sum(1 for i in range(len(files)) if not has_stream[240*i:240*i+int(nt_per_file[i])].any()))}")
print(f"cal-window files                      : {len(cal_names)}")
print(f"recommended exclusion (union)         : {int(exclude.sum())}")

np.savez_compressed(OUT, on_antenna=on_ant, has_rfswitch=has_stream,
                    cal_window_file=cal_file, no_rfswitch_file=no_stream_file,
                    exclude=exclude, files=np.array(names),
                    nt_per_file=nt_per_file)
print(f"wrote {OUT}")
