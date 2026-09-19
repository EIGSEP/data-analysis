"""Per-sample provenance sidecar for beam_explorer_cache.npz.

The explorer cache carries az/el/measured_tx/used but no time axis, so there is
no way from it alone to ask "when did this residual outlier happen, and what
else was going on then". This script rebuilds the same sample grid -- which is
deterministic: sample n = 240 * file_index + sample_idx, per
``load_v007_data`` -- and attaches the provenance columns:

  file_index, sample_idx, times          : who/when each sample is
  motor_az_deg, motor_el_deg, pot_az      : the RAW encoder pointing, before
                                            pointing_table@v1 overwrote az/el
  quality_v1, flags_v1                    : pointing_table@v1's own verdict
  rfi_cat                                 : flags/v0 RFI category bitfield,
                                            OR-reduced over the beam channels

Metadata only -- it does not read the autocorrelation payloads, so it is much
cheaper than load_v007_data.

Run: LD_LIBRARY_PATH=$MAMBA/envs/arp/lib $MAMBA/envs/arp/bin/python3 build_sample_sidecar.py
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import pyarrow.parquet as pq

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/mnt/data02/eigsep/marjum-2026-07/flagging")

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/mnt/data02/eigsep/marjum-2026-07/data")
POINTING_PARQUET = "/mnt/data02/eigsep/marjum-2026-07/curation/pointing_table.parquet"
FILES_SLICE = (-227, -1)
OUT = HERE / "beam_explorer_sidecar.npz"

# same primary-file guard the cache builder uses: data/ also holds derived
# `*.h5.smooth_model.h5` files that a bare *.h5 glob would pick up and that
# would silently shift the negative slice.
_PRIMARY = re.compile(r"^corr_\d{8}_\d{6}Z\.h5$")
files = sorted(p for p in DATA_DIR.glob("*.h5") if _PRIMARY.match(p.name))
files = files[slice(*FILES_SLICE)]
print(f"window: {len(files)} files, {files[0].name} .. {files[-1].name}")

n = len(files) * 240
times = np.zeros(n, dtype=np.float64)
motor_az = np.full(n, np.nan, dtype=np.float32)
motor_el = np.full(n, np.nan, dtype=np.float32)
pot_az = np.full(n, np.nan, dtype=np.float32)
file_index = np.repeat(np.arange(len(files)), 240).astype(np.int32)
sample_idx = np.tile(np.arange(240), len(files)).astype(np.int32)
nt_per_file = np.zeros(len(files), dtype=np.int32)

MOTOR_CAL = None
# The editable install of eigsep_data still points at the pre-rename path
# (.pth -> ~/projects/eigsep/eigsep_data/src), which vanished when the repo was
# renamed eigsep_data -> data-analysis on 2026-09-17, so `import eigsep_data`
# fails environment-wide. Fall back to this repo's own src/ rather than
# reinstalling into the shared env, which other agents are using live.
import importlib.util as _ilu
if _ilu.find_spec("eigsep_data") is None:
    _src = str(Path(__file__).resolve().parents[3] / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from eigsep_data.beam_mapping.diagnostics import MOTOR_CAL, _records_to_array
from eigsep_observing import io

t0 = time.time()
for i, path in enumerate(files):
    with h5py.File(path, "r") as h:
        pass  # io.read_hdf5 handles the layout; just confirm readability
    _, header, metadata = io.read_hdf5(str(path), load_data=False) \
        if "load_data" in io.read_hdf5.__code__.co_varnames else io.read_hdf5(str(path))
    nt = len(header["times"])
    nt_per_file[i] = nt
    sl = slice(240 * i, 240 * i + nt)
    times[sl] = header["times"]
    pot_az[sl] = np.asarray(
        _records_to_array(metadata.get("potmon"), ["pot_az_angle"], nt)).reshape(nt, -1)[:, 0]
    mot = _records_to_array(metadata.get("motor"), ["az_pos", "el_pos"], nt)
    motor_az[sl], motor_el[sl] = mot[:, 0] * MOTOR_CAL, mot[:, 1] * MOTOR_CAL
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)

# --- pointing_table@v1 quality / flags, joined by (file, sample_idx) ---------
quality = np.array(["gap"] * n, dtype=object)
flags_v1 = np.zeros(n, dtype=np.int64)
pt_az = np.full(n, np.nan)
pt_el = np.full(n, np.nan)
df = pq.read_table(POINTING_PARQUET, columns=[
    "file", "sample_idx", "az_deg", "el_deg", "quality", "flags"]).to_pandas()
names = {p.name: i for i, p in enumerate(files)}
df = df[df["file"].isin(names)]
rows = 240 * df["file"].map(names).to_numpy() + df["sample_idx"].to_numpy()
quality[rows] = df["quality"].to_numpy()
flags_v1[rows] = df["flags"].to_numpy()
pt_az[rows] = df["az_deg"].to_numpy()
pt_el[rows] = df["el_deg"].to_numpy()
print(f"pointing_table rows joined: {len(df)}")

# --- flags/v0 RFI categories, OR-reduced over the beam channel range --------
from read_flags import get_flags

CH_LO, CH_HI = 520, 780  # covers the TX comb channels used by the beam fits
rfi_cat = np.zeros(n, dtype=np.uint8)
rfi_avail = np.zeros(n, dtype=bool)
missing = 0
for i, path in enumerate(files):
    try:
        by_input, _ = get_flags(path.name)
    except (FileNotFoundError, KeyError):
        missing += 1
        continue
    cat = by_input.get("4")
    if cat is None:
        missing += 1
        continue
    nt = nt_per_file[i]
    sl = slice(240 * i, 240 * i + nt)
    c = np.asarray(cat)[:nt, CH_LO:CH_HI + 1]
    rfi_cat[sl] = np.bitwise_or.reduce(c, axis=1)
    rfi_avail[sl] = True
print(f"rfi flags: {missing} of {len(files)} files unavailable for input 4")

np.savez_compressed(
    OUT, files=np.array([p.name for p in files]), file_index=file_index,
    sample_idx=sample_idx, nt_per_file=nt_per_file, times=times,
    motor_az_deg=motor_az, motor_el_deg=motor_el, pot_az=pot_az,
    quality_v1=np.array([str(q) for q in quality]), flags_v1=flags_v1,
    pt_az_deg=pt_az, pt_el_deg=pt_el, rfi_cat=rfi_cat, rfi_avail=rfi_avail,
    rfi_channel_range=np.array([CH_LO, CH_HI]),
)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")
