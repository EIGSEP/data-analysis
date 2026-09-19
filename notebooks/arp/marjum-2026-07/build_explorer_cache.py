"""Build a self-contained cache for beam_explorer.ipynb.

Everything expensive (226 HDF5 reads, the data-space RFI mask, the HFSS PCA)
is done once here and written to a single .npz, so the interactive notebook
needs only numpy / healpy / matplotlib / ipywidgets -- no eigsep_data install
and no correlator files on the machine running it.

Run:  LD_LIBRARY_PATH=$MAMBA/envs/arp/lib $MAMBA/envs/arp/bin/python3 build_explorer_cache.py
"""
import glob as _glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# /tmp deliberately not on the path: fit_beam_v2.py is version-controlled
# next to this script since 2026-09-17, and /tmp would shadow it.

# --- guard against non-primary files in the data directory -------------------
# On 2026-09-15 a process wrote 17 derived `*.h5.smooth_model.h5` files into
# marjum-2026-07/data/. They match the `*.h5` glob that load_v007_data and
# data_space_rfi_mask use internally, which (a) breaks reading and (b) silently
# SHIFTS any negative-index file slice, because the file count changed. Filter
# the glob down to strict primary correlator files so the window is defined by
# the data, not by however many extra files happen to be sitting in the folder.
_PRIMARY = re.compile(r"^corr_\d{8}_\d{6}Z\.h5$")
_real_glob = _glob.glob


def _primary_only_glob(pattern, **kw):
    out = _real_glob(pattern, **kw)
    if pattern.endswith("*.h5"):
        out = [p for p in out if _PRIMARY.match(Path(p).name)]
    return out


_glob.glob = _primary_only_glob

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

from eigsep_data.beam_mapping import (
    compute_beam_pca, HFSSBeamSet, data_space_rfi_mask,
)
from eigsep_data.beam_mapping.diagnostics import (
    load_v007_data, channel_validity_masks, gross_power_time_flags, tx_arm_for_channel,
)
import fit_v007_pca_beam as v007
import fit_beam_v2 as v2

CUR = Path("/mnt/data02/eigsep/marjum-2026-07/curation")
OUT = HERE / "beam_explorer_cache.npz"

t0 = time.time()
data = load_v007_data(v2.DATA_DIR, start=v2.FILES_SLICE[0], stop=v2.FILES_SLICE[1])
data = v2.attach_pointing_v1(data, data["files"])
beam = HFSSBeamSet.from_npz(v2.BEAM_FILE)
pca = compute_beam_pca(beam, n_components=4)
clean = data_space_rfi_mask(v2.DATA_DIR, beam.freqs_mhz.min(), beam.freqs_mhz.max(),
                            files_slice=v2.FILES_SLICE, clip_sigma=5.0, min_votes=5)
clean = clean & v2.pointing_v1_valid_mask(data)

# 2026-09-17: the same two corrections the batch pipeline now applies, so the
# RMS the explorer shows interactively is the RMS the report quotes. Without
# these the explorer was drawing from a cache in which calibration samples --
# receiver on a load, not the antenna -- were being fit as beam data, carrying
# a median 71.3% of the residual power.
_on_ant = v2.receiver_on_antenna_mask(data)
_no_glitch = v2.el_solution_glitch_mask(data)
print(f"mask: {int(clean.sum())} -> {int((clean & _on_ant & _no_glitch).sum())} "
      f"(-{int((clean & ~_on_ant).sum())} off-antenna, "
      f"-{int((clean & _on_ant & ~_no_glitch).sum())} EL_SOLUTION_GLITCH)",
      flush=True)
clean = clean & _on_ant & _no_glitch
print(f"load + masks: {time.time()-t0:.0f}s", flush=True)

with open(HERE / "beam_fits_v2_pointingv1geom_report.json") as f:
    rep = json.load(f)
chans = np.array(sorted(int(r["channel"]) for r in rep["channels"]))
arms = np.array([tx_arm_for_channel(int(c)) for c in chans], dtype=np.int8)
freqs = np.array([float(data["freqs"][int(c)]) for c in chans])
a_hfss = np.array([v007.hfss_prior_vector(pca, f) for f in freqs])

Y = np.empty((len(chans), data["az_deg"].size), dtype=np.float32)
USED = np.empty((len(chans), data["az_deg"].size), dtype=bool)
for i, c in enumerate(chans):
    base = channel_validity_masks(data, [int(c)])[:, 0]
    gross, _, _ = gross_power_time_flags(data, [int(c)], 99.0, 5.0)
    USED[i] = base & ~gross & clean
    Y[i] = data["measured_tx"][:, int(c)].astype(np.float32)

# reference geometry: the surveyed positions, and the two fitted headings
tx_enu = np.array(json.load(open(CUR / "transmitter_position.json"))["best_estimate_enu_m"], float)
ant_enu = np.array(json.load(open(CUR / "horizon_profiles.json"))["antenna_enu_m"], float)
with open(HERE / "v007_multichannel_consensus.json") as f:
    old = json.load(f)
with open(HERE / "v007_multichannel_consensus_pointingv1.json") as f:
    new = json.load(f)

np.savez_compressed(
    OUT,
    az_deg=data["az_deg"].astype(np.float32),
    el_deg=data["el_deg"].astype(np.float32),
    # needed to interpolate B7's per-cycle g_rx onto the samples; the explorer
    # had no time axis before 2026-09-17 and so could do nothing time-based
    times=data["times"].astype(np.float64),
    measured_tx=Y,
    used=USED,
    channels=chans,
    arms=arms,
    freqs_mhz=freqs,
    a_hfss=a_hfss,
    beam_cart=pca.components.beam_cart.astype(np.complex64),
    nside=np.array(pca.components.nside),
    explained_variance_ratio=pca.explained_variance_ratio,
    surveyed_delta_enu=(tx_enu - ant_enu),
    heading_old=np.array(old["heading"], float),
    alpha_old=np.array(old["alpha_deg"], float),
    heading_new=np.array(new["heading"], float),
    alpha_new=np.array(new["alpha_deg"], float),
    per_channel_gain_fitted=np.array(
        [r["gain_fitted"] for r in sorted(rep["channels"], key=lambda r: r["channel"])]),
    per_channel_shape_re=np.array(
        [r["shape_correction_real"] for r in sorted(rep["channels"], key=lambda r: r["channel"])]),
    per_channel_shape_im=np.array(
        [r["shape_correction_imag"] for r in sorted(rep["channels"], key=lambda r: r["channel"])]),
)
print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
print(f"total {time.time()-t0:.0f}s")
