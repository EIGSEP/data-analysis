"""Beam-scan window loader on eigsep_data's canonical access path.

Replaces ``load_v007_data`` + ``fit_beam_v2.attach_pointing_v1`` with
``MetadataIndex`` / ``Selection.load_bundle``.  Same outputs, same sample
grid, no HDF5 scraping and no filename slicing.

Why this exists (B39/B42, 2026-09-19)
-------------------------------------
``load_v007_data`` is frozen as a historical loader (its own docstring,
Aaron 2026-09-17).  It selects with a bare ``*.h5`` glob and a negative
slice, which silently re-points whenever a file lands in the data
directory, and it reads pointing from whatever ``pointing_table.parquet``
happens to be on disk without checking its version.  Both have bitten
this campaign.

``load_bundle`` fixes both: a file range is a real selection, and a
product version is *named* and asserted.  On 2026-09-18 the pointing
table regressed v1.2 -> v1.1 and this path raised

    ValueError: pointing@v1.2 requested but
    curation/pointing_table.schema.json declares 'v1.1'.

where the legacy path would have silently disabled
``el_solution_glitch_mask`` (bit 1024 was set on 0 of 1,227,956 rows).
Keep the version pinned; do not relax it to "whatever is current".

The sample grid
---------------
Deterministic and identical to the old loader: ``n = 240*file_index +
row``.  Files with fewer than 240 rows leave their tail **zero-filled**
rather than short — downstream masks and the explorer cache index into
this grid, so the padding is load-bearing, not cosmetic.

Input key
---------
Pass ``key="4"``, not ``antenna="box-air"``.  Inputs 4 and 5 both map to
box-air in this window (the ADC mux copies an even input onto the odd one
above it), so the antenna name is ambiguous.  Verified constant over all
226 window files:
``{"0": "box-gnd", "1": "box-gnd", "4": "box-air", "5": "box-air"}``.
Input numbering is *not* stable campaign-wide, so this pin is scoped to
this window.

Not provided
------------
``comb_off_files`` is returned, but ``accel``/``pot`` come from the index
metadata columns rather than from re-reading the HDF5 metadata group.
"""
import numpy as np

from eigsep_data import MetadataIndex
from eigsep_data.beam_mapping.diagnostics import (
    MOTOR_CAL, comb_present, radiometer_difference_sigma,
)

ROWS_PER_FILE = 240
NCHAN = 1024


def load_window(data_dir, first_file, last_file, key="4",
                pointing_version="v1.2", require_comb=True):
    """Load one file range for one correlator input, with pointing joined.

    Returns the same dict ``load_v007_data`` + ``attach_pointing_v1``
    produce, on the same ``240*file_index + row`` grid.
    """
    idx = MetadataIndex(str(data_dir))
    sel = idx.select(files=(first_file, last_file))
    bundle = sel.load_bundle(key=key,
                             products=[f"pointing@{pointing_version}"])

    meta = bundle.meta.reset_index(drop=True)
    files = list(dict.fromkeys(meta["file"].tolist()))
    file_pos = {name: i for i, name in enumerate(files)}
    # Position on the padded grid, per bundle row.
    slot = (meta["file"].map(file_pos).to_numpy() * ROWS_PER_FILE
            + meta["row"].to_numpy()).astype(np.int64)

    n = len(files) * ROWS_PER_FILE
    times = np.zeros(n, dtype=np.float64)
    accel = np.zeros((n, 3), dtype=np.float32)
    pot = np.zeros(n, dtype=np.float32)
    azm = np.zeros(n, dtype=np.float32)
    elm = np.zeros(n, dtype=np.float32)
    measured_tx = np.zeros((n, NCHAN), dtype=np.float32)
    measured_sigma = np.zeros((n, NCHAN), dtype=np.float32)

    times[slot] = np.asarray(bundle.t, dtype=np.float64)
    for j, col in enumerate(("imu_el_accel_x", "imu_el_accel_y",
                             "imu_el_accel_z")):
        accel[slot, j] = _col(meta, col)
    pot[slot] = _col(meta, "potmon_pot_az_angle")
    azm[slot] = _col(meta, "motor_az_pos")
    elm[slot] = _col(meta, "motor_el_pos")

    freqs = np.asarray(bundle.freqs_mhz)
    df_hz = abs(float(freqs[1] - freqs[0])) * 1e6
    auto_all = np.asarray(bundle.data, dtype=float)

    comb_off_files = []
    for name in files:
        rows = np.flatnonzero(meta["file"].to_numpy() == name)
        if rows.size == 0:
            continue
        auto = auto_all[rows]
        here = slot[rows]
        if require_comb and not comb_present(np.median(auto, axis=0)):
            comb_off_files.append(name)
            times[here] = 0.0          # the loader's existing invalid sentinel
            continue
        measured_tx[here, 1:-1] = auto[:, 1:-1] - 0.5 * (
            auto[:, :-2] + auto[:, 2:])
        integ = float(np.median(_col(meta, "integration_time")[rows]))
        measured_sigma[here, 1:-1] = radiometer_difference_sigma(
            auto[:, 1:-1], auto[:, :-2], auto[:, 2:], df_hz, integ)

    quality = np.array(["gap"] * n, dtype=object)
    flags = np.zeros(n, dtype=np.int64)
    az_v1 = np.full(n, np.nan)
    el_v1 = np.full(n, np.nan)
    az_sigma = np.full(n, np.nan)
    el_sigma = np.full(n, np.nan)
    p = bundle.pointing.reset_index(drop=True)
    az_v1[slot] = p["az_deg"].to_numpy(dtype=float)
    el_v1[slot] = p["el_deg"].to_numpy(dtype=float)
    az_sigma[slot] = p["az_sigma_deg"].to_numpy(dtype=float)
    el_sigma[slot] = p["el_sigma_deg"].to_numpy(dtype=float)
    quality[slot] = p["quality"].to_numpy(dtype=object)
    flags[slot] = p["flags"].to_numpy(dtype=np.int64)

    # Match attach_pointing_v1 exactly: `pointing_v1_finite` is computed
    # BEFORE the fill, then az/el are zero-filled because healpy needs an
    # in-range value for every sample.  Rows with no pointing_table row are
    # excluded downstream by pointing_v1_valid_mask, so the 0/0 dummy never
    # reaches the fit.  Emitting NaN here instead would be more honest but
    # would crash ang2pix and silently change the mask.
    finite = np.isfinite(az_v1) & np.isfinite(el_v1)

    return {
        "files": [str(f) for f in bundle.provenance["used_files"]],
        "times": times,
        "accel": accel,
        "pot": pot,
        "az_deg": np.nan_to_num(az_v1, nan=0.0),
        "el_deg": np.nan_to_num(el_v1, nan=0.0),
        "pointing_v1_finite": finite,
        "motor_az_deg": azm * MOTOR_CAL,
        "motor_el_deg": elm * MOTOR_CAL,
        "measured_tx": measured_tx,
        "measured_sigma": measured_sigma,
        "freqs": freqs,
        "comb_off_files": comb_off_files,
        "quality_v1": quality,
        "flags_v1": flags,
        "az_sigma_v1": az_sigma,
        "el_sigma_v1": el_sigma,
        "provenance": bundle.provenance,
    }


def _col(meta, name):
    return np.asarray(meta[name].to_numpy(), dtype=float)
