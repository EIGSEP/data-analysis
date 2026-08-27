"""Calibrate field VNA S11 captures to the LNA reference plane (or as
deep as each DUT supports) and write one HDF5 file per DUT.

Reads every raw S11 h5 file in a directory (as written by
eigsep_observing's normal antenna-observing VNA loop -- see
eigsep_observing.io.write_s11_file), calibrates each capture using the
VNA's own internal OSL standards (recorded alongside the DUT traces in
each file as "cal:VNAO"/"cal:VNAS"/"cal:VNAL"), de-embeds the
lab-characterized VNA-leg switch path to reach the physical DUT plane,
then embeds the lab-characterized RF/LNA-leg switch path to reach the
LNA input plane. Results are written with
eigsep_data.s11.write_dut_calibration_h5 -- one HDF5 file per DUT.

Calibration chain, per DUT:

    raw S11 --[de-embed VNA's own OSL]--> "vna" plane
            --[de-embed VNA-leg switch path]--> "dut" plane   (skipped
                                                 for load/noise, which
                                                 sit directly at "vna")
            --[embed RF/LNA-leg switch path]--> "lna" plane   (only
                                                 for DUTs with an
                                                 RF-leg switch path;
                                                 load/noise/rec stop
                                                 one plane shallower)

Each DUT's output file gets a "default" alias pointing at the deepest
plane actually reached for that DUT (lna where available, else dut,
else vna) -- so read_final_plane() always has something to return,
even for DUTs that never reach the LNA plane.

Usage::

    python scripts/calibrate_field_s11.py DATADIR SWITCHPATHS OSLDATA
    python scripts/calibrate_field_s11.py DATADIR SWITCHPATHS OSLDATA \\
        --save-dir ./calibrated --pattern "ants11_*.h5"
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from cmt_vna import calkit
from eigsep_observing import io

from eigsep_data.s11 import write_dut_calibration_h5

# Which switch path(s) must be de-embedded (VNA leg) / embedded (RF or
# LNA leg) to get from the VNA's internal-OSL reference plane to each
# DUT's physical location and on to the LNA input. An empty list means
# that DUT sits at the previous plane already -- nothing to de-embed
# or embed.
DEEMBED_DICT = {
    "amb": ["VNAAMB"],
    "ant": ["VNAANT"],
    "load": [],
    "noise": [],
    "sp1_open": ["VNASP1"],
    "sp1_short": ["VNASP1"],
    "sp1": ["VNASP1"],
    "rec": ["VNARF"],
}
EMBED_DICT = {
    "amb": ["RFAMB"],
    "ant": ["RFANT"],
    "load": [],
    "noise": [],
    "sp1_open": ["RFSP1"],
    "sp1_short": ["RFSP1"],
    "sp1": ["RFSP1"],
    "rec": [],
}


def _deepest_plane(key):
    """Deepest calibration plane actually reached for ``key``, given
    DEEMBED_DICT/EMBED_DICT -- used to pick each DUT's "default"
    alias in the output file."""
    if EMBED_DICT.get(key):
        return "lna"
    if DEEMBED_DICT.get(key):
        return "dut"
    return "vna"


def calibrate_field_s11(datadir, switchpaths, osldata, pattern="*.h5"):
    """Calibrate every raw S11 h5 file in ``datadir``.

    Parameters
    ----------
    datadir : Path
        Directory of raw S11 h5 files (as written by
        eigsep_observing's normal VNA observing loop).
    switchpaths : Path
        npz of lab-characterized switch-path S-parameters, keyed by
        switch state name (e.g. "VNAAMB", "RFANT", ...).
    osldata : Path
        npz of the characterized field OSL standards: a "freqs" array
        plus the per-standard S11 arrays used as the ideal/
        characterized reference passed to
        ``calkit.network_sparams``.
    pattern : str, optional
        Glob pattern (relative to ``datadir``) selecting input files.
        Default ``"*.h5"``.

    Returns
    -------
    caled_s11s : dict
        ``{dut: {timestamp: {cal_plane: s11_array}}}``.
    freqs : np.ndarray
        Frequency axis (Hz), from ``osldata``.
    """
    paths = sorted(datadir.glob(pattern))
    if not paths:
        raise ValueError(f"no files matching {pattern!r} in {datadir}")

    # Characterized field OSL model -- the "true" gamma passed as the
    # first argument to calkit.network_sparams. Shares one freq axis
    # across every file processed in this run.
    osl_model = dict(np.load(osldata))
    freqs = osl_model.pop("freqs")
    osl_model = np.array(list(osl_model.values()))

    sparam_dict = dict(np.load(switchpaths))

    # Gather raw captures and internal-OSL sets from every file.
    uncaled_s11s = {}
    caled_s11s = {}
    osls = {"ant": {}, "rec": {}}
    for path in paths:
        data, cal_data, hdr, meta = io.read_s11_file(path)
        for key, s11 in data.items():
            if np.any(s11 == 0):
                continue  # unmeasured/invalid capture
            uncaled_s11s.setdefault(key, {})
            caled_s11s.setdefault(key, {})
            uncaled_s11s[key][hdr["metadata_snapshot_unix"]] = s11
            caled_s11s[key][hdr["metadata_snapshot_unix"]] = {'raw': s11}
        osl = np.array([cal_data["VNAO"], cal_data["VNAS"], cal_data["VNAL"]])
        if np.any(osl == 0):
            continue  # unmeasured/invalid internal OSL set
        osls[hdr["mode"]][hdr["metadata_snapshot_unix"]] = osl

    # Calibrate every capture, walking as far down the chain
    # (vna -> dut -> lna) as that DUT's switch topology allows.
    for key, s11s in uncaled_s11s.items():
        osl_mode = "rec" if key == "rec" else "ant"
        osl_bank = osls[osl_mode]
        if not osl_bank:
            raise ValueError(
                f"no usable internal-OSL captures found for mode "
                f"{osl_mode!r} -- can't calibrate {key!r}"
            )
        osl_times = np.array(list(osl_bank.keys()))
        for time, s11 in s11s.items():
            # use whichever internal-OSL set is closest in time
            time_diffs = np.abs(time - osl_times)
            nearest = osl_times[time_diffs == time_diffs.min()][0]
            osl = osl_bank[nearest]

            vna_sparams = calkit.network_sparams(osl_model, osl)
            vna_port = calkit.de_embed_sparams(vna_sparams, s11)
            caled_s11s[key][time]["vna"] = vna_port

            dut_port = None
            try:
                # de-embed the VNA-leg switch path from vna to dut
                dut_port = calkit.de_embed_sparams(
                    sparams=sparam_dict[DEEMBED_DICT[key][0]],
                    gamma_prime=vna_port,
                )
                caled_s11s[key][time]["dut"] = dut_port
            except IndexError:
                pass  # nothing to de-embed for this DUT
            try:
                # embed the RF/LNA-leg switch path from dut to lna
                lna_port = calkit.de_embed_sparams(
                    sparams=sparam_dict[EMBED_DICT[key][0]],
                    gamma_prime=dut_port,
                )
                caled_s11s[key][time]["lna"] = lna_port
            except IndexError:
                pass  # nothing to embed for this DUT

    return caled_s11s, freqs


def main(argv=None):
    parser = ArgumentParser(
        description=(
            "Calibrate a directory of raw field S11 h5 captures to "
            "the LNA reference plane (or as deep as each DUT "
            "supports) and write one HDF5 file per DUT."
        )
    )
    parser.add_argument(
        "datadir", type=Path, help="directory of raw S11 h5 files"
    )
    parser.add_argument(
        "switchpaths",
        type=Path,
        help="npz of lab-characterized switch-path S-parameters",
    )
    parser.add_argument(
        "osldata",
        type=Path,
        help="npz of characterized field OSL standards (+freqs)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("."),
        help="directory to write the per-DUT calibrated h5 files "
        "into (default: current directory)",
    )
    parser.add_argument(
        "--pattern",
        default="*.h5",
        help="glob pattern (relative to datadir) selecting input "
        "files (default: '*.h5')",
    )
    args = parser.parse_args(argv)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    caled_s11s, freqs = calibrate_field_s11(
        args.datadir,
        args.switchpaths,
        args.osldata,
        pattern=args.pattern,
    )

    # Each DUT's "default" alias should point at the deepest plane it
    # actually reaches (lna / dut / vna). write_dut_calibration_h5
    # only takes one final_plane per call, so group DUTs by their
    # deepest plane and call it once per group.
    by_final_plane = {}
    for key in caled_s11s:
        by_final_plane.setdefault(_deepest_plane(key), {})[key] = caled_s11s[
            key
        ]

    written = {}
    for final_plane, group in by_final_plane.items():
        written.update(
            write_dut_calibration_h5(
                group,
                save_dir=args.save_dir,
                freqs=freqs,
                final_plane=final_plane,
            )
        )

    print(f"wrote {len(written)} DUT file(s) to {args.save_dir}:")
    for dut, path in sorted(written.items()):
        print(f"  {dut}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
