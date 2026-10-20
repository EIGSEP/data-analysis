"""Build ``derived/noise_source/<version>/`` from the 2026-10-08 bench run.

The measurement is ``eigsep_data.noise_source_bench.measure`` and the
product format ``eigsep_data.products.noise_source.write``; this script
holds only what belongs to this run and this campaign: where its files are,
how the bench was wired, the thresholds, and the provenance. The run's
as-built record is ``notebooks/charlie/bench_runs/2026-10-08_interim/``.

Run::

    EIGSEP_CAMPAIGN_ROOT=/mnt/data02/eigsep/marjum-2026-07 \\
        python build_noise_source.py v0001 --products <campaign tree to write into>

Inputs read from the campaign tree given by ``--data`` (default the
campaign root): the run's correlator and S11 files under
``data/bench/2026-10-08_interim/`` (untracked bulk; SHA-256 recorded) and
the lab measurement of the noise-source path's reflection at P,
``data/s11/cal_materials/rfn_sparameters_10_15.npz`` (looked for under
``--products`` first, where it may be tracked on a branch).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eigsep_cal
import eigsep_data
from eigsep_cal import S11
from eigsep_data import MetadataIndex
from eigsep_data.noise_source_bench import BenchSetup, measure
from eigsep_data.products import noise_source

#: The bench run and its as-built record.
RUN = "2026-10-08_interim"
RUN_RECORD = f"data-analysis/notebooks/charlie/bench_runs/{RUN}/README.md"
DATA_SUBDIR = f"data/bench/{RUN}"
GAMMA_NS_FILE = "data/s11/cal_materials/rfn_sparameters_10_15.npz"

#: How the bench was wired (run record, sections 2-3): the hot load on the
#: Amb/Hot Load port (RFAMB, thermistor tempctrl_load), the ambient
#: reference on the Feed port (RFANT, thermistor tempctrl_lna2 on GP27);
#: pull-ups measured 2-wire. The noise diode's board thermistor is
#: rfswitch_therm temp_therm2 (GP28) on this board.
SETUP = BenchSetup(
    hot="RFAMB", ref="RFANT", non="RFNON", noff="RFNOFF",
    hot_s11="amb", ref_s11="ant",
    hot_therm=("tempctrl_load_resistance", 9980.0),
    ref_therm=("tempctrl_lna2_resistance", 9970.0),
    board_therm_c="rfswitch_therm_temp_therm2",
    antenna="box-air",
)

#: Thresholds. A cycle is fitted only if the hot-load thermistor changes by
#: less than STEADY_PER_MIN (K/min): it lags the load while the heater
#: ramps, and on this hardware ~1 K/min is what a plateau allows; heating
#: and cooling cycles are fitted separately as a check. Rows from 20 s
#: before to 5 s after each VNA sweep are excluded (the VNA service
#: disturbs every state there). The mask's thresholds: fit residuals more
#: than 3x the in-band median in radiometer units, or a slope error over
#: 5 %. The smooth fit of the excess: a cubic in frequency.
STEADY_PER_MIN = 1.0
SWEEP_GUARD_S = (20.0, 5.0)
BAND_MHZ = (50.0, 245.0)
CHI2_FACTOR = 3.0
MAX_REL_ERR = 0.05
SMOOTH_DEG = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(path, scope=None):
    """Commit, branch and dirtiness of the repository holding ``path``.

    ``scope`` limits the dirtiness check to one directory. For data-analysis
    it is this script's own: notebooks elsewhere that were committed with
    outputs always read as modified under the repository's nbstripout filter,
    and none of them is code this build runs."""
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], capture_output=True,
                                    text=True).stdout.strip()
    commit = run("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "note": f"{path} is not in a git checkout (installed package)"}
    status = ["status", "--porcelain", "--untracked-files=no"] + (["--", str(scope)] if scope else [])
    info = {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run(*status))}
    if scope:
        info["dirty_checked_in"] = str(Path(scope).relative_to(run("rev-parse", "--show-toplevel")))
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("version", help="product version to write, e.g. v0001")
    p.add_argument("--data", type=Path, default=None,
                   help="campaign tree holding the inputs (default: the campaign root)")
    p.add_argument("--products", type=Path, default=None,
                   help="campaign tree to write derived/noise_source/ into (default: --data)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    data_root = a.data or Path(eigsep_data.get_campaign_root(required=True))
    out_root = a.products or data_root
    run_dir = data_root / DATA_SUBDIR
    gamma_path = next(p for p in (out_root / GAMMA_NS_FILE, data_root / GAMMA_NS_FILE)
                      if p.is_file())

    # every stream: the thermistor resistances are not among the index's curated fields
    index = MetadataIndex(str(run_dir), cache=False, streams="all")
    s11 = S11(str(run_dir), year=2026, pattern="*s11_*.h5")
    s11.calibrate()
    with np.load(gamma_path) as g:
        gamma = {"freqs_mhz": g["freqs"] / 1e6, "on": g["RFNON"], "off": g["RFNOFF"]}

    res = measure(index, s11, gamma, SETUP, steady_per_min=STEADY_PER_MIN,
                  sweep_guard=SWEEP_GUARD_S, band_mhz=BAND_MHZ,
                  chi2_factor=CHI2_FACTOR, max_rel_err=MAX_REL_ERR)

    files = sorted(run_dir.glob("*.h5"))
    digest = hashlib.sha256()
    file_sums = {}
    for f in files:
        file_sums[f.name] = sha256(f)
        digest.update(f"{f.name} {file_sums[f.name]}\n".encode())
    here = Path(__file__).resolve()
    provenance = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "data-analysis/scripts/marjum-2026-07/noise_source/build_noise_source.py",
        "argv": sys.argv[1:],
        "bench_run": RUN,
        "run_record": RUN_RECORD,
        "setup": {k: (list(v) if isinstance(v, tuple) else v) for k, v in SETUP.__dict__.items()},
        "choices": {"steady_per_min": STEADY_PER_MIN, "sweep_guard_s": list(SWEEP_GUARD_S),
                    "band_mhz": list(BAND_MHZ), "chi2_factor": CHI2_FACTOR,
                    "max_rel_err": MAX_REL_ERR, "smooth_deg": SMOOTH_DEG},
        "inputs": {
            "bench_files": {"dir": DATA_SUBDIR, "n_files": len(files),
                            "digest_of_sha256s": digest.hexdigest(), "sha256": file_sums},
            "gamma_ns": {"path": GAMMA_NS_FILE, "sha256": sha256(gamma_path)},
        },
        "code": {
            "data-analysis": git_info(here.parent, scope=here.parent),
            "eigsep_data": git_info(Path(eigsep_data.__file__).parent),
            "eigsep_cal": git_info(Path(eigsep_cal.__file__).parent),
        },
        "model": "eigsep_cal minimal tier: no receiver noise waves, no switch-path loss; "
                 "T_NON receiver-independent with the lab Gamma_NS at P",
    }
    d = noise_source.write(out_root, a.version, freqs=res["freqs"], t_non_k=res["t_non_k"],
                           t_non_sigma_k=res["t_non_sigma_k"], t_off_k=res["t_off_k"],
                           good=res["good"], gamma_ns=res["gamma_ns"], provenance=provenance,
                           validation=res["validation"], smooth_deg=SMOOTH_DEG,
                           overwrite=a.overwrite)
    print(f"wrote {d}")
    print(json.dumps(res["validation"], indent=1, default=float))


if __name__ == "__main__":
    main()
