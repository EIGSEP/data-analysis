"""The Sun's path through box-air's beam over phase C, minute by minute.

Builds ``derived/ground_sky/sun_profile_vNNNN``: for every minute of phase C
with pointing, the Sun column (K per SFU: beam gain toward the Sun ×
knife-edge visibility at the ridge, and without the terrain) at six HFSS beam
frequencies, with the pointing and whether the antenna moved. Pointing is the
minute's median of ``pointing@v2.0`` rows of quality ok or suspect. In a
parked block of ``curation/mode_table.jsonl``, minutes with no such rows take
the block's median. Outside the 07-17 raster the table's azimuth zero inherits
potentiometer slips (memo 003), so the Sun's gain on other days carries that
uncertainty.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python sun_profile.py v0001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eigsep_base.rotations import mount_rotation
from eigsep_sim.design_matrix import HealpixBeam

import sun as sunmod
from common import ERAS, HERE, campaign_root, git_rev, sha256, workspace_root

FREQS = (50.78125, 74.21875, 101.5625, 148.4375, 199.21875, 234.375)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--start", default="2026-07-15T00:30:00Z")
    ap.add_argument("--stop", default="2026-07-18T03:30:00Z")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"sun_profile_{args.version}"
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; versions are immutable")
    t0, t1 = pd.Timestamp(args.start).timestamp(), pd.Timestamp(args.stop).timestamp()
    p = pd.read_parquet(campaign / "curation/pointing_table.parquet",
                        columns=["t_utc_s", "az_deg", "el_deg", "quality"])
    p = p[(p.t_utc_s >= t0) & (p.t_utc_s < t1) & p.quality.isin(["ok", "suspect"])]
    p = p.dropna(subset=["az_deg", "el_deg"])
    p["az"] = np.mod(p.az_deg + 180, 360) - 180
    p["m"] = (p.t_utc_s // 60).astype(int)
    g = p.groupby("m").agg(az=("az", "median"), el=("el_deg", "median"),
                           az_ptp=("az", np.ptp), el_ptp=("el_deg", np.ptp), n=("az", "size"))
    g = g[g.n >= 20].assign(filled=False)
    modes = pd.DataFrame([json.loads(line) for line in open(campaign / "curation/mode_table.jsonl")])
    modes = modes[modes.get("file_first").notna()]
    modes["a"] = pd.to_datetime(modes.t_start_utc).astype("int64") / 1e9
    modes["b"] = pd.to_datetime(modes.t_end_utc).astype("int64") / 1e9
    add = []
    for _, r in modes[(modes.rot_state == "parked") & (modes.b > t0) & (modes.a < t1)].iterrows():
        mins = np.arange(int(r.a // 60), int(r.b // 60) + 1)
        have = g.index.intersection(mins)
        miss = np.setdiff1d(mins, g.index.values)
        if len(have) and len(miss):
            med = g.loc[have, ["az", "el"]].median()
            add.append(pd.DataFrame({"az": med.az, "el": med.el, "az_ptp": 0.0, "el_ptp": 0.0,
                                     "n": 0, "filled": True}, index=miss))
    if add:
        g = pd.concat([g] + add).sort_index()
        g = g[~g.index.duplicated()]
    t = g.index.values * 60.0 + 30
    era = np.full(len(t), "", dtype=object)
    for _, r in modes.iterrows():
        era[(t >= r.a - 60) & (t <= r.b + 60)] = r.height_era
    keep = np.isin(era, list(ERAS))
    t, g, era = t[keep], g[keep], era[keep]
    hb = HealpixBeam.from_npz(workspace_root() / "data-analysis/hfss_beam_maps/bowtie_beam.npz",
                              drop_last=True)
    bf = hb.freqs_hz / 1e6
    beam = hb.select([int(np.argmin(np.abs(bf - x))) for x in FREQS])
    with np.load(campaign / "derived/beam/empirical_raster_v0012/dpss/diagnostics.npz") as d:
        az_off, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    keys = np.array([ERAS[e] for e in era], dtype=object)
    geom = sunmod.sun_geometry(t, keys)
    rot = mount_rotation(g.az.values + az_off, g.el.values, psi)
    out_dir.mkdir(parents=True)
    np.savez_compressed(
        out_dir / "sun_profile.npz", t=t, freqs_mhz=np.array(FREQS), az=g.az.values,
        el=g.el.values, az_ptp=g.az_ptp.values, el_ptp=g.el_ptp.values,
        filled=g.filled.values, era=keys.astype(str),
        col_knife=sunmod.sun_column(beam, rot, geom, "knife").astype(np.float32),
        col_flat=sunmod.sun_column(beam, rot, geom, "flat").astype(np.float32),
        sun_alt=geom["alt"], sun_az=geom["az"], theta=geom["theta_deg"])
    (out_dir / "manifest.json").write_text(json.dumps({"provenance": {
        "product": "ground_sky_sun_profile", "version": args.version,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "data-analysis/scripts/marjum-2026-07/ground_sky/sun_profile.py",
        "argv": sys.argv[1:], "code": {"data-analysis": git_rev(HERE)},
        "inputs": {"pointing_table_sha256": sha256(campaign / "curation/pointing_table.parquet"),
                   "mode_table_sha256": sha256(campaign / "curation/mode_table.jsonl")}},
        "minutes": int(len(t)), "minutes_filled_from_parked_block": int(g.filled.sum())}, indent=1))


if __name__ == "__main__":
    main()
