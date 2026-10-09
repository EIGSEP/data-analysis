"""Create a camera-state NPZ containing only explicitly accepted additions."""

import argparse
import json
from pathlib import Path

import numpy as np

from marjum_mcmc import digest


CAMERA_FIELDS = ("keys", "cameras", "distortion", "groups", "shapes")


def select(source, candidate, accepted, output):
    source_path, candidate_path = Path(source), Path(candidate)
    out = Path(output);out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError("Use a new output directory")
    base = dict(np.load(source_path));trial = dict(np.load(candidate_path))
    base_keys = [str(k) for k in base["keys"]]
    trial_keys = [str(k) for k in trial["keys"]]
    accepted = [str(k) for k in accepted]
    missing = sorted(set(accepted).difference(trial_keys))
    if missing:raise KeyError(f"Candidate fit lacks {missing}")
    if not np.array_equal(base["keys"], trial["keys"][:len(base_keys)]):
        raise ValueError("Candidate does not preserve the source camera prefix")
    keep = np.array([trial_keys.index(k) for k in base_keys + accepted], int)
    state = dict(trial)
    for field in CAMERA_FIELDS:
        state[field] = trial[field][keep]
    np.savez_compressed(out / "fit_selected.npz", **state)
    rejected = [k for k in trial_keys[len(base_keys):] if k not in accepted]
    report = dict(
        source=str(source_path), candidate=str(candidate_path),
        accepted=accepted, rejected=rejected, keys=[str(k) for k in state["keys"]],
        source_sha256=digest(source_path), candidate_sha256=digest(candidate_path),
        source_geometry_preserved=all(np.array_equal(base[f],state[f][:len(base_keys)])
                                      for f in CAMERA_FIELDS),
    )
    (out / "selection.json").write_text(json.dumps(report,indent=2)+"\n")
    return state,report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True);parser.add_argument("--candidate",required=True)
    parser.add_argument("--accepted",nargs="+",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();select(**vars(args))
