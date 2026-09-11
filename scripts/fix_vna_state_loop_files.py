"""One-off migration for S11 h5 files written by an older
eigsep_observing/scripts/vna_state_loop.py.

That older version used raw switch-path names as data keys
(VNAANT/VNAAMB/VNASP1/VNANON/VNANOFF) and set
``header["mode"] = "vna_state_loop"``. Neither is understood by
``scripts/calibrate_field_s11.py``: it expects DUT names ("ant",
"amb", "sp1", "noise", "load", ...) as data keys, and needs
``mode`` to be ``"ant"`` or ``"rec"`` so it can find the matching
internal-OSL bank. Files written by the *current* vna_state_loop.py
already use the right keys/mode -- this script is only for data
collected before that fix.

Rewrites each matching file into ``--save-dir`` with data keys
renamed via STATE_TO_DUT and ``mode`` corrected to ``"ant"``.
Originals are left untouched. Any data key not in STATE_TO_DUT (e.g.
a custom switch path) passes through with its original name -- it
still won't be recognized by calibrate_field_s11.py, but nothing is
dropped silently.

A file recorded before OSL-per-cycle was added to vna_state_loop.py
carries no cal:VNAO/VNAS/VNAL at all; that isn't something this
script can fix (the data was never captured), and
calibrate_field_s11.py will legitimately fail to calibrate those
files with "no usable internal-OSL captures found".

Usage::

    python scripts/fix_vna_state_loop_files.py DATADIR
    python scripts/fix_vna_state_loop_files.py DATADIR --save-dir DATADIR/fixed
"""

from argparse import ArgumentParser
from pathlib import Path

from eigsep_observing import io

# Must match eigsep_observing/scripts/vna_state_loop.py's own mapping.
STATE_TO_DUT = {
    "VNAANT": "ant",
    "VNAAMB": "amb",
    "VNASP1": "sp1",
    "VNANON": "noise",
    "VNANOFF": "load",
}


def fix_file(path, save_dir):
    """Rewrite one file with DUT-name keys and mode="ant".

    Always remaps keys and forces the mode -- no "already compatible"
    shortcut, so a mismatch between what this function assumes and
    what a file actually contains can't hide behind a silent skip.
    Returns ``(out_path, before, after)`` where ``before``/``after``
    are ``(sorted(keys), mode)`` snapshots for the caller to report.

    Note: ``io.write_s11_file`` does not return the path it wrote (it
    returns ``None``) -- ``out_path`` is computed here from ``fname``/
    ``save_dir`` instead of trusting a return value.
    """
    data, cal_data, header, metadata = io.read_s11_file(path)
    before = (sorted(data), header.get("mode"))
    renamed = {STATE_TO_DUT.get(k, k): v for k, v in data.items()}
    header["mode"] = "ant"
    io.write_s11_file(
        renamed,
        header,
        metadata=metadata,
        cal_data=cal_data or None,
        fname=path.name,
        save_dir=save_dir,
    )
    out_path = Path(save_dir) / path.name
    after = (sorted(renamed), header["mode"])
    return out_path, before, after


def main(argv=None):
    parser = ArgumentParser(
        description=(
            "Rewrite old vna_state_loop.py output with "
            "calibrate_field_s11.py-compatible DUT-name keys and "
            "mode. Originals are left untouched."
        )
    )
    parser.add_argument(
        "datadir", type=Path, help="directory of raw h5 files to fix"
    )
    parser.add_argument(
        "--pattern",
        default="*.h5",
        help="glob pattern selecting input files (default: '*.h5')",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="output directory (default: DATADIR/fixed)",
    )
    args = parser.parse_args(argv)

    save_dir = args.save_dir or (args.datadir / "fixed")
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.datadir.glob(args.pattern))
    if not paths:
        raise SystemExit(
            f"no files matching {args.pattern!r} in {args.datadir}"
        )

    n_ok = 0
    n_noop = 0
    for path in paths:
        try:
            out, before, after = fix_file(path, save_dir)
        except Exception as exc:
            print(f"!! {path.name}: failed ({type(exc).__name__}: {exc})")
            continue
        before_keys, before_mode = before
        after_keys, after_mode = after
        if before == after:
            n_noop += 1
            print(
                f"{path.name} -> {out.name}  (no change needed: keys="
                f"{before_keys}, mode={before_mode!r})"
            )
        else:
            print(
                f"{path.name} -> {out.name}\n"
                f"    keys: {before_keys} -> {after_keys}\n"
                f"    mode: {before_mode!r} -> {after_mode!r}"
            )
        n_ok += 1
    print(
        f"processed {n_ok}/{len(paths)} file(s) ({n_noop} needed no "
        f"change); output in {save_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
