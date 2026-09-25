#!/usr/bin/env python3
"""Check that a notebook actually ran, not merely that it has no error cells.

`jupyter nbconvert --execute` can exit 0 having written a notebook whose cells
were never executed -- if the kernel dies on an early cell the traceback goes
to the console and the saved cells keep `execution_count: None` with no
outputs. Checking only for `output_type == "error"` then reports a clean run on
a notebook that did nothing, which is exactly what happened here.

So the check is: every code cell has an execution_count, none carry an error,
and the expected number of figures is present.

    python verify_notebook.py <notebook.ipynb> [min_figures]
"""
import json
import sys


def main(path, min_figures=0):
    nb = json.load(open(path))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    unrun = [i for i, c in enumerate(nb["cells"])
             if c["cell_type"] == "code" and c.get("execution_count") is None]
    errors = [(i, o["ename"], o["evalue"])
              for i, c in enumerate(nb["cells"])
              for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    figures = sum(1 for c in nb["cells"] for o in c.get("outputs", [])
                  if "image/png" in (o.get("data") or {}))

    print(f"{path}: {len(code)} code cells, {len(code) - len(unrun)} executed, "
          f"{figures} figures, {len(errors)} errors")
    ok = True
    if unrun:
        print(f"  NOT EXECUTED: cells {unrun[:20]}")
        ok = False
    for i, name, val in errors:
        print(f"  ERROR cell {i}: {name}: {val}")
        ok = False
    if figures < min_figures:
        print(f"  too few figures: {figures} < {min_figures}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    nbp = sys.argv[1] if len(sys.argv) > 1 else "rfi_flag_prototype.ipynb"
    minf = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sys.exit(main(nbp, minf))
