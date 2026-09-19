"""Build the B29 interactive RFI explorer notebook: point at a range of
campaign files, pull the corresponding B16 DPSS fit + flags/v2 mask, and
render the established 3-panel view (data | data-DPSS | v2-mask x
residual), generalized to an arbitrary file range instead of one fixed
case. Backing code: rfi_explorer_core.py (verified against the
eigsep_data package layout post repo-split), which stays in this repo's
marjum-2026-07/flagging/ -- the notebook itself lives in data-analysis/
(Aaron's direct instruction, 2026-09-17: interactive explorer notebooks
go next to beam_explorer.ipynb, not in a package/checkpoint directory),
a sibling repo, so it imports rfi_explorer_core via EIGSEP_ROOT rather
than a same-repo relative path."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
# data-analysis is a SEPARATE git repo, a sibling of this one under
# EIGSEP_ROOT, not nested inside this worktree -- a same-repo relative
# path would silently create a bogus directory inside a throwaway
# worktree instead (caught by inspecting the actual output location,
# not assumed correct from a successful os.makedirs() call).
EIGSEP_ROOT = os.environ.get("EIGSEP_ROOT", "/mnt/data02/eigsep")
OUT_NB = os.path.join(
    EIGSEP_ROOT, "data-analysis", "notebooks", "arp", "marjum-2026-07",
    "rfi_explorer.ipynb")
os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)

nb = nbf.v4.new_notebook()
C = []
M = nbf.v4.new_markdown_cell
X = nbf.v4.new_code_cell

C.append(M(r"""
# B29: interactive RFI explorer

Point at a range of campaign files and an antenna, and render the same
3-panel view already established in `dpss_smooth_model_and_residual_pca.ipynb`
/ the full-campaign checkpoint -- **data (log color) | data - DPSS model
(residual) | flags/v2 mask x residual (kept-only)** -- generalized to
work across an arbitrary file range instead of one fixed pilot day.

**Run this cell-by-cell top to bottom, then use the controls: pick a
day, drag the range slider to a start/end file within that day, pick an
antenna, click Render.**

### What you need

| | |
|---|---|
| Python packages | `numpy`, `matplotlib`, `h5py`, `ipywidgets` (JupyterLab/Notebook), `eigsep_data` (must be installed -- see `marjum-2026-07/flagging/rfi_explorer_core.py`'s own note on this new coupling) |
| Data | the real campaign tree at `MARJUM_DATA_ROOT` (defaults to `EIGSEP_ROOT/marjum-2026-07`) -- `data/*.h5`, `derived/smooth_model/v0/*.h5` (B16 fit), `flags/v2/*.h5` (mask) |
| Repo layout | this notebook lives in `data-analysis/`; its backing code (`rfi_explorer_core.py`) stays in the sibling `eigsep`/`marjum-2026-07/flagging/` checkout -- set `EIGSEP_ROOT` if that checkout isn't at `/mnt/data02/eigsep` |

**Antenna resolution is transparent, not literal:** "gnd" tries raw key
`0` first, falling back to `3` if `0` isn't live for a given file (same
physical box-gnd, a different SNAP input during campaign Phase
A(late)/B). "air" is always raw key `4`. A file with neither live for
the selected antenna is silently skipped from the range (reported in
the used/skipped counts below the plot), not an error.

**Known limitation, stated plainly rather than discovered later:** this
notebook's live-widget behavior (dragging the range slider, clicking
Render, watching the plot update) can only be verified by a human
actually running it in Jupyter. A static, re-executed export (like the
PDF this checkpoint is delivered as) can only show the plot at
whatever selection was rendered before export -- it cannot demonstrate
the interaction itself. What *is* verified here, the same way as every
other deliverable this week: the underlying data-loading and plotting
functions, executed for real against real data, producing a real,
checkable static snapshot below.
"""))

C.append(X(r"""
import sys, os
EIGSEP_ROOT = os.environ.get("EIGSEP_ROOT", "/mnt/data02/eigsep")
sys.path.insert(0, os.path.join(EIGSEP_ROOT, "marjum-2026-07", "flagging"))
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display

import rfi_explorer_core as RC

ALL_FILES = RC.list_files()
print(f"{len(ALL_FILES)} campaign files available, "
      f"{ALL_FILES[0]} .. {ALL_FILES[-1]}")

DAYS = sorted(set(f[5:13] for f in ALL_FILES))
print("days:", DAYS)
"""))

C.append(M(r"""
## Controls
"""))

C.append(X(r"""
def files_for_day(day):
    return [f for f in ALL_FILES if f[5:13] == day]


def label_for(fname):
    # HH:MM:SS from the filename's own close-time encoding
    hhmmss = fname[14:20]
    return f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}  ({fname})"


day_w = W.Dropdown(options=DAYS, value=DAYS[-1], description="day")
_day_files = files_for_day(day_w.value)
range_w = W.SelectionRangeSlider(
    options=[(label_for(f), f) for f in _day_files],
    index=(0, min(9, len(_day_files) - 1)),
    description="range", layout=W.Layout(width="700px"),
)
antenna_w = W.Dropdown(options=[("box-gnd", "gnd"), ("box-air", "air")],
                        value="air", description="antenna")
render_btn = W.Button(description="Render", button_style="primary")
out = W.Output()


def on_day_change(change):
    day_files = files_for_day(day_w.value)
    range_w.options = [(label_for(f), f) for f in day_files]
    range_w.index = (0, min(9, len(day_files) - 1))


day_w.observe(on_day_change, names="value")


def on_render_click(_b=None):
    start_fname, end_fname = range_w.value
    antenna = antenna_w.value
    with out:
        out.clear_output(wait=True)
        try:
            result = RC.load_range(start_fname, end_fname, antenna, ALL_FILES)
        except ValueError as e:
            print(f"Nothing to show: {e}")
            return
        fig = RC.plot_three_panel(
            result, antenna,
            title_extra=f", {start_fname}..{end_fname}")
        plt.show()
        print(f"used {len(result['used_files'])} files, "
              f"skipped {len(result['skipped_files'])} "
              f"(no live '{antenna}' data or no B16 companion)")


render_btn.on_click(on_render_click)

display(W.VBox([W.HBox([day_w, antenna_w]), range_w, render_btn, out]))
"""))

C.append(M(r"""
## Static default-state snapshot (this is what a re-executed export can show)

Same range/antenna as the pilot notebook's own pilot window, rendered
directly (not via the button, so this cell always produces real output
when the notebook is re-executed non-interactively).
"""))

C.append(X(r"""
default_start, default_end = "corr_20260717_200029Z.h5", "corr_20260717_201948Z.h5"
default_antenna = "air"
result = RC.load_range(default_start, default_end, default_antenna, ALL_FILES)
fig = RC.plot_three_panel(result, default_antenna,
                           title_extra=f", {default_start}..{default_end} (default snapshot)")
plt.show()
print(f"used {len(result['used_files'])} files, skipped {len(result['skipped_files'])}")
"""))

C.append(M(r"""
## Reading the panels

Same convention as the full-campaign checkpoint and the DPSS pilot
notebook: **panel 1** (data, log color) shows the raw band shape and
every comb/RFI feature at full contrast; **panel 2** (data - DPSS model)
is the residual the smooth-band fit leaves behind; **panel 3**
(flags/v2 mask x residual, kept-only) blanks anything v0's category
bits, the static self-comb exclusion, or B16's DPSS-residual-outlier
bit called bad, so what remains is what the pipeline currently
believes is trustworthy signal.

**A file gets silently skipped from a range** if the selected antenna
has no live raw data for it, or if it has no B16 companion in
`derived/smooth_model/v0/` (not yet fit, or genuinely all-zero for
that antenna -- see this week's B16 full-campaign checkpoint for the
full accounting of why files go missing). The used/skipped counts
below each render report this directly rather than silently narrowing
the range.

**STOPPED AT REVIEW GATE — awaiting Aaron's approval.**
"""))

nb["cells"] = C
nb["metadata"] = {
    "kernelspec": {"display_name": "python3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT_NB)
