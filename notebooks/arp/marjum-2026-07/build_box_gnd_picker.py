#!/usr/bin/env python3
"""Generate box_gnd_picker.ipynb, the interactive picker for box-gnd pixels.

The notebook writes hand picks of box-gnd to
memos/memo-002-marjum-2026-07-geometry/box_gnd_picks.json, which the memo's
box-gnd triangulation reads. It fits nothing itself.
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# box-gnd picker (memo 002)

Pick box-gnd in IMG_2157, 2187, 2189, 2191, 2194 and 2246. Picks go to
`memos/memo-002-marjum-2026-07-geometry/box_gnd_picks.json` in the eigsep
repository (`$EIGSEP_REPO`, default: the directory above this data-analysis
checkout, else `~/projects/global21cm/eigsep`) as soon as you
click, in **displayed-image pixels** (EXIF orientation applied, x right,
y down from the top row). The memo's script converts them to the fit
convention (y_fit = H - y).

Points:

| name | what to click | needed |
|---|---|---|
| `feed` | box centre, the same reference point as the box-air `ant_px` labels | yes |
| `arm_a` | tip of the dipole arm with the pink flagging | optional |
| `arm_b` | the opposite tip of that same dipole | optional |

`arm_a` and `arm_b` give the pink-flagged dipole's axis and the bearing from the
box centre toward the pink end.

How to use: run all cells. In each image, use the toolbar's zoom and pan to
find the box, choose the point name with the buttons, then **double-click**
to place it (a single click or a drag is left to zoom and pan). Double-click
again to move it, and use "clear" to remove it. If a fitted position exists
(`box_gnd_fit.json` beside the picks), it is drawn as an orange x for comparison.

Needs `ipympl` (`%matplotlib widget`). HEIC files are read with
`pillow_heif` when it is installed and otherwise converted with ImageMagick.
""")

code(r"""
%matplotlib widget
import json, os, subprocess, tempfile, datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
from PIL import Image, ImageOps

_DA = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
REPO = Path(os.environ.get("EIGSEP_REPO") or next(
    (p for p in (_DA.parent, Path.home() / "projects/global21cm/eigsep") if (p / "memos").is_dir())))
MEMO = REPO / "memos" / "memo-002-marjum-2026-07-geometry"
PICKS = MEMO / "box_gnd_picks.json"
FIT = MEMO / "box_gnd_fit.json"
CAMPAIGN = Path(os.environ.get("EIGSEP_CAMPAIGN_ROOT", REPO / "marjum-2026-07"))
# where the HEICs are; falls back to ~/Downloads if the campaign copy is absent
IMG_DIRS = [CAMPAIGN / "imgs", Path.home() / "projects/global21cm/eigsep/marjum-2026-07/imgs",
            Path.home() / "Downloads"]
KEYS = ["2157", "2187", "2189", "2191", "2194", "2246"]
POINTS = ["feed", "arm_a", "arm_b"]
COLORS = {"feed": "lime", "arm_a": "cyan", "arm_b": "magenta"}
# initial view (x0, x1, y0, y1) in displayed pixels; None = whole frame
ZOOM = {"2189": (900, 1600, 2000, 1450)}
PICKER = "Aaron"


def find(key):
    for d in IMG_DIRS:
        p = d / f"IMG_{key}.HEIC"
        if p.exists():
            return p
    raise FileNotFoundError(f"IMG_{key}.HEIC not in {IMG_DIRS}")


def load(path):
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    except ImportError:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "img.png"
            subprocess.run(["magick", str(path), "-auto-orient", str(out)], check=True)
            return np.array(Image.open(out).convert("RGB"))


def read_picks():
    return json.load(open(PICKS)) if PICKS.exists() else {
        "description": "box-gnd hand picks; displayed-image pixels (EXIF orientation applied), "
                       "x right, y down from the top row; y_fit = height - y",
        "images": {}}


def write_picks(p):
    PICKS.write_text(json.dumps(p, indent=1, sort_keys=True) + "\n")


print("eigsep repo:", REPO, "| branch:", subprocess.run(["git", "-C", str(REPO), "branch", "--show-current"],
      capture_output=True, text=True).stdout.strip())
print("picks file:", PICKS)
print({k: str(find(k)) for k in KEYS})
""")

code(r"""
def picker(key):
    img = load(find(key))
    H, Wd = img.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 6.8))
    fig.canvas.header_visible = False
    ax.imshow(img)
    if ZOOM.get(key):
        x0, x1, y0, y1 = ZOOM[key]
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_title(f"IMG_{key}: double-click to place the selected point")
    marks = {}
    which = W.ToggleButtons(options=POINTS, value="feed", description="point")
    clear = W.Button(description="clear selected")
    status = W.Label()

    def draw():
        for m in marks.values():
            m.remove()
        marks.clear()
        pk = read_picks()["images"].get(key, {}).get("points", {})
        for name, xy in pk.items():
            marks[name] = ax.plot(*xy, "+", ms=18, mew=2, color=COLORS[name])[0]
            marks[name + "_t"] = ax.annotate(name, xy, xytext=(8, 8), textcoords="offset points",
                                             color=COLORS[name], fontsize=9)
        if FIT.exists():
            fit = json.load(open(FIT)).get("projected_px", {}).get(key, {})
            for name, xy in fit.items():
                marks["fit_" + name] = ax.plot(*xy, "x", ms=12, mew=2, color="orange")[0]
        status.value = f"IMG_{key}: " + (", ".join(f"{n}=({v[0]:.1f}, {v[1]:.1f})" for n, v in pk.items()) or "no picks")
        fig.canvas.draw_idle()

    def onclick(ev):
        if ev.inaxes is not ax or not ev.dblclick:
            return
        p = read_picks()
        rec = p["images"].setdefault(key, {"width": Wd, "height": H, "points": {}})
        rec["points"][which.value] = [round(float(ev.xdata), 1), round(float(ev.ydata), 1)]
        rec["picked_by"] = PICKER
        rec["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        write_picks(p)
        draw()

    def onclear(_):
        p = read_picks()
        p["images"].get(key, {}).get("points", {}).pop(which.value, None)
        write_picks(p)
        draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    clear.on_click(onclear)
    draw()
    return W.VBox([W.HBox([which, clear]), status, fig.canvas])
""")

for k in ["2157", "2187", "2189", "2191", "2194", "2246"]:
    md(f"## IMG_{k}")
    code(f'picker("{k}")')

md("## Current picks")
code(r"""print(json.dumps(read_picks(), indent=1))""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "box_gnd_picker.ipynb")
print("wrote", HERE / "box_gnd_picker.ipynb")
