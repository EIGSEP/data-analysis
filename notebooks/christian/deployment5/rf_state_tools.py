"""Browse/load deployment-5 filtered corr files gated on RF-switch state.

The sidecar ``rfswitch_index.npz`` (built 2026-08-12 from ``metadata/rfswitch``)
holds one state code per integration for every readable file, in true time
order (filename time, acc_cnt tiebreak). States:

    RFANT  antenna          RFAMB  ambient load     RFNON  noise source
    VNAS/VNAL/VNARF/VNAO    VNA cal states
    UNKNOWN                 no/short metadata, None entries, transition rows

1842 files carry no rfswitch metadata at all (most of Jul 15) -- their rows
are UNKNOWN, so state="RFANT" silently excludes them. Switch transitions are
buffered by ~2 UNKNOWN rows, so gated rows are clean single-state data.
"""

import fnmatch
import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered"
INDEX = os.path.join(HERE, "rfswitch_index.npz")


class StateIndex:
    def __init__(self, path=INDEX):
        z = np.load(path)
        self.states = list(z["states"])
        self.names = z["names"]
        self.offsets = z["offsets"]
        self.codes = z["codes"]
        self.fname_t = z["fname_t"]

    def rows(self, i, state):
        """Within-file row numbers of `state` for file i."""
        c = self.codes[self.offsets[i] : self.offsets[i + 1]]
        return np.flatnonzero(c == self.states.index(state))

    def files_with(self, state, min_rows=1, patterns=None):
        """Index positions of files with >= min_rows of `state`.

        patterns: optional list of filename globs (e.g. ["corr_20260717*"]).
        """
        code = self.states.index(state)
        keep = []
        for i in range(len(self.names)):
            if patterns is not None and not any(
                fnmatch.fnmatch(str(self.names[i]), p) for p in patterns
            ):
                continue
            c = self.codes[self.offsets[i] : self.offsets[i + 1]]
            if (c == code).sum() >= min_rows:
                keep.append(i)
        return np.array(keep, dtype=int)


def load_gated(state, patterns=None, keys=None, data_dir=DATA_DIR, index=None,
               with_t_load=False):
    """All integrations of `state` across matching files.

    Returns dict with:
      spec    {key: (n, 1024) float32}   autos only (keys present in each file)
      t       (n,) filename unix time (file-close; all rows in a file share it)
      int_t   (n,) integration time [s]
      file_i  (n,) index-position of source file
      t_load  (n,) load thermistor T_now [C] (NaN where absent), if with_t_load
      freqs   (1024,) MHz
    """
    idx = index or StateIndex()
    sel = idx.files_with(state, patterns=patterns)
    if not len(sel):
        raise ValueError(f"no files matching patterns with state {state}")
    if keys is None:
        keys = ["0", "1", "2", "3", "4", "5"]
    out = {k: [] for k in keys}
    t, int_t, file_i, t_load = [], [], [], []
    freqs = None
    for n, i in enumerate(sel):
        rows = idx.rows(i, state)
        fn = os.path.join(data_dir, str(idx.names[i]))
        try:
            with h5py.File(fn, "r") as f:
                if freqs is None:
                    freqs = f["header/freqs"][:]
                it = float(f["header"].attrs["integration_time"])
                present = [k for k in keys if k in f["data"] and len(k) == 1]
                for k in present:
                    out[k].append(f["data"][k][:][rows].astype(np.float32))
                for k in keys:
                    if k not in present:
                        out[k].append(np.full((len(rows), 1024), np.nan, np.float32))
                if with_t_load:
                    tl = np.full(len(rows), np.nan)
                    if "metadata/tempctrl_load" in f:
                        raw = f["metadata/tempctrl_load"][()]
                        if isinstance(raw, bytes):
                            raw = raw.decode()
                        recs = json.loads(raw)
                        for j, r in enumerate(rows):
                            if r < len(recs) and isinstance(recs[r], dict):
                                v = recs[r].get("T_now")
                                if v is not None:
                                    tl[j] = v
                    t_load.append(tl)
        except Exception:
            continue
        t.append(np.full(len(rows), idx.fname_t[i]))
        int_t.append(np.full(len(rows), it))
        file_i.append(np.full(len(rows), i, dtype=int))
        if n % 20 == 0:
            print(f"\r{n}/{len(sel)} files", end="", flush=True)
    print(f"\r{len(sel)}/{len(sel)} files")
    res = {
        "spec": {k: np.concatenate(v) for k, v in out.items()
                 if v and not np.isnan(np.concatenate(v)).all()},
        "t": np.concatenate(t),
        "int_t": np.concatenate(int_t),
        "file_i": np.concatenate(file_i),
        "freqs": freqs,
    }
    if with_t_load:
        res["t_load"] = np.concatenate(t_load)
    return res


def visits(t, gap_s=600):
    """Group row times into contiguous visits: returns (n,) visit id."""
    v = np.zeros(len(t), dtype=int)
    if len(t):
        v[1:] = np.cumsum(np.diff(t) > gap_s)
    return v


class StateBrowser:
    """Flip through per-file autocorr data gated on switch state.

    Usage in a ``%matplotlib widget`` notebook::

        b = StateBrowser(state="RFANT")           # or RFAMB / RFNON / ...
        b = StateBrowser(state="RFAMB", keys=["0", "4"], patterns=["corr_20260717*"])

    Slider / prev-next / Play flip through files containing the state;
    top panel = waterfall of the gated rows (chosen key), bottom = mean
    spectrum per key with a 10-90 percentile envelope.
    """

    def __init__(self, state="RFANT", keys=("0", "4"), patterns=None,
                 data_dir=DATA_DIR, min_rows=1, vmin=4, vmax=6.6, show=True):
        import matplotlib.pyplot as plt

        self.idx = StateIndex()
        self.data_dir = data_dir
        self.keys = list(keys)
        self.min_rows = min_rows
        self.patterns = patterns
        self.state = state
        self.vmin, self.vmax = vmin, vmax
        self.sel = self.idx.files_with(state, min_rows, patterns)
        if not len(self.sel):
            raise ValueError(f"no files with >= {min_rows} rows of {state}")
        with h5py.File(os.path.join(data_dir, str(self.idx.names[self.sel[0]])), "r") as f:
            self.freqs = f["header/freqs"][:]

        self.fig, (self.ax_w, self.ax_s) = plt.subplots(
            2, 1, figsize=(9, 7), layout="constrained",
            gridspec_kw={"height_ratios": [1, 1.4]})
        self.im = None
        self._build_controls() if show else None
        self.goto(0)

    # ---- data ----
    def _read(self, pos):
        i = self.sel[pos]
        rows = self.idx.rows(i, self.state)
        fn = os.path.join(self.data_dir, str(self.idx.names[i]))
        spec = {}
        with h5py.File(fn, "r") as f:
            for k in self.keys:
                if k in f["data"] and len(k) == 1:
                    spec[k] = f["data"][k][:][rows].astype(np.float32)
        return i, rows, spec

    # ---- drawing ----
    def goto(self, pos):
        import matplotlib.pyplot as plt

        self.pos = int(np.clip(pos, 0, len(self.sel) - 1))
        i, rows, spec = self._read(self.pos)
        name = str(self.idx.names[i])
        tstr = datetime.fromtimestamp(
            self.idx.fname_t[i], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        self.ax_w.clear()
        self.ax_s.clear()
        wkey = next((k for k in self.keys if k in spec), None)
        if wkey is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.log10(np.abs(spec[wkey]))
            self.im = self.ax_w.imshow(
                img, aspect="auto", cmap="plasma", interpolation="none",
                vmin=self.vmin, vmax=self.vmax,
                extent=[self.freqs.min(), self.freqs.max(), len(rows), 0])
            self.ax_w.set_ylabel(f"{self.state} row (key {wkey})")
        self.ax_w.set_title(
            f"[{self.pos + 1}/{len(self.sel)}] {name}  {tstr} UTC  "
            f"{len(rows)} x {self.state}", fontsize=10)

        for k, d in spec.items():
            m = np.nanmean(d, axis=0)
            lo, hi = np.nanpercentile(d, [10, 90], axis=0)
            (ln,) = self.ax_s.plot(self.freqs, m, lw=1, label=f"key {k}")
            self.ax_s.fill_between(self.freqs, lo, hi, alpha=0.25,
                                   color=ln.get_color(), lw=0)
        self.ax_s.set_yscale("log")
        self.ax_s.set_xlabel("Frequency [MHz]")
        self.ax_s.set_ylabel("Power [counts]")
        self.ax_s.set_ylim(10 ** self.vmin / 30, 10 ** self.vmax * 3)
        self.ax_s.legend(loc="upper right", fontsize=8)
        self.fig.canvas.draw_idle()

    # ---- widgets ----
    def _build_controls(self):
        import ipywidgets as w
        from IPython.display import display

        self.slider = w.IntSlider(0, 0, len(self.sel) - 1, description="file",
                                  layout=w.Layout(width="55%"))
        play = w.Play(interval=400, min=0, max=len(self.sel) - 1, step=1)
        w.jslink((play, "value"), (self.slider, "value"))
        prev = w.Button(description="prev", layout=w.Layout(width="70px"))
        nxt = w.Button(description="next", layout=w.Layout(width="70px"))
        prev.on_click(lambda _: setattr(self.slider, "value",
                                        max(0, self.slider.value - 1)))
        nxt.on_click(lambda _: setattr(self.slider, "value",
                                       min(len(self.sel) - 1, self.slider.value + 1)))
        self.slider.observe(lambda ch: self.goto(ch["new"]), names="value")
        display(w.HBox([play, prev, nxt, self.slider]))
