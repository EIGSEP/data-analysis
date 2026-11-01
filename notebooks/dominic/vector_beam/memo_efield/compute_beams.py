"""
Solve the vector beam on every tooth under the three mounts of ``mounts.py`` and
score each against HFSS (``vb.compare_to_truth``). No per-dwell phase
calibration. The closure g is refitted per tooth and per mount (its phase flips
by ~180 deg with the handedness). Maps of the three ``MAP_D`` teeth are saved
for mount L.

Output: output/scores.json, output/beams_maps.npz (~10 min).
"""

import json
import time

import numpy as np

from eigsep_data.beam_mapping import vector_beam as vb
from mounts import MAP_D, OUT, geometry, load, truth

z, ch = load()
GEO = {name: geometry(name, z) for name in ("R", "L0", "L")}
OUT.mkdir(exist_ok=True)
rows, maps = [], {}
for c in ch:
    t0 = time.time()
    out = dict(d=c["d"], freq=c["freq"], arm=c["arm"])
    for name, geo in GEO.items():
        g = vb.fit_closure_g(*geo, c["Ex"], c["Ey"], c["v"])
        sol = vb.solve_sampled_beam(*geo, c["Ex"], c["Ey"], c["v"], g=g)
        t = truth(z, c["k"], sol["pix"])
        sc = vb.compare_to_truth(sol["u"], t)
        out[name] = None if sc is None else {
            key: [val.real, val.imag] if isinstance(val, complex) else val for key, val in sc.items()}
        out[name + "_npix"] = int(len(sol["pix"]))
        out[name + "_g"] = [float(np.abs(g)), float(np.degrees(np.angle(g)))]
        if c["d"] in MAP_D and name == "L":
            maps[c["d"]] = dict(pix=sol["pix"], u=sol["u"], truth=t, n_visits=sol["n_visits"])
    rows.append(out)
    fmt = lambda n: "none" if out[n] is None else " ".join(
        f"{k} {out[n][k]:.3f}" for k in ("x", "y", "z", "power"))
    print(f"d{c['d']} {c['freq']:.1f} MHz: " + " | ".join(f"{n} [{fmt(n)}]" for n in GEO)
          + f"  {time.time() - t0:.0f}s", flush=True)
json.dump(rows, open(OUT / "scores.json", "w"), indent=1)
np.savez(OUT / "beams_maps.npz", map_d=np.array(MAP_D),
         **{f"{key}_{d}": val for d, m in maps.items() for key, val in m.items()})
