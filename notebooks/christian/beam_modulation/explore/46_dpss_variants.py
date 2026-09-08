"""Why the global DPSS fit loses to a 13-channel median, and whether it can be fixed.

Same transmitter-off truth as 45. Varies three things: fitting in linear power vs
log power, fitting across the FM hole vs in sub-bands either side of it, and the
delay half-width (capped by the 256 ns alias of the periodic comb mask).
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/home/christian/Documents/research/hera_filters")
from hera_filters import dspec  # noqa: E402

DATA = Path("/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered")
OFF = ["corr_20260717_174008Z.h5", "corr_20260717_174217Z.h5", "corr_20260717_174426Z.h5",
       "corr_20260717_174635Z.h5", "corr_20260717_174844Z.h5"]
COMB_RESIDUE, COMB_SPILL, FM_BAND = 8, (7, 8, 9, 15, 0, 1), (86.0, 110.0)

d = np.concatenate([h5py.File(DATA / n, "r")["data/4"][:].astype(np.float64) for n in OFF])
d[d <= 0] = np.nan
chan = np.arange(d.shape[1])
freq = chan * 0.244140625
RADIOM = 1.0 / np.sqrt(0.244140625e6 * 0.537)
finite = np.isfinite(d).all(0)
is_fm = (freq > FM_BAND[0]) & (freq < FM_BAND[1])


def run(lo, hi, tau_ns, logfit):
    band = (freq > lo) & (freq < hi) & finite
    cl = band & ~np.isin(chan % 16, COMB_SPILL) & ~is_fm
    tn = band & (chan % 16 == COMB_RESIDUE) & ~is_fm
    if tn.sum() < 3:
        return None
    xb = freq[band] * 1e6
    y = d[:, band]
    y = np.log10(y) if logfit else y
    basis, _ = dspec.dpss_operator(xb, [0.0], [tau_ns * 1e-9], eigenval_cutoff=[1e-9])
    basis = np.real(basis)
    m = cl[band]
    model = (basis @ (np.linalg.pinv(basis[m]) @ y[:, m].T)).T
    if logfit:
        model = 10 ** model
    t = tn[band]
    return np.nanmedian(np.abs(model[:, t] / d[:, tn] - 1.0)), basis.shape[1], int(tn.sum())


print(" fit space  band            tau_ns  modes  tones   err_at_tone   vs_radiom")
for logfit in (False, True):
    for label, bands in (("48-235 (over FM)", [(48.0, 235.0)]),
                         ("50-86 + 110-200", [(50.0, 86.0), (110.0, 200.0)])):
        for tau_ns in (100.0, 200.0, 300.0):
            errs, nm, nt = [], [], 0
            for lo, hi in bands:
                r = run(lo, hi, tau_ns, logfit)
                if r:
                    errs.append(r[0] * r[2]); nm.append(r[1]); nt += r[2]
            if not errs:
                continue
            e = sum(errs) / nt
            print(f" {'log10' if logfit else 'linear':<9}  {label:<15} {tau_ns:6.0f}  "
                  f"{'+'.join(str(v) for v in nm):>5}  {nt:5d}  {100*e:10.3f} %  {e/RADIOM:9.2f}x")

fl = np.array([np.nanmedian(d[:, [c + o for o in range(-6, 7) if 3 <= abs(o) <= 6
                                  and finite[c + o] and (c + o) % 16 not in COMB_SPILL]], 1)
               for c in chan[(chan % 16 == COMB_RESIDUE) & (freq > 50) & (freq < 200) & ~is_fm]]).T
tt = chan[(chan % 16 == COMB_RESIDUE) & (freq > 50) & (freq < 200) & ~is_fm]
e = np.nanmedian(np.abs(fl / d[:, tt] - 1.0))
print(f"\n flanking median (current method)                        {100*e:10.3f} %  {e/RADIOM:9.2f}x")
