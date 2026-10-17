"""Mock binned data for injection tests of ``global_fit.py`` (read back with ``--mock-y``).

    inject_mock.py BINNED_NPZ FIT_RESID_NPZ OUT_NPZ [--seed S] [--add-resid RESID_NPZ --roll N]

The mock is a global fit's own prediction at every bin it kept, model = y (1 - resid_frac),
so the fit's T_gnd and gain are the truth, plus Gaussian radiometer noise y / sqrt(B tau N);
bins without a prediction are NaN. ``--add-resid`` multiplies in a fit's real residuals,
rolled by N bins among the bins that have them: they keep their size and time structure
but lose their alignment with pointing.
"""
import argparse
import numpy as np
from common import CHANNEL_MHZ

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("binned"); ap.add_argument("resid"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--add-resid", default=None)
ap.add_argument("--roll", type=int, default=1500)
args = ap.parse_args()
B = np.load(args.binned); R = np.load(args.resid)
idx = np.searchsorted(B["t"], R["t"]); assert np.allclose(B["t"][idx], R["t"])
fi = [int(np.argmin(np.abs(B["freqs_mhz"] - f))) for f in R["freqs_mhz"]]
rng = np.random.default_rng(args.seed)
y = np.full((len(B["t"]), len(fi)), np.nan)
for j, c in enumerate(fi):
    m = B["y"][idx, c] * (1 - R["resid_frac"][j])
    sig = m / np.sqrt(CHANNEL_MHZ * 1e6 * B["tau"][idx] * np.maximum(B["nsamp"][idx, c], 1))
    y[idx, j] = m + sig * rng.standard_normal(len(m))
if args.add_resid:
    A = np.load(args.add_resid); ia = np.searchsorted(B["t"], A["t"])
    for j in range(len(fi)):
        r = A["resid_frac"][j]; ok = np.flatnonzero(np.isfinite(r) & np.isfinite(y[ia, j]))
        y[ia[ok], j] *= 1 + np.roll(r[ok], args.roll)
np.savez(args.out, t=B["t"], freqs_mhz=R["freqs_mhz"], y=y)
print(args.out, "bins with data per frequency", np.isfinite(y).sum(0))
