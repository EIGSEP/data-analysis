"""Shared loader for the 2026-07-17 motor raster beam-modulation exploration."""
import numpy as np, h5py
from pathlib import Path

REPO = Path("/home/christian/Documents/research/eigsep/data-analysis")
SIDECAR = REPO/"notebooks/christian/deployment5/motor_scan_20260717_key4.h5"
OUT = REPO/"notebooks/christian/beam_modulation/explore"

def load():
    with h5py.File(SIDECAR,"r") as f:
        d = f["data4"][:].astype(np.float64)
        t = f["times"][:]
        cpd = f.attrs["counts_per_deg"]
        az = f["az_counts"][:]/cpd
        el = f["el_counts"][:]/cpd
        azt = f["az_target_counts"][:]/cpd
        elt = f["el_target_counts"][:]/cpd
        fr = f["freqs"][:]                 # MHz, 0.244140625 spacing
        pot = f["pot_az_angle_deg"][:]
        rfsw = f["rfswitch"][:].astype(str)
    d[d<=0] = np.nan
    tm = (t-t[0])/60.0
    return dict(d=d, t=t, tm=tm, az=az, el=el, azt=azt, elt=elt,
                fr=fr, pot=pot, rfsw=rfsw, ch=np.arange(d.shape[1]))

# 2026 (deployment-5) comb: strong tones on ch%16==8, weaker set on ch%16==0
COMB_MAIN, COMB_SUB = 8, 0
def comb_masks(ch):
    tones = np.isin(ch % 16, [COMB_MAIN])
    tones_all = np.isin(ch % 16, [COMB_MAIN, COMB_SUB])
    contaminated = np.isin(ch % 16, [7,8,9, 15,0,1])   # tones + -+1 spillover
    return tones, tones_all, contaminated

def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10*np.log10(x)
