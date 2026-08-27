"""Extract key 0 (stationary antenna) over the same raster window -> control sidecar.
Also survey the comb level in key0/key4 across the whole of Jul 17 to see when the
comb transmitter is on."""
import json, numpy as np, h5py
from pathlib import Path
from scipy.ndimage import median_filter
DATA=Path("/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered")
LO,HI="corr_20260717_202824Z.h5","corr_20260717_212831Z.h5"
names=sorted(p.name for p in DATA.glob("corr_*.h5"))
ras=[n for n in names if LO<=n<=HI]
print(f"raster files: {len(ras)}")

d0=[];t0=[]
for n in ras:
    with h5py.File(DATA/n,"r") as f:
        d0.append(f["data/0"][:]); t0.append(f["header/times"][:])
d0=np.concatenate(d0).astype(np.float64); t0=np.concatenate(t0)
np.savez_compressed("key0_raster.npz",d0=d0.astype(np.float32),t0=t0)
print("key0:",d0.shape,"saved key0_raster.npz")

# --- comb-on timeline over all of Jul 17 -------------------------------------
ch=np.arange(1024); fr=ch*0.244140625; band=(fr>60)&(fr<200)
def comb_excess(spec):
    with np.errstate(divide="ignore",invalid="ignore"):
        db=10*np.log10(spec)
    ex=db-median_filter(db,size=33,mode="nearest")
    return np.nanmean(ex[band&((ch%16)==8)])
jul17=[n for n in names if n.startswith("corr_20260717_")]
print(f"\nJul 17 files: {len(jul17)}; comb excess (ch%16==8) per file, every 8th file:")
print(f"{'file':30s} {'key0':>7s} {'key4':>7s}  run_tag")
for n in jul17[::8]:
    with h5py.File(DATA/n,"r") as f:
        tag=f["header"].attrs.get("run_tag","")
        r=[]
        for k in ("0","4"):
            s=f[f"data/{k}"][:].astype(np.float64); s[s<=0]=np.nan
            r.append(comb_excess(np.nanmedian(s,axis=0)))
    print(f"{n:30s} {r[0]:+7.2f} {r[1]:+7.2f}  {tag}")
