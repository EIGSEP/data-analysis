"""Signal-injection on/off: median spectra of both receivers before and during the
comb transmission, and the per-tone injected-signal-to-continuum ratio."""
import numpy as np, h5py
from pathlib import Path
DATA=Path("/home/christian/Documents/research/eigsep/data-analysis/data/deployment5_filtered")
ch=np.arange(1024); fr=ch*0.244140625

OFF=["corr_20260717_174008Z.h5","corr_20260717_174217Z.h5","corr_20260717_174426Z.h5",
     "corr_20260717_174635Z.h5","corr_20260717_174844Z.h5"]
ON =["corr_20260717_203032Z.h5","corr_20260717_203241Z.h5","corr_20260717_203450Z.h5",
     "corr_20260717_203659Z.h5","corr_20260717_203908Z.h5"]
def med(names,key):
    a=[]
    for n in names:
        with h5py.File(DATA/n,"r") as f:
            d=f[f"data/{key}"][:].astype(np.float64); d[d<=0]=np.nan; a.append(d)
    return np.nanmedian(np.concatenate(a),axis=0)

out={}
for lab,names in [("off",OFF),("on",ON)]:
    for key,nm in [("0","gnd"),("4","air")]:
        out[f"{lab}_{nm}"]=med(names,key)
        print(f"{lab:3s} {nm}: median spectrum ok")
np.savez_compressed("inject_onoff.npz",freq=fr,**out)

TONE=(ch%16)==8
tc=ch[TONE&(fr>50)&(fr<200)]
print(f"\nper-tone injected-signal-to-continuum [dB]  ({len(tc)} tones)")
print(f"{'MHz':>7} {'gnd off':>8} {'gnd on':>8} {'air off':>8} {'air on':>8}")
res={}
for nm in ("gnd","air"):
    for lab in ("off","on"):
        s=out[f"{lab}_{nm}"]
        r=[]
        for c in tc:
            nb=np.r_[c-6:c-2,c+3:c+7]; nb=nb[((nb%16)!=8)&((nb%16)!=0)]
            r.append(10*np.log10(np.nanmean(s[c-1:c+2])*3/np.nanmedian(s[nb])/3))
        res[f"{lab}_{nm}"]=np.array(r)
for i,c in enumerate(tc[::4]):
    j=i*4
    print(f"{fr[c]:7.1f} {res['off_gnd'][j]:8.2f} {res['on_gnd'][j]:8.2f}"
          f" {res['off_air'][j]:8.2f} {res['on_air'][j]:8.2f}")
print(f"\nband median 50-200 MHz: gnd off {np.median(res['off_gnd']):+.2f} -> on {np.median(res['on_gnd']):+.2f} dB")
print(f"                        air off {np.median(res['off_air']):+.2f} -> on {np.median(res['on_air']):+.2f} dB")
np.savez_compressed("inject_ratio.npz",tone_freq=fr[tc],**res)
