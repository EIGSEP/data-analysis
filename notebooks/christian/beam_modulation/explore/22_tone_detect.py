"""Is the injected tone actually detected at each frequency in this window?
Raw channel power = tone + sky continuum, so a weak tone shows sky modulation,
not injection. Check tone-over-continuum at the peak of the rotation."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,fr,ch=z["el"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
sel=np.zeros(len(el),bool)
for i in range(16,20): sel[SEGS[i][0]:SEGS[i][1]]=True
peak=sel&(np.abs(el)<30)                    # near the response maximum
tone_ch=[c for c in ch[(ch%16)==8] if 50<fr[c]<200]
print(f"{'MHz':>7} {'tone/cont at peak':>18} {'at null':>9}   raw-channel depth")
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y):
    y=y-np.nanmedian(y[sel])
    return np.array([np.nanmedian(y[sel&(ie==j)&np.isfinite(y)])
                     if (sel&(ie==j)&np.isfinite(y)).sum()>3 else np.nan for j in range(len(Ec))])
null=sel&(np.abs(np.abs(el)-90)<25)
for c in tone_ch:
    if 86<fr[c]<110: continue
    nb=[c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
    tp=C.db(np.nanmedian(d4[peak][:,c]))-C.db(np.nanmedian(d4[peak][:,nb]))
    tn=C.db(np.nanmedian(d4[null][:,c]))-C.db(np.nanmedian(d4[null][:,nb]))
    p=fold(C.db(d4[:,c]))
    print(f"{fr[c]:7.1f} {tp:18.2f} {tn:9.2f}   {np.nanmax(p)-np.nanmin(p):6.1f} dB")
