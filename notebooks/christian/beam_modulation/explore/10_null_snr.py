"""Is the deep null real, or is the tone just falling below the noise floor there?
Compare tone power against the local continuum in the same spectrum."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); d4,el,az,fr,ch=z["d"],z["el"],z["az"],z["fr"],z["ch"]
S=np.load("tone_series.npz"); T4,T0,tf,tch=S["T4"],S["T0"],S["freq"],S["tone_ch"]

sel=(tf>=150)&(tf<=190)
g=C.db(np.nanmedian(T4[:,sel],axis=1))-C.db(np.nanmedian(T0[:,sel],axis=1))
g-=np.nanmedian(g)
# continuum reference in the same channels, same integrations
cont=np.array([np.nanmedian(d4[:,[c-6,c-5,c-4,c+4,c+5,c+6]],axis=1) for c in tch[sel]]).T
tone=np.array([np.nanmean(d4[:,c-1:c+2],axis=1)*3 for c in tch[sel]]).T
snr=C.db(np.nanmedian(tone,axis=1))-C.db(np.nanmedian(cont,axis=1))   # tone-over-continuum

q=np.nanpercentile(g,[1,5,25,50,75,95,99])
print("referenced gain percentiles [dB]:", " ".join(f"{v:+.1f}" for v in q))
for lab,m in [("in the null  (gain < -8 dB)",g<-8),("at the peak  (gain > +6 dB)",g>6),
              ("everything else",(g>=-8)&(g<=6))]:
    m=m&np.isfinite(snr)
    print(f"  {lab:28s} n={m.sum():5d}  tone-over-continuum median {np.nanmedian(snr[m]):+6.2f} dB"
          f"  (5th pct {np.nanpercentile(snr[m],5):+.2f})")
print("\n-> if the null population still sits well above 0 dB tone-over-continuum,")
print("   the null depth is a genuine beam null, not a detection limit.")
