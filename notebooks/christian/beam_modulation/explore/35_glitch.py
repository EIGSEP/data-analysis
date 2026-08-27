"""Identify the residual outlier in the az 0 deg panel near -55 deg."""
import numpy as np
from scipy.ndimage import median_filter
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
a,b=SEGS[36]
print(f"az 0 rotation: samples {a}..{b}, t {tm[a]:.1f}-{tm[b-1]:.1f} min")
tone=[c for c in ch[(ch%16)==8] if 50<fr[c]<200 and not (86<fr[c]<110)]
win=(el[a:b]>-70)&(el[a:b]<-40)
print(f"\nchannels whose power in the -70..-40 deg window sits >2 dB below the sweep median:")
for c in tone:
    s=C.db(d4[a:b,c]); s=s-np.nanmedian(s)
    d=np.nanmin(s[win])
    if d<-2: print(f"  {fr[c]:7.1f} MHz (ch {c:4d}): min {d:6.2f} dB, "
                   f"{int(np.nansum(s[win]<-2))} of {win.sum()} samples affected")
# is it broadband (a dropout) or narrow (RFI)?
i0=a+int(np.argmin(np.where(win,C.db(d4[a:b,tone[0]])-np.nanmedian(C.db(d4[a:b,tone[0]])),0)))
sub=(el>-70)&(el<-40); sub[:a]=False; sub[b:]=False
allch=C.db(d4[sub])-C.db(np.nanmedian(d4[a:b],axis=0))
worst=np.nanmin(np.nanmedian(allch,axis=0))
print(f"\nacross ALL 1024 channels in that window: median deviation min {worst:.2f} dB, "
      f"max {np.nanmax(np.nanmedian(allch,axis=0)):.2f} dB")
frac=np.nanmean(np.nanmedian(allch,axis=0)<-1)
print(f"fraction of all channels more than 1 dB low: {frac*100:.1f}%")
print("-> broadband suppression across most channels = a dropped/short integration,")
print("   not narrowband RFI; flagging whole integrations is the right fix.")
# find the offending integrations
rowmed=np.nanmedian(C.db(d4[a:b])-C.db(np.nanmedian(d4[a:b],axis=0)),axis=1)
bad=np.where(rowmed<-1.0)[0]
print(f"\nintegrations in this rotation with median level >1 dB low: {len(bad)} "
      f"(indices {bad[:10]}{'...' if len(bad)>10 else ''}), levels "
      f"{np.array2string(rowmed[bad][:6],precision=2)}")
