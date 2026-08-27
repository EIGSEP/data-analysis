"""Pick a clean set of rotations: strong modulation, low RFI, away from FM.
Folding all 53 sweeps mixes azimuth (the response depends on it), so restrict
to a few consecutive passes at nearly the same azimuth."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]
S=np.load("tone_series.npz"); T4,T0,tf=S["T4"],S["T0"],S["freq"]

turn=np.where(np.diff(np.sign(np.diff(el)))!=0)[0]+1
segs=[(a,b) for a,b in zip(np.r_[0,turn],np.r_[turn,len(el)]) if b-a>100]
print(f"{len(segs)} full rotations\n")
NOFM=(tf<86)|(tf>110)
g=C.db(np.nanmedian(T4[:,NOFM],axis=1)); g-=np.nanmedian(g)
d4=z["d"]; _,_,contam=C.comb_masks(ch)
sky=(((fr>=55)&(fr<=85))|((fr>=115)&(fr<=195)))&(~contam)
ysky=C.db(np.nanmedian(d4[:,sky],axis=1)); ysky-=np.nanmedian(ysky)

print(f"{'#':>3} {'t0':>6} {'az':>7} {'depth':>7} {'spikes':>7}  (spikes = |jump|>4 dB between samples)")
for i,(a,b) in enumerate(segs):
    y=g[a:b]; jump=np.nansum(np.abs(np.diff(y))>4)
    print(f"{i:3d} {tm[a]:6.1f} {np.nanmedian(az[a:b]):7.1f} "
          f"{np.nanmax(y)-np.nanmin(y):7.1f} {jump:7d}")
np.savez("segs.npz",segs=np.array(segs))
