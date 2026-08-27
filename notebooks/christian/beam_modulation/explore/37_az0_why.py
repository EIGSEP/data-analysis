"""The two nulls separately, across azimuth, to explain the az=0 panel."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
tone=[c for c in ch[(ch%16)==8] if 150<fr[c]<200]
print(f"{'rot':>4} {'az':>6} {'dir':>5} | {'null on -90 side':>17} {'null on +90 side':>17} | "
      f"{'width(-) ':>9} {'width(+)':>9}")
for ri in range(28,44):
    a,b=SEGS[ri]
    s=C.db(np.nanmedian(d4[a:b][:,tone],axis=1)); s-=np.nanmedian(s)
    e=el[a:b]; d="up" if e[-1]>e[0] else "down"
    out=[]
    for lo,hi in [(-160,-20),(20,160)]:
        m=(e>lo)&(e<hi)
        if m.sum()<5: out.append((np.nan,np.nan,np.nan)); continue
        j=int(np.nanargmin(np.where(m,s,np.nan)))
        half=s[j]/2                      # angular width at half the null depth
        w=np.nansum(m&(s<half))*np.nanmedian(np.abs(np.diff(e)))
        out.append((e[j],s[j],w))
    print(f"{ri:4d} {np.nanmedian(az[a:b]):6.0f} {d:>5} | {out[0][0]:7.0f}deg {out[0][1]:+7.1f}dB "
          f"| {out[1][0]:7.0f}deg {out[1][1]:+7.1f}dB | {out[0][2]:8.0f}d {out[1][2]:8.0f}d")
print("\nthe two nulls are not equally deep, and which one dominates swings with azimuth;")
print("near az=0 they are closest to equal, and the deeper one is also the narrowest.")
