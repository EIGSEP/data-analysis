"""Does the independent azimuth potentiometer explain the az=0 rotation?
Hypothesis: motor counts say az=0 but the platform was actually further round,
where neighbouring rotations do show deep narrow nulls."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,pot,fr,ch=z["tm"],z["el"],z["az"],z["pot"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
tone=[c for c in ch[(ch%16)==8] if 150<fr[c]<200]

print(f"{'rot':>4} {'motor az':>9} {'pot az':>8} {'pot-motor':>10} {'pot spread':>11} "
      f"{'deeper null':>12} {'angle':>7}")
rows=[]
for ri in range(28,44):
    a,b=SEGS[ri]
    s=C.db(np.nanmedian(d4[a:b][:,tone],axis=1)); s-=np.nanmedian(s)
    e=el[a:b]
    dd=[]
    for lo,hi in [(-160,-20),(20,160)]:
        m=(e>lo)&(e<hi)
        j=int(np.nanargmin(np.where(m,s,np.nan))); dd.append((s[j],e[j]))
    deep=min(dd)
    ma,pa=np.nanmedian(az[a:b]),np.nanmedian(pot[a:b])
    sp=np.nanpercentile(pot[a:b],95)-np.nanpercentile(pot[a:b],5)
    rows.append((ri,ma,pa,deep[0],deep[1]))
    print(f"{ri:4d} {ma:9.0f} {pa:8.1f} {pa-ma:10.1f} {sp:11.1f} {deep[0]:11.1f} dB {deep[1]:7.0f}")

print("\nsorted by POTENTIOMETER azimuth instead of motor counts:")
print(f"{'pot az':>8} {'motor az':>9} {'deeper null':>12}")
for ri,ma,pa,dep,ang in sorted(rows,key=lambda r:r[2]):
    print(f"{pa:8.1f} {ma:9.0f} {dep:11.1f} dB")
m=np.array([r[1] for r in rows]); p=np.array([r[2] for r in rows]); dpt=np.array([r[3] for r in rows])
def rough(x,y):
    o=np.argsort(x); return np.nanmean(np.abs(np.diff(y[o],2)))
print(f"\nsmoothness of null depth vs azimuth (mean |2nd difference|, lower = smoother):")
print(f"   ordered by motor counts : {rough(m,dpt):.2f} dB")
print(f"   ordered by potentiometer: {rough(p,dpt):.2f} dB")
