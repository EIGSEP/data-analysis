"""How many consecutive rotations can be folded before azimuth drift dominates?"""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,az,tm=z["el"],z["az"],z["tm"]
S=np.load("tone_series.npz"); T4,tf=S["T4"],S["freq"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
k=int(np.argmin(np.abs(tf-130.9)))
y=C.db(T4[:,k])

print(f"{'n_rot':>6} {'az span':>8} {'depth':>7} {'16-84 spread':>13} {'up-vs-down RMS':>15}")
for n in (2,3,4,6,8,12,16):
    s=np.zeros(len(el),bool); c=18-n//2
    for i in range(c,c+n): s[SEGS[i][0]:SEGS[i][1]]=True
    yy=y-np.nanmedian(y[s])
    mu=np.array([np.nanmedian(yy[s&(ie==j)]) if (s&(ie==j)).sum()>2 else np.nan for j in range(len(Ec))])
    sp=np.array([np.diff(np.nanpercentile(yy[s&(ie==j)],[16,84]))[0] if (s&(ie==j)).sum()>3 else np.nan
                 for j in range(len(Ec))])
    v=np.gradient(el); up,dn=s&(v>1),s&(v<-1)
    pu=np.array([np.nanmedian(yy[up&(ie==j)]) if (up&(ie==j)).sum()>1 else np.nan for j in range(len(Ec))])
    pd=np.array([np.nanmedian(yy[dn&(ie==j)]) if (dn&(ie==j)).sum()>1 else np.nan for j in range(len(Ec))])
    g=np.isfinite(pu)&np.isfinite(pd)
    print(f"{n:6d} {np.nanmax(az[s])-np.nanmin(az[s]):8.0f} {np.nanmax(mu)-np.nanmin(mu):7.1f}"
          f" {np.nanmedian(sp):13.1f} {np.sqrt(np.nanmean((pu[g]-pd[g])**2)):15.2f}")
