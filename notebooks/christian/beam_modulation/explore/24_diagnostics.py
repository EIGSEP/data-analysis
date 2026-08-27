"""Answers to: (1) what is being averaged over azimuth, (2) do the two panels correlate."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
TARGET=[84.0,123.0,158.2,189.5]; OFF=4
tone_ch=ch[(ch%16)==8]
pairs=[(lambda c:(c,c+OFF))(int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])) for t in TARGET]

def fold(y,mask):
    y=y-np.nanmedian(y[mask])
    return np.array([np.nanmedian(y[mask&(ie==j)&np.isfinite(y)])
                     if (mask&(ie==j)&np.isfinite(y)).sum()>2 else np.nan for j in range(len(Ec))])

# ---- (1) what the 4 rotations actually are ----------------------------------
print("(1) the window is 4 rotations, each at a DIFFERENT azimuth (5 deg step per pass):")
for i in range(16,20):
    a,b=SEGS[i]
    print(f"    rotation {i}: az {np.nanmedian(az[a:b]):+7.1f} deg, t {tm[a]:5.1f}-{tm[b-1]:5.1f} min, {b-a} integrations")
print("\n    so yes, the folded curve averages 15 deg of azimuth. Per-rotation spread:")
for (ct,cn) in pairs:
    ps=[]
    for i in range(16,20):
        m=np.zeros(len(el),bool); m[SEGS[i][0]:SEGS[i][1]]=True
        ps.append(fold(C.db(d4[:,ct]),m))
    ps=np.array(ps)
    single=np.nanmax(ps,axis=1)-np.nanmin(ps,axis=1)
    allm=np.zeros(len(el),bool)
    for i in range(16,20): allm[SEGS[i][0]:SEGS[i][1]]=True
    p4=fold(C.db(d4[:,ct]),allm)
    print(f"    {fr[ct]:6.1f} MHz: single-rotation depths {np.array2string(single,precision=1)}"
          f"  4-rot avg {np.nanmax(p4)-np.nanmin(p4):.1f} dB"
          f"  rot-to-rot rms {np.nanmedian(np.nanstd(ps,axis=0)):.2f} dB")

# ---- (2) correlation between the two panels ---------------------------------
allm=np.zeros(len(el),bool)
for i in range(16,20): allm[SEGS[i][0]:SEGS[i][1]]=True
print("\n(2) Pearson r between the tone curve and its neighbour-channel curve:")
for (ct,cn) in pairs:
    a,b=fold(C.db(d4[:,ct]),allm),fold(C.db(d4[:,cn]),allm)
    g=np.isfinite(a)&np.isfinite(b)
    r=np.corrcoef(a[g],b[g])[0,1]
    sl=np.polyfit(b[g],a[g],1)[0]
    print(f"    {fr[ct]:6.1f} / {fr[cn]:6.1f} MHz   r = {r:+.2f}   slope = {sl:5.1f} dB per dB")
print("\n    cross-frequency: neighbour channels against each other")
nb=[fold(C.db(d4[:,cn]),allm) for _,cn in pairs]
for i in range(len(nb)):
    for j in range(i+1,len(nb)):
        g=np.isfinite(nb[i])&np.isfinite(nb[j])
        print(f"    {fr[pairs[i][1]]:6.1f} vs {fr[pairs[j][1]]:6.1f} MHz   r = {np.corrcoef(nb[i][g],nb[j][g])[0,1]:+.2f}")
