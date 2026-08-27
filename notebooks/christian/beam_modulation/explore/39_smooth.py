"""Where in the scan does the response vary smoothly with azimuth?"""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,pot,fr,ch=z["tm"],z["el"],z["az"],z["pot"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
tone=[c for c in ch[(ch%16)==8] if 150<fr[c]<200]
dep=[];azm=[];potsp=[]
for ri in range(len(SEGS)):
    a,b=SEGS[ri]
    s=C.db(np.nanmedian(d4[a:b][:,tone],axis=1)); s-=np.nanmedian(s)
    dep.append(np.nanmax(s)-np.nanmin(s)); azm.append(np.nanmedian(az[a:b]))
    potsp.append(np.nanpercentile(pot[a:b],95)-np.nanpercentile(pot[a:b],5))
dep=np.array(dep); azm=np.array(azm); potsp=np.array(potsp)
for lo,hi,lab in [(0,30,"rotations 0-29  (az -180..-35, t 0-35 min)"),
                  (30,52,"rotations 30-51 (az -30..+75, t 36-62 min)")]:
    d=dep[lo:hi]
    print(f"{lab}\n   depth {d.min():.1f}-{d.max():.1f} dB, "
          f"mean |2nd difference| {np.mean(np.abs(np.diff(d,2))):.2f} dB, "
          f"pot wander within a rotation {np.median(potsp[lo:hi]):.1f} deg")
print("\nazimuth wander within a single rotation (5-95 pct of the potentiometer):")
print(f"   median over the whole scan: {np.median(potsp):.1f} deg")
print("   the platform is suspended and the IMU returned errors during this scan,")
print("   so commanded azimuth is not a precise attitude measurement.")
print(f"\ncandidate clean trio from the smooth first half:")
for target in (-180,-110,-40):
    i=int(np.argmin(np.abs(azm-target)))
    a,b=SEGS[i]
    print(f"   rotation {i:2d}: motor az {azm[i]:+5.0f} deg, pot az {np.nanmedian(pot[a:b]):+6.1f} deg, "
          f"t {tm[a]:4.1f} min, depth {dep[i]:4.1f} dB")
