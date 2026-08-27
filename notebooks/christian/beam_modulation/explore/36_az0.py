"""Why is the az=0 rotation different? Three discriminators:
  (a) does the STATIONARY ground receiver show the same dip at the same time?
  (b) is the dip achromatic (dropout) or frequency-dependent (a null)?
  (c) does it appear at the same rotation angle in the neighbouring azimuths?"""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
SEGS=np.load("segs.npz")["segs"]
a,b=SEGS[36]
tone=[c for c in ch[(ch%16)==8] if 150<fr[c]<200]
lo  =[c for c in ch[(ch%16)==8] if 50<fr[c]<85]

# (a) ground receiver over the same integrations
g4=C.db(np.nanmedian(d4[a:b][:,tone],axis=1)); g4-=np.nanmedian(g4)
g0=C.db(np.nanmedian(d0[a:b][:,tone],axis=1)); g0-=np.nanmedian(g0)
w=(el[a:b]>-70)&(el[a:b]<-40)
print("(a) during the dip window (rotation angle -70..-40 deg):")
print(f"    suspended receiver : {np.nanmin(g4[w]):+6.2f} dB   (sweep median 0)")
print(f"    ground receiver    : {np.nanmin(g0[w]):+6.2f} dB   rms over whole sweep {np.nanstd(g0):.2f} dB")
print("    -> ground is unaffected, so this is not a correlator dropout or a gain glitch.\n")

# (b) chromaticity
print("(b) depth of the dip vs frequency:")
for c in [c for c in ch[(ch%16)==8] if 50<fr[c]<200 and not (86<fr[c]<110)][::5]:
    s=C.db(d4[a:b,c]); s-=np.nanmedian(s)
    print(f"    {fr[c]:7.1f} MHz  {np.nanmin(s[w]):+7.2f} dB")
print("    -> depth grows smoothly with frequency: a beam null, not a dropout.\n")

# (c) neighbouring azimuths
print("(c) same feature in the neighbouring rotations?")
print(f"    {'rot':>4} {'az':>6} {'angle of deepest point':>24} {'depth':>8}")
for ri in range(33,40):
    aa,bb=SEGS[ri]
    s=C.db(np.nanmedian(d4[aa:bb][:,tone],axis=1)); s-=np.nanmedian(s)
    j=int(np.nanargmin(s))
    print(f"    {ri:4d} {np.nanmedian(az[aa:bb]):6.0f} {el[aa:bb][j]:24.0f} {s[j]:8.2f} dB")
print("\n(d) motor telemetry during the dip -- was the platform actually where it says?")
print(f"    rotation-angle samples in the dip window: "
      f"{np.array2string(el[a:b][w],precision=0,max_line_width=100)}")
print(f"    time step across the window: {np.diff(tm[a:b][w]).min()*60:.2f}-{np.diff(tm[a:b][w]).max()*60:.2f} s "
      f"(nominal 0.54 s)")
