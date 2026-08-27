"""What azimuths does the scan actually cover, and which are clean?"""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
print(f"azimuth actually scanned: {np.nanmin(az):.0f} to {np.nanmax(az):.0f} deg "
      f"(the raster was stopped before completing the full +-180 range)")
print("so in a 0-360 convention the scan covers 180 -> 360/0 -> 80;")
print("az 90 deg was never reached. Nearest available quarter points: -180, -90, 0, +80.\n")
tone=[c for c in ch[(ch%16)==8] if 150<fr[c]<200]
g=C.db(np.nanmedian(d4[:,tone],axis=1))
print(f"{'rot':>4} {'t0':>6} {'az':>7} {'az(0-360)':>10} {'depth':>7} {'spikes':>7}")
for i,(a,b) in enumerate(SEGS):
    y=g[a:b]-np.nanmedian(g[a:b])
    a0=np.nanmedian(az[a:b])
    print(f"{i:4d} {tm[a]:6.1f} {a0:7.1f} {a0%360:10.1f} "
          f"{np.nanmax(y)-np.nanmin(y):7.1f} {int(np.nansum(np.abs(np.diff(y))>4)):7d}")
