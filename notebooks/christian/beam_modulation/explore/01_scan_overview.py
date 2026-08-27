"""Fig 01: what the raster actually is — pointing vs time, and the raw waterfall."""
import sys; sys.path.insert(0,".")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C = SourceFileLoader("C","00_common.py").load_module()

z = C.load(); d,tm,az,el,fr = z["d"],z["tm"],z["az"],z["el"],z["fr"]
print(f"spectra {d.shape}, {tm[-1]:.1f} min, rfswitch={set(z['rfsw'])}")
print(f"az {np.nanmin(az):.0f}..{np.nanmax(az):.0f}  el {np.nanmin(el):.0f}..{np.nanmax(el):.0f}")

# identify el sweeps (turning points of the fast axis)
turn = np.where(np.diff(np.sign(np.diff(el)))!=0)[0]+1
segs = [(a,b) for a,b in zip(np.r_[0,turn],np.r_[turn,len(el)]) if b-a>40]
print(f"el sweeps with >40 samples: {len(segs)}; median length {np.median([b-a for a,b in segs]):.0f} samples"
      f" = {np.median([tm[b-1]-tm[a] for a,b in segs])*60:.0f} s")

fig,ax = plt.subplots(3,1,figsize=(13,10),height_ratios=[1,1,2.2],sharex=True)
ax[0].plot(tm,el,lw=0.6,c="C0"); ax[0].plot(tm,z["elt"],lw=0.6,c="k",alpha=.4)
ax[0].set_ylabel("el [deg]"); ax[0].grid(alpha=.3)
ax[1].plot(tm,az,lw=1.0,c="C3"); ax[1].plot(tm,z["pot"],lw=0.6,c="0.5")
ax[1].set_ylabel("az [deg]\n(grey = pot)"); ax[1].grid(alpha=.3)
im = ax[2].pcolormesh(fr, tm, C.db(d), cmap="turbo",
                      vmin=np.nanpercentile(C.db(d),2), vmax=np.nanpercentile(C.db(d),99.5))
ax[2].set_ylabel("min since scan start"); ax[2].set_xlabel("frequency [MHz]")
ax[2].invert_yaxis()
fig.colorbar(im,ax=ax[2],label="dB (arb)")
for a,b in segs[:60]: ax[2].axhline(tm[a],c="w",lw=0.3,alpha=.35)
fig.suptitle("2026-07-17 motor raster, key 4 (suspended bowtie) — pointing and raw waterfall")
fig.tight_layout(); fig.savefig("01_scan_overview.png",dpi=110)
print("saved 01_scan_overview.png")
