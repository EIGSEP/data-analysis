"""Fig 09: the non-transmitter channels. Same reduction, comb channels masked,
key4 referenced to key0 to remove common-mode gain/RFI."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); d4,tm,el,az,fr,ch=z["d"],z["tm"],z["el"],z["az"],z["fr"],z["ch"]
d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
_,_,contam=C.comb_masks(ch)
FM=(fr>86)&(fr<110)                                  # FM broadcast, heavily contaminated

EE=np.arange(-180,181,6.0); AE=np.arange(-182.5,82.6,5.0)
ie=np.digitize(el,EE)-1; ia=np.digitize(az,AE)-1
def gmap(y):
    H=np.full((len(EE)-1,len(AE)-1),np.nan)
    for i in range(len(EE)-1):
        for j in range(len(AE)-1):
            m=(ie==i)&(ia==j)&np.isfinite(y)
            if m.sum(): H[i,j]=np.nanmedian(y[m])
    return H

BANDS=[(50,85),(115,145),(150,195)]
fig,axes=plt.subplots(2,3,figsize=(14,7.5),sharex=True,sharey=True)
for k,(lo,hi) in enumerate(BANDS):
    m=(fr>=lo)&(fr<=hi)&(~contam)&(~FM)
    y4=C.db(np.nanmedian(d4[:,m],axis=1)); y0=C.db(np.nanmedian(d0[:,m],axis=1))
    for r,(lbl,y) in enumerate([("key 4 raw",y4),("key4 - key0 (referenced)",y4-y0)]):
        y=y-np.nanmedian(y); H=gmap(y)
        im=axes[r,k].pcolormesh(AE,EE,H,cmap="RdYlBu_r",vmin=-1.5,vmax=1.5)
        pk=np.nanpercentile(H,98)-np.nanpercentile(H,2)
        axes[r,k].set_title(f"{lbl}  {lo}-{hi} MHz\nspread(2-98pct) {pk:.2f} dB, n_ch={m.sum()}",fontsize=8)
        print(f"{lbl:26s} {lo:3d}-{hi:3d} MHz  spread {pk:5.2f} dB  "
              f"min {np.nanmin(H):+5.2f} max {np.nanmax(H):+5.2f}")
for a in axes[1]: a.set_xlabel("azimuth [deg]")
for a in axes[:,0]: a.set_ylabel("elevation-axis angle [deg]")
axes[0,0].set_yticks([-180,-90,0,90,180])
fig.colorbar(im,ax=axes,label="dB rel. median",fraction=.025)
fig.suptitle("Non-transmitter (diffuse sky) channels — same maps, note the 1.5 dB colour scale")
fig.savefig("09_sky_channels.png",dpi=120,bbox_inches="tight")
print("saved 09_sky_channels.png")
