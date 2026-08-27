"""Fig 06: the raster mapped in (az, el) -- referenced comb power = relative beam gain
toward the fixed transmitter, plus the same map for the diffuse-sky channels."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); d4,tm,el,az,fr,ch=z["d"],z["tm"],z["el"],z["az"],z["fr"],z["ch"]
S=np.load("tone_series.npz"); T4,T0,tf=S["T4"],S["T0"],S["freq"]
_,_,contam=C.comb_masks(ch)

AE=np.arange(-182.5,82.6,5.0); EE=np.arange(-180,181,4.0)
def grid(y):
    H=np.full((len(EE)-1,len(AE)-1),np.nan)
    ia=np.digitize(az,AE)-1; ie=np.digitize(el,EE)-1
    for i in range(len(EE)-1):
        for j in range(len(AE)-1):
            m=(ie==i)&(ia==j)&np.isfinite(y)
            if m.sum()>1: H[i,j]=np.nanmedian(y[m])
    return H

PANELS=[]
for lo,hi in [(60,90),(110,140),(150,190)]:
    sel=(tf>=lo)&(tf<=hi)
    r=C.db(np.nanmedian(T4[:,sel],axis=1))-C.db(np.nanmedian(T0[:,sel],axis=1))
    PANELS.append((f"comb tones {lo}-{hi} MHz\n(key4 referenced to key0)",grid(r-np.nanmedian(r)),8))
m=(fr>=150)&(fr<=190)&(~contam)
ysky=C.db(np.nanmedian(d4[:,m],axis=1)); ysky-=np.nanmedian(ysky)
PANELS.append(("diffuse sky 150-190 MHz\n(comb channels masked)",grid(ysky),1.5))

fig,axes=plt.subplots(1,4,figsize=(17,4.4),sharey=True)
for ax,(ttl,H,v) in zip(axes,PANELS):
    im=ax.pcolormesh(AE,EE,H,cmap="RdYlBu_r",vmin=-v,vmax=v)
    ax.set_title(ttl,fontsize=9); ax.set_xlabel("azimuth [deg]")
    fig.colorbar(im,ax=ax,label="dB rel. median",fraction=.046)
    print(f"{ttl.splitlines()[0]:38s} filled {np.isfinite(H).sum():4d}/{H.size}  "
          f"range(2-98pct) {np.nanpercentile(H,2):+6.2f} .. {np.nanpercentile(H,98):+6.2f} dB")
axes[0].set_ylabel("elevation-axis angle [deg]"); axes[0].set_yticks([-180,-90,0,90,180])
fig.suptitle("Raster mapped in platform orientation")
fig.tight_layout(); fig.savefig("06_beam_map.png",dpi=115)
print("saved 06_beam_map.png")
