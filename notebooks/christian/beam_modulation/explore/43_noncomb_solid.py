"""Make the non-comb detection solid: average several rotations to beat the noise,
and run the identical reduction on the stationary ground receiver as a null test."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
_,_,contam=C.comb_masks(ch)
FM=(fr>86)&(fr<110)

def foldm(y,rots):
    m=np.zeros(len(el),bool)
    for i in rots: m[SEGS[i][0]:SEGS[i][1]]=True
    p=np.array([np.nanmedian(y[m&(ie==j)&np.isfinite(y)]) if (m&(ie==j)&np.isfinite(y)).sum()>2 else np.nan
                for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])

BANDS={"115-145 MHz":(115,145),"150-195 MHz":(150,195),"55-85 MHz":(55,85)}
GROUPS={"1 rotation (az -90)":[18],"5 rotations (az -100..-80)":range(16,21),
        "9 rotations (az -110..-70)":range(14,23)}
print(f"{'band':>12} {'rotations':>28} {'suspended':>11} {'ground':>9} {'ratio':>7}")
for lbl,(lo,hi) in BANDS.items():
    msk=(fr>=lo)&(fr<=hi)&(~contam)&(~FM)
    y4=C.db(np.nanmedian(d4[:,msk],axis=1)); y0=C.db(np.nanmedian(d0[:,msk],axis=1))
    for gl,rots in GROUPS.items():
        p4,p0=foldm(y4,rots),foldm(y0,rots)
        d4d=np.nanmax(p4)-np.nanmin(p4); d0d=np.nanmax(p0)-np.nanmin(p0)
        print(f"{lbl:>12} {gl:>28} {d4d:10.2f}dB {d0d:8.2f}dB {d4d/d0d:7.1f}")
    print()

rots=range(14,23)
fig,ax=plt.subplots(figsize=(4.4,3.2))
for (lbl,(lo,hi)),c in zip(BANDS.items(),["#3b7dd8","#c0392b","#2e9e6b"]):
    msk=(fr>=lo)&(fr<=hi)&(~contam)&(~FM)
    ax.plot(Ec,foldm(C.db(np.nanmedian(d4[:,msk],axis=1)),rots),c=c,lw=1.4,label=f"{lbl} (suspended)")
    ax.plot(Ec,foldm(C.db(np.nanmedian(d0[:,msk],axis=1)),rots),c=c,lw=.9,ls=":",alpha=.8)
ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
ax.axhline(0,c="0.75",lw=.5,ls=":")
ax.set_xlabel("Platform rotation angle [deg]",fontsize=8.5)
ax.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=8.5)
ax.set_title("non-comb channels, 9 rotations\nsolid: suspended   dotted: ground",fontsize=8.5)
ax.legend(fontsize=6.3,loc="lower center"); ax.tick_params(labelsize=7.5)
fig.tight_layout(); fig.savefig("43_noncomb_solid.png",dpi=200,bbox_inches="tight")
print("saved 43_noncomb_solid.png")
