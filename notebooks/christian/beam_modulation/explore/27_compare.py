"""Single rotation (one azimuth) vs 4-rotation fold, and r annotated on the panels."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,az,fr,ch=z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
TARGET=[84.0,123.0,158.2,189.5]; OFF=4
tone_ch=ch[(ch%16)==8]
pairs=[(lambda c:(c,c+OFF))(int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])) for t in TARGET]
COLS=[plt.cm.plasma(x) for x in (0.08,0.38,0.62,0.85)]

def mask_of(rots):
    m=np.zeros(len(el),bool)
    for i in rots: m[SEGS[i][0]:SEGS[i][1]]=True
    return m
def fold(y,m,minn):
    p=np.array([np.nanmedian(y[m&(ie==j)&np.isfinite(y)])
                if (m&(ie==j)&np.isfinite(y)).sum()>=minn else np.nan for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])

for tag,rots,minn,sub in [("27_1rot",[18],1,"one rotation, azimuth $-90^\\circ$"),
                          ("27_4rot",range(16,20),3,"four rotations, azimuth $-100^\\circ$ to $-85^\\circ$")]:
    m=mask_of(rots)
    fig,ax=plt.subplots(1,2,figsize=(7.0,3.0),sharex=True)
    print(f"\n{tag}: {sub}")
    for k,((ct,cn),col) in enumerate(zip(pairs,COLS)):
        pt,pn=fold(C.db(d4[:,ct]),m,minn),fold(C.db(d4[:,cn]),m,minn)
        g=np.isfinite(pt)&np.isfinite(pn); r=np.corrcoef(pt[g],pn[g])[0,1]
        ax[0].plot(Ec,pt,c=col,lw=1.2,label=f"{fr[ct]:.0f} MHz")
        ax[1].plot(Ec,pn,c=col,lw=1.2,label=f"{fr[cn]:.0f} MHz   $r$={r:+.2f}")
        print(f"   {fr[ct]:6.1f} MHz  tone depth {np.nanmax(pt)-np.nanmin(pt):5.1f} dB   "
              f"neighbour depth {np.nanmax(pn)-np.nanmin(pn):4.2f} dB   r={r:+.2f}")
    for a,t in zip(ax,["injected comb tone","neighbouring channel, no injection"]):
        a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.25,lw=.5)
        a.set_xlabel("Platform rotation angle [deg]",fontsize=8)
        a.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=8)
        a.tick_params(labelsize=7); a.axhline(0,c="0.75",lw=.5,ls=":")
        a.set_title(t,fontsize=8.5)
    ax[0].legend(fontsize=6.8,ncol=2,loc="lower center",framealpha=.92,columnspacing=1.0)
    ax[1].legend(fontsize=6.0,loc="lower center",framealpha=.92)
    fig.suptitle(sub,fontsize=8,y=1.02,color="0.35")
    fig.tight_layout(); fig.savefig(f"{tag}.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("\nsaved 27_1rot.png, 27_4rot.png")
