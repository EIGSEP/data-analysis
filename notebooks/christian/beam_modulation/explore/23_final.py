"""Final candidate: four frequencies where the injected tone genuinely dominates
its channel (tone/continuum +4.4 to +23.5 dB at the response peak), each paired
with the channel 1 MHz away. Raw channel power, no baseline subtraction, no bands."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]
d4=z["d"]; d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
SEGS=np.load("segs.npz")["segs"]
N0,N=16,4
sel=np.zeros(len(el),bool)
for i in range(N0,N0+N): sel[SEGS[i][0]:SEGS[i][1]]=True
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y):
    y=y-np.nanmedian(y[sel])
    return np.array([np.nanmedian(y[sel&(ie==j)&np.isfinite(y)])
                     if (sel&(ie==j)&np.isfinite(y)).sum()>3 else np.nan for j in range(len(Ec))])

TARGET=[84.0,123.0,158.2,189.5]; OFF=4
COLS=["#3b7dd8","#2e9e6b","#e08a1e","#c0392b"]
tone_ch=ch[(ch%16)==8]
pairs=[(lambda c:(c,c+OFF))(int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])) for t in TARGET]
print(f"{N} consecutive rotations at az {np.nanmin(az[sel]):.0f}..{np.nanmax(az[sel]):.0f} deg, "
      f"{sel.sum()} integrations\n")
print(f"{'tone':>20} {'depth':>8} {'tone/cont':>10}  |  {'neighbour':>20} {'depth':>8}")
peak=sel&(np.abs(el)<30)
for (ct,cn),col in zip(pairs,COLS):
    nb=[ct+o for o in (-6,-5,-4,-3,3,4,5,6) if (ct+o)%16 not in (7,8,9,15,0,1)]
    tc=C.db(np.nanmedian(d4[peak][:,ct]))-C.db(np.nanmedian(d4[peak][:,nb]))
    pt,pn=fold(C.db(d4[:,ct])),fold(C.db(d4[:,cn]))
    print(f"ch{ct:4d} {fr[ct]:8.2f} MHz {np.nanmax(pt)-np.nanmin(pt):7.1f} dB {tc:9.1f} dB  |"
          f"  ch{cn:4d} {fr[cn]:8.2f} MHz {np.nanmax(pn)-np.nanmin(pn):7.2f} dB")
gnd=fold(C.db(np.nanmedian(d0[:,[c for c,_ in pairs]],axis=1)))
print(f"\nground receiver, same four tones: depth {np.nanmax(gnd)-np.nanmin(gnd):.2f} dB")

for tag,gref in [("23_final",False),("23_final_gnd",True)]:
    fig,ax=plt.subplots(1,2,figsize=(7.0,3.0),sharex=True)
    for (ct,cn),col in zip(pairs,COLS):
        ax[0].plot(Ec,fold(C.db(d4[:,ct])),c=col,lw=1.4,label=f"{fr[ct]:.0f} MHz")
        ax[1].plot(Ec,fold(C.db(d4[:,cn])),c=col,lw=1.4,label=f"{fr[cn]:.0f} MHz")
    if gref:
        ax[0].plot(Ec,gnd,c="0.45",lw=1.0,ls="--",label="ground rx")
    for a,ttl in zip(ax,["injected comb tone","neighbouring channel, no injection"]):
        a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.3)
        a.set_xlabel("Platform rotation angle [deg]",fontsize=8)
        a.set_title(ttl,fontsize=8.5); a.tick_params(labelsize=7)
        a.set_ylabel("Received power [dB]",fontsize=8)
        a.legend(fontsize=6.8,ncol=2,loc="upper center",framealpha=.92,columnspacing=1.0)
    ax[0].set_ylim(-16,14); ax[1].set_ylim(-1.0,1.2)
    fig.tight_layout(); fig.savefig(f"{tag}.png",dpi=200,bbox_inches="tight")
print("saved 23_final.png, 23_final_gnd.png")
