"""Paired channels: each injected comb tone beside its neighbouring channel.
Same frequency, same integrations, one carries the injected signal and one does not.
No scatter bands. FM band avoided."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]
d4=z["d"]; d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
SEGS=np.load("segs.npz")["segs"]

N0,N=16,4                                    # 4 consecutive rotations at fixed azimuth
sel=np.zeros(len(el),bool)
for i in range(N0,N0+N): sel[SEGS[i][0]:SEGS[i][1]]=True
print(f"{N} rotations at az {np.nanmin(az[sel]):.0f}..{np.nanmax(az[sel]):.0f} deg "
      f"(5 deg step per pass), t {tm[sel][0]:.1f}-{tm[sel][-1]:.1f} min, {sel.sum()} integrations")

E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y):
    y=y-np.nanmedian(y[sel])
    return np.array([np.nanmedian(y[sel&(ie==j)&np.isfinite(y)])
                     if (sel&(ie==j)&np.isfinite(y)).sum()>3 else np.nan for j in range(len(Ec))])

# tone channels (ch%16==8) nearest the requested frequencies, plus neighbour at +4 ch
TARGET=[60.5,80.1,130.9,177.7]; OFFSET=4      # +4 ch = +0.98 MHz, clear of +-1 spillover
COLS=["#3b7dd8","#2e9e6b","#e08a1e","#c0392b"]
tone_ch=ch[(ch%16)==8]
pairs=[]
for t in TARGET:
    c=int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))]); pairs.append((c,c+OFFSET))
print(f"\n{'tone':>18}  {'neighbour':>18}   {'tone depth':>11} {'neigh depth':>12}")
prof={}
for (ct,cn),col in zip(pairs,COLS):
    pt,pn=fold(C.db(d4[:,ct])),fold(C.db(d4[:,cn]))
    prof[ct]=(pt,pn)
    print(f"  ch{ct:4d} {fr[ct]:7.1f} MHz   ch{cn:4d} {fr[cn]:7.1f} MHz "
          f"{np.nanmax(pt)-np.nanmin(pt):10.1f} dB {np.nanmax(pn)-np.nanmin(pn):10.2f} dB")
# per-bin noise on a single neighbour channel, for the caption
cn=pairs[-1][1]; y=C.db(d4[:,cn]); y-=np.nanmedian(y[sel])
sc=[np.nanstd(y[sel&(ie==j)])/np.sqrt(max((sel&(ie==j)).sum(),1)) for j in range(len(Ec))]
print(f"\nsingle-channel standard error per 5 deg bin: {np.nanmedian(sc):.3f} dB "
      f"(median {np.nanmedian([(sel&(ie==j)).sum() for j in range(len(Ec))]):.0f} integrations/bin)")
gnd=fold(C.db(np.nanmedian(d0[:,[c for c,_ in pairs]],axis=1)))
print(f"ground receiver, same tones: depth {np.nanmax(gnd[0:])-np.nanmin(gnd):.2f} dB")

def style(a,ttl):
    a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.3)
    a.set_xlabel("Platform rotation angle [deg]",fontsize=8); a.tick_params(labelsize=7)
    a.set_title(ttl,fontsize=8)

for tag,with_gnd in [("18_paired","")   ,("18_paired_gnd","g")]:
    fig,ax=plt.subplots(1,2,figsize=(7.0,2.9),sharex=True)
    for (ct,cn),col in zip(pairs,COLS):
        pt,pn=prof[ct]
        ax[0].plot(Ec,pt,c=col,lw=1.3,label=f"{fr[ct]:.0f} MHz")
        ax[1].plot(Ec,pn,c=col,lw=1.3,label=f"{fr[cn]:.0f} MHz")
    if with_gnd:
        for a in ax: a.plot(Ec,gnd,c="0.45",lw=1.0,ls="--",label="ground rx")
    style(ax[0],"injected comb tones"); style(ax[1],"neighbouring channels (no injection)")
    ax[0].set_ylabel("Received power [dB]",fontsize=8)
    ax[1].set_ylabel("Received power [dB]",fontsize=8)
    ax[0].legend(fontsize=6.3,ncol=2,loc="lower center",framealpha=.92)
    ax[1].legend(fontsize=6.3,ncol=2,loc="lower center",framealpha=.92)
    fig.tight_layout(); fig.savefig(f"{tag}.png",dpi=200,bbox_inches="tight")
print("saved 18_paired.png, 18_paired_gnd.png")
