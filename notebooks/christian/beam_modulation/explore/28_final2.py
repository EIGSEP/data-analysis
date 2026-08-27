"""Single rotation, plasma palette, referenced to 0 deg.
Two variants of the right panel: the single neighbouring channel, or the mean of the
eight flanking non-comb channels (r is unchanged by this, so it costs no information)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,az,fr,ch=z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
ROT=18
m=np.zeros(len(el),bool); m[SEGS[ROT][0]:SEGS[ROT][1]]=True
print(f"one rotation (#{ROT}) at az {np.nanmedian(az[m]):.0f} deg, {m.sum()} integrations, "
      f"{np.nanmedian([(m&(ie==j)).sum() for j in range(len(Ec))]):.0f} per 5 deg bin")
def fold(y):
    p=np.array([np.nanmedian(y[m&(ie==j)&np.isfinite(y)]) if (m&(ie==j)&np.isfinite(y)).sum() else np.nan
                for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])
TARGET=[84.0,123.0,158.2,189.5]
tone_ch=ch[(ch%16)==8]
COLS=[plt.cm.plasma(x) for x in (0.08,0.38,0.62,0.85)]
rows=[]
for t in TARGET:
    c=int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])
    n8=[c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
    rows.append((c,c+4,fold(C.db(d4[:,c])),fold(C.db(d4[:,c+4])),fold(C.db(np.nanmedian(d4[:,n8],axis=1)))))
print(f"\n{'tone':>10} {'depth':>7} | {'1 nbr ch':>9} {'depth':>7} {'r':>6} | {'8 ch mean':>10} {'depth':>7} {'r':>6}")
for ct,cn,pt,p1,p8 in rows:
    r1=np.corrcoef(*[v[np.isfinite(pt)&np.isfinite(p1)] for v in (pt,p1)])[0,1]
    r8=np.corrcoef(*[v[np.isfinite(pt)&np.isfinite(p8)] for v in (pt,p8)])[0,1]
    print(f"{fr[ct]:7.1f}MHz {np.nanmax(pt)-np.nanmin(pt):6.1f} | {fr[cn]:8.1f}M "
          f"{np.nanmax(p1)-np.nanmin(p1):6.2f} {r1:+6.2f} | {'':10s} {np.nanmax(p8)-np.nanmin(p8):6.2f} {r8:+6.2f}")

for tag,idx,rlab in [("28_single_nbr",3,"single neighbouring channel"),
                     ("28_mean_nbr",4,"mean of 8 flanking channels")]:
    fig,ax=plt.subplots(1,2,figsize=(7.0,3.0),sharex=True)
    for (ct,cn,pt,p1,p8),col in zip(rows,COLS):
        pn=(p1,p8)[idx-3]
        g=np.isfinite(pt)&np.isfinite(pn); r=np.corrcoef(pt[g],pn[g])[0,1]
        ax[0].plot(Ec,pt,c=col,lw=1.25,label=f"{fr[ct]:.0f} MHz")
        ax[1].plot(Ec,pn,c=col,lw=1.25,label=f"{fr[ct]:.0f} MHz   $r$={r:+.2f}")
    for a,t in zip(ax,["injected comb tone",rlab+", no injection"]):
        a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.25,lw=.5)
        a.set_xlabel("Platform rotation angle [deg]",fontsize=8)
        a.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=8)
        a.tick_params(labelsize=7); a.axhline(0,c="0.75",lw=.5,ls=":"); a.set_title(t,fontsize=8.5)
    ax[0].legend(fontsize=6.8,ncol=2,loc="lower center",framealpha=.92,columnspacing=1.0)
    ax[1].legend(fontsize=6.0,loc="lower center",framealpha=.92)
    fig.tight_layout(); fig.savefig(f"{tag}.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("\nsaved 28_single_nbr.png, 28_mean_nbr.png")
