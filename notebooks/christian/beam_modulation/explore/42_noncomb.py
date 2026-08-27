"""Vazquez's question: do the NON-COMB channels show the same structure?
Same two rotations, same reduction, same frequency grouping -- the only change is
that each curve comes from the flanking non-comb channels instead of the tone."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def nb_of(c): return [c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
def fold(y,a,b):
    s=y[a:b].copy()
    smth=median_filter(np.nan_to_num(s,nan=np.nanmedian(s)),size=5,mode="nearest")
    s[np.abs(s-smth)>3.0]=np.nan
    idx=ie[a:b]
    p=np.array([np.nanmedian(s[(idx==j)&np.isfinite(s)]) if ((idx==j)&np.isfinite(s)).sum() else np.nan
                for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])

a18,b18=SEGS[18]
peak=np.zeros(len(el),bool); peak[a18:b18]=True; peak&=np.abs(el)<30
TONES=[]
for c in [c for c in ch[(ch%16)==8] if 50<fr[c]<200]:
    if 86<fr[c]<110: continue
    nb=nb_of(c)
    tc=C.db(np.nanmedian(d4[peak][:,c]))-C.db(np.nanmedian(d4[peak][:,nb]))
    if np.nanmedian(np.abs(np.diff(fold(C.db(np.nanmedian(d4[:,nb],axis=1)),a18,b18),2)))<0.06 and tc>=3.0:
        TONES.append(c)
fq=np.array([fr[c] for c in TONES])
norm=Normalize(fq.min(),fq.max()); sm=ScalarMappable(norm,plt.cm.plasma)
print(f"{len(TONES)} tone/non-comb pairs, {fq.min():.0f}-{fq.max():.0f} MHz\n")

ROTS=[0,18]
print(f"{'az':>6} {'MHz':>7} {'tone depth':>11} {'non-comb depth':>15} {'ratio':>7} {'r':>7}")
res={}
for ri in ROTS:
    a,b=SEGS[ri]; a0=np.nanmedian(az[a:b]); res[ri]={}
    for c in TONES:
        pt=fold(C.db(d4[:,c]),a,b)
        pn=fold(C.db(np.nanmedian(d4[:,nb_of(c)],axis=1)),a,b)
        res[ri][c]=(pt,pn)
    dt=[np.nanmax(v[0])-np.nanmin(v[0]) for v in res[ri].values()]
    dn=[np.nanmax(v[1])-np.nanmin(v[1]) for v in res[ri].values()]
    rr=[np.corrcoef(*[u[np.isfinite(v[0])&np.isfinite(v[1])] for u in v])[0,1] for v in res[ri].values()]
    for k,c in enumerate(TONES[::5]):
        j=TONES.index(c)
        print(f"{a0:6.0f} {fr[c]:7.1f} {dt[j]:10.1f} dB {dn[j]:14.2f} dB {dt[j]/dn[j]:7.1f} {rr[j]:+7.2f}")
    print(f"{'':6} {'median':>7} {np.median(dt):10.1f} dB {np.median(dn):14.2f} dB "
          f"{np.median(dt)/np.median(dn):7.1f} {np.median(rr):+7.2f}\n")
# ground receiver control on the same non-comb channels
for ri in ROTS:
    a,b=SEGS[ri]
    allnb=sorted({x for c in TONES for x in nb_of(c)})
    g=fold(C.db(np.nanmedian(d0[:,allnb],axis=1)),a,b)
    print(f"ground rx, non-comb channels, az {np.nanmedian(az[a:b]):+.0f}: "
          f"depth {np.nanmax(g)-np.nanmin(g):.2f} dB")

fig,axes=plt.subplots(2,2,figsize=(7.0,5.0),sharex=True)
for k,ri in enumerate(ROTS):
    a,b=SEGS[ri]; a0=np.nanmedian(az[a:b])
    for c in TONES:
        pt,pn=res[ri][c]
        axes[0,k].plot(Ec,pt,c=sm.to_rgba(fr[c]),lw=.85,alpha=.95)
        axes[1,k].plot(Ec,pn,c=sm.to_rgba(fr[c]),lw=.85,alpha=.95)
    axes[0,k].set_title(f"azimuth ${a0:+.0f}^\\circ$",fontsize=9)
for r,lab in enumerate(["injected comb tone","non-comb channels"]):
    for k in range(2):
        ax=axes[r,k]
        ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
        ax.axhline(0,c="0.75",lw=.5,ls=":"); ax.tick_params(labelsize=7.5)
    axes[r,0].set_ylabel(f"{lab}\nrel. $0^\\circ$ [dB]",fontsize=8.5)
for k in range(2):
    axes[1,k].set_xlabel("Platform rotation angle [deg]",fontsize=8.5)
    axes[1,k].sharey(axes[1,0]); axes[0,k].sharey(axes[0,0])
axes[1,0].set_ylim(-2.2,1.2)
cb=fig.colorbar(sm,ax=axes,pad=.02,fraction=.035); cb.set_label("Frequency [MHz]",fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
fig.savefig("42_noncomb.png",dpi=200,bbox_inches="tight")
print("\nsaved 42_noncomb.png")
