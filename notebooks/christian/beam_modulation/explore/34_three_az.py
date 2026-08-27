"""Three panels at the azimuth quarter points the raster actually reached:
-180, -90, 0 deg (equivalently 180, 270, 0 in a 0-360 convention; +90 was never scanned)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def nb_of(c): return [c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
NF=[0]
def fold(y,a,b):
    s=y[a:b].copy()
    smth=median_filter(np.nan_to_num(s,nan=np.nanmedian(s)),size=5,mode="nearest")
    bad=np.abs(s-smth)>3.0; NF[0]+=int(bad.sum()); s[bad]=np.nan
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
print(f"{len(TONES)} tones, {fq.min():.0f}-{fq.max():.0f} MHz")

NF[0]=0
PANELS=[0,18,36]
fig,axes=plt.subplots(1,3,figsize=(7.4,2.8),sharex=True,sharey=True)
for ax,ri in zip(axes,PANELS):
    a,b=SEGS[ri]; a0=np.nanmedian(az[a:b]); dep=[]
    for c in TONES:
        p=fold(C.db(d4[:,c]),a,b); ax.plot(Ec,p,c=sm.to_rgba(fr[c]),lw=.8,alpha=.95)
        dep.append(np.nanmax(p)-np.nanmin(p))
    ax.set_title(f"azimuth ${a0:+.0f}^\\circ$",fontsize=9)
    ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
    ax.axhline(0,c="0.75",lw=.5,ls=":"); ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Platform rotation angle [deg]",fontsize=8.5)
    print(f"  az {a0:+5.0f} deg (t={tm[a]:4.1f} min): depth {min(dep):4.1f} to {max(dep):4.1f} dB")
axes[0].set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=8.5)
cb=fig.colorbar(sm,ax=axes,pad=.015,fraction=.035); cb.set_label("Frequency [MHz]",fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
fig.savefig("34_three_az.png",dpi=200,bbox_inches="tight")
print(f"{NF[0]} samples flagged ({NF[0]/(3*len(TONES)*128)*100:.2f}%)")
print("saved 34_three_az.png")
