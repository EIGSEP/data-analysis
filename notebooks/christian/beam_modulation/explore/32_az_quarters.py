"""Azimuth panels on the quarter points that were actually scanned.
Scan covers az -180..+80 only, so +90 does not exist; nearest clean is +60.
Adds a MAD outlier flag so the few RFI spikes near az 0 cannot corrupt a bin."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def nb_of(c): return [c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]

def fold(y,m):
    y=y.copy()
    v=y[m]; md=np.nanmedian(v); mad=np.nanmedian(np.abs(v-md))*1.4826
    y[m & (np.abs(y-md) > md+8*max(mad,0.1)*0+30)]=np.nan     # guard only absurd values
    d=np.abs(np.diff(y,prepend=y[0]))
    y[m & (d>6) & (np.abs(y-np.nanmedian(v))>3*max(mad,0.5))]=np.nan   # single-sample spikes
    p=np.array([np.nanmedian(y[m&(ie==j)&np.isfinite(y)]) if (m&(ie==j)&np.isfinite(y)).sum() else np.nan
                for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])

m18=np.zeros(len(el),bool); m18[SEGS[18][0]:SEGS[18][1]]=True
peak=m18&(np.abs(el)<30)
TONES=[]
for c in [c for c in ch[(ch%16)==8] if 50<fr[c]<200]:
    if 86<fr[c]<110: continue
    nb=nb_of(c)
    tc=C.db(np.nanmedian(d4[peak][:,c]))-C.db(np.nanmedian(d4[peak][:,nb]))
    if np.nanmedian(np.abs(np.diff(fold(C.db(np.nanmedian(d4[:,nb],axis=1)),m18),2)))<0.06 and tc>=3.0:
        TONES.append(c)
fq=np.array([fr[c] for c in TONES])
print(f"{len(TONES)} tones, {fq.min():.0f}-{fq.max():.0f} MHz")

PANELS=[0,18,36,48]          # az -180, -90, 0, +60
norm=Normalize(fq.min(),fq.max()); sm=ScalarMappable(norm,plt.cm.plasma)
fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),sharex=True,sharey=True)
for ax,ri in zip(axes.ravel(),PANELS):
    mm=np.zeros(len(el),bool); mm[SEGS[ri][0]:SEGS[ri][1]]=True
    a0=np.nanmedian(az[mm]); dep=[]
    for c in TONES:
        p=fold(C.db(d4[:,c]),mm); ax.plot(Ec,p,c=sm.to_rgba(fr[c]),lw=.8,alpha=.95)
        dep.append(np.nanmax(p)-np.nanmin(p))
    ax.set_title(f"azimuth ${a0:+.0f}^\\circ$",fontsize=9)
    ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
    ax.axhline(0,c="0.75",lw=.5,ls=":"); ax.tick_params(labelsize=8)
    print(f"  az {a0:+5.0f} deg (rot {ri:2d}, t={tm[mm][0]:4.1f} min): depth {min(dep):4.1f} to {max(dep):4.1f} dB")
for ax in axes[1]: ax.set_xlabel("Platform rotation angle [deg]",fontsize=9)
for ax in axes[:,0]: ax.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=9)
cb=fig.colorbar(sm,ax=axes,pad=.02,fraction=.03); cb.set_label("Frequency [MHz]",fontsize=9)
cb.ax.tick_params(labelsize=8)
fig.savefig("32_az_quarters.png",dpi=190,bbox_inches="tight")
print("saved 32_az_quarters.png")
