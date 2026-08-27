"""Final: tone family vs rotation angle. Robust RFI flagging (deviation from a
rolling median along the sweep), quarter-point azimuths that were actually scanned."""
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

NFLAG=[0]
def fold(y,a,b):
    """flag samples deviating >3 dB from a 5-sample rolling median along the sweep"""
    s=y[a:b].copy()
    sm=median_filter(np.nan_to_num(s,nan=np.nanmedian(s)),size=5,mode="nearest")
    bad=np.abs(s-sm)>3.0
    NFLAG[0]+=int(bad.sum()); s[bad]=np.nan
    idx=ie[a:b]
    p=np.array([np.nanmedian(s[(idx==j)&np.isfinite(s)]) if ((idx==j)&np.isfinite(s)).sum() else np.nan
                for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])

a18,b18=SEGS[18]
peak=np.zeros(len(el),bool); peak[a18:b18]=True; peak&= np.abs(el)<30
TONES=[]
for c in [c for c in ch[(ch%16)==8] if 50<fr[c]<200]:
    if 86<fr[c]<110: continue
    nb=nb_of(c)
    tc=C.db(np.nanmedian(d4[peak][:,c]))-C.db(np.nanmedian(d4[peak][:,nb]))
    r=np.nanmedian(np.abs(np.diff(fold(C.db(np.nanmedian(d4[:,nb],axis=1)),a18,b18),2)))
    if r<0.06 and tc>=3.0: TONES.append(c)
fq=np.array([fr[c] for c in TONES])
print(f"{len(TONES)} tones, {fq.min():.0f}-{fq.max():.0f} MHz, spacing {16*0.244140625:.2f} MHz")
norm=Normalize(fq.min(),fq.max()); sm=ScalarMappable(norm,plt.cm.plasma)

# ---- single azimuth ----------------------------------------------------------
NFLAG[0]=0
fig,ax=plt.subplots(figsize=(6.2,4.2))
dep=[]
for c in TONES:
    p=fold(C.db(d4[:,c]),a18,b18); ax.plot(Ec,p,c=sm.to_rgba(fr[c]),lw=.9,alpha=.95)
    dep.append(np.nanmax(p)-np.nanmin(p))
ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
ax.axhline(0,c="0.75",lw=.5,ls=":"); ax.tick_params(labelsize=8)
ax.set_xlabel("Platform rotation angle [deg]",fontsize=9)
ax.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=9)
cb=fig.colorbar(sm,ax=ax,pad=.02); cb.set_label("Frequency [MHz]",fontsize=9); cb.ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig("33_single_az.png",dpi=190,bbox_inches="tight"); plt.close(fig)
print(f"single azimuth (-90 deg): depth {min(dep):.1f} to {max(dep):.1f} dB, {NFLAG[0]} samples flagged")

# ---- four azimuths -----------------------------------------------------------
NFLAG[0]=0
PANELS=[0,18,36,48]
fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),sharex=True,sharey=True)
for ax,ri in zip(axes.ravel(),PANELS):
    a,b=SEGS[ri]; a0=np.nanmedian(az[a:b]); dep=[]
    for c in TONES:
        p=fold(C.db(d4[:,c]),a,b); ax.plot(Ec,p,c=sm.to_rgba(fr[c]),lw=.8,alpha=.95)
        dep.append(np.nanmax(p)-np.nanmin(p))
    ax.set_title(f"azimuth ${a0:+.0f}^\\circ$",fontsize=9)
    ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
    ax.axhline(0,c="0.75",lw=.5,ls=":"); ax.tick_params(labelsize=8)
    print(f"  az {a0:+5.0f} deg: depth {min(dep):4.1f} to {max(dep):4.1f} dB")
for ax in axes[1]: ax.set_xlabel("Platform rotation angle [deg]",fontsize=9)
for ax in axes[:,0]: ax.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=9)
cb=fig.colorbar(sm,ax=axes,pad=.02,fraction=.03); cb.set_label("Frequency [MHz]",fontsize=9)
cb.ax.tick_params(labelsize=8)
fig.savefig("33_four_az.png",dpi=190,bbox_inches="tight")
print(f"{NFLAG[0]} samples flagged across the four panels "
      f"({NFLAG[0]/(4*len(TONES)*128)*100:.2f}% of samples)")
print("saved 33_single_az.png, 33_four_az.png")
