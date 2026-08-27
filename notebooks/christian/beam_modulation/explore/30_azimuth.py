"""Azimuth panels: the same tone family at four different azimuths.
Also: is the apparent frequency trend in raw channel power real, or dilution?"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,az,fr,ch=z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def nb_of(c): return [c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
def fold(y,m):
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
print(f"{len(TONES)} tones, {fr[TONES[0]]:.0f}-{fr[TONES[-1]]:.0f} MHz")

# ---- dilution check with an SNR gate ----------------------------------------
print("\nIs the frequency trend in RAW channel power real, or tone/continuum dilution?")
print(f"{'MHz':>7} {'tone/cont peak':>15} {'tone/cont null':>15} {'raw depth':>10} {'sub depth':>10} {'sub valid?':>11}")
def tone_only(c):
    v=np.nanmean(d4[:,c-1:c+2],axis=1)*3-np.nanmedian(d4[:,nb_of(c)],axis=1)*3
    v[v<=0]=np.nan; return v
null=m18&(np.abs(np.abs(el)-90)<25)
for c in TONES[::4]:
    nb=nb_of(c)
    tp=C.db(np.nanmedian(d4[peak][:,c]))-C.db(np.nanmedian(d4[peak][:,nb]))
    tn=C.db(np.nanmedian(d4[null][:,c]))-C.db(np.nanmedian(d4[null][:,nb]))
    pr,ps=fold(C.db(d4[:,c]),m18),fold(C.db(tone_only(c)),m18)
    ok="yes" if tn>3 else "NO (null lost in continuum)"
    print(f"{fr[c]:7.1f} {tp:15.1f} {tn:15.1f} {np.nanmax(pr)-np.nanmin(pr):10.1f}"
          f" {np.nanmax(ps)-np.nanmin(ps):10.1f}   {ok}")

# ---- azimuth panels ----------------------------------------------------------
PANELS=[0,8,18,28]
fq=np.array([fr[c] for c in TONES]); norm=Normalize(fq.min(),fq.max())
sm=ScalarMappable(norm,plt.cm.plasma)
fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),sharex=True,sharey=True)
print()
for ax,ri in zip(axes.ravel(),PANELS):
    mm=np.zeros(len(el),bool); mm[SEGS[ri][0]:SEGS[ri][1]]=True
    dep=[]
    for c in TONES:
        p=fold(C.db(d4[:,c]),mm); ax.plot(Ec,p,c=sm.to_rgba(fr[c]),lw=.8,alpha=.95)
        dep.append(np.nanmax(p)-np.nanmin(p))
    a0=np.nanmedian(az[mm])
    ax.set_title(f"azimuth ${a0:+.0f}^\\circ$",fontsize=9)
    ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
    ax.axhline(0,c="0.75",lw=.5,ls=":"); ax.tick_params(labelsize=8)
    print(f"  azimuth {a0:+.0f} deg: depth {min(dep):4.1f} to {max(dep):4.1f} dB")
for ax in axes[1]: ax.set_xlabel("Platform rotation angle [deg]",fontsize=9)
for ax in axes[:,0]: ax.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=9)
cb=fig.colorbar(sm,ax=axes,pad=.02,fraction=.03); cb.set_label("Frequency [MHz]",fontsize=9)
cb.ax.tick_params(labelsize=8)
fig.savefig("30_azimuth4.png",dpi=190,bbox_inches="tight")
print("\nsaved 30_azimuth4.png")
