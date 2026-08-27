"""Many tones as a family of 1-D curves instead of an image.
Also checks a subtlety: raw channel power = tone + sky continuum, so at low
frequency (where the tone barely exceeds the continuum) the channel depth is
limited by dilution, not by the beam. Compare raw vs continuum-subtracted."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,az,fr,ch=z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
ROT=18; m=np.zeros(len(el),bool); m[SEGS[ROT][0]:SEGS[ROT][1]]=True
def fold(y):
    p=np.array([np.nanmedian(y[m&(ie==j)&np.isfinite(y)]) if (m&(ie==j)&np.isfinite(y)).sum() else np.nan
                for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])
def nb_of(c): return [c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]

# clean, non-FM tones where the injected tone dominates its channel
peak=m&(np.abs(el)<30)
TONES=[]
for c in [c for c in ch[(ch%16)==8] if 50<fr[c]<200]:
    if 86<fr[c]<110: continue
    nb=nb_of(c)
    tc=C.db(np.nanmedian(d4[peak][:,c]))-C.db(np.nanmedian(d4[peak][:,nb]))
    rough=np.nanmedian(np.abs(np.diff(fold(C.db(np.nanmedian(d4[:,nb],axis=1))),2)))
    if rough<0.06: TONES.append((c,tc))
print(f"{len(TONES)} clean non-FM tones, {fr[TONES[0][0]]:.0f}-{fr[TONES[-1][0]]:.0f} MHz")
DOM=[(c,tc) for c,tc in TONES if tc>=3.0]
print(f"{len(DOM)} of them with tone >= 3 dB over continuum ({fr[DOM[0][0]]:.0f}-{fr[DOM[-1][0]]:.0f} MHz)")

def tone_only(c):
    """linear tone power with the local continuum removed"""
    v=np.nanmean(d4[:,c-1:c+2],axis=1)*3-np.nanmedian(d4[:,nb_of(c)],axis=1)*3
    v[v<=0]=np.nan; return v

for tag,use,mode,ttl in [
    ("29_raw_dom",DOM,"raw","raw channel power, tones dominating their channel"),
    ("29_raw_all",TONES,"raw","raw channel power, all clean tones"),
    ("29_sub_all",TONES,"sub","continuum-subtracted tone power, all clean tones")]:
    fq=np.array([fr[c] for c,_ in use])
    norm=Normalize(fq.min(),fq.max()); sm=ScalarMappable(norm,plt.cm.plasma)
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    dep=[]
    for c,_ in use:
        p=fold(C.db(d4[:,c])) if mode=="raw" else fold(C.db(tone_only(c)))
        ax.plot(Ec,p,c=sm.to_rgba(fr[c]),lw=.9,alpha=.95)
        dep.append(np.nanmax(p)-np.nanmin(p))
    ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
    ax.axhline(0,c="0.75",lw=.5,ls=":")
    ax.set_xlabel("Platform rotation angle [deg]",fontsize=9)
    ax.set_ylabel("Power relative to $0^\\circ$ [dB]",fontsize=9); ax.tick_params(labelsize=8)
    cb=fig.colorbar(sm,ax=ax,pad=.02); cb.set_label("Frequency [MHz]",fontsize=9); cb.ax.tick_params(labelsize=8)
    ax.set_title(ttl,fontsize=9)
    fig.tight_layout(); fig.savefig(f"{tag}.png",dpi=190,bbox_inches="tight"); plt.close(fig)
    print(f"  {tag}: depth {min(dep):.1f} to {max(dep):.1f} dB over {fq.min():.0f}-{fq.max():.0f} MHz")

print("\ndilution check -- raw channel depth is capped by the tone/continuum ratio:")
print(f"{'MHz':>7} {'tone/cont peak':>15} {'raw depth':>10} {'subtracted depth':>17}")
for c,tc in TONES[::4]:
    pr,ps=fold(C.db(d4[:,c])),fold(C.db(tone_only(c)))
    print(f"{fr[c]:7.1f} {tc:15.1f} {np.nanmax(pr)-np.nanmin(pr):10.1f} {np.nanmax(ps)-np.nanmin(ps):17.1f}")
