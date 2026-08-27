"""Iteration: (1) single vs 4-rotation fold, (2) correlation displays,
(3) explicit dB reference, (4) colour options."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
TARGET=[84.0,123.0,158.2,189.5]; OFF=4
tone_ch=ch[(ch%16)==8]
pairs=[(lambda c:(c,c+OFF))(int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])) for t in TARGET]
allm=np.zeros(len(el),bool)
for i in range(16,20): allm[SEGS[i][0]:SEGS[i][1]]=True

def fold(y,mask,minn=1,ref="zero"):
    """ref='zero' -> dB relative to the response at rotation angle 0 (bin nearest 0)"""
    p=np.array([np.nanmedian(y[mask&(ie==j)&np.isfinite(y)])
                if (mask&(ie==j)&np.isfinite(y)).sum()>=minn else np.nan for j in range(len(Ec))])
    if ref=="zero":
        j0=np.argmin(np.abs(Ec)); p=p-np.nanmedian(p[max(j0-1,0):j0+2])
    else:
        p=p-np.nanmedian(p)
    return p

# ---- (1) single rotation vs the 4-rotation fold ------------------------------
print("(1) single rotation (one azimuth) vs 4-rotation fold, tone channels:")
print(f"{'MHz':>7}  " + "  ".join(f"az{np.nanmedian(az[SEGS[i][0]:SEGS[i][1]]):+.0f}" for i in range(16,20))
      + "   4-rot   rot-to-rot rms")
for ct,cn in pairs:
    ps=[]
    for i in range(16,20):
        m=np.zeros(len(el),bool); m[SEGS[i][0]:SEGS[i][1]]=True
        ps.append(fold(C.db(d4[:,ct]),m))
    ps=np.array(ps); p4=fold(C.db(d4[:,ct]),allm)
    dep=[np.nanmax(p)-np.nanmin(p) for p in ps]
    print(f"{fr[ct]:7.1f}  " + "  ".join(f"{d:5.1f}" for d in dep)
          + f"   {np.nanmax(p4)-np.nanmin(p4):5.1f}   {np.nanmedian(np.nanstd(ps,axis=0)):.2f} dB")

TONE=[fold(C.db(d4[:,c]),allm) for c,_ in pairs]
NBR =[fold(C.db(d4[:,c]),allm) for _,c in pairs]
RS=[]
for t,n in zip(TONE,NBR):
    g=np.isfinite(t)&np.isfinite(n); RS.append(np.corrcoef(t[g],n[g])[0,1])
print("\n(2) tone vs neighbour Pearson r: " + ", ".join(f"{fr[c]:.0f} MHz {r:+.2f}" for (c,_),r in zip(pairs,RS)))

LABS=[f"{fr[c]:.0f} MHz" for c,_ in pairs]
PAL={"current":["#3b7dd8","#2e9e6b","#e08a1e","#c0392b"],
     "viridis":[plt.cm.viridis(x) for x in (0.05,0.35,0.62,0.88)],
     "plasma" :[plt.cm.plasma(x)  for x in (0.10,0.40,0.65,0.88)],
     "muted"  :["#4c6ea8","#5c9e7d","#c9a227","#a34a3a"]}

def two_panel(cols,fn,lw=1.2,title=True):
    fig,ax=plt.subplots(1,2,figsize=(7.0,3.0),sharex=True)
    for k,c in enumerate(cols):
        ax[0].plot(Ec,TONE[k],c=c,lw=lw,label=LABS[k])
        ax[1].plot(Ec,NBR[k],c=c,lw=lw,label=LABS[k])
    for a,t in zip(ax,["injected comb tone","neighbouring channel, no injection"]):
        a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.25,lw=.5)
        a.set_xlabel("Platform rotation angle [deg]",fontsize=8)
        a.set_ylabel("Power relative to 0$^\\circ$ [dB]",fontsize=8)
        a.tick_params(labelsize=7); a.axhline(0,c="0.75",lw=.5,ls=":")
        if title: a.set_title(t,fontsize=8.5)
        a.legend(fontsize=6.8,ncol=2,loc="lower center",framealpha=.92,columnspacing=1.0)
    fig.tight_layout(); fig.savefig(fn,dpi=200,bbox_inches="tight"); plt.close(fig)

for nm,cols in PAL.items(): two_panel(cols,f"25_pal_{nm}.png")
print("\nwrote colour options: " + ", ".join(f"25_pal_{n}.png" for n in PAL))

# ---- (2) correlation displays ------------------------------------------------
cols=PAL["viridis"]
# option A: normalised shape overlay
fig,ax=plt.subplots(figsize=(3.6,3.0))
for k,c in enumerate(cols):
    n=lambda p:(p-np.nanmin(p))/(np.nanmax(p)-np.nanmin(p))
    ax.plot(Ec,n(TONE[k]),c=c,lw=1.2,label=f"{LABS[k]}  r={RS[k]:+.2f}")
    ax.plot(Ec,n(NBR[k]),c=c,lw=1.0,ls="--")
ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.grid(alpha=.25,lw=.5)
ax.set_xlabel("Platform rotation angle [deg]",fontsize=8)
ax.set_ylabel("Normalised response",fontsize=8); ax.tick_params(labelsize=7)
ax.legend(fontsize=6.2,loc="lower center",framealpha=.92)
ax.set_title("solid: injected tone   dashed: neighbour",fontsize=7.5)
fig.tight_layout(); fig.savefig("25_corr_overlay.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# option B: scatter, neighbour vs tone, per angle bin
fig,ax=plt.subplots(figsize=(3.4,3.0))
for k,c in enumerate(cols):
    ax.plot(NBR[k],TONE[k],".",ms=3.5,c=c,label=f"{LABS[k]}  r={RS[k]:+.2f}")
ax.set_xlabel("neighbouring channel [dB]",fontsize=8)
ax.set_ylabel("injected tone [dB]",fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=.25,lw=.5)
ax.legend(fontsize=6.2,framealpha=.92)
fig.tight_layout(); fig.savefig("25_corr_scatter.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("wrote 25_corr_overlay.png, 25_corr_scatter.png")
