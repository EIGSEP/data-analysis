"""Simple version: response vs rotation angle. Rotation angle on the x-axis,
a few individual comb channels, FM band avoided."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]
d4=z["d"]; d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
S=np.load("tone_series.npz"); T4,T0,tf,tc=S["T4"],S["T0"],S["freq"],S["tone_ch"].astype(int)
_,_,contam=C.comb_masks(ch)

SEGS=np.load("segs.npz")["segs"]
USE=range(14,30)                       # az -110..-35, 21-26 dB depth, no RFI spikes
sel=np.zeros(len(el),bool)
for i in USE: sel[SEGS[i][0]:SEGS[i][1]]=True
print(f"using rotations {USE.start}-{USE.stop-1}: az {np.nanmin(az[sel]):.0f}..{np.nanmax(az[sel]):.0f} deg, "
      f"{sel.sum()} integrations, {tm[sel][0]:.1f}-{tm[sel][-1]:.1f} min")

E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y):
    """median and 16-84 pct spread across the used rotations, per angle bin"""
    mu=np.full(len(Ec),np.nan); lo=np.full(len(Ec),np.nan); hi=np.full(len(Ec),np.nan)
    for j in range(len(Ec)):
        v=y[sel&(ie==j)&np.isfinite(y)]
        if len(v)>3: mu[j],lo[j],hi[j]=np.nanmedian(v),*np.nanpercentile(v,[16,84])
    return mu,lo,hi

FRQ=[60.5,80.1,130.9,177.7]            # comb tones, all outside the 86-110 MHz FM band
COLS=["#3b7dd8","#2e9e6b","#e08a1e","#c0392b"]
print("\ncomb-tone modulation vs rotation angle:")
prof={}
for f0 in FRQ:
    k=int(np.argmin(np.abs(tf-f0)))
    y=C.db(T4[:,k]); y-=np.nanmedian(y[sel])
    mu,lo,hi=fold(y); prof[f0]=(mu,lo,hi)
    print(f"  {tf[k]:6.1f} MHz (ch {tc[k]:4d})  depth {np.nanmax(mu)-np.nanmin(mu):5.1f} dB   "
          f"median 16-84 spread {np.nanmedian(hi-lo):4.1f} dB")

# non-comb, FM excluded
SKY={"55-85 MHz":(55,85),"115-195 MHz":(115,195)}
print("\nnon-comb (sky) modulation vs rotation angle:")
sky_prof={}
for lbl,(a,b) in SKY.items():
    m=(fr>=a)&(fr<=b)&(~contam)
    y=C.db(np.nanmedian(d4[:,m],axis=1)); y-=np.nanmedian(y[sel])
    sky_prof[lbl]=fold(y)
    print(f"  {lbl:12s} n_ch={m.sum():3d}  depth {np.nanmax(sky_prof[lbl][0])-np.nanmin(sky_prof[lbl][0]):5.2f} dB")
# ground reference on the same tones
yg=C.db(np.nanmedian(T0[:,(tf<86)|(tf>110)],axis=1)); yg-=np.nanmedian(yg[sel])
gnd=fold(yg); print(f"\nground receiver, same tones: depth {np.nanmax(gnd[0])-np.nanmin(gnd[0]):.2f} dB")

# ---- v1: single panel, comb only --------------------------------------------
fig,ax=plt.subplots(figsize=(3.5,2.8))
for f0,c in zip(FRQ,COLS):
    mu,lo,hi=prof[f0]
    ax.fill_between(Ec,lo,hi,color=c,alpha=.16,lw=0)
    ax.plot(Ec,mu,c=c,lw=1.2,label=f"{f0:.0f} MHz")
ax.set_xlim(-180,180); ax.set_xticks([-180,-90,0,90,180])
ax.set_xlabel("Platform rotation angle [deg]",fontsize=8)
ax.set_ylabel("Received tone power [dB]",fontsize=8)
ax.tick_params(labelsize=7); ax.grid(alpha=.3)
ax.legend(fontsize=6.5,ncol=2,loc="lower center",framealpha=.92)
fig.savefig("15_v1_comb_only.png",dpi=200,bbox_inches="tight")

# ---- v2: two panels, comb + non-comb ----------------------------------------
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9),sharex=True)
for f0,c in zip(FRQ,COLS):
    mu,lo,hi=prof[f0]
    ax[0].fill_between(Ec,lo,hi,color=c,alpha=.16,lw=0)
    ax[0].plot(Ec,mu,c=c,lw=1.2,label=f"{f0:.0f} MHz")
ax[0].set_ylabel("Received tone power [dB]",fontsize=8)
ax[0].set_title("injected comb tones",fontsize=8)
ax[0].legend(fontsize=6.5,ncol=2,loc="lower center",framealpha=.92)
for (lbl,(mu,lo,hi)),c in zip(sky_prof.items(),["#3b7dd8","#c0392b"]):
    ax[1].fill_between(Ec,lo,hi,color=c,alpha=.16,lw=0)
    ax[1].plot(Ec,mu,c=c,lw=1.2,label=lbl)
ax[1].plot(Ec,gnd[0],c="0.45",lw=1.0,ls="--",label="ground receiver")
ax[1].set_ylabel("Received power [dB]",fontsize=8)
ax[1].set_title("comb channels masked (sky)",fontsize=8)
ax[1].legend(fontsize=6.5,loc="lower center",framealpha=.92)
for a in ax:
    a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.3)
    a.set_xlabel("Platform rotation angle [deg]",fontsize=8); a.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig("15_v2_comb_and_sky.png",dpi=200,bbox_inches="tight")
print("\nsaved 15_v1_comb_only.png, 15_v2_comb_and_sky.png")
