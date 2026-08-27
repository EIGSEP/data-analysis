"""Simple figure, take 2: 4 consecutive rotations (~15 deg of azimuth) so the
azimuth dependence does not blur the curve. Rotation angle on the x-axis."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]
d4=z["d"]; d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
S=np.load("tone_series.npz"); T4,T0,tf,tc=S["T4"],S["T0"],S["freq"],S["tone_ch"].astype(int)
_,_,contam=C.comb_masks(ch)
SEGS=np.load("segs.npz")["segs"]

N0,N=16,4                                   # rotations 16-19, az about -100..-85
sel=np.zeros(len(el),bool)
for i in range(N0,N0+N): sel[SEGS[i][0]:SEGS[i][1]]=True
print(f"{N} rotations, az {np.nanmin(az[sel]):.0f}..{np.nanmax(az[sel]):.0f} deg, "
      f"t {tm[sel][0]:.1f}-{tm[sel][-1]:.1f} min, {sel.sum()} integrations")

E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y):
    y=y-np.nanmedian(y[sel])
    f=lambda q:[np.nanpercentile(y[sel&(ie==j)&np.isfinite(y)],q)
                if (sel&(ie==j)&np.isfinite(y)).sum()>3 else np.nan for j in range(len(Ec))]
    return np.array(f(50)),np.array(f(16)),np.array(f(84))

FRQ=[60.5,80.1,130.9,177.7]; COLS=["#3b7dd8","#2e9e6b","#e08a1e","#c0392b"]
prof={}
print("\ncomb tones (FM band avoided):")
for f0 in FRQ:
    k=int(np.argmin(np.abs(tf-f0)))
    prof[f0]=fold(C.db(T4[:,k]))
    print(f"  {tf[k]:6.1f} MHz  depth {np.nanmax(prof[f0][0])-np.nanmin(prof[f0][0]):5.1f} dB")
SKY={"55-85 MHz":(55,85),"115-195 MHz":(115,195)}
sky={}
print("non-comb channels:")
for lbl,(a,b) in SKY.items():
    m=(fr>=a)&(fr<=b)&(~contam)
    sky[lbl]=fold(C.db(np.nanmedian(d4[:,m],axis=1)))
    print(f"  {lbl:12s} n_ch={m.sum():3d}  depth {np.nanmax(sky[lbl][0])-np.nanmin(sky[lbl][0]):5.2f} dB")
gnd=fold(C.db(np.nanmedian(T0[:,(tf<86)|(tf>110)],axis=1)))
print(f"ground receiver: depth {np.nanmax(gnd[0])-np.nanmin(gnd[0]):.2f} dB")

def style(a):
    a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.3)
    a.set_xlabel("Platform rotation angle [deg]",fontsize=8); a.tick_params(labelsize=7)

fig,ax=plt.subplots(figsize=(3.5,2.8))
for f0,c in zip(FRQ,COLS):
    mu,lo,hi=prof[f0]; ax.fill_between(Ec,lo,hi,color=c,alpha=.15,lw=0)
    ax.plot(Ec,mu,c=c,lw=1.3,label=f"{f0:.0f} MHz")
ax.plot(Ec,gnd[0],c="0.4",lw=1.0,ls="--",label="ground ref.")
style(ax); ax.set_ylabel("Received tone power [dB]",fontsize=8)
ax.legend(fontsize=6.3,ncol=2,loc="lower center",framealpha=.92)
fig.savefig("17_v1.png",dpi=200,bbox_inches="tight")

fig,ax=plt.subplots(1,2,figsize=(7.0,2.9),sharex=True)
for f0,c in zip(FRQ,COLS):
    mu,lo,hi=prof[f0]; ax[0].fill_between(Ec,lo,hi,color=c,alpha=.15,lw=0)
    ax[0].plot(Ec,mu,c=c,lw=1.3,label=f"{f0:.0f} MHz")
ax[0].plot(Ec,gnd[0],c="0.4",lw=1.0,ls="--",label="ground ref.")
ax[0].set_ylabel("Received power [dB]",fontsize=8)
ax[0].set_title("injected comb tones, suspended receiver",fontsize=8)
ax[0].legend(fontsize=6.3,ncol=2,loc="lower center",framealpha=.92)
for (lbl,(mu,lo,hi)),c in zip(sky.items(),["#3b7dd8","#c0392b"]):
    ax[1].fill_between(Ec,lo,hi,color=c,alpha=.15,lw=0); ax[1].plot(Ec,mu,c=c,lw=1.3,label=lbl)
ax[1].plot(Ec,gnd[0],c="0.4",lw=1.0,ls="--",label="ground ref.")
ax[1].set_title("comb channels masked (sky continuum)",fontsize=8)
ax[1].legend(fontsize=6.3,loc="lower center",framealpha=.92)
for a in ax: style(a)
fig.tight_layout(); fig.savefig("17_v2.png",dpi=200,bbox_inches="tight")
print("\nsaved 17_v1.png, 17_v2.png")
