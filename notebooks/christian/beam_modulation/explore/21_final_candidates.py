"""Final candidates.
  21_curves.png    : paired tone / neighbour channels vs rotation angle (no bands)
  21_waterfall.png : the same rotations as a frequency-vs-angle waterfall, RFI flagged
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az,fr,ch=z["tm"],z["el"],z["az"],z["fr"],z["ch"]
d4=z["d"]; d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan
SEGS=np.load("segs.npz")["segs"]
N0,N=16,4
sel=np.zeros(len(el),bool)
for i in range(N0,N0+N): sel[SEGS[i][0]:SEGS[i][1]]=True
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y,mask=None):
    y=y-np.nanmedian(y[sel])
    s=sel if mask is None else (sel&mask)
    return np.array([np.nanmedian(y[s&(ie==j)&np.isfinite(y)])
                     if (s&(ie==j)&np.isfinite(y)).sum()>3 else np.nan for j in range(len(Ec))])

TARGET=[60.5,84.0,130.9,169.9]; OFF=4
COLS=["#3b7dd8","#2e9e6b","#e08a1e","#c0392b"]
tone_ch=ch[(ch%16)==8]
pairs=[(int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))]),) for t in TARGET]
pairs=[(c[0],c[0]+OFF) for c in pairs]
print(f"{N} rotations, az {np.nanmin(az[sel]):.0f}..{np.nanmax(az[sel]):.0f} deg\n")
print(f"{'tone':>22} {'depth':>8}  |  {'neighbour':>22} {'depth':>8}")
for (ct,cn),col in zip(pairs,COLS):
    pt,pn=fold(C.db(d4[:,ct])),fold(C.db(d4[:,cn]))
    print(f"  ch{ct:4d} {fr[ct]:8.2f} MHz {np.nanmax(pt)-np.nanmin(pt):7.1f} dB  |"
          f"    ch{cn:4d} {fr[cn]:8.2f} MHz {np.nanmax(pn)-np.nanmin(pn):7.2f} dB")
gnd=fold(C.db(np.nanmedian(d0[:,[c for c,_ in pairs]],axis=1)))
print(f"\nground receiver, same four tones: depth {np.nanmax(gnd)-np.nanmin(gnd):.2f} dB")

# ---------------- curves ------------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(7.0,3.0),sharex=True)
for (ct,cn),col in zip(pairs,COLS):
    ax[0].plot(Ec,fold(C.db(d4[:,ct])),c=col,lw=1.4,label=f"{fr[ct]:.0f} MHz")
    ax[1].plot(Ec,fold(C.db(d4[:,cn])),c=col,lw=1.4)
ax[0].plot(Ec,gnd,c="0.45",lw=1.0,ls="--",label="ground rx")
for a,ttl in zip(ax,["injected comb tone","neighbouring channel (+1 MHz)"]):
    a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180]); a.grid(alpha=.3)
    a.set_xlabel("Platform rotation angle [deg]",fontsize=8)
    a.set_title(ttl,fontsize=8.5); a.tick_params(labelsize=7)
    a.set_ylabel("Received power [dB]",fontsize=8)
ax[0].legend(fontsize=6.8,ncol=2,loc="upper center",framealpha=.92,columnspacing=1.0)
ax[0].set_ylim(-14,12)
fig.tight_layout(); fig.savefig("21_curves.png",dpi=200,bbox_inches="tight")

# ---------------- waterfall ---------------------------------------------------
band=(fr>=50)&(fr<=200)
R=C.db(d4)-C.db(np.nanmedian(d4[sel],axis=0))          # per-channel detrend over the window
W=np.full((band.sum(),len(Ec)),np.nan)
for j in range(len(Ec)):
    m=sel&(ie==j)
    if m.sum()>3: W[:,j]=np.nanmedian(R[m][:,band],axis=0)
med=np.nanmedian(W,axis=1); mad=np.nanmedian(np.abs(W-med[:,None]),axis=1)*1.4826
bad=np.abs(W-med[:,None])>8*np.maximum(mad,0.05)[:,None]
badrow=np.nanmean(bad,axis=1)>0.25
W[badrow]=np.nan; W[bad]=np.nan
print(f"\nwaterfall: {badrow.sum()} channels flagged entirely, "
      f"{np.isnan(W).sum()/W.size*100:.1f}% of cells flagged")
fig,a=plt.subplots(figsize=(4.6,3.4))
im=a.pcolormesh(Ec,fr[band],W,cmap="RdBu_r",vmin=-8,vmax=8,shading="nearest")
a.set_xlim(-180,180); a.set_xticks([-180,-90,0,90,180])
a.set_xlabel("Platform rotation angle [deg]",fontsize=8)
a.set_ylabel("Frequency [MHz]",fontsize=8); a.tick_params(labelsize=7)
fig.colorbar(im,ax=a,label="dB rel. channel median")
fig.tight_layout(); fig.savefig("21_waterfall.png",dpi=200,bbox_inches="tight")
print("saved 21_curves.png, 21_waterfall.png")
