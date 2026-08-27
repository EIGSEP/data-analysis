"""Fig 03: fold every el sweep onto elevation. Comb tones vs diffuse sky.
El is the fast axis and covers a full turn (-180..+180), so one sweep is a
complete great-circle cut through the antenna beam."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); d,tm,el,az,fr,ch=z["d"],z["tm"],z["el"],z["az"],z["fr"],z["ch"]
tones,tones_all,contam=C.comb_masks(ch)

turn=np.where(np.diff(np.sign(np.diff(el)))!=0)[0]+1
segs=[(a,b) for a,b in zip(np.r_[0,turn],np.r_[turn,len(el)]) if b-a>40]
E=np.arange(-180,181,4.0); Ec=0.5*(E[:-1]+E[1:])

def fold(y):
    """median-normalised per sweep, then binned on el; returns (nsweep, nbin)"""
    P=np.full((len(segs),len(Ec)),np.nan)
    for i,(a,b) in enumerate(segs):
        yy=y[a:b]-np.nanmedian(y[a:b]); ee=el[a:b]
        idx=np.digitize(ee,E)-1
        for j in range(len(Ec)):
            m=idx==j
            if m.sum(): P[i,j]=np.nanmedian(yy[m])
    return P

BANDS=[(55,75),(120,140),(150,175)]
fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
for k,(lo,hi) in enumerate(BANDS):
    for r,(lbl,mask) in enumerate([("comb tones",tones),("diffuse sky (non-comb)",~contam)]):
        m=(fr>=lo)&(fr<=hi)&mask
        y=C.db(np.nanmedian(d[:,m],axis=1)) if r else C.db(np.nanmean(d[:,m],axis=1))
        P=fold(y); mu=np.nanmedian(P,axis=0); sd=np.nanstd(P,axis=0)
        ax=axes[r,k]
        for p in P: ax.plot(Ec,p,lw=0.4,c="0.75",alpha=.5)
        ax.plot(Ec,mu,lw=1.8,c="C3" if r==0 else "C0")
        ax.fill_between(Ec,mu-sd,mu+sd,color="C3" if r==0 else "C0",alpha=.2,lw=0)
        depth=np.nanmax(mu)-np.nanmin(mu)
        ax.set_title(f"{lbl}  {lo}-{hi} MHz\ndepth {depth:.2f} dB, scatter {np.nanmedian(sd):.2f} dB, n_ch={m.sum()}",fontsize=8)
        ax.axvline(0,c="0.6",lw=.6,ls=":"); ax.grid(alpha=.3); ax.set_xlim(-180,180)
        ax.set_xticks([-180,-90,0,90,180])
        print(f"{lbl:24s} {lo}-{hi} MHz  depth {depth:6.2f} dB  scatter {np.nanmedian(sd):5.2f}  SNR {depth/np.nanmedian(sd):5.1f}")
axes[0,0].set_ylabel("dB rel. sweep median"); axes[1,0].set_ylabel("dB rel. sweep median")
for a in axes[1]: a.set_xlabel("elevation-axis angle [deg]")
fig.suptitle(f"Folded on elevation: {len(segs)} sweeps (grey), median (colour)")
fig.tight_layout(); fig.savefig("03_fold_el.png",dpi=115)
print("saved 03_fold_el.png")
