"""Fig 08: (a) clean beam map with bins matched to the sample step,
(b) the 'all frequencies' view -- referenced comb power vs (elevation, frequency)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az=z["tm"],z["el"],z["az"]
S=np.load("tone_series.npz"); T4,T0,tf=S["T4"],S["T0"],S["freq"]
R=C.db(T4)-C.db(T0)                                  # (nint, ntone) referenced gain
R-=np.nanmedian(R,axis=0)

EE=np.arange(-180,181,6.0); AE=np.arange(-182.5,82.6,5.0)
ia=np.digitize(az,AE)-1; ie=np.digitize(el,EE)-1
def gmap(y):
    H=np.full((len(EE)-1,len(AE)-1),np.nan)
    for i in range(len(EE)-1):
        for j in range(len(AE)-1):
            m=(ie==i)&(ia==j)&np.isfinite(y)
            if m.sum(): H[i,j]=np.nanmedian(y[m])
    return H
def eprof(y):
    return np.array([np.nanmedian(y[(ie==i)&np.isfinite(y)]) if ((ie==i)&np.isfinite(y)).sum()>5
                     else np.nan for i in range(len(EE)-1)])

band=(tf>=150)&(tf<=190); H=gmap(np.nanmedian(R[:,band],axis=1))
W=np.array([eprof(R[:,k]) for k in range(R.shape[1])])   # (ntone, nel)
print(f"map: filled {np.isfinite(H).sum()}/{H.size}, peak {np.nanmax(H):+.1f} dB at "
      f"az={AE[np.nanargmax(H)%(len(AE)-1)]:.0f} el={EE[np.nanargmax(H)//(len(AE)-1)]:.0f}")
print("\nper-tone el-profile depth (max-min), every 4th tone:")
for k in range(0,len(tf),4):
    print(f"  {tf[k]:6.1f} MHz  depth {np.nanmax(W[k])-np.nanmin(W[k]):5.1f} dB")

fig,ax=plt.subplots(1,2,figsize=(13,4.6))
im=ax[0].pcolormesh(AE,EE,H,cmap="RdYlBu_r",vmin=-10,vmax=10)
ax[0].set_xlabel("azimuth [deg]"); ax[0].set_ylabel("elevation-axis angle [deg]")
ax[0].set_yticks([-180,-90,0,90,180]); ax[0].set_title("beam map, comb tones 150-190 MHz",fontsize=10)
fig.colorbar(im,ax=ax[0],label="dB rel. median")
Ec=.5*(EE[:-1]+EE[1:])
im=ax[1].pcolormesh(Ec,tf,W,cmap="RdYlBu_r",vmin=-10,vmax=10,shading="nearest")
ax[1].set_xlabel("elevation-axis angle [deg]"); ax[1].set_ylabel("frequency [MHz]")
ax[1].set_xticks([-180,-90,0,90,180]); ax[1].set_title("every comb tone, az-averaged el cut",fontsize=10)
fig.colorbar(im,ax=ax[1],label="dB rel. median")
fig.suptitle("Beam modulation: 2-D map and its frequency dependence")
fig.tight_layout(); fig.savefig("08_map_and_freq.png",dpi=120)
print("saved 08_map_and_freq.png")
