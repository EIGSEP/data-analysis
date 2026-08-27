"""Fix the up/down sweep striping: the reported motor position lags the true one,
so ascending and descending el sweeps land the same beam feature at different el.
Measure the lag by maximising up/down profile agreement."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el,az=z["tm"],z["el"],z["az"]
S=np.load("tone_series.npz"); T4,T0,tf=S["T4"],S["T0"],S["freq"]
sel=(tf>=110)&(tf<=190)
y=C.db(np.nanmedian(T4[:,sel],axis=1))-C.db(np.nanmedian(T0[:,sel],axis=1))

vel=np.gradient(el)                                  # deg per integration
up,dn=vel>1.0, vel<-1.0
print(f"ascending {up.sum()}  descending {dn.sum()}  slow/turnaround {(~up&~dn).sum()}")
print(f"|d el| while sweeping: {np.nanmedian(np.abs(vel[up|dn])):.2f} deg/integration")

E=np.arange(-180,181,4.0); Ec=.5*(E[:-1]+E[1:])
def prof(mask,shift):
    e=el+shift*vel                                   # shift in integrations
    idx=np.digitize(e,E)-1
    return np.array([np.nanmedian(y[mask&(idx==j)]) if (mask&(idx==j)).sum()>3 else np.nan
                     for j in range(len(Ec))])
shifts=np.arange(-4,4.01,0.25); rms=[]
for s in shifts:
    a,b=prof(up,s),prof(dn,s); g=np.isfinite(a)&np.isfinite(b)
    rms.append(np.sqrt(np.nanmean((a[g]-b[g])**2)))
rms=np.array(rms); best=shifts[np.argmin(rms)]
print(f"best lag correction: {best:+.2f} integrations = {best*np.nanmedian(np.abs(vel[up|dn])):+.2f} deg")
print(f"up-vs-down RMS mismatch: {rms[np.argmin(np.abs(shifts))]:.2f} dB uncorrected -> {rms.min():.2f} dB corrected")

fig,ax=plt.subplots(1,3,figsize=(14,3.8))
ax[0].plot(shifts,rms,"o-",ms=3); ax[0].axvline(best,c="C3",ls="--")
ax[0].set_xlabel("shift [integrations]"); ax[0].set_ylabel("up-vs-down RMS [dB]"); ax[0].grid(alpha=.3)
for k,s in enumerate([0.0,best]):
    ax[k+1].plot(Ec,prof(up,s),c="C0",label="ascending el")
    ax[k+1].plot(Ec,prof(dn,s),c="C3",label="descending el")
    ax[k+1].set_title(f"shift {s:+.2f}  (RMS {rms[np.argmin(np.abs(shifts-s))]:.2f} dB)",fontsize=9)
    ax[k+1].set_xlabel("el [deg]"); ax[k+1].grid(alpha=.3); ax[k+1].legend(fontsize=8)
    ax[k+1].set_xticks([-180,-90,0,90,180])
ax[1].set_ylabel("dB rel. median")
fig.suptitle("Pointing-lag calibration from up/down sweep agreement (comb tones 110-190 MHz)")
fig.tight_layout(); fig.savefig("07_lag.png",dpi=120)
np.save("best_lag.npy",best); print("saved 07_lag.png")
