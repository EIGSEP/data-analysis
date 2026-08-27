"""Fig 05: per-tone beam response. Key 4 (rotating) referenced to key 0 (stationary).

For each comb tone: tone power minus a local baseline from the two neighbouring
non-tone channel groups (removes continuum/RFI), on both keys; then key4/key0
divides out transmitter drift and any common-mode gain change."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); d4,tm,el,az,fr,ch=z["d"],z["tm"],z["el"],z["az"],z["fr"],z["ch"]
d0=np.load("key0_raster.npz")["d0"].astype(np.float64); d0[d0<=0]=np.nan

TONE=(ch%16)==8
tone_ch=ch[TONE & (fr>50) & (fr<200)]
print(f"{len(tone_ch)} comb tones between 50 and 200 MHz "
      f"({fr[tone_ch[0]]:.1f} .. {fr[tone_ch[-1]]:.1f} MHz, spacing {16*0.244140625:.3f} MHz)")

def tone_power(d):
    """linear tone power above local baseline, per tone, per integration"""
    out=np.full((d.shape[0],len(tone_ch)),np.nan)
    for i,c in enumerate(tone_ch):
        nb=np.r_[c-6:c-2, c+3:c+7]              # clear of the tone and its +-1 spill
        nb=nb[(nb>=0)&(nb<1024)&((nb%16)!=8)&((nb%16)!=0)]
        out[:,i]=np.nanmean(d[:,c-1:c+2],axis=1)*3 - np.nanmedian(d[:,nb],axis=1)*3
    return out

T4,T0=tone_power(d4),tone_power(d0)
T4[T4<=0]=np.nan; T0[T0<=0]=np.nan
g4,g0=C.db(np.nanmedian(T4,axis=1)),C.db(np.nanmedian(T0,axis=1))
ratio=g4-g0
for nm,v in [("key4 (rotating)",g4),("key0 (stationary)",g0),("key4 - key0",ratio)]:
    v=v[np.isfinite(v)]
    print(f"  {nm:20s} band-median tone power: pk-pk(1-99pct) {np.percentile(v,99)-np.percentile(v,1):6.2f} dB"
          f"   std {v.std():5.2f} dB")

fig,ax=plt.subplots(4,1,figsize=(13,9),sharex=True)
ax[0].plot(tm,el,lw=.6,c="C0"); ax[0].set_ylabel("el [deg]"); ax[0].grid(alpha=.3)
ax[1].plot(tm,g4-np.nanmedian(g4),lw=.7,c="C3"); ax[1].set_ylabel("key 4 (rotating)\n[dB]"); ax[1].grid(alpha=.3)
ax[2].plot(tm,g0-np.nanmedian(g0),lw=.7,c="C0"); ax[2].set_ylabel("key 0 (stationary)\n[dB]"); ax[2].grid(alpha=.3)
ax[3].plot(tm,ratio-np.nanmedian(ratio),lw=.7,c="k"); ax[3].set_ylabel("key4 - key0\n[dB]"); ax[3].grid(alpha=.3)
for a in ax[1:]: a.set_ylim(-20,20); a.axhline(0,c="0.6",lw=.6,ls=":")
ax[3].set_xlabel("minutes since scan start")
fig.suptitle("Comb-tone power through the raster: rotating antenna modulates, stationary one does not")
fig.tight_layout(); fig.savefig("05_tone_ratio.png",dpi=115)
np.savez_compressed("tone_series.npz",tone_ch=tone_ch,freq=fr[tone_ch],T4=T4,T0=T0,tm=tm,el=el,az=az)
print("saved 05_tone_ratio.png + tone_series.npz")
