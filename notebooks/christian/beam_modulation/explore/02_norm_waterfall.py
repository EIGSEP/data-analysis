"""Fig 02: per-channel-normalised waterfall of the raster.
Divide every channel by its own time-median -> bandpass and gain drop out,
what is left is the beam sweeping across the sky/ground/transmitter."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C = SourceFileLoader("C","00_common.py").load_module()

z=C.load(); d,tm,el,az,fr,ch = z["d"],z["tm"],z["el"],z["az"],z["fr"],z["ch"]
tones,tones_all,contam = C.comb_masks(ch)

R = C.db(d/np.nanmedian(d,axis=0))          # dB relative to each channel's own median
band=(fr>=50)&(fr<=200)
print("normalised waterfall spread (5-95 pct), 50-200 MHz:",
      f"{np.nanpercentile(R[:,band],5):+.2f} .. {np.nanpercentile(R[:,band],95):+.2f} dB")
print("  comb tones only :",f"{np.nanpercentile(R[:,band&tones],5):+.2f} .. {np.nanpercentile(R[:,band&tones],95):+.2f} dB")
print("  non-comb only   :",f"{np.nanpercentile(R[:,band&~contam],5):+.2f} .. {np.nanpercentile(R[:,band&~contam],95):+.2f} dB")

fig=plt.figure(figsize=(14,9))
gs=fig.add_gridspec(1,3,width_ratios=[1,6,6],wspace=0.06)
a0=fig.add_subplot(gs[0]); a0.plot(el,tm,lw=0.5,c="C0"); a0.plot(az,tm,lw=1.2,c="C3")
a0.set_ylim(tm[-1],0); a0.set_xlabel("el(blue)/az(red)\n[deg]"); a0.set_ylabel("min since scan start"); a0.grid(alpha=.3)
for k,(sel,ttl) in enumerate({ "all channels":slice(None) }.items()): pass
for k,(ttl,mask) in enumerate([("all channels",np.ones_like(ch,bool)),
                               ("non-comb channels only",~contam)]):
    a=fig.add_subplot(gs[k+1],sharey=a0)
    Rm=np.where(mask[None,:],R,np.nan)
    im=a.pcolormesh(fr,tm,Rm,cmap="RdBu_r",vmin=-3,vmax=3)
    a.set_xlim(45,205); a.set_xlabel("frequency [MHz]"); a.set_title(ttl,fontsize=10)
    if k: plt.setp(a.get_yticklabels(),visible=False)
    else: plt.setp(a.get_yticklabels(),visible=False)
fig.colorbar(im,ax=fig.axes[1:],label="dB rel. channel time-median",pad=0.02,fraction=0.03)
fig.suptitle("Per-channel-normalised raster waterfall — beam modulation is the horizontal banding")
fig.savefig("02_norm_waterfall.png",dpi=110,bbox_inches="tight")
print("saved 02_norm_waterfall.png")
