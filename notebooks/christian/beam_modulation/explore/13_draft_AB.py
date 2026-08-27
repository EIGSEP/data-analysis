"""Two improved drafts.
 A: (a) spectrum on/off | (b) per-tone detection with rotation range | (c) time series + zoom inset
 B: same top row, (c) time series, (d) modulation depth vs frequency (quantitative, no angular shape)
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); tm,el=z["tm"],z["el"]
S=np.load("tone_series.npz"); T4,T0,tf=S["T4"],S["T0"],S["freq"]
OO=np.load("inject_onoff.npz"); R=np.load("inject_ratio.npz"); fr=OO["freq"]
AIR,GND,OFFC="#c0392b","#1f6fb4","#8a8a8a"

# per-tone tone/continuum through the whole rotation, suspended receiver
d4=z["d"]; ch=z["ch"]; tc=(S["tone_ch"]).astype(int)
snr4=np.full((d4.shape[0],len(tc)),np.nan)
for i,c in enumerate(tc):
    nb=np.r_[c-6:c-2,c+3:c+7]; nb=nb[((nb%16)!=8)&((nb%16)!=0)]
    snr4[:,i]=C.db(np.nanmean(d4[:,c-1:c+2],axis=1))-C.db(np.nanmedian(d4[:,nb],axis=1))
lo4,hi4=np.nanpercentile(snr4,2,axis=0),np.nanpercentile(snr4,98,axis=0)

g4=C.db(np.nanmedian(T4,axis=1)); g4-=np.nanmedian(g4)
g0=C.db(np.nanmedian(T0,axis=1)); g0-=np.nanmedian(g0)
pk4=np.nanpercentile(g4,99)-np.nanpercentile(g4,1); rms0=np.nanstd(g0)

# per-tone modulation depth over the rotation, both receivers
def depth(T):
    v=C.db(T); v-=np.nanmedian(v,axis=0)
    return np.nanpercentile(v,99,axis=0)-np.nanpercentile(v,1,axis=0)
d_air,d_gnd=depth(T4),depth(T0)
print("modulation depth (1-99 pct) per tone:")
for k in range(0,len(tf),6):
    print(f"  {tf[k]:6.1f} MHz   suspended {d_air[k]:5.1f} dB   ground {d_gnd[k]:5.2f} dB")
print(f"  band median      suspended {np.median(d_air):5.1f} dB   ground {np.median(d_gnd):5.2f} dB")

def top_row(fig,gs):
    a=fig.add_subplot(gs[0,0]); m=(fr>=150)&(fr<=182)
    a.plot(fr[m],C.db(OO["on_gnd"][m]),lw=.9,c="k",label="transmitter on",zorder=3)
    a.plot(fr[m],C.db(OO["off_gnd"][m]),lw=1.3,c=OFFC,ls="--",label="transmitter off",zorder=4)
    a.set_xlabel("frequency [MHz]",fontsize=8); a.set_ylabel("power [dB, arb.]",fontsize=8)
    a.legend(fontsize=6.5,loc="upper left",framealpha=.9); a.tick_params(labelsize=7); a.grid(alpha=.25)
    a.set_title("(a)  injected comb, ground receiver",fontsize=8,loc="left")
    b=fig.add_subplot(gs[0,1])
    b.fill_between(R["tone_freq"],lo4,hi4,color=AIR,alpha=.22,lw=0,
                   label="suspended, range over rotation")
    b.plot(R["tone_freq"],R["on_gnd"],"o-",ms=2.5,lw=.9,c=GND,label="ground, fixed")
    b.plot(R["tone_freq"],R["off_gnd"],"-",lw=.9,c=OFFC,ls="--",label="transmitter off")
    b.axhline(0,c="0.75",lw=.6,ls=":")
    b.set_xlabel("frequency [MHz]",fontsize=8); b.set_ylabel("tone / continuum [dB]",fontsize=8)
    b.legend(fontsize=6.3,loc="upper left",framealpha=.9); b.tick_params(labelsize=7); b.grid(alpha=.25)
    b.set_title("(b)  injected signal detected across the band",fontsize=8,loc="left")

def series(c,t0=None,t1=None,lg=True,ttl=""):
    ce=c.twinx(); ce.plot(tm,el,lw=.6,c="0.8",zorder=0)
    ce.set_ylim(-190,760); ce.set_yticks([-180,-90,0,90,180]); ce.tick_params(labelsize=6.5,colors="0.55")
    ce.set_ylabel("rotation [deg]",fontsize=7,color="0.55")
    c.plot(tm,g4,lw=.7 if t0 is None else 1.1,c=AIR,label=f"suspended, rotating ({pk4:.0f} dB pk-pk)")
    c.plot(tm,g0,lw=.9,c=GND,label=f"ground, fixed ({rms0:.2f} dB rms)")
    c.set_zorder(ce.get_zorder()+1); c.patch.set_visible(False)
    c.set_xlim(t0 if t0 is not None else 0, t1 if t1 is not None else tm[-1])
    c.set_ylim(-24,13); c.set_ylabel("injected-tone power\n[dB rel. median]",fontsize=8)
    c.tick_params(labelsize=7); c.grid(alpha=.25)
    if lg: c.legend(fontsize=7,loc="lower left",ncol=2,framealpha=.92)
    c.set_title(ttl,fontsize=8,loc="left")

# ---------------- draft A ----------------
fig=plt.figure(figsize=(7.1,5.4)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.3],hspace=.45,wspace=.30)
top_row(fig,gs)
c=fig.add_subplot(gs[1,:]); series(c,ttl="(c)  response to platform rotation — identical receivers, same injected signal")
c.set_xlabel("time since start of rotation sequence [min]",fontsize=8)
ins=c.inset_axes([0.60,0.055,0.38,0.42])
ins.plot(tm,g4,lw=.9,c=AIR); ins.plot(tm,g0,lw=.9,c=GND)
ins.set_xlim(20,24); ins.set_ylim(-22,12); ins.tick_params(labelsize=5.5); ins.grid(alpha=.25)
ins.set_title("4 min detail",fontsize=6)
c.indicate_inset_zoom(ins,ec="0.4",lw=.6)
fig.savefig("13_draft_A.png",dpi=160,bbox_inches="tight")

# ---------------- draft B ----------------
fig=plt.figure(figsize=(7.1,6.4)); gs=fig.add_gridspec(3,2,height_ratios=[1,1,1],hspace=.52,wspace=.30)
top_row(fig,gs)
c=fig.add_subplot(gs[1,:]); series(c,ttl="(c)  response to platform rotation over the full sequence")
c.set_xticklabels([]); c.set_xlabel("")
c2=fig.add_subplot(gs[2,0]); series(c2,20,24,lg=False,ttl="(d)  4-minute detail")
c2.set_xlabel("time [min]",fontsize=8)
d=fig.add_subplot(gs[2,1])
d.plot(tf,d_air,"o-",ms=2.5,lw=.9,c=AIR,label="suspended")
d.plot(tf,d_gnd,"o-",ms=2.5,lw=.9,c=GND,label="ground")
d.set_yscale("log"); d.set_ylim(0.05,40)
d.set_xlabel("frequency [MHz]",fontsize=8); d.set_ylabel("modulation depth [dB]",fontsize=8)
d.legend(fontsize=6.5,loc="upper left"); d.tick_params(labelsize=7); d.grid(alpha=.25,which="both")
d.set_title("(e)  rotation-induced modulation vs frequency",fontsize=8,loc="left")
fig.savefig("13_draft_B.png",dpi=160,bbox_inches="tight")
print("saved 13_draft_A.png, 13_draft_B.png")
