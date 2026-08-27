"""DRAFT A: the two instrument novelties, quantified.
(a) signal injection: transmitter off vs on, spectrum
(b) signal injection: per-tone detection across the band, both receivers
(c) rotation: suspended receiver modulates, matched ground receiver does not
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()

z=C.load(); tm,el=z["tm"],z["el"]
S=np.load("tone_series.npz"); T4,T0,tf=S["T4"],S["T0"],S["freq"]
OO=np.load("inject_onoff.npz"); R=np.load("inject_ratio.npz")
fr=OO["freq"]

AIR,GND="#c0392b","#1f6fb4"
g4=C.db(np.nanmedian(T4,axis=1)); g4-=np.nanmedian(g4)
g0=C.db(np.nanmedian(T0,axis=1)); g0-=np.nanmedian(g0)
pk4=np.nanpercentile(g4,99)-np.nanpercentile(g4,1)
pk0=np.nanpercentile(g0,99)-np.nanpercentile(g0,1)
print(f"pk-pk (1-99pct): suspended {pk4:.1f} dB   ground {pk0:.2f} dB   ratio {10**((pk4-pk0)/20):.0f}x")
print(f"ground rms {np.nanstd(g0):.2f} dB")

fig=plt.figure(figsize=(7.1,5.0))
gs=fig.add_gridspec(2,2,height_ratios=[1,1.15],hspace=0.42,wspace=0.28)

# --- (a) spectrum, transmitter off vs on -------------------------------------
a=fig.add_subplot(gs[0,0])
m=(fr>=150)&(fr<=182)
a.plot(fr[m],C.db(OO["off_gnd"][m]),lw=.9,c="0.6",label="transmitter off")
a.plot(fr[m],C.db(OO["on_gnd"][m]),lw=.9,c="k",label="transmitter on")
a.set_xlabel("frequency [MHz]",fontsize=8); a.set_ylabel("power [dB, arb.]",fontsize=8)
a.legend(fontsize=6.5,loc="upper left",framealpha=.9); a.tick_params(labelsize=7); a.grid(alpha=.25)
a.set_title("(a)  injected comb, ground receiver",fontsize=8,loc="left")

# --- (b) per-tone detection across the band ----------------------------------
b=fig.add_subplot(gs[0,1])
b.plot(R["tone_freq"],R["on_gnd"],"o-",ms=2.5,lw=.9,c=GND,label="ground (fixed)")
b.plot(R["tone_freq"],R["on_air"],"o-",ms=2.5,lw=.9,c=AIR,label="suspended")
b.plot(R["tone_freq"],R["off_gnd"],"o-",ms=2,lw=.7,c="0.6",label="transmitter off")
b.axhline(0,c="0.7",lw=.6,ls=":")
b.set_xlabel("frequency [MHz]",fontsize=8); b.set_ylabel("tone / continuum [dB]",fontsize=8)
b.legend(fontsize=6.5,loc="upper left",framealpha=.9); b.tick_params(labelsize=7); b.grid(alpha=.25)
b.set_title("(b)  injected signal detected, 50–200 MHz",fontsize=8,loc="left")

# --- (c) rotation response ----------------------------------------------------
c=fig.add_subplot(gs[1,:])
ce=c.twinx()
ce.plot(tm,el,lw=.5,c="0.78",zorder=0)
ce.set_ylabel("rotation angle [deg]",fontsize=8,color="0.5")
ce.tick_params(labelsize=7,colors="0.5"); ce.set_ylim(-190,900)
ce.set_yticks([-180,-90,0,90,180])
c.plot(tm,g4,lw=.6,c=AIR,label=f"suspended, rotating  ({pk4:.0f} dB pk-pk)")
c.plot(tm,g0,lw=.8,c=GND,label=f"ground, fixed  ({np.nanstd(g0):.2f} dB rms)")
c.set_zorder(ce.get_zorder()+1); c.patch.set_visible(False)
c.set_xlabel("time since start of rotation sequence [min]",fontsize=8)
c.set_ylabel("injected-tone power\n[dB rel. median]",fontsize=8)
c.set_ylim(-24,14); c.set_xlim(0,tm[-1])
c.legend(fontsize=7,loc="lower left",ncol=2,framealpha=.9); c.tick_params(labelsize=7); c.grid(alpha=.25)
c.set_title("(c)  response to platform rotation, same injected signal, two identical receivers",fontsize=8,loc="left")

fig.savefig("12_draft_A.png",dpi=160,bbox_inches="tight")
print("saved 12_draft_A.png")
