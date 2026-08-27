"""Scan every comb tone: modulation depth and cleanliness, to pick 4 good frequencies."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,fr,ch=z["el"],z["freqs"] if "freqs" in z else z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
sel=np.zeros(len(el),bool)
for i in range(16,20): sel[SEGS[i][0]:SEGS[i][1]]=True
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y):
    y=y-np.nanmedian(y[sel])
    return np.array([np.nanmedian(y[sel&(ie==j)&np.isfinite(y)])
                     if (sel&(ie==j)&np.isfinite(y)).sum()>3 else np.nan for j in range(len(Ec))])
rough=lambda p: np.nanmedian(np.abs(np.diff(p,2)))
tone_ch=[c for c in ch[(ch%16)==8] if 50<fr[c]<200]
print(f"{'MHz':>7} {'ch':>5} {'tone depth':>11} {'tone rough':>11} {'nbr depth':>10} {'nbr rough':>10}  FM?")
rows=[]
for c in tone_ch:
    pt=fold(C.db(d4[:,c])); pn=fold(C.db(d4[:,c+4]))
    fm="FM" if 86<fr[c]<110 else ""
    rows.append((fr[c],c,np.nanmax(pt)-np.nanmin(pt),rough(pt),np.nanmax(pn)-np.nanmin(pn),rough(pn),fm))
    print(f"{fr[c]:7.1f} {c:5d} {rows[-1][2]:10.1f} {rows[-1][3]:11.3f} {rows[-1][4]:9.2f} {rows[-1][5]:10.3f}  {fm}")
ok=[r for r in rows if not r[6] and r[5]<0.06]
print(f"\nclean (non-FM, neighbour roughness < 0.06 dB): {len(ok)} tones")
print("  " + ", ".join(f"{r[0]:.0f}" for r in ok))
