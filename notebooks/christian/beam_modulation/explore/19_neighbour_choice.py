"""Which neighbouring channel is cleanest? Compare offsets around each tone."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,az,fr,ch=z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
sel=np.zeros(len(el),bool)
for i in range(16,20): sel[SEGS[i][0]:SEGS[i][1]]=True
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
def fold(y):
    y=y-np.nanmedian(y[sel])
    return np.array([np.nanmedian(y[sel&(ie==j)&np.isfinite(y)])
                     if (sel&(ie==j)&np.isfinite(y)).sum()>3 else np.nan for j in range(len(Ec))])
def rough(p):  # point-to-point roughness = RFI/noise proxy
    return np.nanmedian(np.abs(np.diff(p,2)))

tone_ch=ch[(ch%16)==8]
print(f"{'tone':>12}  offsets -7..+7 (residues 7,8,9,15,0,1 skipped): roughness [dB]")
best={}
for t in [60.5,80.1,130.9,177.7]:
    c=int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])
    row=[]
    for o in range(-7,8):
        cc=c+o
        if cc<0 or cc>=1024 or (cc%16) in (7,8,9,15,0,1): continue
        row.append((o,rough(fold(C.db(d4[:,cc])))))
    row.sort(key=lambda r:r[1])
    best[c]=row[0][0]
    print(f"  {fr[c]:7.1f} MHz  " + "  ".join(f"{o:+d}:{r:.3f}" for o,r in sorted(row)))
    print(f"{'':14s}best offset {row[0][0]:+d} ({row[0][1]:.3f})   +4 gives {dict(row)[4]:.3f}")

# does averaging the two flanking groups help?
print("\naveraging the flanking non-comb channels (c-6..c-3 and c+3..c+6):")
for t in [60.5,80.1,130.9,177.7]:
    c=int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])
    nb=[c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
    p=fold(C.db(np.nanmedian(d4[:,nb],axis=1)))
    p4=fold(C.db(d4[:,c+4]))
    print(f"  {fr[c]:7.1f} MHz  single(+4) rough {rough(p4):.3f} depth {np.nanmax(p4)-np.nanmin(p4):.2f}"
          f"   |  {len(nb)}-ch mean rough {rough(p):.3f} depth {np.nanmax(p)-np.nanmin(p):.2f}")
