"""Is the low tone-vs-neighbour correlation at high frequency real, or single-channel noise?
Recompute r with progressively more neighbour channels averaged."""
import numpy as np
from importlib.machinery import SourceFileLoader
C=SourceFileLoader("C","00_common.py").load_module()
z=C.load(); el,az,fr,ch=z["el"],z["az"],z["fr"],z["ch"]; d4=z["d"]
SEGS=np.load("segs.npz")["segs"]
E=np.arange(-180,181,5.0); Ec=.5*(E[:-1]+E[1:]); ie=np.digitize(el,E)-1
allm=np.zeros(len(el),bool)
for i in range(16,20): allm[SEGS[i][0]:SEGS[i][1]]=True
def fold(y):
    p=np.array([np.nanmedian(y[allm&(ie==j)&np.isfinite(y)])
                if (allm&(ie==j)&np.isfinite(y)).sum()>2 else np.nan for j in range(len(Ec))])
    j0=np.argmin(np.abs(Ec)); return p-np.nanmedian(p[j0-1:j0+2])
def r_of(a,b):
    g=np.isfinite(a)&np.isfinite(b); return np.corrcoef(a[g],b[g])[0,1]

tone_ch=ch[(ch%16)==8]
_,_,contam=C.comb_masks(ch)
print(f"{'MHz':>7} {'1 ch':>7} {'8 ch':>7} {'+-2MHz':>8} {'+-10MHz':>9} {'50-200 band':>12}   depth(8ch)")
wide=(((fr>=55)&(fr<=85))|((fr>=115)&(fr<=195)))&(~contam)
pw=fold(C.db(np.nanmedian(d4[:,wide],axis=1)))
for t in [84.0,123.0,158.2,189.5]:
    c=int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])
    pt=fold(C.db(d4[:,c]))
    n1=fold(C.db(d4[:,c+4]))
    n8=[c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
    p8=fold(C.db(np.nanmedian(d4[:,n8],axis=1)))
    m2=(np.abs(fr-fr[c])<=2)&(~contam); p2=fold(C.db(np.nanmedian(d4[:,m2],axis=1)))
    m10=(np.abs(fr-fr[c])<=10)&(~contam); p10=fold(C.db(np.nanmedian(d4[:,m10],axis=1)))
    print(f"{fr[c]:7.1f} {r_of(pt,n1):7.2f} {r_of(pt,p8):7.2f} {r_of(pt,p2):8.2f}"
          f" {r_of(pt,p10):9.2f} {r_of(pt,pw):12.2f}   {np.nanmax(p8)-np.nanmin(p8):.2f} dB")
print("\nneighbour-vs-neighbour across frequency (8-ch averages), a noise-free check of")
print("whether the sky modulation itself has a common shape:")
ps={}
for t in [84.0,123.0,158.2,189.5]:
    c=int(tone_ch[np.argmin(np.abs(fr[tone_ch]-t))])
    n8=[c+o for o in (-6,-5,-4,-3,3,4,5,6) if (c+o)%16 not in (7,8,9,15,0,1)]
    ps[fr[c]]=fold(C.db(np.nanmedian(d4[:,n8],axis=1)))
k=list(ps)
for i in range(len(k)):
    for j in range(i+1,len(k)):
        print(f"   {k[i]:6.1f} vs {k[j]:6.1f} MHz  r = {r_of(ps[k[i]],ps[k[j]]):+.2f}")
