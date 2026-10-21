import numpy as np, sys
z = np.load(sys.argv[1]); sel=z["sel"]; nc = z["Ex_all"].shape[1]
print(" d   arm  nfill  bracketed%   absurd(>100x): n  bracketed  | ratio max among bracketed / unbracketed")
for k,d in enumerate(z["d_idx"]):
    v = z["valid"][k]&sel
    for arm in ("Ex","Ey"):
        A=z[arm+"_all"]; M=z[arm+"_meas"]
        fill = v & ~M[:,d]
        if fill.sum()==0: continue
        lo = M[:,d-1] if d>0 else np.zeros(len(v),bool); hi = M[:,d+1] if d+1<nc else np.zeros(len(v),bool)
        br = lo & hi
        p=np.abs(A[:,d])**2; med=np.median(p[v]); ab = fill & (p>100*med)
        r_br = (p[fill&br]/med).max() if (fill&br).any() else np.nan
        r_un = (p[fill&~br]/med).max() if (fill&~br).any() else np.nan
        print(f"d{d} {arm} {fill.sum():5d} {100*br[fill].mean():8.1f}%   {ab.sum():5d} {br[ab].sum():6d}       | {r_br:9.3g} / {r_un:9.3g}")
