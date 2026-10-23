import numpy as np, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, healpy as hp
S = sys.argv[1]; L = np.load(f"{S}/variants_loo.npz")
fig = plt.figure(figsize=(14, 8))
for r, d in enumerate((82, 87)):
    for c, (var, lab) in enumerate((("baseline", "before"), ("bracketed", "after"), ("bracketed", "HFSS"))):
        pix, u, tr = L[f"{var}_d{d}_pix"], L[f"{var}_d{d}_u"], L[f"{var}_d{d}_truth"]
        nside = hp.npix2nside(len(L[f"{var}_d{d}_truth"]) * 0 + 12 * 32**2)
        p = np.sum(np.abs(tr if lab == "HFSS" else u)**2, 1); p = p / np.nanmedian(p)
        m = np.full(12 * nside**2, hp.UNSEEN); m[pix] = 10 * np.log10(np.maximum(p, 1e-6))
        hp.orthview(m, rot=(0, 90, 0), half_sky=True, sub=(2, 3, 3*r + c + 1), cmap="viridis", min=-15, max=10,
                    title=f"d{d} {lab}  (dB re median, max {10*np.log10(np.nanmax(p)):.0f} dB)", unit="dB", badcolor="#e8e7e3", bgcolor="#fcfcfb")
fig.savefig(f"{S}/fig4_maps.png", dpi=105, facecolor="#fcfcfb")
