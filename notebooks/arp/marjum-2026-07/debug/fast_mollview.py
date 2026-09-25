"""Fast Mollweide rendering via direct projection + imshow.

``healpy.mollview`` is very slow when called hundreds of times (each
call redoes matplotlib figure/axes/colorbar/graticule setup from
scratch) -- ~0.5-1s+ per panel, dominating runtime for a many-row
comparison grid. Projecting the map to a 2D array once via the same
projector healpy uses internally, then drawing it with a single
``imshow``, is the same math without that overhead (measured: ~0.007s
per projection).
"""
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

_PROJECTORS = {}


def _projector(nside, xsize=300):
    key = (nside, xsize)
    if key not in _PROJECTORS:
        _PROJECTORS[key] = hp.projector.MollweideProj(xsize=xsize)
    return _PROJECTORS[key]


def fast_mollview(ax, m, nside, vmin=None, vmax=None, cmap="viridis",
                  title=None, xsize=300, bad_color="0.8"):
    """Draw ``m`` (a HEALPix map, may contain ``hp.UNSEEN``) onto ``ax``."""
    proj = _projector(nside, xsize)
    img = proj.projmap(m, lambda x, y, z: hp.vec2pix(nside, x, y, z))
    img = np.asarray(img, dtype=float)
    img[~np.isfinite(img)] = np.nan
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(bad_color)
    ax.imshow(img, origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax,
             aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
