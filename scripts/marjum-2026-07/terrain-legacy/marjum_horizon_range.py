"""Per-camera horizon spread across the posterior draws.

For each camera, ray-trace the model sky/ground boundary at a handful of
posterior draws and overlay them on the photograph together with the actual
segmentation. This shows what the sampler's uncertainty in a camera pose
actually does to the predicted horizon -- which is the physically meaningful
consequence of the focal-length stiffness diagnosed in the joint run.

Draws are chosen, not rendered exhaustively: the median draw plus the extremes
of the two coordinates that mix worst for that camera (log focal length and
altitude). Five traces per camera keeps the render affordable while bracketing
the spread.

Usage: marjum_horizon_range.py RUN_DIR [--keys 2159 2213] [--out DIR]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from eigsep_terrain.img import HorizonImage  # noqa: E402
from eigsep_terrain.marjum_dem import MarjumDEM as DEM  # noqa: E402
from marjum_bundle import working_grid  # noqa: E402
from marjum_camera import rays as camera_rays  # noqa: E402

DECIMATE = 6


def select_draws(flat, base, names, key):
    """Median draw plus the extremes of this camera's two stiffest coordinates.

    Returns [(label, camera_vector), ...]. `flat` is (draw, param) pooled over
    chains; `base` is the index of this camera's first coordinate.
    """
    logf = flat[:, base + 6]
    u = flat[:, base + 2]
    picks = [
        ('median', int(np.argsort(logf)[len(logf) // 2])),
        ('min focal', int(np.argmin(logf))),
        ('max focal', int(np.argmax(logf))),
        ('min altitude', int(np.argmin(u))),
        ('max altitude', int(np.argmax(u))),
    ]
    out, seen = [], set()
    for label, i in picks:
        if i in seen:
            continue
        seen.add(i)
        cam = flat[i, base:base + 7].copy()
        cam[6] = np.exp(cam[6])          # sampled as log focal
        out.append((label, cam))
    return out


def model_sky(img, cam, distort, dem, decimate=DECIMATE):
    # ray_distance takes ray *directions* and reads the origin off the image's
    # own pose, so the pose must be set for each draw. Without this every ray
    # is traced from a stale origin and the whole frame comes back as ground.
    img.set_prms(list(cam))
    rr, cc = np.mgrid[0:img.npix_y:decimate, 0:img.npix_x:decimate]
    world = camera_rays(cam, (img.npix_y, img.npix_x),
                        np.c_[cc.ravel(), rr.ravel()], distort)
    world = world.T.reshape(3, *rr.shape)
    return np.isnan(img.ray_distance(dem, world))


def render(run_dir, keys=None, out_dir='horizon_range'):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    with np.load(run_dir / 'combined.npz', allow_pickle=True) as z:
        draws = np.asarray(z['draws'], float)
        names = [str(s) for s in z['names']]
        all_keys = [str(s) for s in z['keys']]
        ess = np.asarray(z['ess_bulk'], float)
        rhat = np.asarray(z['rhat'], float)
    flat = draws.reshape(-1, draws.shape[-1])

    manifest = json.loads((run_dir / 'manifest.json').read_text())
    state = Path(manifest['state_file'])
    with np.load(state) as s:
        st_keys = [str(k) for k in s['keys']]
        distortion = dict(zip(st_keys, np.asarray(s['distortion'], float)))

    meta = json.loads(Path('meta.json').read_text())
    dem = working_grid(DEM(cache_file='marjum_dem_sw.npz'))

    todo = keys or all_keys
    summary = {}
    for key in todo:
        base = all_keys.index(key) * 7
        picks = select_draws(flat, base, names, key)
        img = HorizonImage(f'marjum-2026-07/IMG_{key}.HEIC', meta=meta,
                           px_smooth=150, px_dist=30)
        sl = slice(None, None, DECIMATE)
        actual = img.sky_mask[sl, sl]

        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.imshow(img.img[sl, sl], origin='lower')
        ax.contour(actual.astype(float), levels=[.5], colors='lime', linewidths=1.6)
        colors = plt.cm.autumn(np.linspace(0, .85, len(picks)))
        spread = []
        for (label, cam), col in zip(picks, colors):
            m = model_sky(img, cam, distortion[key], dem)
            ax.contour(m.astype(float), levels=[.5], colors=[col], linewidths=1.0)
            valid = m.any(0) & ~m.all(0)
            rows = np.where(valid, np.argmax(m, axis=0), np.nan)
            spread.append(rows)
        spread = np.array(spread, float)
        # Columns that are entirely sky or entirely ground in every draw carry
        # no horizon and are dropped rather than warned about.
        import warnings
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            band = np.nanmax(spread, axis=0) - np.nanmin(spread, axis=0)
        band_px = float(np.nanmedian(band) * DECIMATE) if np.isfinite(band).any() else float('nan')
        summary[key] = dict(
            median_horizon_spread_px=round(band_px, 1),
            max_horizon_spread_px=round(float(np.nanmax(band) * DECIMATE), 1) if np.isfinite(band).any() else None,
            logf_ess=round(float(ess[base + 6]), 1), logf_rhat=round(float(rhat[base + 6]), 3),
            draws=[label for label, _ in picks])
        handles = [plt.Line2D([], [], color='lime', lw=1.6, label='actual segmentation')]
        handles += [plt.Line2D([], [], color=c, lw=1.0, label=l)
                    for (l, _), c in zip(picks, colors)]
        ax.legend(handles=handles, fontsize=6, loc='lower left', framealpha=.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'IMG_{key}: horizon across posterior draws  '
                     f'(median spread {band_px:.0f} px; logf ESS {ess[base+6]:.0f}, '
                     f'Rhat {rhat[base+6]:.2f})', fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f'horizon_range_{key}.png', dpi=110)
        plt.close(fig)
        del img
        gc.collect()
        print(f'  {key}: median spread {band_px:6.1f} px  logf ESS {ess[base+6]:5.0f}', flush=True)

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--keys', nargs='*', default=None)
    ap.add_argument('--out', default='horizon_range')
    a = ap.parse_args()
    render(a.run_dir, a.keys, a.out)
