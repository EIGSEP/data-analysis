"""Independent raster-horizon validation of saved CV initialization candidates."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eigsep_terrain.img import HorizonImage
from eigsep_terrain.imageio import load_image
from eigsep_terrain.fitio import load_fit
from eigsep_terrain.marjum_dem import MarjumDEM
from marjum_bundle import PRM_ORDER, project


def validate(before, after, output, dem_file='marjum_dem_sw.npz', decimate=12):
    """Use the existing ray tracer, not the optimizer's sampled skyline."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    p0, a0, _ = load_fit(before)
    p1, a1, _ = load_fit(after)
    dem = MarjumDEM(cache_file=dem_file)
    meta = json.loads(Path('meta.json').read_text())
    rows = []
    for key in p1:
        # Small raster only; don't construct full-resolution ray grids or smooth masks.
        rgb = np.flipud(load_image(f'marjum-2026-07/IMG_{key}.HEIC'))
        shape = rgb.shape[:2]
        display = rgb[::decimate, ::decimate].copy()
        del rgb
        with np.load(f'img_seg_IMG_{key}.npz') as z:
            sky = np.flipud(z['skymask'])[::decimate, ::decimate].astype(bool)
        img = HorizonImage.__new__(HorizonImage)
        img.img = np.empty((*shape, 0))
        rr, cc = np.meshgrid(np.arange(0, shape[0], decimate), np.arange(0, shape[1], decimate), indexing='ij')
        actual = img._raster_boundary(sky)
        actual_valid = np.isfinite(actual) & (actual < sky.shape[0]-1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        metrics = dict(key=key)
        for label, poses, ant, ax in zip(['before', 'after'], [p0, p1], [a0, a1], axes):
            p = np.array([poses[key][k] for k in PRM_ORDER])
            img.prms = dict(zip(PRM_ORDER, p))
            model = np.isnan(img.ray_distance(dem, img.get_rays((rr, cc))))
            predicted = img._raster_boundary(model)
            pred_valid = np.isfinite(predicted) & (predicted < sky.shape[0]-1)
            valid = actual_valid & pred_valid
            # Explicitly report missing/clipped columns, never silently improve by dropping them.
            error = (predicted[valid]-actual[valid])*decimate
            metrics[label] = dict(horizon_rms_px=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                                  matched_columns=int(valid.sum()), actual_columns=int(actual_valid.sum()),
                                  missing_columns=int((actual_valid & ~pred_valid).sum()),
                                  sky_disagreement_fraction=float(np.mean(sky != model)))
            ax.imshow(display, origin='lower', extent=(0, shape[1], 0, shape[0]))
            ax.contour(cc, rr, sky.astype(float), levels=[.5], colors='lime', linewidths=.8)
            ax.contour(cc, rr, model.astype(float), levels=[.5], colors='red', linewidths=.8)
            if 'ant_px' in meta.get(key, {}):
                q = np.array(meta[key]['ant_px'])
                pred, depth = project(p, shape, ant)
                metrics[label]['antenna_px'] = float(np.linalg.norm(pred[0]-q))
                metrics[label]['antenna_depth_m'] = float(depth[0])
                ax.plot(*q, 'm+', ms=10)
                ax.plot(*pred[0], 'cx', ms=8)
            ax.set(xlim=(0, shape[1]), ylim=(0, shape[0]), title=f'{key} {label}', xticks=[], yticks=[])
        fig.suptitle('Horizon: green observed, red DEM. Antenna: magenta pick, cyan prediction.')
        fig.tight_layout()
        fig.savefig(out/f'{key}.png', dpi=130)
        plt.close(fig)
        rows.append(metrics)
        print(key, metrics, flush=True)
    (out/'raster_validation.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', default='fit_bundle_v5_2231_polished.npz')
    p.add_argument('--after', default='cv_initialization/fit_cv.npz')
    p.add_argument('--output', default='cv_initialization/validation')
    p.add_argument('--dem-file', default='marjum_dem_sw.npz')
    args = p.parse_args()
    validate(**vars(args))
