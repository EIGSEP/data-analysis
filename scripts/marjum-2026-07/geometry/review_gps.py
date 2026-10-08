"""Compare old- and UTM-target camera fits with raw HEIC GPS positions."""
from pathlib import Path
import hashlib
import json

import numpy as np

from eigsep_data.paths import get_campaign_root

CAMPAIGN = Path(get_campaign_root(required=True)).resolve()
HERE = CAMPAIGN / 'derived/geometry_posterior/v0004'
OLD = CAMPAIGN / 'derived/geometry_posterior/v0003'
OLD_EXIF = CAMPAIGN / 'derived/geometry_posterior/v0002/inputs/marjum_2026_07_exif_joint.npz'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def summary(residual, sigma):
    distance = np.linalg.norm(residual, axis=1)
    weight = 1 / np.maximum(sigma, 10)**2
    return dict(n=len(residual),
                mean_e_n_m=np.mean(residual, axis=0).tolist(),
                weighted_mean_e_n_m=np.average(residual, axis=0, weights=weight).tolist(),
                rms_horizontal_m=float(np.sqrt(np.mean(distance**2))),
                largest_horizontal_m=float(distance.max()),
                largest_camera_index=int(distance.argmax()),
                sum_squared_standardized=float(np.sum((distance / np.maximum(sigma, 10))**2)))


def main():
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    new_manifest = json.loads((HERE / 'manifest.json').read_text())
    assert sha(OLD / 'provisional_geometry.npz') == old_manifest['geometry_sha256']
    assert sha(HERE / 'provisional_geometry.npz') == new_manifest['geometry_sha256']
    assert sha(HERE / 'inputs/exif_utm.npz') == new_manifest['exif_utm_sha256']
    with np.load(OLD / 'provisional_geometry.npz') as old, \
         np.load(HERE / 'provisional_geometry.npz') as new, \
         np.load(OLD_EXIF, allow_pickle=True) as old_exif, \
         np.load(HERE / 'inputs/exif_utm.npz', allow_pickle=True) as new_exif:
        assert list(old['keys']) == list(new['keys'])
        fitted_bias = np.asarray(new['gps_bias'], float)
        old_index = {str(k): i for i, k in enumerate(old_exif['keys'])}
        new_index = {str(k): i for i, k in enumerate(new_exif['keys'])}
        rows = []
        for i, key in enumerate(old['keys']):
            a, b = old_index[key], new_index[key]
            old_gps = np.array([old_exif['e_gps'][a], old_exif['n_gps'][a]])
            new_gps = np.array([new_exif['e_gps'][b], new_exif['n_gps'][b]])
            if not np.isfinite(new_gps).all():
                continue
            old_camera = old['camera'][i, :2]
            new_camera = new['camera'][i, :2]
            rows.append(dict(key=str(key), gps_error_m=float(new_exif['h_err_m'][b]),
                             old_fit_old_gps=(old_camera-old_gps).tolist(),
                             old_fit_utm_gps=(old_camera-new_gps).tolist(),
                             utm_fit_utm_gps=(new_camera-new_gps).tolist(),
                             utm_fit_with_bias_utm_gps=(new_camera+fitted_bias-new_gps).tolist(),
                             camera_move_m=(new_camera-old_camera).tolist()))
        sigma = np.array([row['gps_error_m'] for row in rows])
        groups = {name: np.array([row[name] for row in rows])
                  for name in ('old_fit_old_gps', 'old_fit_utm_gps', 'utm_fit_utm_gps',
                               'utm_fit_with_bias_utm_gps')}
        eras = {}
        for era, keep in [('transmitter', np.array([int(row['key']) < 2209 for row in rows])),
                          ('91m', np.array([int(row['key']) >= 2209 for row in rows]))]:
            eras[era] = {name: summary(values[keep], sigma[keep])
                         for name, values in groups.items()}
            for name, entry in eras[era].items():
                keys = [row['key'] for row, selected in zip(rows, keep) if selected]
                entry['largest_camera'] = keys[entry.pop('largest_camera_index')]
        result = dict(status='provisional_in_sample_comparison',
                      scope='27 modeled HEIC cameras with GPS; no GPS held out of either fit',
                      old_frame='v0001 tangent-plane raster grid; cached HEIC priors',
                      new_frame='v0002 EPSG:6341 raster grid; raw HEIC latitude/longitude',
                      old_geometry_sha256=old_manifest['geometry_sha256'],
                      new_geometry_sha256=new_manifest['geometry_sha256'],
                      old_exif_sha256=sha(OLD_EXIF),
                      new_exif_sha256=sha(HERE / 'inputs/exif_utm.npz'),
                      review_source_sha256=sha(Path(__file__)),
                      fitted_utm_gps_bias_e_n_m=fitted_bias.tolist(),
                      summary={name: summary(values, sigma) for name, values in groups.items()},
                      by_era=eras,
                      cameras_improved_by_utm=int(np.sum(
                          np.linalg.norm(groups['old_fit_utm_gps'], axis=1)
                          < np.linalg.norm(groups['old_fit_old_gps'], axis=1))),
                      per_camera=rows)
        for name, row in result['summary'].items():
            row['largest_camera'] = rows[row.pop('largest_camera_index')]['key']
    target = HERE / 'gps_comparison.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
