"""Extend the EXIF prior cache to the transmitter-era images.

`marjum_2026_07_exif.npz` covers only the antenna-era images (2209-2245). The
six transmitter-era cameras (2159/2171/2172/2198/2199/2203) are absent, so a
joint fit over all 29 usable poses has no GPS, altitude or heading prior for
them. That gap is not cosmetic: camera 2159 was free to settle 158 degrees from
its own compass bearing and ~5x too far from the dish -- a false terrain minimum
that a heading prior would have disfavoured outright.

Those HEICs do carry GPS and compass; only the cache was missing them.

Existing entries are copied through byte-identically so that anything fitted
against the original cache stays reproducible. Only genuinely missing keys are
added.

Position needs a projection. The original cache's e_gps/n_gps are reconstructed
here as EPSG:6341 easting/northing minus a constant offset, calibrated on the
images the cache already covers. That reconstruction is good to ~1.5 m RMS, not
exact -- the original derivation is not recorded anywhere in the tree -- so new
entries carry that reconstruction scatter added in quadrature to their reported
GPS error. In practice the prior sigma has a 10 m floor
(`Config.gps_independent_floor_m`), so the correction is well inside it.

Heading needs no projection and is exact: the model's convention,
`heading = pi/2 - deg2rad(GPSImgDirection)`, reproduces 2203's fitted azimuth to
4 degrees and refit-B's 2159 to 3 degrees.

Usage: marjum_exif_extend.py [OUT.npz]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path('marjum_2026_07_exif.npz')
DEFAULT_OUT = Path('marjum_2026_07_exif_joint.npz')
IMAGES = Path('marjum-2026-07')
EPSG = 'EPSG:6341'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_exif(key):
    """GPS/compass/focal straight from the HEIC, or None when absent."""
    import pillow_heif
    pillow_heif.register_heif_opener()
    from PIL import Image
    from PIL.ExifTags import GPSTAGS

    path = IMAGES / f'IMG_{key}.HEIC'
    if not path.exists():
        return None
    exif = Image.open(path).getexif()
    gps = {GPSTAGS.get(t, t): v for t, v in exif.get_ifd(0x8825).items()}
    out = dict(key=key, datetime=exif.get(306))
    if 'GPSLatitude' in gps:
        def dms(t):
            return sum(float(x) / 60 ** i for i, x in enumerate(t))
        out['lat'] = dms(gps['GPSLatitude']) * (1 if gps['GPSLatitudeRef'] == 'N' else -1)
        out['lon'] = dms(gps['GPSLongitude']) * (1 if gps['GPSLongitudeRef'] == 'E' else -1)
        out['alt'] = float(gps.get('GPSAltitude', np.nan))
        out['heading_deg'] = float(gps.get('GPSImgDirection', np.nan))
        out['h_err_m'] = float(gps.get('GPSHPositioningError', np.nan))
    ifd = exif.get_ifd(0x8769)
    f35 = ifd.get(41989)
    out['focal_35mm'] = float(f35) if f35 else np.nan
    return out


def calibrate(src_keys, e_gps, n_gps, u_gps, has_gps, transformer):
    """Recover the constant offset between EPSG:6341 and the cache's ENU grid."""
    de, dn, du = [], [], []
    for k, e, n, u, h in zip(src_keys, e_gps, n_gps, u_gps, has_gps):
        if not h or not np.isfinite(e):
            continue
        ex = read_exif(k)
        if ex is None or 'lat' not in ex:
            continue
        X, Y = transformer.transform(ex['lon'], ex['lat'])
        de.append(X - e); dn.append(Y - n); du.append(ex['alt'] - u)
    de, dn, du = np.array(de), np.array(dn), np.array(du)
    return dict(n=len(de),
                offset=(float(de.mean()), float(dn.mean()), float(du.mean())),
                scatter=(float(de.std()), float(dn.std()), float(du.std())))


def main(out_path=DEFAULT_OUT):
    import pyproj
    out_path = Path(out_path)
    with np.load(SRC, allow_pickle=True) as z:
        src = {k: z[k] for k in z.files}
    src_keys = [str(k) for k in src['keys']]

    tr = pyproj.Transformer.from_crs('EPSG:4326', EPSG, always_xy=True)
    cal = calibrate(src_keys, src['e_gps'], src['n_gps'], src['u_gps'], src['has_gps'], tr)
    off_e, off_n, off_u = cal['offset']
    scat = float(np.hypot(cal['scatter'][0], cal['scatter'][1]))
    print(f"calibrated on {cal['n']} cached images: offset E {off_e:.3f} N {off_n:.3f} "
          f"U {off_u:.3f}; horizontal reconstruction scatter {scat:.2f} m")

    present = sorted(p.stem.replace('IMG_', '') for p in IMAGES.glob('IMG_*.HEIC'))
    missing = [k for k in present if k not in src_keys]
    print(f'{len(src_keys)} cached keys; {len(present)} images on disk; adding {len(missing)}: {missing}')

    keys = list(src_keys)
    cols = {name: list(np.asarray(src[name])) for name in
            ('e_gps', 'n_gps', 'u_gps', 'heading_deg', 'h_err_m', 'focal_35mm', 'has_gps', 'datetime')}
    added = {}
    for k in missing:
        ex = read_exif(k)
        if ex is None:
            continue
        if 'lat' in ex and np.isfinite(ex['lat']):
            X, Y = tr.transform(ex['lon'], ex['lat'])
            e, n, u = X - off_e, Y - off_n, ex['alt'] - off_u
            # The reconstruction is not exact; charge its scatter to this
            # entry's stated error rather than to the model.
            herr = float(np.hypot(ex['h_err_m'], scat)) if np.isfinite(ex['h_err_m']) else np.nan
            hg = True
        else:
            e = n = u = herr = np.nan
            hg = False
        keys.append(k)
        cols['e_gps'].append(e); cols['n_gps'].append(n); cols['u_gps'].append(u)
        cols['heading_deg'].append(ex.get('heading_deg', np.nan))
        cols['h_err_m'].append(herr)
        cols['focal_35mm'].append(ex.get('focal_35mm', np.nan))
        cols['has_gps'].append(hg)
        cols['datetime'].append(ex.get('datetime'))
        added[k] = dict(e=round(float(e), 2), n=round(float(n), 2), u=round(float(u), 2),
                        heading_deg=ex.get('heading_deg'), h_err_m=None if herr != herr else round(herr, 2),
                        focal_35mm=ex.get('focal_35mm'))
        print(f"  {k}: E {e:9.2f} N {n:10.2f} U {u:8.2f}  heading {ex.get('heading_deg')}  "
              f"h_err {herr:.1f}  f35 {ex.get('focal_35mm')}")

    payload = {name: np.array(vals, dtype=object if name == 'datetime' else None)
               for name, vals in cols.items()}
    payload['keys'] = np.array(keys)
    np.savez_compressed(out_path, **payload)

    # Verify the copied entries really are unchanged.
    with np.load(out_path, allow_pickle=True) as z:
        new_keys = [str(k) for k in z['keys']]
        for name in ('e_gps', 'n_gps', 'u_gps', 'heading_deg', 'h_err_m', 'focal_35mm'):
            a = np.asarray(src[name], float)
            b = np.asarray(z[name], float)[[new_keys.index(k) for k in src_keys]]
            assert np.array_equal(a, b, equal_nan=True), f'{name} changed for pre-existing keys'
    print('verified: every pre-existing entry is unchanged')

    prov = dict(product='marjum_exif_joint', source=str(SRC), source_sha256=sha256(SRC),
                crs=EPSG, calibration=cal, added=added, output=str(out_path),
                note=('Extends the antenna-era EXIF cache to the transmitter-era images so a '
                      'joint fit can give all 29 usable poses the same GPS/altitude/heading '
                      'priors. Pre-existing entries copied unchanged; new positions are '
                      'reconstructed via EPSG:6341 minus a calibrated constant offset, with the '
                      'reconstruction scatter added in quadrature to their GPS error.'))
    Path(str(out_path) + '.provenance.json').write_text(json.dumps(prov, indent=2, default=str) + '\n')
    print('wrote', out_path, 'and its provenance sidecar')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
