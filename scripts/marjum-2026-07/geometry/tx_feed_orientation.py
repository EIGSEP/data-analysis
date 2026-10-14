"""Transmitter-feed orientation from the IMG_2203 feed-corner picks, at a geometry's poses.

Ported from the retired ``terrain/_tx_feed_orientation.py`` (terrain branch
``tx-feed-orientation``, ``bb6c990``; archived in ``terrain-archive.git``),
which made ``derived/tx_feed_orientation/v0001`` at the historical release pose
of IMG_2203 and the historical joint-MCMC draws. The feed model is unchanged:
the four picked corners are a planar square seen through the camera's pose,
perspective and radial distortion (both planar branches kept, the near-horizontal
one adopted); the rx6 axis joins the quadrants marked 6. The IMG_2203/IMG_2211
shared-rock triangulation is kept as a per-draw diagnostic.

Two modes:

``--geometry v0001``
    Reproduce v0001 from its frozen ``inputs/`` (baseline camera from
    ``fit_transmitter.npz``, draws from ``combined.npz``, every 100th draw,
    sensitivities at chain 0's middle draw). Bearings are taken on the raster
    axes, as v0001 reported them.
``--geometry v0004``
    Geometry release v0004: baseline cameras from
    ``imgs/fits/v0004_marjum_geometry/cameras.jsonl``; draws from the v0004
    chains (``derived/geometry_posterior/v0004/run/chain_*.npz``, every
    ``--stride``-th retained sweep); sensitivities at the release pose. The
    raster axes are UTM grid axes, so each bearing is reported both as a grid
    bearing and as a true bearing (grid + the DEM's grid convergence at the
    transmitter).

The camera model is the frozen ``marjum_camera.py``/``marjum_bundle.py`` in
``eigsep_terrain/marjum_geometry`` (``marjum_camera.py`` is byte-identical to the
terrain ``986e8cf`` copy v0001 used; ``marjum_bundle.rotation`` is unchanged).

Usage:
  tx_feed_orientation.py --geometry v0001|v0004 --out DIR [--stride N]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

PACKAGE = Path(importlib.util.find_spec('eigsep_terrain').submodule_search_locations[0])
MODELS = PACKAGE / 'marjum_geometry'
sys.path.insert(0, str(MODELS))
from marjum_bundle import rotation  # noqa: E402
from marjum_camera import normalized, project, rays  # noqa: E402

KEYS = ('2203', '2211')
# Unit square in cyclic order: image top, right, bottom, left. Its diagonals
# cross the opposite quadrants marked 6 and 1; the seams are NOT dipole axes.
SQUARE = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], float)
CV_TO_BODY = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])


def campaign_root():
    from eigsep_data.paths import get_campaign_root
    return Path(get_campaign_root(required=True))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_stamp(path):
    run = lambda *a: subprocess.run(['git', '-C', str(path), *a], capture_output=True,
                                    text=True).stdout.strip()
    return run('rev-parse', 'HEAD') + ('-dirty' if run('status', '--porcelain',
                                                        '--untracked-files=no') else '')


def bottom_up(xy, shape):
    xy = np.array(xy, float)
    xy[..., 1] = shape[0] - 1 - xy[..., 1]
    return xy


def bearing(v):
    return float(np.degrees(np.arctan2(v[0], v[1])) % 180)


def feed_solutions(camera, shape, distortion, corners, aspect=1.):
    square = SQUARE.copy()
    square[:, 0] *= aspect
    q = normalized(camera, shape, bottom_up(corners, shape), distortion)
    q[:, 1] *= -1  # Standard OpenCV image convention, y down.
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(square, q, np.eye(3), None,
                                             flags=cv2.SOLVEPNP_IPPE)
    out = []
    for rv, tv in zip(rvecs, tvecs):
        def residual(params):
            rc = cv2.Rodrigues(params[:3])[0]
            xc = square @ rc.T + params[3:]
            xw = camera[:3] + (xc @ CV_TO_BODY.T) @ rotation(camera).T
            return (project(camera, shape, xw, distortion)[0] - bottom_up(corners, shape)).ravel()
        fitted = least_squares(residual, np.r_[rv.ravel(), tv.ravel()], max_nfev=60)
        r = cv2.Rodrigues(fitted.x[:3])[0]
        tv = fitted.x[3:]
        xyz_cv = square @ r.T + tv.ravel()
        if np.min(xyz_cv[:, 2]) <= 0:
            continue
        world = rotation(camera) @ CV_TO_BODY @ r
        xyz = camera[:3] + (xyz_cv @ CV_TO_BODY.T) @ rotation(camera).T
        predicted = project(camera, shape, xyz, distortion)[0]
        res = predicted - bottom_up(corners, shape)
        axis6 = world @ (np.array([aspect, 1., 0]) / np.hypot(aspect, 1))
        axis1 = world @ (np.array([aspect, -1., 0]) / np.hypot(aspect, 1))
        out.append(dict(axis6_deg=bearing(axis6), axis1_deg=bearing(axis1),
                        tilt_deg=float(np.degrees(np.arccos(np.clip(abs(world[2, 2]), 0, 1)))),
                        corner_rms_px=float(np.sqrt(np.mean(res ** 2))),
                        center_distance_in_side_lengths=float(np.linalg.norm(tv)),
                        axis6_enu=axis6.tolist(), axis1_enu=axis1.tolist(),
                        normal_enu=world[:, 2].tolist()))
    # The mounting in the dish identifies the approximately horizontal face.
    return sorted(out, key=lambda v: v['tilt_deg'])


def rock_fit(cameras, shapes, distortion, rock):
    xy = [bottom_up(rock[i], shapes[i]) for i in range(2)]
    d = np.array([rays(cameras[i], shapes[i], [xy[i]], distortion[i])[0] for i in range(2)])
    centers = cameras[:, :3]
    lengths = np.linalg.lstsq(np.column_stack([d[0], -d[1]]), centers[1] - centers[0],
                              rcond=None)[0]
    starts = centers + lengths[:, None] * d

    def residual(pos):
        return np.concatenate([project(cameras[i], shapes[i], pos, distortion[i])[0][0] - xy[i]
                               for i in range(2)])
    fit = least_squares(residual, starts.mean(axis=0), max_nfev=80)
    depth = [project(cameras[i], shapes[i], fit.x, distortion[i])[1][0] for i in range(2)]
    return dict(enu_m=fit.x.tolist(), residual_px=fit.fun.tolist(),
                sse_px2=float(fit.fun @ fit.fun), positive_depth=bool(min(depth) > 0),
                ray_gap_m=float(np.linalg.norm(starts[0] - starts[1])),
                intersection_deg=float(np.degrees(np.arccos(np.clip(abs(d[0] @ d[1]), 0, 1)))),
                success=bool(fit.success))


def axial_unwrap(angles, valid):
    anchor = np.degrees(np.angle(np.mean(np.exp(2j * np.radians(angles[valid]))))) / 2
    return anchor + (angles - anchor + 90) % 180 - 90


# ------------------------------------------------------------------ geometries

def geometry_v0001(campaign, stride):
    """v0001's frozen inputs, read exactly as the original script read them."""
    root = campaign / 'derived/tx_feed_orientation/v0001/inputs'
    with np.load(root / 'fit_transmitter.npz') as z:
        keys = list(z['keys'])
        ids = [keys.index(k) for k in KEYS]
        cameras, shapes, distortion = [z[k][ids] for k in ['cameras', 'shapes', 'distortion']]
    with np.load(root / 'combined.npz') as z:
        names = list(z['names'])
        draws = z['draws'][:, ::stride]
        trace_keys = list(z['keys'])
        trace_ids = [trace_keys.index(k) for k in KEYS]
        diag = [names.index(f'cam{k}_{v}') for k in KEYS
                for v in ['e', 'n', 'u', 'th', 'ph', 'ti', 'logf']]
        convergence = dict(max_rhat=float(z['rhat'][diag].max()),
                           min_ess_bulk=float(z['ess_bulk'][diag].min()))
    samples = []
    for chain, trace in enumerate(draws):
        for j, row in enumerate(trace):
            cams = np.array([row[7 * i:7 * i + 7] for i in trace_ids])
            cams[:, 6] = np.exp(cams[:, 6])
            samples.append((chain, j * stride, cams))
    mid = np.array([draws[0, len(draws[0]) // 2, 7 * i:7 * i + 7] for i in trace_ids])
    mid[:, 6] = np.exp(mid[:, 6])
    sources = [root / n for n in ('fit_transmitter.npz', 'combined.npz', 'tx_feed_picks.json')]
    return dict(baseline=cameras, shapes=shapes, distortion=distortion, samples=samples,
                sensitivity_camera=mid, convergence=convergence, gamma_deg=0.0,
                picks=root / 'tx_feed_picks.json', sources=sources,
                pose_label='historical release pose (cv_transmitter_refit_2159B)',
                draws_label='historical joint MCMC (joint_posterior_v1)',
                sensitivity_label='chain 0 middle draw', bearing_frame='raster axes')


def geometry_v0004(campaign, stride):
    """Release v0004 cameras and the v0004 chains."""
    release = campaign / 'imgs/fits/v0004_marjum_geometry'
    product = campaign / 'derived/geometry_posterior/v0004'
    manifest = json.loads((product / 'manifest.json').read_text())
    rows = {json.loads(l)['image'][4:8]: json.loads(l)
            for l in (release / 'cameras.jsonl').read_text().splitlines()}
    cams, shapes, dist = [], [], []
    for k in KEYS:
        c = rows[k]['camera']
        o = c['orientation_rad']
        cams.append([*c['position_enu_m'], o['theta'], o['phi'], o['tilt'], c['focal_length_px']])
        shapes.append([c['image_shape_px']['height'], c['image_shape_px']['width']])
        dist.append([c['radial_distortion']['k1'], c['radial_distortion']['k2']])
    cams, shapes, dist = np.array(cams), np.array(shapes), np.array(dist)
    with np.load(product / 'provisional_geometry.npz') as g:
        names = [str(n) for n in g['global_names']]
        tx = np.asarray(g['transmitter'], float)
    cols = [[names.index(f'cam{k}_{v}') for v in ('e', 'n', 'u', 'th', 'ph', 'ti', 'logf')]
            for k in KEYS]
    samples = []
    for chain in manifest['chains']:
        path = product / 'run' / f'chain_{chain}.npz'
        if sha256(path) != manifest['chain_sha256'][str(chain)]:
            raise SystemExit(f'{path} does not match the v0004 manifest')
        with np.load(path) as z:
            G = np.asarray(z['globals'], float)
        for j in range(0, len(G), stride):
            c = np.array([G[j, ix] for ix in cols])
            c[:, 6] = np.exp(c[:, 6])
            samples.append((int(chain), j, c))
    report = json.loads((product / 'run/convergence_report.json').read_text())
    rec = [r for r in report['coordinates'] if r['name'].split('_')[0] in ('cam2203', 'cam2211')]
    convergence = dict(max_rhat=max(r['rhat'] for r in rec), min_ess_bulk=min(r['bulk_ess'] for r in rec))
    from eigsep_terrain.dem import DEM
    dem = DEM(cache_file=str(campaign / 'derived/dem/v0002/marjum_dem.npz'))
    gamma = float(np.degrees(dem.grid_convergence(tx[0], tx[1])))
    picks = campaign / 'derived/tx_feed_orientation/v0001/inputs/tx_feed_picks.json'
    sources = [release / 'cameras.jsonl', product / 'manifest.json',
               product / 'provisional_geometry.npz', product / 'run/convergence_report.json',
               picks, campaign / 'derived/dem/v0002/marjum_dem.npz']
    return dict(baseline=cams, shapes=shapes, distortion=dist, samples=samples,
                sensitivity_camera=cams, convergence=convergence, gamma_deg=gamma,
                picks=picks, sources=sources,
                pose_label='release v0004 pose', draws_label='geometry_posterior v0004 chains',
                sensitivity_label='release v0004 pose', bearing_frame='UTM grid; true = grid + gamma')


# ------------------------------------------------------------------ run

def run(geo, out, stride, mode):
    picks = json.loads(Path(geo['picks']).read_text())
    rock = [picks['rock_topdown_px'][k] for k in KEYS]
    corners = picks['feed_corners_topdown_px']['2203']
    shapes, distortion = geo['shapes'], geo['distortion']
    baseline = dict(rock=rock_fit(geo['baseline'], shapes, distortion, rock),
                    feed=feed_solutions(geo['baseline'][0], shapes[0], distortion[0], corners))
    records = []
    for chain, draw, cams in geo['samples']:
        records.append(dict(chain=chain, draw=draw,
                            rock=rock_fit(cams, shapes, distortion, rock),
                            feed=feed_solutions(cams[0], shapes[0], distortion[0], corners)))
    valid = np.array([r['rock']['positive_depth'] and r['rock']['success'] and bool(r['feed'])
                      for r in records])
    sse = np.array([r['rock']['sse_px2'] for r in records])
    a6 = np.array([r['feed'][0]['axis6_deg'] if r['feed'] else np.nan for r in records])
    a1 = np.array([r['feed'][0]['axis1_deg'] if r['feed'] else np.nan for r in records])
    u6, u1 = axial_unwrap(a6, valid), axial_unwrap(a1, valid)
    chains = sorted({r['chain'] for r in records})
    summary = dict(
        product='tx_feed_orientation', geometry=mode,
        status='feed orientation from Aaron\'s IMG_2203 corner picks; draw spread is over '
               'non-converged chains, not a confidence interval',
        pose=geo['pose_label'], draws=geo['draws_label'],
        sensitivity_pose=geo['sensitivity_label'], bearing_frame=geo['bearing_frame'],
        grid_convergence_deg=geo['gamma_deg'],
        picks=picks, baseline=baseline, convergence=geo['convergence'],
        sampled_draws=len(records), stride=stride, compatibility={})
    for sigma in picks['rock_sigma_sensitivity_px']:
        logw = -.5 * sse / sigma ** 2
        logw[~valid] = -np.inf
        w = np.exp(logw - np.max(logw))
        w /= w.sum()
        mu = float(w @ u6)
        summary['compatibility'][str(sigma)] = dict(
            profile_weight_count=float(1 / (w @ w)), axis6_mean_deg=mu % 180,
            axis6_sd_deg=float(np.sqrt(w @ ((u6 - mu) ** 2))),
            chain_weight={str(c): float(sum(x for x, r in zip(w, records) if r['chain'] == c))
                          for c in chains})
    summary['unweighted_axis6'] = dict(
        mean_deg=float(np.mean(u6[valid]) % 180), sd_deg=float(np.std(u6[valid])),
        per_chain_mean_deg={str(c): float(np.mean(u6[[r['chain'] == c for r in records]]) % 180)
                            for c in chains})
    summary['unweighted_axis1'] = dict(mean_deg=float(np.mean(u1[valid]) % 180),
                                       sd_deg=float(np.std(u1[valid])))
    summary['rock_residual_norm_quantiles_px'] = np.quantile(np.sqrt(sse), [.05, .5, .95]).tolist()
    summary['positive_depth_fraction'] = float(np.mean(valid))
    rng = np.random.default_rng(20260930)
    cam = geo['sensitivity_camera'][0]
    summary['corner_pick_sensitivity'] = {}
    for sigma in [2., 5., 10.]:
        vals = np.array([feed_solutions(cam, shapes[0], distortion[0],
                                        np.array(corners) + rng.normal(0, sigma, (4, 2)))[0]['axis6_deg']
                         for _ in range(100)])
        summary['corner_pick_sensitivity'][str(sigma)] = dict(
            median_deg=float(np.median(vals)), sd_deg=float(np.std(vals)),
            q05_q95_deg=np.quantile(vals, [.05, .95]).tolist())
    summary['aspect_sensitivity'] = {str(a): feed_solutions(cam, shapes[0], distortion[0], corners, a)
                                     for a in [.9, 1., 1.1]}
    g = geo['gamma_deg']
    summary['true_bearings_deg'] = dict(
        baseline_axis6=(baseline['feed'][0]['axis6_deg'] + g) % 180,
        baseline_axis1=(baseline['feed'][0]['axis1_deg'] + g) % 180,
        draws_axis6_mean=(summary['unweighted_axis6']['mean_deg'] + g) % 180,
        draws_axis1_mean=(summary['unweighted_axis1']['mean_deg'] + g) % 180)
    summary['generated_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    summary['code'] = dict(script=str(Path(__file__).name), sha256=sha256(Path(__file__)),
                           data_analysis_commit=git_stamp(Path(__file__).parent),
                           eigsep_terrain_commit=git_stamp(PACKAGE),
                           marjum_camera_sha256=sha256(MODELS / 'marjum_camera.py'),
                           marjum_bundle_sha256=sha256(MODELS / 'marjum_bundle.py'))
    campaign = campaign_root()
    summary['inputs'] = {str(Path(p).relative_to(campaign.parent)) if str(p).startswith(str(campaign.parent))
                         else str(p): sha256(p) for p in geo['sources']}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'draw_diagnostics.json').write_text(json.dumps(records) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--geometry', choices=['v0001', 'v0004'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stride', type=int, default=None,
                   help='draw stride (default 100 for v0001, as it was run; 20 for v0004)')
    args = p.parse_args()
    campaign = campaign_root()
    stride = args.stride or (100 if args.geometry == 'v0001' else 20)
    geo = (geometry_v0001 if args.geometry == 'v0001' else geometry_v0004)(campaign, stride)
    s = run(geo, args.out, stride, args.geometry)
    print(json.dumps({k: s[k] for k in ('unweighted_axis6', 'unweighted_axis1', 'true_bearings_deg',
                                        'sampled_draws', 'convergence')}, indent=1))


if __name__ == '__main__':
    main()
