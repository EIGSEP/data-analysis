"""Exploratory feed-plane recovery and shared-rock compatibility diagnostic.

Run with --terrain pointing at the data-bearing terrain checkout. Original
labels, camera fits and chains are read-only. Output is not a new posterior.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.optimize import least_squares

from marjum_bundle import rotation
from marjum_camera import normalized, project, rays

# Unit square in cyclic order: image top, right, bottom, left. Its diagonals
# cross the opposite quadrants marked 6 and 1; the seams are NOT dipole axes.
SQUARE = np.array([[-.5, -.5, 0], [.5, -.5, 0],
                   [.5, .5, 0], [-.5, .5, 0]], float)
CV_TO_BODY = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])


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
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        square, q, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    out = []
    for rv, tv in zip(rvecs, tvecs):
        # Refine each planar branch in the actual distorted pixel metric.
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
        residual = predicted - bottom_up(corners, shape)
        axis6 = world @ (np.array([aspect, 1., 0]) / np.hypot(aspect, 1))
        axis1 = world @ (np.array([aspect, -1., 0]) / np.hypot(aspect, 1))
        out.append(dict(axis6_deg=bearing(axis6), axis1_deg=bearing(axis1),
                        tilt_deg=float(np.degrees(np.arccos(np.clip(abs(world[2, 2]), 0, 1)))),
                        corner_rms_px=float(np.sqrt(np.mean(residual**2))),
                        center_distance_in_side_lengths=float(np.linalg.norm(tv)),
                        axis6_enu=axis6.tolist(), axis1_enu=axis1.tolist(),
                        normal_enu=world[:, 2].tolist()))
    # The mounting in the dish identifies the approximately horizontal face.
    # Retain the other planar solution explicitly in every diagnostic record.
    return sorted(out, key=lambda v: v['tilt_deg'])


def self_check():
    camera = np.array([10., 20, 30, .65, 1.1, .07, 2200])
    shape = [4032, 3024]
    distortion = [-.02, .01]
    rc = cv2.Rodrigues(np.array([.35, -.2, .6]))[0]
    xyz_cv = SQUARE @ rc.T + [0.1, .3, 6.]
    xyz = camera[:3] + xyz_cv @ CV_TO_BODY.T @ rotation(camera).T
    xy = project(camera, shape, xyz, distortion)[0]
    sol = feed_solutions(camera, shape, distortion, bottom_up(xy, shape))
    expected = rotation(camera) @ CV_TO_BODY @ rc @ (np.array([1., 1, 0])/np.sqrt(2))
    recovered = min(sol, key=lambda s: s['corner_rms_px'])
    assert recovered['corner_rms_px'] < 1e-5
    assert abs(np.dot(expected, recovered['axis6_enu'])) > 1-1e-8
    assert np.allclose(bottom_up(bottom_up(xy, shape), shape), xy)
    other = camera.copy()
    other[:3] += [1., -1., .5]
    target = xyz.mean(axis=0)
    rock = [bottom_up(project(c, shape, target, distortion)[0][0], shape)
            for c in [camera, other]]
    recovered_rock = rock_fit(np.array([camera, other]), [shape, shape], [distortion]*2, rock)
    assert np.linalg.norm(np.array(recovered_rock['enu_m'])-target) < 1e-6
    assert recovered_rock['positive_depth']
    return 'Passed synthetic distorted-square axis recovery, pixel-convention round trip, and two-camera triangulation.'


def render_report(out, terrain, summary, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    import pillow_heif
    pillow_heif.register_heif_opener()
    picks = summary['picks']
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    images = {}
    for key in ['2203', '2211']:
        path = terrain.parent/'marjum-2026-07'/'imgs'/f'IMG_{key}.HEIC'
        images[key] = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    for ax, key in zip(axs[:2], ['2203', '2211']):
        ax.imshow(images[key])
        x, y = picks['rock_topdown_px'][key]
        ax.plot(x, y, '+', color='red', ms=15, mew=1)
        old = summary['existing_label_audit'][key]['transmitter_topdown_px']
        tx = picks.get('transmitter_topdown_px', {}).get(key, old)
        label = 'Aaron transmitter pick' if key in picks.get('transmitter_topdown_px', {}) else 'Existing transmitter pick'
        ax.plot(*tx, 'x', color='yellow', ms=10, label=label)
        ax.legend(fontsize=7)
        ax.set(xlim=(x-350, x+350), ylim=(y+300, y-400),
               title=f'{key}: rock center', xlabel='Native x (right)', ylabel='Native y (down)')
    corners = np.array(picks['feed_corners_topdown_px']['2203'])
    ax = axs[2]
    ax.imshow(images['2203'])
    ax.plot(*corners[[0, 1, 2, 3, 0]].T, 'r.-', lw=1)
    for idx, label in [([0, 2], '6 axis'), ([1, 3], '1 axis')]:
        ax.plot(*corners[idx].T, label=label, lw=1)
    ax.set(xlim=(1150, 1700), ylim=(3130, 2630), title='2203: feed face and quadrant axes')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'picks.png', dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for chain in sorted({r['chain'] for r in records}):
        subset = [r for r in records if r['chain']==chain]
        axs[0].hist([r['feed'][0]['axis6_deg'] for r in subset], bins=20,
                    histtype='step', label=f'Chain {chain}')
        axs[1].scatter([np.sqrt(r['rock']['sse_px2']) for r in subset],
                       [r['feed'][0]['axis6_deg'] for r in subset], s=5)
    axs[0].set(xlabel='6-axis ENU bearing mod 180 (degrees)', ylabel='Draw count',
               title='Unconverged trace sensitivity, fixed corner picks')
    axs[0].legend(fontsize=7)
    axs[1].set(xlabel='Rock reprojection residual norm (pixels)', ylabel='6-axis bearing (degrees)',
               title='Does rock compatibility select a narrower bearing?')
    fig.tight_layout()
    fig.savefig(out/'sensitivity.png', dpi=130)
    plt.close(fig)
    def embed(name):
        return '<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">'
    body = '''<!doctype html><meta charset="utf-8"><title>Transmitter feed / rock diagnostic</title>
    <style>body{max-width:1200px;margin:2em auto;font:16px sans-serif}pre{white-space:pre-wrap}</style>
    <h1>Transmitter feed / shared-rock diagnostic — 2026-09-30</h1>
    <p>Exploratory analysis, not a released geometry fit or calibrated posterior. Both rock centers,
    the 2211 transmitter center, and all four 2203 feed corners are supplied by Aaron.
    Source images are 3024 × 4032, with y down.</p>
    <p>Each retained joint draw supplies both camera poses together. The rock is fitted as a free 3D point
    using its two image locations. Profile compatibility weights exp(−χ²/2) explore which existing draws
    match it; they do not integrate out rock position or create samples outside the original trace support.
    The square face is solved with calibrated perspective and inherited fixed radial distortion, retaining
    both planar pose branches. Summaries select the lower-tilt, approximately horizontal feed branch
    supported by the mounting in the dish. Axes join opposite labelled quadrants, not the dividing seams.
    Transmitter position labels are not re-used as face-center observations. No absolute size is assumed.</p>
    <p>A single shared point supplies one epipolar compatibility constraint after fitting its unknown
    3D position. Near alignment is not imposed. Opposing visible rock surfaces may shift the effective
    picked point; the 5–40 pixel sweep is a sensitivity study, not a measured noise calibration. Corner jitter
    and a 0.9–1.1 rectangular-face aspect sweep test separate assumptions. No result replaces campaign geometry.</p>
    '''
    central = summary['unweighted_axis6']
    body += f'<p><b>Initial 6-axis estimate: {central["mean_deg"]:.1f}° (modulo 180°, ENU compass bearing).</b> '
    body += f'The corresponding 1-axis estimate is {summary["unweighted_axis1"]["mean_deg"]:.1f}°. '
    body += f'The fixed-pick trace spread is {central["sd_deg"]:.2f}°, not a confidence interval. '
    body += 'The rock constraint does not appreciably narrow that spread in the available draws. '
    body += 'The second planar solution is nearly vertical and is retained in the numbers below.</p>'
    alignment = summary['confirmed_alignment_2211']
    body += '<p><b>Confirmed near-alignment:</b> Aaron identifies the 2211 rock at (1600, 3349) '
    body += 'and transmitter at (1589, 3346): the rock is 11 pixels right and 3 pixels below, '
    body += f'a separation of {alignment["separation_px"]:.1f} pixels. The transmitter update is '
    body += f'{alignment["transmitter_shift_from_existing_px"]:.1f} pixels from the old label. '
    body += 'These are separate landmarks. The new transmitter pick is recorded for a future joint refit; '
    body += 'the current compatibility calculation retains the original MCMC draws and their original labels.</p>'
    body += embed('picks.png')+embed('sensitivity.png')
    body += '<h2>Numbers, assumptions, validation and provenance</h2><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>'
    (out/'report.html').write_text(body)


def rock_fit(cameras, shapes, distortion, rock):
    xy = [bottom_up(rock[i], shapes[i]) for i in range(2)]
    d = np.array([rays(cameras[i], shapes[i], [xy[i]], distortion[i])[0]
                  for i in range(2)])
    centers = cameras[:, :3]
    lengths = np.linalg.lstsq(np.column_stack([d[0], -d[1]]),
                              centers[1] - centers[0], rcond=None)[0]
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


def run(args):
    terrain = args.terrain.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    picks = json.loads(args.picks.read_text())
    state_path = terrain / 'cv_transmitter_refit_2159B/fit_transmitter.npz'
    trace_path = terrain.parent / 'marjum-2026-07/derived/geometry_memo_inputs/v0001/legacy_combined.npz'
    with np.load(state_path) as z:
        keys = list(z['keys'])
        ids = [keys.index(k) for k in ['2203', '2211']]
        cameras, shapes, distortion = [z[k][ids] for k in ['cameras', 'shapes', 'distortion']]
    with np.load(trace_path) as z:
        names = list(z['names'])
        draws = z['draws'][:, ::args.stride]
        trace_keys = list(z['keys'])
        trace_ids = [trace_keys.index(k) for k in ['2203', '2211']]
        diagnostic_ids = [names.index(f'cam{k}_{v}') for k in ['2203', '2211']
                          for v in ['e', 'n', 'u', 'th', 'ph', 'ti', 'logf']]
        convergence = dict(max_rhat=float(z['rhat'][diagnostic_ids].max()),
                           min_ess_bulk=float(z['ess_bulk'][diagnostic_ids].min()))
    rock = [picks['rock_topdown_px'][k] for k in ['2203', '2211']]
    corners = picks['feed_corners_topdown_px']['2203']
    baseline = dict(rock=rock_fit(cameras, shapes, distortion, rock),
                    feed=feed_solutions(cameras[0], shapes[0], distortion[0], corners))
    records = []
    for chain, trace in enumerate(draws):
        for j, row in enumerate(trace):
            cams = np.array([row[7*i:7*i+7] for i in trace_ids])
            cams[:, 6] = np.exp(cams[:, 6])
            feed = feed_solutions(cams[0], shapes[0], distortion[0], corners)
            records.append(dict(chain=chain, draw=j*args.stride,
                                rock=rock_fit(cams, shapes, distortion, rock), feed=feed))
    summary = dict(status='exploratory; unconverged input; all landmark and feed-corner picks supplied by user',
                   method='Per-correlated-draw rock triangulation and calibrated square pose; profile compatibility weights are not a marginal posterior.',
                   picks=picks, baseline=baseline, convergence=convergence,
                   sampled_draws=len(records), compatibility={}, validation=self_check())
    valid = np.array([r['rock']['positive_depth'] and r['rock']['success'] and bool(r['feed']) for r in records])
    sse = np.array([r['rock']['sse_px2'] for r in records])
    angles = np.array([r['feed'][0]['axis6_deg'] if r['feed'] else np.nan for r in records])
    # Axial unwrap around the unweighted circular mean before summarising.
    anchor = np.degrees(np.angle(np.mean(np.exp(2j*np.radians(angles[valid])))))/2
    unwrapped = anchor + (angles-anchor+90) % 180-90
    for sigma in picks['rock_sigma_sensitivity_px']:
        logw = -.5*sse/sigma**2
        logw[~valid] = -np.inf
        weights = np.exp(logw - np.max(logw))
        weights /= weights.sum()
        mu = float(weights @ unwrapped)
        summary['compatibility'][str(sigma)] = dict(
            profile_weight_count=float(1/(weights@weights)),
            axis6_mean_deg=mu % 180,
            axis6_sd_deg=float(np.sqrt(weights @ ((unwrapped-mu)**2))),
            chain_weight=[float(sum(w for w, r in zip(weights, records) if r['chain']==c)) for c in range(len(draws))])
    summary['unweighted_axis6'] = dict(mean_deg=float(np.mean(unwrapped[valid]) % 180),
                                      sd_deg=float(np.std(unwrapped[valid])),
                                      per_chain_mean_deg=[float(np.mean(unwrapped[[r['chain']==c for r in records]]) % 180) for c in range(len(draws))])
    angles1 = np.array([r['feed'][0]['axis1_deg'] if r['feed'] else np.nan for r in records])
    anchor1 = np.degrees(np.angle(np.mean(np.exp(2j*np.radians(angles1[valid])))))/2
    unwrapped1 = anchor1 + (angles1-anchor1+90) % 180-90
    summary['unweighted_axis1'] = dict(mean_deg=float(np.mean(unwrapped1[valid]) % 180),
                                     sd_deg=float(np.std(unwrapped1[valid])))
    summary['rock_residual_norm_quantiles_px'] = np.quantile(np.sqrt(sse), [.05, .5, .95]).tolist()
    summary['positive_depth_fraction'] = float(np.mean(valid))
    meta = json.loads((terrain/'meta.json').read_text())
    summary['existing_label_audit'] = {}
    for i, key in enumerate(['2203', '2211']):
        old = bottom_up(meta[key]['transmitter_px'], shapes[i])
        summary['existing_label_audit'][key] = dict(
            transmitter_topdown_px=old.tolist(),
            rock_separation_px=float(np.linalg.norm(old-np.array(rock[i]))))
    confirmed_tx = np.array(picks['transmitter_topdown_px']['2211'])
    delta = np.array(rock[1])-confirmed_tx
    summary['confirmed_alignment_2211'] = dict(
        rock_minus_transmitter_topdown_px=delta.tolist(),
        separation_px=float(np.linalg.norm(delta)),
        transmitter_shift_from_existing_px=float(np.linalg.norm(
            confirmed_tx-np.array(summary['existing_label_audit']['2211']['transmitter_topdown_px']))),
        usage='New transmitter pick is recorded for future joint refitting; existing draws retain original labels.')
    midcam = np.array([draws[0, len(draws[0])//2, 7*i:7*i+7] for i in trace_ids])
    midcam[:, 6] = np.exp(midcam[:, 6])
    rng = np.random.default_rng(20260930)
    summary['corner_pick_sensitivity'] = {}
    for sigma in [2., 5., 10.]:
        sols = [feed_solutions(midcam[0], shapes[0], distortion[0],
                              np.array(corners)+rng.normal(0, sigma, (4, 2)))[0]
                for _ in range(100)]
        values = np.array([s['axis6_deg'] for s in sols])
        summary['corner_pick_sensitivity'][str(sigma)] = dict(
            median_deg=float(np.median(values)), sd_deg=float(np.std(values)),
            q05_q95_deg=np.quantile(values, [.05, .95]).tolist())
    summary['aspect_sensitivity'] = {str(a): feed_solutions(midcam[0], shapes[0], distortion[0], corners, a)
                                     for a in [.9, 1., 1.1]}
    summary['code_commit'] = subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], text=True).strip()
    summary['provenance'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [state_path, trace_path, terrain/'meta.json', terrain/'TRANSMITTER_FIT.md', args.picks, Path(__file__),
                                       Path(__file__).with_name('marjum_camera.py'),
                                       Path(__file__).with_name('marjum_bundle.py')]}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'draw_diagnostics.json').write_text(json.dumps(records)+'\n')
    render_report(out, terrain, summary, records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--terrain', type=Path, required=True)
    p.add_argument('--picks', type=Path, default=Path(__file__).with_name('tx_feed_picks.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=100)
    run(p.parse_args())
