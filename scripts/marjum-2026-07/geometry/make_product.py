"""Rebuild the joint Marjum geometry against the EPSG:6341 DEM cache.

Run ``prepare``, five independent ``chain`` jobs, then ``finalize``.  Inputs
are content-addressed; a changed input or sampler cannot resume a checkpoint.
The old-frame v0003 point fit is used only to initialize the new target.

Moved on 2026-10-08 from ``marjum-2026-07/derived/geometry_posterior/v0004/``.
Set ``EIGSEP_CAMPAIGN_ROOT`` (or ``eigsep_data.set_campaign_root``) to the
campaign; outputs still go to its v0004 product directory. The five model
files come from ``eigsep_terrain/marjum_geometry/`` on the import path; the
manifest still names their old ``terrain/`` paths, and ``PINS.json`` there
maps each to its byte-identical copy. The manifest's pin for this driver is
the original file, which is checked from git at ``ORIGINAL_DRIVER_COMMIT``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def _campaign():
    from eigsep_data.paths import get_campaign_root
    return Path(get_campaign_root(required=True)).resolve()


CAMPAIGN = _campaign()
# Manifest paths are relative to the workspace that holds the campaign.
ROOT = CAMPAIGN.parent
HERE = CAMPAIGN / 'derived/geometry_posterior/v0004'
OLD = CAMPAIGN / 'derived/geometry_posterior/v0003'
# The model inputs the run read, copied byte-for-byte from the unpublished
# geometry_posterior/v0002/inputs (recorded paths still name that directory).
INPUT = HERE / 'inputs/model_v0002'
RECORDED_INPUT = 'derived/geometry_posterior/v0002/inputs/'
DEM_V0001 = CAMPAIGN / 'derived/dem/v0001/marjum_dem.npz'
DEM = CAMPAIGN / 'derived/dem/v0002/marjum_dem.npz'
CHAINS = (0, 1, 5, 6, 7)
PACKAGE = Path(importlib.util.find_spec('eigsep_terrain').submodule_search_locations[0])
PACKAGE_REPO = PACKAGE.parents[1]
MODELS = PACKAGE / 'marjum_geometry'
sys.path.insert(0, str(MODELS))
PINS = json.loads((MODELS / 'PINS.json').read_text())
LEGACY_MODELS = {item['legacy_path']: MODELS / Path(item['path']).name
                 for item in PINS['files']}
ORIGINAL_DRIVER = 'marjum-2026-07/derived/geometry_posterior/v0004/make_product.py'
ORIGINAL_DRIVER_COMMIT = 'dd37b15'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sources():
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    dem_manifest = json.loads((DEM.parent / 'manifest.json').read_text())
    assert sha(OLD / 'provisional_geometry.npz') == old_manifest['geometry_sha256']
    assert sha(DEM) == dem_manifest['files']['marjum_dem.npz']['sha256']
    frozen = json.loads((INPUT / 'input_manifest.json').read_text())
    for name, item in frozen['files'].items():
        # The frozen DEM copy is the published dem/v0001, not copied into v0004.
        path = DEM_V0001 if name == 'dem' else INPUT / Path(item['frozen']).name
        assert sha(path) == item['sha256']
    for name, item in frozen['feature_cache']['files'].items():
        assert sha(INPUT / 'cv_features' / name) == item['sha256']
    old_run = OLD / 'full_20261006/joint'
    old_report = json.loads((OLD / 'full_20261006/convergence_report.json').read_text())
    for chain in CHAINS:
        assert sha(old_run / f'chain_{chain}.npz') == old_report['chain_sha256'][str(chain)]
    return old_manifest, frozen


def git_sha(repo, commit, path):
    """SHA-256 of ``path`` as committed at ``commit`` in ``repo``."""
    blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{path}'])
    return hashlib.sha256(blob).hexdigest()


def resolve(recorded):
    """Current file for a path recorded by the original v0004 run.

    Recorded paths are workspace-relative manifest keys or the absolute
    paths in chain checkpoints; both may name the retired ``terrain/``.
    """
    parts = Path(recorded).parts
    for start, name in enumerate(parts):
        if name in ('terrain', 'eigsep_terrain', CAMPAIGN.name):
            break
    else:
        raise KeyError(recorded)
    key = Path(*parts[start:]).as_posix()
    if key in LEGACY_MODELS:
        return LEGACY_MODELS[key]
    if key.startswith('eigsep_terrain/src/eigsep_terrain/'):
        return PACKAGE / key.split('eigsep_terrain/src/eigsep_terrain/', 1)[1]
    if key.startswith(CAMPAIGN.name + '/'):
        rel = key.split('/', 1)[1]
        if rel.startswith(RECORDED_INPUT):
            return INPUT / rel[len(RECORDED_INPUT):]
        return CAMPAIGN / rel
    raise KeyError(recorded)


def model_files():
    """Hashes keyed by the paths the original run recorded."""
    names = [f'terrain/{name}' for name in (
        'marjum_mcmc_b21.py', 'marjum_mcmc.py', 'marjum_bundle.py',
        'marjum_camera.py', 'marjum_fitio.py')]
    names += [f'eigsep_terrain/src/eigsep_terrain/{name}'
              for name in ('dem.py', 'marjum_dem.py', 'exif.py')]
    files = {name: sha(resolve(name)) for name in names}
    files[ORIGINAL_DRIVER] = git_sha(ROOT, ORIGINAL_DRIVER_COMMIT, ORIGINAL_DRIVER)
    return files


def package_files_at(commit):
    """The pinned eigsep_terrain files as committed at ``commit``."""
    prefix = 'eigsep_terrain/src/eigsep_terrain/'
    return {key: git_sha(PACKAGE_REPO, commit, 'src/eigsep_terrain/' + key[len(prefix):])
            for key in model_files() if key.startswith(prefix)}


def load_model():
    import marjum_mcmc_b21 as b
    old_launch = json.loads((OLD / 'full_20261006/launch.json').read_text())
    config = b.Config(**old_launch['signature']['config'])
    model = b.Posterior(state_file=INPUT / 'fit_transmitter.npz',
                        dem_file=str(DEM), meta_file=INPUT / 'meta.json',
                        exif_file=HERE / 'inputs/exif_utm.npz',
                        feature_dir=INPUT / 'cv_features', config=config)
    return b, model


def prepare():
    from eigsep_terrain.exif import read_exif
    from eigsep_terrain.marjum_dem import MarjumDEM
    old_manifest, frozen = sources()
    product_inputs = HERE / 'inputs'
    product_inputs.mkdir(parents=True, exist_ok=True)
    dem = MarjumDEM(cache_file=str(DEM))
    assert int(dem.crs.to_epsg()) == 6341
    assert np.array_equal(dem.survey_offset, [0, 0, 3])
    assert np.array_equal(dem.raster_origin, [291000.25, 4345000.25])
    with np.load(INPUT / 'marjum_2026_07_exif_joint.npz', allow_pickle=True) as saved:
        exif = {name: saved[name].copy() for name in saved.files}
    latitude = np.full(len(exif['keys']), np.nan)
    longitude = latitude.copy()
    raw_error = latitude.copy()
    for i, key in enumerate(exif['keys']):
        photo = CAMPAIGN / 'imgs' / f'IMG_{key}.HEIC'
        if not photo.exists():
            continue
        entry = read_exif(photo)
        if entry['lat'] is None or entry['lon'] is None:
            continue
        latitude[i], longitude[i] = entry['lat'], entry['lon']
        e, n, _ = dem.latlon_to_raster(entry['lat'], entry['lon'])
        exif['e_gps'][i], exif['n_gps'][i] = e, n
        if entry['h_err'] is not None:
            raw_error[i] = entry['h_err']
            exif['h_err_m'][i] = entry['h_err']
    exif['latitude_deg'] = latitude
    exif['longitude_deg'] = longitude
    exif['raw_h_err_m'] = raw_error
    assert np.isfinite(exif['e_gps']).sum() == 39
    target = product_inputs / 'exif_utm.npz'
    if target.exists():
        with np.load(target, allow_pickle=True) as existing:
            assert set(existing.files) == set(exif)
            for name, value in exif.items():
                np.testing.assert_equal(existing[name], value)
    else:
        np.savez_compressed(target, **exif)

    b, model = load_model()
    old_run = OLD / 'full_20261006/joint'
    starts = []
    for chain in CHAINS:
        with np.load(old_run / f'chain_{chain}.npz') as saved:
            state = np.asarray(saved['final'], float).copy()
            assert list(saved['camera_keys']) == model.keys
        bias = 7 * model.nc + 3 + model.ntx
        state[bias:bias+2] = 0
        assert np.isfinite(model.logp(state)), chain
        starts.append(state)
    start_file = product_inputs / 'starting_states.npz'
    if start_file.exists():
        with np.load(start_file) as existing:
            np.testing.assert_equal(existing['states'], starts)
            np.testing.assert_equal(existing['chains'], CHAINS)
    else:
        np.savez_compressed(start_file, states=np.stack(starts), chains=CHAINS)

    record = dict(product='geometry_posterior', version='v0004',
                  status='prepared', created_utc=datetime.now(timezone.utc).isoformat(),
                  coordinate_frame='EPSG:6341 UTM raster grid, southwest pixel centre origin',
                  survey_offset_subtracted_m=[0, 0, 3],
                  old_geometry_sha256=old_manifest['geometry_sha256'],
                  old_input_manifest_sha256=sha(INPUT / 'input_manifest.json'),
                  dem_sha256=sha(DEM), exif_utm_sha256=sha(target),
                  start_states_sha256=sha(start_file), model_files=model_files(),
                  package_commit=subprocess.check_output(
                      ['git', '-C', str(PACKAGE_REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                  config=asdict(model.config), chains=list(CHAINS),
                  exif_gps_count=int(np.isfinite(exif['e_gps']).sum()),
                  note='Raw HEIC latitude/longitude and horizontal errors re-extracted; '
                       'other EXIF fields remain from the pinned joint cache. '
                       'Old-frame chain endpoints initialize only the new target.')
    manifest = HERE / 'input_manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        for key in ('old_geometry_sha256', 'old_input_manifest_sha256',
                    'dem_sha256', 'exif_utm_sha256', 'start_states_sha256',
                    'model_files', 'config', 'chains'):
            assert old[key] == record[key], key
        # A later package commit is accepted when the pinned files are unchanged.
        assert package_files_at(old['package_commit']) == {
            k: v for k, v in record['model_files'].items() if k.startswith('eigsep_terrain/')}
    else:
        write_json(manifest, record)
    print('Prepared 39 HEIC GPS fixes and 5 finite new-target starts.', flush=True)


def verify():
    p = json.loads((HERE / 'input_manifest.json').read_text())
    assert sha(OLD / 'provisional_geometry.npz') == p['old_geometry_sha256']
    assert sha(INPUT / 'input_manifest.json') == p['old_input_manifest_sha256']
    assert sha(DEM) == p['dem_sha256']
    assert sha(HERE / 'inputs/exif_utm.npz') == p['exif_utm_sha256']
    assert sha(HERE / 'inputs/starting_states.npz') == p['start_states_sha256']
    assert model_files() == p['model_files']
    # The working package files match the manifest (above) and the recorded commit.
    assert package_files_at(p['package_commit']) == {
        k: v for k, v in p['model_files'].items() if k.startswith('eigsep_terrain/')}
    return p


def run_chain(chain, tune, draws, resume):
    import marjum_mcmc_b21 as b
    p = verify()
    assert chain in CHAINS and (tune, draws) == (300, 2000)
    _, model = load_model()
    with np.load(HERE / 'inputs/starting_states.npz') as saved:
        start = saved['states'][list(CHAINS).index(chain)]
    output = HERE / 'run'
    output.mkdir(exist_ok=True)
    b.run_chain(model, chain, tune, draws, seed=20261007, out=output,
                start=start, resume=resume, difference_step_factor=0.1,
                joint_directions=None, thin_points=10, shift_every=5,
                checkpoint_every=100)
    print(f'Chain {chain} complete: {output / f"chain_{chain}.npz"}', flush=True)


def finalize():
    import arviz as az
    p = verify()
    _, model = load_model()
    traces, candidates, hashes = [], [], {}
    for chain in CHAINS:
        path = HERE / 'run' / f'chain_{chain}.npz'
        with np.load(path) as saved:
            traces.append(saved['globals'])
            for (sweep, logp), points in zip(saved['logp'], saved['landmarks']):
                state = np.r_[saved['globals'][int(sweep)], points.ravel()]
                candidates.append((float(logp), chain, int(sweep), state))
            state = saved['final']
            candidates.append((float(model.logp(state)), chain, 1999, state))
        hashes[str(chain)] = sha(path)
    traces = np.stack(traces)
    assert traces.shape == (5, 2000, model.ng)
    coords = az.convert_to_dataset(traces)
    rhat = np.asarray(az.rhat(coords).to_array()).ravel()
    bulk = np.asarray(az.ess(coords, method='bulk').to_array()).ravel()
    tail = np.asarray(az.ess(coords, method='tail').to_array()).ravel()
    passed = np.isfinite(rhat) & np.isfinite(bulk) & np.isfinite(tail) & (rhat <= 1.01) & (bulk >= 400) & (tail >= 400)
    best, chain, sweep, state = max(candidates, key=lambda item: item[0])
    assert np.isclose(best, model.logp(state), rtol=0, atol=1e-5)
    cam, ant, tx, bias, extra, tx_extra, landmarks = model.unpack(state)
    output = HERE / 'provisional_geometry.npz'
    assert not output.exists(), 'refusing to replace a published fit'
    np.savez_compressed(output, state=state, camera=cam, antenna=ant,
                        transmitter=tx, gps_bias=bias, antenna_extra_px=extra,
                        transmitter_extra_px=tx_extra, landmarks=landmarks,
                        keys=np.asarray(model.keys))
    result = dict(status='provisional_point_fit_only',
                  selected=dict(chain=chain, retained_sweep=sweep, exact_log_density=best),
                  convergence=dict(pass_count=int(passed.sum()), rhat_fail=int((rhat > 1.01).sum()),
                                   bulk_fail=int((bulk < 400).sum()), tail_fail=int((tail < 400).sum()),
                                   rhat_max=float(np.nanmax(rhat)), bulk_min=float(np.nanmin(bulk)),
                                   tail_min=float(np.nanmin(tail))),
                  chains_sha256=hashes, geometry_sha256=sha(output),
                  input_manifest_sha256=sha(HERE / 'input_manifest.json'),
                  note='Five new-target chains; no posterior uncertainty is published if convergence fails.')
    write_json(HERE / 'manifest.json', dict(p, **result))
    print(json.dumps(result['convergence'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'chain', 'finalize'))
    parser.add_argument('--chain', type=int)
    parser.add_argument('--tune', type=int, default=300)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'chain':
        if args.chain is None:
            parser.error('--chain is required')
        run_chain(args.chain, args.tune, args.draws, args.resume)
    else:
        finalize()


if __name__ == '__main__':
    main()
