"""Publish the highest-density valid UTM-target state and full diagnostics.

The old v0003 selected state is a candidate because the new target changes
GPS placement and bias without changing the elevation pixels. Its GPS bias is
optimized conditionally; the photo geometry is left untouched. Chain states
are compared against that candidate at the same exact new-target density.
"""
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np

import make_product as product


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def diagnostics(draws, names):
    dataset = az.convert_to_dataset(draws)
    arrays = {key: np.asarray(getattr(az, function)(dataset, **kwargs).to_array().values).reshape(-1)
              for key, function, kwargs in [('rhat', 'rhat', {}),
                                             ('bulk_ess', 'ess', {'method': 'bulk'}),
                                             ('tail_ess', 'ess', {'method': 'tail'})]}
    assert all(len(value) == len(names) for value in arrays.values())
    good = np.isfinite(arrays['rhat']) & np.isfinite(arrays['bulk_ess']) & np.isfinite(arrays['tail_ess'])
    passed = good & (arrays['rhat'] <= 1.01) & (arrays['bulk_ess'] >= 400) & (arrays['tail_ess'] >= 400)
    worst = int(np.nanargmax(arrays['rhat']))
    summary = dict(pass_count=int(passed.sum()),
                   rhat_fail=int((~good | (arrays['rhat'] > 1.01)).sum()),
                   bulk_fail=int((~good | (arrays['bulk_ess'] < 400)).sum()),
                   tail_fail=int((~good | (arrays['tail_ess'] < 400)).sum()),
                   rhat_max=float(arrays['rhat'][worst]), worst_name=str(names[worst]),
                   bulk_min=float(np.nanmin(arrays['bulk_ess'])),
                   tail_min=float(np.nanmin(arrays['tail_ess'])))
    rows = [dict(name=str(name), rhat=float(arrays['rhat'][i]),
                 bulk_ess=float(arrays['bulk_ess'][i]), tail_ess=float(arrays['tail_ess'][i]),
                 pass_all=bool(passed[i])) for i, name in enumerate(names)]
    return summary, rows


def carried_candidate(model):
    with np.load(product.OLD / 'provisional_geometry.npz') as old:
        assert list(old['keys']) == model.keys
        state = old['state'].copy()
        old_bias = old['gps_bias'].tolist()
        names = old['global_names'].copy()
    cameras = model.unpack(state)[0]
    use = model.has_gps
    weight = 1 / model.gps_sigma[use]**2
    precision = weight.sum() + 1 / model.config.gps_common_sigma_m**2
    bias = ((model.gps[use] - cameras[use, :2]) * weight[:, None]).sum(axis=0) / precision
    bias_slice = slice(7*model.nc + 3 + model.ntx, 7*model.nc + 5 + model.ntx)
    np.testing.assert_allclose(state[bias_slice], old_bias)
    state[bias_slice] = bias
    return state, bias, names, old_bias


def main():
    inputs = product.verify()
    assert json.loads((product.HERE / 'run_integrity.json').read_text())['status'] == 'pass'
    _, model = product.load_model()
    carried, carried_bias, names, old_bias = carried_candidate(model)
    carried_logp = float(model.logp(carried))
    candidates = [(carried_logp, dict(kind='v0003_selected_with_utm_gps_bias'), carried)]
    traces, hashes = [], {}
    for chain in product.CHAINS:
        path = product.HERE / 'run' / f'chain_{chain}.npz'
        with np.load(path) as saved:
            globals_ = np.asarray(saved['globals'])
            logp = np.asarray(saved['logp'])
            landmarks = np.asarray(saved['landmarks'])
            final = np.asarray(saved['final'])
            assert globals_.shape == (2000, model.ng)
            assert logp.shape == (200, 2) and landmarks.shape == (200, model.npoint, 3)
            assert np.array_equal(logp[:, 0], np.arange(0, 2000, 10))
            for i, (sweep, density) in enumerate(logp):
                if np.isfinite(density):
                    state = np.r_[globals_[int(sweep)], landmarks[i].ravel()]
                    candidates.append((float(density), dict(kind='retained_draw', chain=chain,
                                                             retained_sweep=int(sweep)), state))
            candidates.append((float(model.logp(final)),
                               dict(kind='final_draw', chain=chain, retained_sweep=1999), final))
            traces.append(globals_)
        hashes[str(chain)] = product.sha(path)
    traces = np.stack(traces)
    full, rows = diagnostics(traces, names)
    halves = [diagnostics(traces[:, block, :], names)[0]
              for block in (slice(0, 1000), slice(1000, 2000))]
    density, selected, state = max(candidates, key=lambda item: item[0])
    assert np.isclose(model.logp(state), density, rtol=0, atol=1e-5)
    camera, antenna, transmitter, bias, ant_extra, tx_extra, landmarks = model.unpack(state)
    output = product.HERE / 'provisional_geometry.npz'
    assert not output.exists(), 'refusing to replace a published fit'
    np.savez_compressed(output, state=state, camera=camera, antenna=antenna,
                        transmitter=transmitter, gps_bias=bias,
                        antenna_extra_px=ant_extra, transmitter_extra_px=tx_extra,
                        landmarks=landmarks, keys=np.asarray(model.keys),
                        global_names=names)
    report = dict(status='provisional_point_fit_only',
                  scope='five chains, 300 warmup and 2000 retained sweeps each',
                  target='float32 DEM v0002 EPSG:6341, raw HEIC UTM GPS, diagonal EXIF, mixed focal priors',
                  convergence_thresholds=dict(rhat_max=1.01, bulk_ess_min=400, tail_ess_min=400),
                  full=full, retained_halves=halves,
                  coordinates=rows,
                  selected=dict(selected, exact_log_density=float(density),
                                candidates=len(candidates)),
                  carried_candidate=dict(exact_log_density=carried_logp,
                                         old_frame_bias_e_n_m=old_bias,
                                         new_target_conditional_bias_e_n_m=carried_bias.tolist()),
                  run_best_log_density=float(max(x[0] for x in candidates[1:])),
                  caveat='The chains did not establish a posterior unless all convergence checks pass. '
                         'The carried v0003 photo geometry is compared on the new target, not refitted. '
                         'No posterior widths are published.')
    report_file = product.HERE / 'run/convergence_report.json'
    write_json(report_file, report)
    manifest = dict(inputs, status=report['status'], target=report['target'],
                    selected=report['selected'], convergence=full,
                    carried_candidate=report['carried_candidate'],
                    run_best_log_density=report['run_best_log_density'],
                    geometry_file=output.name, geometry_sha256=product.sha(output),
                    convergence_report_sha256=product.sha(report_file),
                    run_integrity_sha256=product.sha(product.HERE / 'run_integrity.json'),
                    chain_sha256=hashes,
                    finalizer_sha256=product.sha(Path(__file__)))
    write_json(product.HERE / 'manifest.json', manifest)
    print(json.dumps(dict(selected=report['selected'], full=full,
                          carried_candidate=report['carried_candidate'],
                          run_best_log_density=report['run_best_log_density']), indent=2))


if __name__ == '__main__':
    main()
