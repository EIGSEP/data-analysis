"""Evaluate all saved full states and publish a provisional point geometry.

Run only after all five full chains complete. The full-vector diagnostics are
recorded even when they fail; no posterior summaries are released.
"""
from pathlib import Path
import argparse
import csv
import json
import hashlib

import arviz as az
import numpy as np

import marjum_mcmc_b21 as b
from marjum_mcmc_b21_combine import label_coordinates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def diagnostics(draws, names):
    dataset = az.convert_to_dataset(draws)
    arrays = {key: np.asarray(getattr(az, function)(dataset, **kwargs).to_array().values).reshape(-1)
              for key, function, kwargs in [('rhat', 'rhat', {}),
                                             ('bulk_ess', 'ess', {'method': 'bulk'}),
                                             ('tail_ess', 'ess', {'method': 'tail'})]}
    assert all(len(values) == len(names) for values in arrays.values())
    good = np.isfinite(arrays['rhat']) & np.isfinite(arrays['bulk_ess']) & np.isfinite(arrays['tail_ess'])
    passed = good & (arrays['rhat'] <= 1.01) & (arrays['bulk_ess'] >= 400) & (arrays['tail_ess'] >= 400)
    worst = int(np.nanargmax(arrays['rhat']))
    summary = dict(pass_count=int(passed.sum()), rhat_fail=int((~good | (arrays['rhat'] > 1.01)).sum()),
                   bulk_fail=int((~good | (arrays['bulk_ess'] < 400)).sum()),
                   tail_fail=int((~good | (arrays['tail_ess'] < 400)).sum()),
                   rhat_max=float(arrays['rhat'][worst]), worst_name=names[worst],
                   bulk_min=float(np.nanmin(arrays['bulk_ess'])),
                   tail_min=float(np.nanmin(arrays['tail_ess'])))
    rows = [dict(name=name, rhat=float(arrays['rhat'][i]), bulk_ess=float(arrays['bulk_ess'][i]),
                 tail_ess=float(arrays['tail_ess'][i]), pass_all=bool(passed[i]))
            for i, name in enumerate(names)]
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run, root = args.run.resolve(), args.root.resolve()
    launch = json.loads((run/'launch.json').read_text())
    sig = launch['signature']
    assert sig['starts'] == [0, 1, 5, 6, 7] and sig['draws'] == 2000
    assert len(launch['workers']) == 5
    assert all(json.loads((run/'joint'/f'completion_{c}.json').read_text())['result_sha256'] ==
               sha(run/'joint'/f'chain_{c}.npz') for c in sig['starts'])
    inp = root/'inputs'
    model = b.Posterior(state_file=inp/'fit_transmitter.npz', dem_file=str(inp/'marjum_dem.npz'),
                        meta_file=inp/'meta.json', exif_file=inp/'marjum_2026_07_exif_joint.npz',
                        feature_dir=inp/'cv_features', config=b.Config(**sig['config']))
    assert {str(f):sha(f) for f in model.input_files} == sig['model_input_sha256']
    names = label_coordinates(model.keys, joint=True)
    assert len(names) == model.ng == 213
    traces, candidates, chain_hashes = [], [], {}
    for chain in sig['starts']:
        path = run/'joint'/f'chain_{chain}.npz'
        chain_hashes[str(chain)] = sha(path)
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
                    candidates.append((float(density), chain, int(sweep), state))
            candidates.append((float(model.logp(final)), chain, 1999, final))
            traces.append(globals_)
    traces = np.stack(traces)
    assert np.isfinite(traces).all() and candidates
    full, rows = diagnostics(traces, names)
    halves = [diagnostics(traces[:, sl, :], names)[0] for sl in [slice(0,1000),slice(1000,2000)]]
    with (run/'all_coordinate_diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    density, chain, sweep, state = max(candidates, key=lambda item: item[0])
    verified = float(model.logp(state))
    assert np.isclose(verified, density, rtol=0, atol=1e-5), (verified, density)
    cam, ant, tx, bias, extra, tx_extra, points = model.unpack(state)
    output = run.parent/'provisional_geometry.npz'
    np.savez_compressed(output, state=state, camera=cam, antenna=ant, transmitter=tx,
                        gps_bias=bias, antenna_extra_px=extra, transmitter_extra_px=tx_extra,
                        landmarks=points, keys=np.asarray(model.keys), global_names=np.asarray(names))
    report = dict(status='provisional_point_fit_only', scope='Five chains, 300 warmup + 2000 retained each; corrected float32 DEM, diagonal EXIF, 3% main/tele and 25% ultrawide log-f priors',
                  convergence_thresholds=dict(rhat_max=1.01, bulk_ess_min=400, tail_ess_min=400),
                  full=full, retained_halves=halves, selected=dict(chain=chain, retained_sweep=sweep,
                  exact_log_density=verified, selection='largest exact target log density among 200 thinned complete states plus final complete state per chain',
                  candidates=len(candidates)),
                  caveat='The chains were expected to fail convergence. This is a provisional fitted state; neither the chain means nor posterior widths are released.',
                  chain_sha256=chain_hashes, coordinate_diagnostics_sha256=sha(run/'all_coordinate_diagnostics.csv'),
                  geometry_sha256=sha(output), input_manifest_sha256=sig['input_manifest_sha256'],
                  start_states_sha256=sig['starts_sha256'], source_commit=sig['code_commit'],
                  source_sha256=sig['code_sha256'])
    write(run/'convergence_report.json', report)
    product = dict(version='v0003', type='provisional_geometry_point_fit', parent='v0002',
                   target='corrected float32 DEM, diagonal EXIF, mixed focal priors',
                   selected=report['selected'], convergence=full, retained_halves=halves,
                   geometry_file=output.name, geometry_sha256=sha(output),
                   run_dir=run.name, run_report_sha256=sha(run/'convergence_report.json'),
                   start_states_sha256=sig['starts_sha256'], input_manifest_sha256=sig['input_manifest_sha256'],
                   source_commit=sig['code_commit'])
    write(run.parent/'manifest.json', product)
    (run.parent/'README.md').write_text('# Geometry v0003 — provisional full-run point fit\n\n'
        'Five corrected-target chains used 300 warmup and 2,000 retained sweeps each, starting from the latest saved valid full state for each chain. '
        'The full state with highest saved exact log density is `provisional_geometry.npz`. '
        'This is a fitted geometry under the current model, not a converged posterior or uncertainty estimate.\n\n'
        f"All-coordinate diagnostics and retained halves are in `{run.name}/convergence_report.json`; {full['rhat_fail']}/213 globals fail R-hat 1.01, "
        f"{full['bulk_fail']} fail bulk ESS 400, and {full['tail_fail']} fail tail ESS 400. "
        'Every coordinate is in `all_coordinate_diagnostics.csv`. The run, source and input hashes are in `manifest.json` and the run launch record.\n\n'
        '## Recent changes\n\n- 2026-10-06: Saved the provisional highest-density complete geometry from the new five-chain full run, with its convergence failures.\n')
    print(json.dumps(dict(geometry=str(output),selected=report['selected'],convergence=full)), flush=True)


if __name__ == '__main__':
    main()
