"""Combine `marjum_mcmc_b21` chains and report convergence.

The sampler writes one `chain_N.npz` per process and deliberately does not
combine them (`main` refuses to run more than one chain per process). This is
the missing combination step: it stacks the retained draws, computes split-Rhat
and effective sample size per coordinate, and summarises the antenna posterior
-- which is the quantity the run exists to produce.

Convergence is reported per coordinate and then reduced to the worst case,
because a posterior is only as converged as its worst-mixing direction. Rhat
and ESS come from arviz, which is already this project's trace tooling.

Usage: marjum_mcmc_b21_combine.py RUN_DIR [-o OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Thresholds. Rhat 1.01 is the modern (Vehtari et al. 2021) recommendation
# rather than the older 1.1; ESS 400 gives ~5% Monte Carlo error on a 90%
# interval endpoint, which is finer than any geometry claim made from it.
RHAT_MAX = 1.01
ESS_MIN = 400.0


def load_run(run_dir):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob('chain_*.npz'), key=lambda p: int(p.stem.split('_')[1]))
    if not files:
        raise SystemExit(f'no chain_*.npz in {run_dir}')
    chains, starts, acceptance, logps = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            chains.append(np.asarray(z['globals'], float))
            starts.append(np.asarray(z['start'], float))
            logps.append(np.asarray(z['logp'], float))
            acceptance.append(json.loads(str(z['acceptance'])))
    n = min(len(c) for c in chains)
    if len({len(c) for c in chains}) > 1:
        print(f'chains have unequal length {[len(c) for c in chains]}; truncating to {n}')
    draws = np.stack([c[:n] for c in chains])           # (chain, draw, param)
    return dict(files=[str(f) for f in files], draws=draws, starts=np.array(starts),
                acceptance=acceptance, logp=logps, run_dir=str(run_dir))


def convergence(draws):
    """Split-Rhat and ESS per coordinate, via arviz."""
    import arviz as az
    ds = az.convert_to_dataset(draws[:, :, None, :].squeeze(2)
                               if draws.ndim == 4 else draws)
    rhat = az.rhat(ds).to_array().values.squeeze()
    ess_bulk = az.ess(ds, method='bulk').to_array().values.squeeze()
    ess_tail = az.ess(ds, method='tail').to_array().values.squeeze()
    return np.atleast_1d(rhat), np.atleast_1d(ess_bulk), np.atleast_1d(ess_tail)


def label_coordinates(keys, joint=False):
    """Names for the global vector: see Posterior's coordinate layout.

    Antenna-only: [cameras][antenna 3][gps bias 2][antenna scatter 1].
    Joint:        [cameras][antenna 3][transmitter 3][gps bias 2]
                  [antenna scatter 1][transmitter scatter 1].
    """
    names = []
    for k in keys:
        names += [f'cam{k}_{c}' for c in ('e', 'n', 'u', 'th', 'ph', 'ti', 'logf')]
    names += ['antenna_e', 'antenna_n', 'antenna_u']
    if joint:
        names += ['transmitter_e', 'transmitter_n', 'transmitter_u']
    names += ['gps_bias_e', 'gps_bias_n', 'antenna_extra_px']
    if joint:
        names += ['transmitter_extra_px']
    return names


def summarise(run_dir, out=None):
    run = load_run(run_dir)
    draws = run['draws']
    nchain, ndraw, nparam = draws.shape

    manifest_path = Path(run_dir) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    keys = manifest.get('keys') or []
    joint = bool(manifest.get('joint')) or (len(keys)*7 + 10 == nparam)
    if not keys:
        keys = [str(i) for i in range((nparam - (10 if joint else 6)) // 7)]
    names = label_coordinates(keys, joint)
    if len(names) != nparam:
        print(f'warning: {len(names)} coordinate names for {nparam} parameters; using indices')
        names = [f'p{i}' for i in range(nparam)]

    rhat, ess_bulk, ess_tail = convergence(draws)

    worst_r = int(np.nanargmax(rhat))
    worst_e = int(np.nanargmin(ess_bulk))
    converged = bool(np.nanmax(rhat) <= RHAT_MAX and np.nanmin(ess_bulk) >= ESS_MIN)

    ai = names.index('antenna_e')
    antenna = draws[:, :, ai:ai + 3].reshape(-1, 3)
    # Between-chain spread of the per-chain antenna means, as a blunt check
    # that dispersed starts actually met rather than merely each settling.
    chain_means = draws[:, :, ai:ai + 3].mean(axis=1)

    report = dict(
        run_dir=run['run_dir'], chains=nchain, draws_per_chain=ndraw, parameters=nparam,
        chain_files=run['files'],
        geometry_release=manifest.get('geometry_release'),
        thresholds=dict(rhat_max=RHAT_MAX, ess_min=ESS_MIN),
        converged=converged,
        rhat_max=float(np.nanmax(rhat)), rhat_max_at=names[worst_r],
        ess_bulk_min=float(np.nanmin(ess_bulk)), ess_bulk_min_at=names[worst_e],
        ess_tail_min=float(np.nanmin(ess_tail)),
        n_rhat_above_threshold=int(np.sum(rhat > RHAT_MAX)),
        n_ess_below_threshold=int(np.sum(ess_bulk < ESS_MIN)),
        worst_coordinates=[
            dict(name=names[i], rhat=float(rhat[i]), ess_bulk=float(ess_bulk[i]),
                 ess_tail=float(ess_tail[i]))
            for i in np.argsort(-rhat)[:15]
        ],
        antenna=dict(
            mean=antenna.mean(0).round(4).tolist(),
            sd=antenna.std(0, ddof=1).round(4).tolist(),
            q05=np.percentile(antenna, 5, axis=0).round(4).tolist(),
            q50=np.percentile(antenna, 50, axis=0).round(4).tolist(),
            q95=np.percentile(antenna, 95, axis=0).round(4).tolist(),
            rhat=[float(rhat[ai + j]) for j in range(3)],
            ess_bulk=[float(ess_bulk[ai + j]) for j in range(3)],
            per_chain_mean=chain_means.round(3).tolist(),
            between_chain_sd_of_means=chain_means.std(0, ddof=1).round(4).tolist(),
            note=('Posterior over the 91 m-era antenna position in working-grid ENU metres. '
                  'This IS a posterior, unlike the deterministic bracket products -- but it is '
                  'conditioned on the fixed camera network, fixed distortion and the sampled-ray '
                  'skyline surrogate, so it is not a total error budget.'),
        ),
        acceptance=run['acceptance'],
    )

    if joint and 'transmitter_e' in names:
        ti = names.index('transmitter_e')
        tx = draws[:, :, ti:ti + 3].reshape(-1, 3)
        tx_chain_means = draws[:, :, ti:ti + 3].mean(axis=1)
        report['transmitter'] = dict(
            mean=tx.mean(0).round(4).tolist(),
            sd=tx.std(0, ddof=1).round(4).tolist(),
            q05=np.percentile(tx, 5, axis=0).round(4).tolist(),
            q50=np.percentile(tx, 50, axis=0).round(4).tolist(),
            q95=np.percentile(tx, 95, axis=0).round(4).tolist(),
            rhat=[float(rhat[ti + j]) for j in range(3)],
            ess_bulk=[float(ess_bulk[ti + j]) for j in range(3)],
            per_chain_mean=tx_chain_means.round(3).tolist(),
            between_chain_sd_of_means=tx_chain_means.std(0, ddof=1).round(4).tolist(),
            note=('Posterior over the transmitter position, sampled jointly with the antenna '
                  'and all 29 camera poses. Six of the eight transmitter-labelled views are '
                  'fitted against this same point, so their agreement is not independent '
                  'evidence; 2210 and 2211 are the two that are not conditioned on it.'))
        for label, key in (('antenna_extra_px', 'antenna_scatter_px'),
                           ('transmitter_extra_px', 'transmitter_scatter_px')):
            if label in names:
                si = names.index(label)
                v = draws[:, :, si].ravel()
                report[key] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                   q05=float(np.percentile(v, 5)), q95=float(np.percentile(v, 95)),
                                   rhat=float(rhat[si]), ess_bulk=float(ess_bulk[si]),
                                   note=('Excess label scatter beyond the assumed 3 px, in pixels. '
                                         'A large value means the manual picks for that target are '
                                         'mutually inconsistent by more than the assumed precision.'))
    if not converged:
        report['verdict'] = (
            f'NOT CONVERGED: max Rhat {report["rhat_max"]:.4f} at {report["rhat_max_at"]}, '
            f'min bulk ESS {report["ess_bulk_min"]:.0f} at {report["ess_bulk_min_at"]}. '
            'Do not quote the antenna interval as a posterior.')
    else:
        report['verdict'] = (
            f'Converged by both criteria: max Rhat {report["rhat_max"]:.4f}, '
            f'min bulk ESS {report["ess_bulk_min"]:.0f} over {nparam} coordinates.')

    out = Path(out) if out else Path(run_dir) / 'convergence.json'
    out.write_text(json.dumps(report, indent=2) + '\n')

    combined = Path(run_dir) / 'combined.npz'
    np.savez_compressed(combined, draws=draws, rhat=rhat, ess_bulk=ess_bulk,
                        ess_tail=ess_tail, names=np.array(names), keys=np.array(keys))

    print(report['verdict'])
    print(f'antenna mean {report["antenna"]["mean"]}  sd {report["antenna"]["sd"]}')
    if 'transmitter' in report:
        print(f'transmitter mean {report["transmitter"]["mean"]}  sd {report["transmitter"]["sd"]}')
    if 'transmitter_scatter_px' in report:
        t = report['transmitter_scatter_px']
        print(f'transmitter excess label scatter {t["mean"]:.1f} px (90% {t["q05"]:.1f}-{t["q95"]:.1f})')
    print(f'wrote {out} and {combined}')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()
    summarise(a.run_dir, a.out)
