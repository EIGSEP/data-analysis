"""Freeze the inputs of a ``marjum_mcmc_b21`` run, then launch its chains.

The sampler reads its state, DEM, labels, EXIF priors and horizon features from
paths relative to wherever it is started, and the repaired 2159/2199 state needs
a feature cache that exists nowhere as one directory: the repair wrote only the
two changed ``sift_*.npz`` files.  This script removes both failure modes.

``prepare RUN_DIR`` copies every input into ``RUN_DIR/inputs/``, builds the
overlay feature cache (``cv_features/`` with the repaired files substituted),
checks each source against the SHA-256 pinned for the run, evaluates the joint
density at the start state, and writes ``RUN_DIR/inputs/input_manifest.json``.
It refuses to overwrite a frozen input with different bytes.

``launch RUN_DIR --stage pilot|production`` re-hashes the frozen inputs against
that manifest, then starts one detached single-threaded process per chain from
this directory, writing to ``RUN_DIR/<stage>/`` and ``RUN_DIR/logs/``.  Pilot
and production go to different directories so their chains cannot be combined
by accident; an existing stage directory is rejected.

Usage, from anywhere::

    python terrain/marjum_mcmc_b21_prepare.py prepare
    python terrain/marjum_mcmc_b21_prepare.py launch --stage pilot --chains 8 --tune 300 --draws 200
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

TERRAIN = Path(__file__).resolve().parent
WORKSPACE = TERRAIN.parent
DEFAULT_RUN = WORKSPACE/'marjum-2026-07/derived/geometry_posterior/v0001'
PYTHON = '/home/aparsons/.local/share/mamba/envs/arp/bin/python3.11'

# geometry_posterior v0001 sources.  The DEM is the int32 cache the 2159/2199
# repair was fitted against (its report.json input_sha256), not the float32
# cache now at terrain/marjum_dem.npz: mixing those changes the likelihood.
# Its basename must stay marjum_dem.npz, because marjum_bundle.working_grid
# selects the grid anchor by file name.
SOURCES = dict(
    state=('_repair_tight_final/fit_transmitter.npz', 'fit_transmitter.npz'),
    dem=('/home/aparsons/projects/eigsep_old/terrain/marjum_dem.npz', 'marjum_dem.npz'),
    meta=('meta.json', 'meta.json'),
    exif=('marjum_2026_07_exif_joint.npz', 'marjum_2026_07_exif_joint.npz'),
)
PINS = dict(
    state='5ae3909a7b42abddc6035c9473eb6f49bdfc7b49f0bc569bb3bb2a3ee8b31a21',
    dem='4d1c65c45cddd2e0e5d5a9225e3215af1bb0e78cefffe6bf11f88f34876b9fd8',
    meta='4c0e5de2e0f07d0fdd00fc0f4485df0d0d080d8df576c62e4307aab18ec002fb',
    exif='51d2b39356aca110b316539ead6b6173368618f521342e899935a85840299eb2',
)
FEATURE_BASE = 'cv_features'
FEATURE_OVERLAY = '_repair_tight_final/cv_features'
STAGE_SEEDS = dict(pilot=20261001, production=20261002)
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_state(repo):
    def run(*cmd):
        return subprocess.run(['git', '-C', str(repo), *cmd], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return dict(commit=run('rev-parse', 'HEAD'), branch=run('rev-parse', '--abbrev-ref', 'HEAD'),
                    dirty_tracked=sorted(line[3:] for line in run('status', '--porcelain',
                                                                  '--untracked-files=no').splitlines()))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def code_state():
    files = ['marjum_mcmc_b21.py', 'marjum_mcmc.py', 'marjum_bundle.py', 'marjum_camera.py',
             'marjum_fitio.py', Path(__file__).name]
    return dict(terrain=git_state(TERRAIN), eigsep_terrain=git_state(WORKSPACE/'eigsep_terrain'),
                sha256={f: sha256(TERRAIN/f) for f in files})


def freeze(src, dst, want=None):
    """Copy ``src`` to ``dst`` and return its hash; never replace different bytes."""
    have = sha256(src)
    if want is not None and have != want:
        raise SystemExit(f'{src}: sha256 {have} does not match pinned {want}')
    if dst.exists():
        if sha256(dst) != have:
            raise SystemExit(f'{dst} exists with different contents; use a fresh run directory')
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256(dst) != have:
            raise SystemExit(f'{dst}: copy does not match its source')
    return have


def inputs_dir(run):
    return Path(run).resolve()/'inputs'


def prepare(args):
    run = Path(args.run_dir).resolve()
    inp = inputs_dir(run)
    record = dict(run_dir=str(run), prepared_utc=_now(), code=code_state(), files={})
    for name, (src, dst) in SOURCES.items():
        src = (TERRAIN/src).resolve()
        record['files'][name] = dict(source=str(src), frozen=str((inp/dst).relative_to(run)),
                                     sha256=freeze(src, inp/dst, PINS[name]),
                                     bytes=src.stat().st_size)

    base, overlay = TERRAIN/FEATURE_BASE, TERRAIN/FEATURE_OVERLAY
    replaced = sorted(p.name for p in overlay.glob('sift_*.npz'))
    if not replaced:
        raise SystemExit(f'no repaired features in {overlay}')
    features = {}
    for path in sorted(base.glob('sift_*.npz')):
        src = overlay/path.name if path.name in replaced else path
        features[path.name] = dict(source=str(src), sha256=freeze(src, inp/'cv_features'/path.name))
    missing = set(replaced) - set(features)
    if missing:
        raise SystemExit(f'overlay files with no base counterpart: {sorted(missing)}')
    record['feature_cache'] = dict(frozen='inputs/cv_features', base=str(base), overlay=str(overlay),
                                   replaced=replaced, n_files=len(features), files=features)

    record['check'] = start_check(inp)
    target = inp/'input_manifest.json'
    if target.exists():
        old = json.loads(target.read_text())
        if _hashes(old) != _hashes(record):
            raise SystemExit(f'{target} records different inputs; use a fresh run directory')
        print(f'{target} already matches; leaving it unchanged')
    else:
        target.write_text(json.dumps(record, indent=2) + '\n')
        print(f'wrote {target}')
    print(json.dumps(record['check'], indent=2))


def _hashes(record):
    return ({k: v['sha256'] for k, v in record['files'].items()},
            {k: v['sha256'] for k, v in record['feature_cache']['files'].items()})


def start_check(inp):
    """Load the frozen inputs exactly as the sampler will and evaluate the start."""
    cwd = os.getcwd()
    os.chdir(TERRAIN)
    sys.path.insert(0, str(TERRAIN))
    try:
        import numpy as np
        import marjum_mcmc_b21 as b21
        model = b21.Posterior(state_file=inp/SOURCES['state'][1], dem_file=str(inp/SOURCES['dem'][1]),
                              meta_file=inp/SOURCES['meta'][1], exif_file=inp/SOURCES['exif'][1],
                              feature_dir=inp/'cv_features')
        z = model.start_vector()
        cam = model.unpack(z)[0]
        clearance = cam[:, 2] - model.terrain.height(cam[:, 0], cam[:, 1])
        logp = model.logp(z)
        if not np.isfinite(logp):
            raise SystemExit('joint density is not finite at the start state')
        return dict(cameras=model.nc, keys=model.keys, landmarks=model.npoint,
                    observations=int(len(model.xy)), parameters_global=model.ng,
                    antenna_labelled=model.n_ant_label, transmitter_labelled=model.n_tx_label,
                    unpriored=model.unpriored, start_logp=float(logp),
                    camera_clearance_m=dict(min=float(clearance.min()), max=float(clearance.max())))
    finally:
        os.chdir(cwd)


def launch(args):
    run = Path(args.run_dir).resolve()
    inp = inputs_dir(run)
    record = json.loads((inp/'input_manifest.json').read_text())
    for name, info in record['files'].items():
        if sha256(run/info['frozen']) != info['sha256']:
            raise SystemExit(f'frozen input {name} changed since prepare')
    for name, info in record['feature_cache']['files'].items():
        if sha256(inp/'cv_features'/name) != info['sha256']:
            raise SystemExit(f'frozen feature {name} changed since prepare')

    out, logs = run/args.stage, run/'logs'
    if out.exists():
        raise SystemExit(f'{out} exists; a stage is launched once')
    out.mkdir(parents=True)
    logs.mkdir(exist_ok=True)
    seed = STAGE_SEEDS[args.stage] if args.seed is None else args.seed
    env = dict(os.environ, **{v: '1' for v in THREAD_VARS})
    common = [args.python, '-u', str(TERRAIN/'marjum_mcmc_b21.py'), '--output', str(out),
              '--tune', str(args.tune), '--draws', str(args.draws), '--seed', str(seed),
              '--state-file', str(inp/SOURCES['state'][1]), '--dem-file', str(inp/SOURCES['dem'][1]),
              '--meta-file', str(inp/SOURCES['meta'][1]), '--exif-file', str(inp/SOURCES['exif'][1]),
              '--feature-dir', str(inp/'cv_features')]
    # The first process writes RUN/<stage>/manifest.json; start it alone so the
    # others do not race to write it.
    chains = []
    for index in range(args.chains):
        cmd = common + ['--chain', str(index)]
        log = logs/f'{args.stage}_chain{index}.log'
        with open(log, 'w') as f:
            proc = subprocess.Popen(cmd, cwd=TERRAIN, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        chains.append(dict(chain=index, pid=proc.pid, log=str(log.relative_to(run)), argv=cmd))
        if index == 0:
            _wait_for(out/'manifest.json', proc)
    info = dict(stage=args.stage, launched_utc=_now(), host=os.uname().nodename, cwd=str(TERRAIN),
                threads={v: '1' for v in THREAD_VARS}, seed=seed, tune=args.tune, draws=args.draws,
                code=code_state(), chains=chains)
    (logs/f'{args.stage}_launch.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'{args.stage}: {len(chains)} chains -> {out}')
    print('pids:', ' '.join(str(c['pid']) for c in chains))
    print(f'stop with: kill {" ".join(str(c["pid"]) for c in chains)}')


def _wait_for(path, proc, timeout=600):
    import time
    start = time.monotonic()
    while not path.exists():
        if proc.poll() is not None:
            raise SystemExit(f'chain 0 exited ({proc.returncode}) before writing {path}; see its log')
        if time.monotonic() - start > timeout:
            raise SystemExit(f'{path} not written after {timeout} s')
        time.sleep(2)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('run_dir', nargs='?', default=str(DEFAULT_RUN))
    p.set_defaults(func=prepare)
    p = sub.add_parser('launch')
    p.add_argument('run_dir', nargs='?', default=str(DEFAULT_RUN))
    p.add_argument('--stage', choices=sorted(STAGE_SEEDS), required=True)
    p.add_argument('--chains', type=int, default=8)
    p.add_argument('--tune', type=int, required=True)
    p.add_argument('--draws', type=int, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--python', default=PYTHON)
    p.set_defaults(func=launch)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
