"""Supervise one approved eight-chain pilot; verify inputs and preserve logs.

Run under the host's user service manager. No production run or interpretation
is triggered on completion. Resume only with the same code and frozen inputs.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terrain-cwd', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    inp, out, cwd = args.inputs.resolve(), args.output.resolve(), args.terrain_cwd.resolve()
    code = Path(__file__).resolve().parent
    record = json.loads((inp/'input_manifest.json').read_text())
    for item in record['files'].values():
        path = inp.parent/item['frozen']
        if digest(path) != item['sha256']:
            raise ValueError(f'frozen input changed: {path}')
    for name, item in record['feature_cache']['files'].items():
        if digest(inp/'cv_features'/name) != item['sha256']:
            raise ValueError(f'frozen feature changed: {name}')
    git = lambda *cmd: subprocess.check_output(['git', '-C', str(code), *cmd], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('pilot code must be committed before launch')
    signature = dict(code_commit=git('rev-parse', 'HEAD'),
                     code_sha256={name: digest(code/name) for name in [
                         'marjum_mcmc_b21.py', 'marjum_mcmc_b21_pilot.py', 'marjum_mcmc.py',
                         'marjum_bundle.py', 'marjum_camera.py', 'marjum_fitio.py']},
                     inputs=str(inp), input_manifest_sha256=digest(inp/'input_manifest.json'),
                     tune=300, draws=200, seed=20261001, chains=8, checkpoint_every=100)
    if args.resume:
        previous = json.loads((out/'launch.json').read_text())
        if previous['signature'] != signature:
            raise ValueError('resume code or inputs differ from original pilot')
    else:
        out.mkdir(parents=True, exist_ok=False)
    logs = out/'logs'
    logs.mkdir(exist_ok=True)
    if not args.resume:
        (out/'.gitignore').write_text('*.npz\n*.pkl\n*.pkl.tmp\n')
        (out/'README.md').write_text(
            '# Repaired log-f camera pilot\n\n'
            'Approved diagnostic: eight chains, each 300 warmup and 200 retained\n'
            'sweeps, seed 20261001. This is not a validated posterior.\n\n'
            '`launch.json` records exact commands, PIDs and code/input hashes;\n'
            '`manifest.json` records the model. `status.json` records completion\n'
            'or failure. Each chain writes `logs/chain_N.log`, atomic restart\n'
            'checkpoints `checkpoint_N.pkl` every 100 sweeps, and `chain_N.npz`\n'
            'at completion. Bulk arrays/checkpoints move only with eigsep-sync.\n\n'
            'Resume with the identical supervisor command plus `--resume`. Only\n'
            'use this run\'s own trusted checkpoints; these are Python pickles.\n'
            'Do not combine these chains with the earlier raw-f pilot.\n\n'
            'Completion does not authorize interpretation or production.\n\n'
            '## Recent changes\n\n'
            '- 2026-10-02: Launched the approved repaired-sampler pilot with\n'
            '  durable logs and exact restart checkpoints.\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    common = [sys.executable, '-u', str(code/'marjum_mcmc_b21.py'),
              '--output', str(out), '--tune', '300', '--draws', '200', '--seed', '20261001',
              '--checkpoint-every', '100', '--state-file', str(inp/'fit_transmitter.npz'),
              '--dem-file', str(inp/'marjum_dem.npz'), '--meta-file', str(inp/'meta.json'),
              '--exif-file', str(inp/'marjum_2026_07_exif_joint.npz'),
              '--feature-dir', str(inp/'cv_features')]
    info = dict(signature=signature, supervisor_pid=os.getpid(), host=os.uname().nodename,
                cwd=str(cwd), started_utc=now(), state='starting', chains=[])
    write_json(out/'launch.json', info)
    processes = {}
    try:
        for index in range(8):
            if args.resume and (out/f'chain_{index}.npz').exists():
                # An interruption during the final write must not make a
                # truncated result look like a completed chain.
                with zipfile.ZipFile(out/f'chain_{index}.npz') as saved:
                    if saved.testzip() is not None or not {
                            'globals.npy', 'final.npy', 'camera_factor.npy'} <= set(saved.namelist()):
                        raise ValueError(f'chain {index} result is incomplete; recover its checkpoint')
                info['chains'].append(dict(chain=index, state='already_complete'))
                continue
            command = common + ['--chain', str(index)]
            if args.resume and (out/f'checkpoint_{index}.pkl').exists():
                command.append('--resume')
            log = logs/f'chain_{index}.log'
            with log.open('a' if args.resume else 'x') as stream:
                process = subprocess.Popen(command, cwd=cwd, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL)
            processes[index] = process
            info['chains'].append(dict(chain=index, pid=process.pid, argv=command,
                                       log=str(log), state='running'))
            write_json(out/'launch.json', info)
            # Only the first worker writes the shared model manifest. Wait for
            # complete JSON before starting another worker, avoiding file races.
            if len(processes) == 1:
                deadline = time.monotonic()+240
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'chain {index} exited before its manifest; see {log}')
                    try:
                        json.loads((out/'manifest.json').read_text())
                        break
                    except (FileNotFoundError, json.JSONDecodeError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('model manifest startup timed out')
                        time.sleep(.5)
        info['state'] = 'running'
        write_json(out/'launch.json', info)
        print(json.dumps(info), flush=True)
        pending = dict(processes)
        while pending:
            for index, process in list(pending.items()):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                item = next(c for c in info['chains'] if c['chain'] == index)
                item.update(exit_code=exit_code, state='complete' if exit_code == 0 else 'failed')
                del pending[index]
                write_json(out/'status.json', info)
                if exit_code:
                    raise RuntimeError(f'chain {index} failed; see {item["log"]}')
            if pending:
                time.sleep(5)
        info.update(state='complete', finished_utc=now())
        write_json(out/'status.json', info)
        print('Pilot complete; interpretation requires separate review.', flush=True)
    except BaseException as exc:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        info.update(state='failed', error=str(exc), finished_utc=now())
        write_json(out/'status.json', info)
        raise


if __name__ == '__main__':
    main()
