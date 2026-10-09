"""Five-chain float32 diagnostic using freshly validated collective directions.

The paired pilot (joint_pilot_20261003) showed the frozen cam2223_n/cam2224_n
directions are one collective network mode (cosine 0.97) that cuts start-to-start
separation for nearly every camera, while neither arm converged (ESS ~3 in 200
draws). This run uses explicitly selected validated directions and all five
supported float32 starts (old endpoints 0, 1, 5, 6, 7), 300 warmup and 2000
retained sweeps, to test whether rank R-hat and ESS approach the review's
1.01 / 400 criteria. It is not a validated posterior. Checkpoints contain all
RNG and adapted-kernel state. Run under the host user supervisor.
"""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time
import numpy as np
import marjum_mcmc_b21 as b
from marjum_mcmc_b21_coupling import Linearization,sha


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['joint']);p.add_argument('--chain',type=int,choices=[0,1,5,6,7])
    p.add_argument('--smoke',action='store_true');p.add_argument('--geometry',default='fine_geometry_refined_20261003',help='validated geometry directory under --root; its diagnostic config defines the target');p.add_argument('--joint-names',default='cam2223_n',help='comma-separated validated directions from the refined geometry');p.add_argument('--resume',action='store_true')
    a=p.parse_args();code=Path(__file__).resolve().parent;root=a.root.resolve();out=a.output.resolve()
    inp=root/'inputs';geo=root/a.geometry;old=root.parent/'v0001'
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    for rel in ['../v0002/inputs/input_manifest.json']+['../v0002/fine_geometry_refined_20261003/'+n for n in ['diagnostic.json','geometry.npz','supported_endpoints.npz']]:
        assert sha(old/rel)==pins[rel],rel
    frozen=json.loads((inp/'input_manifest.json').read_text())
    for item in frozen['files'].values():assert sha(root/item['frozen'])==item['sha256']
    for name,item in frozen['feature_cache']['files'].items():assert sha(inp/'cv_features'/name)==item['sha256']
    validated=json.loads((geo/'diagnostic.json').read_text());assert validated['derivative_gate_pass']
    assert validated['status']=='complete'
    assert sha(geo/'geometry.npz')==validated['geometry_sha256']
    assert sha(geo/'supported_endpoints.npz')==validated['supported_endpoints_sha256']
    for name,want in validated['provenance']['source_hashes'].items():
        assert sha(code/name)==want, f'recompute geometry after changing {name}'
    model_inputs=validated['provenance'].get('model_input_sha256')
    if not model_inputs:raise ValueError('recompute geometry with model/dependency provenance before launch')
    for path,want in model_inputs.items():assert sha(Path(path))==want,path
    assert set(a.joint_names.split(',')) <= {r['coordinate'] for r in validated['derivative_checks']}
    config=validated['provenance']['config'];tune,draws=(2,3) if a.smoke else (300,2000)
    signature=dict(code_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        code_sha256={n:sha(code/n) for n in ['marjum_mcmc_b21.py','marjum_mcmc_b21_joint_run.py','marjum_mcmc_b21_coupling.py','marjum_mcmc.py','marjum_camera.py','marjum_bundle.py','marjum_fitio.py']},
        model_input_sha256=model_inputs,
        input_manifest_sha256=sha(inp/'input_manifest.json'),geometry_sha256=sha(geo/'geometry.npz'),
        starts_sha256=sha(geo/'supported_endpoints.npz'),geometry=a.geometry,config=config,tune=tune,draws=draws,seed=20261003,
        starts=[0,1,5,6,7],arms=['joint'],difference_step_factor=1e-4,joint_scale=.005,joint_every=1,
        joint_names=a.joint_names.split(','),checkpoint_every=25,thin_points=10,shift_every=5)
    if a.worker:
        m=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),meta_file=inp/'meta.json',
            exif_file=inp/'marjum_2026_07_exif_joint.npz',feature_dir=inp/'cv_features',config=b.Config(**config))
        assert {str(f):sha(f) for f in m.input_files}==model_inputs,'worker model differs from validated geometry'
        with np.load(geo/'supported_endpoints.npz') as saved:start=saved[f'chain_{a.chain}'].copy()
        linear=Linearization(m,start)
        with np.load(geo/'geometry.npz') as saved:
            directions=np.array([np.r_[saved[n][2,:linear.ng]*linear.scale,np.zeros(2),saved[n][2,linear.ng:]] for n in signature['joint_names']])
            for name,direction in zip(signature['joint_names'],directions):
                np.testing.assert_allclose(start+.005*direction,linear.expand(.005*saved[name][2]),rtol=0,atol=1e-10)
        folder=out/a.worker;folder.mkdir(exist_ok=True)
        # Every worker writes separate metadata; no concurrent shared-file mutation.
        write(folder/f'manifest_{a.chain}.json',dict(signature=signature,arm=a.worker,chain=a.chain,keys=m.keys,
            initial_logp=float(m.logp(start)),input_sha256={str(f):sha(f) for f in m.input_files},
            focal_priors={k:dict(centre_px=float(f),log_sigma=float(s)) for k,f,s in zip(m.keys,m.focal,m.focal_sigma)}))
        clock=time.monotonic();cpu=time.process_time()
        b.run_chain(m,a.chain,tune,draws,signature['seed'],folder,thin_points=10,shift_every=5,
            checkpoint_every=25,resume=a.resume,start=start,difference_step_factor=1e-4,
            joint_directions=directions if a.worker=='joint' else None,joint_scale=.005,joint_every=1)
        with np.load(folder/f'chain_{a.chain}.npz') as saved:
            assert saved['globals'].shape==(draws,m.ng) and np.isfinite(saved['globals']).all()
            assert np.isfinite(m.logp(saved['final']))
            report=dict(arm=a.worker,chain=a.chain,seconds=time.monotonic()-clock,cpu_seconds=time.process_time()-cpu,
                joint_acceptance=json.loads(str(saved['joint_acceptance'])),acceptance=json.loads(str(saved['acceptance'])),
                camera_geometry_history=json.loads(str(saved['camera_geometry_history'])),result_sha256=sha(folder/f'chain_{a.chain}.npz'))
        write(folder/f'completion_{a.chain}.json',report);return
    if subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('commit pilot code before launch')
    if a.resume:
        previous=json.loads((out/'launch.json').read_text())
        assert previous['signature']==signature,'resume source/input/config mismatch'
    else:out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir(exist_ok=True)
    (out/'.gitignore').write_text('**/*.npz\n**/*.pkl\n**/*.pkl.tmp\n')
    (out/'README.md').write_text('# Longer float32 run with the collective joint move\n\n'
        'Five supported starts (old endpoints 0, 1, 5, 6, 7), fine block updates\n'
        f'plus the frozen collective directions {a.joint_names}, {tune} warmup + {draws} retained\n'
        'sweeps. Configuration is inherited from the validated geometry, including\n'
        'the diagonal EXIF centres and any camera-specific focal-prior widths.\n'
        'Tests rank R-hat <= 1.01 and bulk/tail\n'
        'ESS >= 400; not a validated posterior. Full configuration, source hashes,\n'
        'commands and PIDs are in launch.json; status.json records completion.\n'
        'Checkpoints save all random streams and adaptation every 25 sweeps.\n'
        'Resume with the same command plus --resume; trusted local pickles only.\n'
        'All globals are recorded each sweep; landmarks and log density every 10.\n'
        'Report every global, worst cases first; passing marginal checks is\n'
        'necessary, not sufficient, for joint convergence.\n\n'
        '## Recent changes\n\n- 2026-10-04: Prepared the corrected-EXIF diagnostic with matching geometry and dependency checks.\n')
    info=dict(signature=signature,supervisor_pid=os.getpid(),started_utc=now(),state='starting',workers=[])
    processes={}
    try:
        for arm in signature['arms']:
            for chain in signature['starts']:
                result=out/arm/f'chain_{chain}.npz'
                if a.resume and (out/arm/f'completion_{chain}.json').exists():
                    done=json.loads((out/arm/f'completion_{chain}.json').read_text());assert sha(result)==done['result_sha256']
                    info['workers'].append(dict(arm=arm,chain=chain,state='already_complete'));continue
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--output',str(out),'--worker',arm,'--chain',str(chain),'--joint-names',a.joint_names,'--geometry',a.geometry]
                if a.smoke:command.append('--smoke')
                if a.resume and (out/arm/f'checkpoint_{chain}.pkl').exists():command.append('--resume')
                log=out/'logs'/f'{arm}_{chain}.log'
                with log.open('a' if a.resume else 'x') as stream:
                    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                key=(arm,chain);processes[key]=child
                info['workers'].append(dict(arm=arm,chain=chain,pid=child.pid,command=command,log=str(log),state='running'))
        info['state']='running';write(out/'launch.json',info);write(out/'status.json',info)
        print(json.dumps(info),flush=True)
        while processes:
            for key,child in list(processes.items()):
                status=child.poll()
                if status is None:continue
                item=next(w for w in info['workers'] if (w['arm'],w['chain'])==key)
                item.update(exit_code=status,state='complete' if status==0 else 'failed');del processes[key]
                write(out/'status.json',info)
                if status:raise RuntimeError(f'{key} failed; see {item["log"]}')
            if processes:time.sleep(3)
        info.update(state='complete',finished_utc=now());write(out/'status.json',info)
    except BaseException as exc:
        for child in processes.values():child.terminate()
        for child in processes.values():child.wait()
        info.update(state='failed',error=str(exc),finished_utc=now());write(out/'status.json',info);raise

if __name__=='__main__':main()
