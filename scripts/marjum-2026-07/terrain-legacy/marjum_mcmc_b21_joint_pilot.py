"""Paired float32 pilot: fine block updates with and without frozen joint moves.

Four workers: two declared supported starts, each in both arms. Optional
baseline geometry compares an existing frozen-direction kernel against new
directions, with identical current starting states, target and schedules.
No production or interpretation is triggered by completion. Checkpoints contain
all RNG and adapted-kernel state. Run under the approved host user supervisor.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
import marjum_mcmc_b21 as b
from marjum_mcmc_b21_combine import label_coordinates
from marjum_mcmc_b21_coupling import Linearization,sha


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare_local_mixture(root, output, code):
    """Freeze the reviewed two-height/six-angle mixture without refitting it."""
    names=['cam2159_ph','cam2159_th','cam2222_ti']
    fine=root/'fine_geometry_height_angles_20261005'
    local=root/'local_geometry_transfer_20261005'
    cell=root/'cell_aware_derivatives_20261006'
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    for folder,files in [(fine,['diagnostic.json','geometry.npz','supported_endpoints.npz']),
                         (local,['diagnostic.json','geometry_local_5.npz','geometry_local_6.npz','supported_endpoints.npz']),
                         (cell,['diagnostic.json'])]:
        for filename in files:
            rel='../v0002/'+folder.name+'/'+filename
            assert sha(folder/filename)==pins[rel],rel
    f=json.loads((fine/'diagnostic.json').read_text())
    l=json.loads((local/'diagnostic.json').read_text())
    c=json.loads((cell/'diagnostic.json').read_text())
    assert f['status']==l['status']==c['status']=='complete' and c['all_pass']
    assert f['provenance']['config']==l['provenance']['config']==c['provenance']['config']
    assert f['provenance']['model_input_sha256']==l['provenance']['model_input_sha256']==c['provenance']['model_input_sha256']
    assert sha(fine/'geometry.npz')==f['geometry_sha256']
    assert sha(local/'supported_endpoints.npz')==l['supported_endpoints_sha256']
    # Each selected local angle has two passing, within-cell derivative checks
    # at both receiving starts. Selection remains fixed throughout sampling.
    checks={(r['chain'],r['bank'],r['coordinate']):r for r in c['cases']}
    directions={};derivatives=[];probes=[]
    with np.load(fine/'geometry.npz') as source:
        for name in ['transmitter_u','transmitter_n']:
            directions[name]=source[name].copy()
    for bank in ['local_5','local_6']:
        with np.load(local/f'geometry_{bank}.npz') as source:
            for name in names:
                alias=f'{name}_{bank}'
                directions[alias]=source[name].copy()
                for chain in [5,6]:
                    row=checks[chain,bank,name]
                    assert row['geometry_eligible'] and all(x['pass_check'] and x['crossing_count']==0 for x in row['checks'])
                    derivatives.append(dict(coordinate=alias,chain=chain,bank=bank,source_coordinate=name,
                                            checks=row['checks'],steps_scaled=row['steps_scaled']))
    assert len(directions)==8 and len(derivatives)==12
    for name in ['transmitter_u','transmitter_n']:
        derivatives.append(dict(coordinate=name,source='fine_geometry_height_angles_20261005'))
    for row in l['probes']:
        if row['chain'] in [5,6] and row['bank'] in ['local_5','local_6'] and row['coordinate'] in names:
            probes.append(dict(row,coordinate=f"{row['coordinate']}_{row['bank']}"))
    for row in f['probes']:
        if row['chain'] in [5,6] and row['coordinate'] in ['transmitter_u','transmitter_n']:
            probes.append(row)
    lookup={(r['chain'],r['coordinate'],r['mode'],r['amplitude_scaled']):r for r in probes}
    for name in directions:
        for chain in [5,6]:
            for amplitude in [.005,.05]:
                for mode in ['block_only','globals_landmarks']:
                    assert lookup[chain,name,mode,amplitude]['both_supported'],(chain,name,mode,amplitude)
    assert len(probes)==64
    assert not output.exists()
    output.mkdir(parents=True)
    np.savez(output/'geometry.npz',**directions)
    shutil.copyfile(local/'supported_endpoints.npz',output/'supported_endpoints.npz')
    commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    sources=['marjum_mcmc_b21_joint_pilot.py']
    result=dict(status='complete',proposal_validation='fixed_local_bank_mixture',derivative_gate_pass=True,
        geometry_sha256=sha(output/'geometry.npz'),supported_endpoints_sha256=sha(output/'supported_endpoints.npz'),
        probe_chains=[5,6],derivative_checks=derivatives,probes=probes,
        source_products={str(folder.name):{filename:sha(folder/filename) for filename in files}
                         for folder,files in [(fine,['diagnostic.json','geometry.npz','supported_endpoints.npz']),
                                              (local,['diagnostic.json','geometry_local_5.npz','geometry_local_6.npz','supported_endpoints.npz']),
                                              (cell,['diagnostic.json'])]},
        provenance=dict(commit=commit,source_hashes={n:sha(code/n) for n in sources},
            config=f['provenance']['config'],model_input_sha256=f['provenance']['model_input_sha256']))
    write(output/'diagnostic.json',result)
    (output/'README.md').write_text('# Fixed local-angle mixture geometry\n\n'
        'Eight frozen proposal vectors: the archived transmitter height and northing vectors, plus three camera-angle vectors trained at each of chains 5 and 6. '
        'Every vector is selected with probability 1/8 independently of the current state. '
        'The saved arrays and endpoints are copied from the pinned source products; no new fit or target is made.\n\n'
        'The two-step, within-DEM-cell derivative validation passed at both receiving endpoints for the six angle vectors. '
        'Signed proposals at amplitudes 0.005 and 0.05 have valid support at both endpoints, but the angles do not uniformly reduce the even penalty when transferred. '
        'diagnostic.json records every source hash and signed support result.\n\n'
        '## Recent changes\n\n- 2026-10-06: Froze the reviewed local camera-angle mixture for a paired pilot.\n')
    print(json.dumps({'geometry':str(output),'sha256':result['geometry_sha256'],'directions':list(directions)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['blocks','joint','previous']);p.add_argument('--chain',type=int)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--geometry',default='fine_geometry_refined_20261003')
    p.add_argument('--joint-names',default='cam2223_n,cam2224_n')
    p.add_argument('--baseline-geometry')
    p.add_argument('--baseline-names',default='cam2223_n,transmitter_n')
    p.add_argument('--starts',default='0,1')
    p.add_argument('--seed',type=int,default=20261003)
    p.add_argument('--prepare-local-mixture',action='store_true')
    p.add_argument('--state-aware-selector',action='store_true',
                   help='Use endpoint-angle-dependent bank weights with the exact reverse/forward correction')
    a=p.parse_args();code=Path(__file__).resolve().parent;root=a.root.resolve();out=a.output.resolve()
    if a.prepare_local_mixture:
        if subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise RuntimeError('commit packet builder before freezing geometry')
        prepare_local_mixture(root,out,code);return
    inp=root/'inputs';geo=root/a.geometry;old=root.parent/'v0001'
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    for rel in ['../v0002/inputs/input_manifest.json']+['../v0002/'+a.geometry+'/'+n for n in ['diagnostic.json','geometry.npz','supported_endpoints.npz']]:
        assert sha(old/rel)==pins[rel],rel
    frozen=json.loads((inp/'input_manifest.json').read_text())
    for item in frozen['files'].values():assert sha(root/item['frozen'])==item['sha256']
    for name,item in frozen['feature_cache']['files'].items():assert sha(inp/'cv_features'/name)==item['sha256']
    target_prefix_hash=None
    def validated_geometry(folder, selected):
        nonlocal target_prefix_hash
        result=json.loads((folder/'diagnostic.json').read_text())
        assert result['status']=='complete' and result['derivative_gate_pass']
        for name in ['diagnostic.json','geometry.npz','supported_endpoints.npz']:
            assert sha(folder/name)==pins['../v0002/'+folder.name+'/'+name]
        assert sha(folder/'geometry.npz')==result['geometry_sha256']
        assert sha(folder/'supported_endpoints.npz')==result['supported_endpoints_sha256']
        assert set(selected) <= {r['coordinate'] for r in result['derivative_checks']}
        # Archived vectors remain valid symmetric proposals when their target
        # matches, even if the diagnostic helper has since been extended.
        provenance=result['provenance']
        for name,want in provenance['source_hashes'].items():
            blob=subprocess.check_output(['git','-C',str(code),'show',provenance['commit']+':'+name])
            assert hashlib.sha256(blob).hexdigest()==want
        model_inputs=provenance.get('model_input_sha256')
        if not model_inputs:raise ValueError('geometry lacks model/dependency hashes; recompute before launch')
        for path,want in model_inputs.items():
            if sha(Path(path))==want:continue
            if not a.state_aware_selector or Path(path).resolve()!=code/'marjum_mcmc_b21.py':
                raise ValueError(f'geometry target input changed: {path}')
            archived=subprocess.check_output(['git','-C',str(code),'show',
                                              provenance['commit']+':marjum_mcmc_b21.py'])
            assert hashlib.sha256(archived).hexdigest()==want
            old_prefix=archived.split(b'\nclass Chain',1)[0]
            current_prefix=(code/'marjum_mcmc_b21.py').read_bytes().split(
                b'\ndef validated_joint_selector',1)[0]
            assert old_prefix==current_prefix,'model target or geometry code changed'
            target_prefix_hash=hashlib.sha256(current_prefix).hexdigest()
        return result
    selected=a.joint_names.split(',')
    validated=validated_geometry(geo,selected)
    config=validated['provenance']['config'];tune,draws=(2,3) if a.smoke else (100,200)
    arm_geometry={'joint':a.geometry};arm_names={'joint':selected}
    if a.baseline_geometry:
        # A local-bank mixture is valid as a fixed symmetric proposal even if
        # finite-step gain does not transfer between endpoints.
        probes=validated['probes']
        lookup={(r['chain'],r['coordinate'],r['mode'],r.get('amplitude_scaled',r.get('amplitude_m'))):r for r in probes}
        for name in selected:
            for chain in validated['probe_chains']:
                for amplitude in [.005,.05]:
                    block=lookup[chain,name,'block_only',amplitude]
                    joint=lookup[chain,name,'globals_landmarks',amplitude]
                    assert block['both_supported'] and joint['both_supported']
                    if validated.get('proposal_validation')!='fixed_local_bank_mixture':
                        assert joint['even_penalty']<block['even_penalty'],(chain,name,amplitude)
        previous=validated_geometry(root/a.baseline_geometry,a.baseline_names.split(','))
        assert previous['provenance']['config']==config
        assert previous['provenance']['model_input_sha256']==validated['provenance']['model_input_sha256']
        arms=['previous','joint']
        arm_geometry['previous']=a.baseline_geometry;arm_names['previous']=a.baseline_names.split(',')
    else:arms=['blocks','joint']
    starts=[int(v) for v in a.starts.split(',')]
    assert len(starts)==2 and len(set(starts))==2
    with np.load(geo/'supported_endpoints.npz') as saved:
        assert all(f'chain_{c}' in saved for c in starts)
        selector=None
        if a.state_aware_selector:
            assert validated.get('proposal_validation')=='fixed_local_bank_mixture'
            assert starts==[5,6] and len(selected)==8
            groups=[1 if n.endswith('_local_5') else 2 if n.endswith('_local_6') else 0 for n in selected]
            assert groups.count(0)==2 and groups.count(1)==groups.count(2)==3
            with np.load(inp/'fit_transmitter.npz') as fitted:
                names=label_coordinates(fitted['keys'].astype(str).tolist(),joint=True)
            indices=[names.index(n) for n in ['cam2159_ph','cam2159_th','cam2222_ti']]
            centers=np.stack([saved[f'chain_{c}'][indices] for c in starts])
            bandwidth=float(np.linalg.norm(centers[1]-centers[0])/2)
            selector=b.validated_joint_selector(dict(indices=indices,centers=centers.tolist(),
                groups=groups,bandwidth=bandwidth,neutral_mass=.25,bank_floor=.05),
                len(saved[f'chain_{starts[0]}']),len(selected))
    if a.worker:
        assert a.worker in arms and a.chain in starts
    signature=dict(code_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        code_sha256={n:sha(code/n) for n in ['marjum_mcmc_b21.py','marjum_mcmc_b21_joint_pilot.py','marjum_mcmc_b21_coupling.py','marjum_mcmc_b21_combine.py','marjum_mcmc.py','marjum_camera.py','marjum_bundle.py','marjum_fitio.py']},
        input_manifest_sha256=sha(inp/'input_manifest.json'),geometry_sha256=sha(geo/'geometry.npz'),
        starts_sha256=sha(geo/'supported_endpoints.npz'),config=config,tune=tune,draws=draws,seed=a.seed,
        model_input_sha256={path:sha(Path(path)) for path in validated['provenance']['model_input_sha256']},
        archived_model_input_sha256=validated['provenance']['model_input_sha256'],
        target_prefix_sha256=target_prefix_hash,
        arm_geometry=arm_geometry,arm_names=arm_names,
        arm_geometry_sha256={arm:sha(root/folder/'geometry.npz') for arm,folder in arm_geometry.items()},
        starts=starts,arms=arms,difference_step_factor=1e-4,joint_scale=.005,joint_every=1,
        joint_names=selected,joint_selector=selector,checkpoint_every=25,thin_points=10,shift_every=5)
    if a.worker:
        m=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),meta_file=inp/'meta.json',
            exif_file=inp/'marjum_2026_07_exif_joint.npz',feature_dir=inp/'cv_features',config=b.Config(**config))
        assert {str(f):sha(f) for f in m.input_files}==signature['model_input_sha256']
        with np.load(geo/'supported_endpoints.npz') as saved:start=saved[f'chain_{a.chain}'].copy()
        linear=Linearization(m,start)
        directions=None
        if a.worker in arm_geometry:
            arm_geo=root/arm_geometry[a.worker];chosen=arm_names[a.worker]
            with np.load(arm_geo/'geometry.npz') as saved:
                directions=np.array([np.r_[saved[n][2,:linear.ng]*linear.scale,np.zeros(2),saved[n][2,linear.ng:]] for n in chosen])
                for name,direction in zip(chosen,directions):
                    np.testing.assert_allclose(start+.005*direction,linear.expand(.005*saved[name][2]),rtol=0,atol=1e-10)
        folder=out/a.worker;folder.mkdir(exist_ok=True)
        # Every worker writes separate metadata; no concurrent shared-file mutation.
        write(folder/f'manifest_{a.chain}.json',dict(signature=signature,arm=a.worker,chain=a.chain,keys=m.keys,
            initial_logp=float(m.logp(start)),input_sha256={str(f):sha(f) for f in m.input_files}))
        clock=time.monotonic();cpu=time.process_time()
        b.run_chain(m,a.chain,tune,draws,signature['seed'],folder,thin_points=10,shift_every=5,
            checkpoint_every=25,resume=a.resume,start=start,difference_step_factor=1e-4,
            joint_directions=directions,joint_scale=.005,joint_every=1,
            joint_selector=selector if a.worker=='joint' else None)
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
    selector_description=('state-aware with reverse/forward selection correction' if selector else
                          'uniform over the fixed directions')
    (out/'README.md').write_text('# Paired collective-direction pilot\n\n'
        f'Two starts {starts}, each in both arms {arms}, {tune} warmup and {draws} retained sweeps.\n'
        f'Arm directions: {json.dumps(arm_names)}. Geometry: {json.dumps(arm_geometry)}.\n'
        'Both arms use the same current endpoints, target, random seeds, fine block steps,\n'
        'initial joint scale, one joint attempt per sweep, and adaptation schedule.\n'
        f'This compares whole frozen-direction kernels. The candidate selector is {selector_description}.\n'
        'Full hashes, configuration, commands and PIDs are in launch.json.\n'
        'Checkpoints save RNG and adaptation every 25 sweeps; resume with --resume.\n'
        'Globals are saved every sweep; landmarks and log density every 10.\n'
        'Compare transmitter-height movement, chain separation, all-coordinate diagnostics,\n'
        'and ESS per CPU time, with worst cases. Two short chains cannot certify convergence.\n\n'
        '## Recent changes\n\n- 2026-10-05: Prepared paired existing-kernel versus height/northing comparison.\n')
    info=dict(signature=signature,supervisor_pid=os.getpid(),started_utc=now(),state='starting',workers=[])
    processes={}
    try:
        for arm in signature['arms']:
            for chain in signature['starts']:
                result=out/arm/f'chain_{chain}.npz'
                if a.resume and (out/arm/f'completion_{chain}.json').exists():
                    done=json.loads((out/arm/f'completion_{chain}.json').read_text());assert sha(result)==done['result_sha256']
                    info['workers'].append(dict(arm=arm,chain=chain,state='already_complete'));continue
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--output',str(out),'--worker',arm,'--chain',str(chain),'--geometry',a.geometry,'--joint-names',a.joint_names,'--starts',a.starts,'--seed',str(a.seed)]
                if a.baseline_geometry:command+=['--baseline-geometry',a.baseline_geometry,'--baseline-names',a.baseline_names]
                if a.state_aware_selector:command.append('--state-aware-selector')
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
