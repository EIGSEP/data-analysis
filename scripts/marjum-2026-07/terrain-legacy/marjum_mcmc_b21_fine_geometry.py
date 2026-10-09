"""Recompute fine-step geometry on a supported float32 state and test transfer.

Screens all eight old final states without altering them. Training uses chain 0;
validation uses the first other supported endpoint in ascending chain order.
This selection is based on support only, before testing directional performance.
With --source-run, use that completed run's target and final states, and check
transfer at every supported endpoint. Probe amplitudes are reported in both
scaled coordinates and physical units (metres, radians, or log focal length).
--training-chains compares independent local banks with a pinned archived bank
at the declared receiving endpoints, with checkpointed progress and a time cap.
"""
from pathlib import Path
import argparse,hashlib,json,signal,subprocess,time
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import save_npz
import marjum_mcmc_b21 as b
from marjum_mcmc_b21_coupling import Linearization,geometry,sha
from marjum_mcmc_b21_dem_diagnostic import support_details
from marjum_mcmc_b21_combine import label_coordinates


def local_transfer(model, states, args, out, save, start, code):
    """Compare independently trained geometry with a pinned archived control."""
    training=[int(c) for c in args.training_chains.split(',')]
    probes=[int(c) for c in args.probe_chains.split(',')]
    focus=args.coordinates.split(',')
    assert len(set(training))==len(training) and len(set(probes))==len(probes)
    assert training and probes and set(training+probes)<=set(states)
    if args.smoke:
        training=training[:1];focus=focus[:2]
    archive=args.root/args.baseline_geometry
    baseline=json.loads((archive/'diagnostic.json').read_text())
    bp=baseline['provenance'];pins=out['provenance']['input_pins']
    for name in ['diagnostic.json','geometry.npz','supported_endpoints.npz']:
        assert sha(archive/name)==pins['../v0002/'+args.baseline_geometry+'/'+name]
    assert baseline['status']=='complete' and baseline['derivative_gate_pass']
    assert bp['config']==out['provenance']['config']
    assert bp['model_input_sha256']==out['provenance']['model_input_sha256']
    for name,want in bp['source_hashes'].items():
        blob=subprocess.check_output(['git','-C',str(code),'show',bp['commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==want
    with np.load(archive/'geometry.npz',allow_pickle=False) as saved:
        banks={'archived_0':{n:saved[n].copy() for n in focus}}
    out.update(training_chains=training,probe_chains=probes,coordinates=focus,
        smoke=args.smoke,geometry_banks={},stage='validated_inputs')
    out['provenance']['baseline_geometry']=dict(path=str(archive),diagnostic_sha256=sha(archive/'diagnostic.json'),geometry_sha256=sha(archive/'geometry.npz'),commit=bp['commit'])
    selected=sorted(set(training+probes))
    np.savez_compressed(args.output/'supported_endpoints.npz',**{f'chain_{c}':states[c] for c in selected})
    out['supported_endpoints_sha256']=sha(args.output/'supported_endpoints.npz');save()
    lines={c:Linearization(model,states[c]) for c in selected};jacs={}
    names=label_coordinates(model.keys,joint=True)[:lines[selected[0]].ng]
    scales={n:float(lines[selected[0]].scale[names.index(n)]) for n in focus}
    units={n:('rad' if n.endswith(('_th','_ph','_ti')) else 'log_pixels' if n.endswith('_logf') else 'm') for n in focus}
    out['coordinate_scales']=scales;out['coordinate_units']=units
    # Smoke checks one new Jacobian; full diagnostic checks derivatives at both
    # receiving endpoints, including derivatives along transferred directions.
    for chain in training:
        line=lines[chain];line.step*=args.step_multiplier
        out['stage']=f'jacobian_{chain}';save()
        jac=approx_derivative(line.residual,np.zeros(line.nvar),method='3-point',abs_step=line.step,sparsity=line.sparsity)
        jacs[chain]=jac;save_npz(args.output/f'jacobian_{chain}.npz',jac)
        directions,widths,checks=geometry(line,jac=jac,stable=True,focus=focus)
        key=f'local_{chain}';banks[key]=directions
        out['geometry_banks'][key]=dict(training_chain=chain,widths=widths,geometry_checks=checks,jacobian_sha256=sha(args.output/f'jacobian_{chain}.npz'))
        print('computed geometry',chain,flush=True);save()
    out['geometry_banks']['archived_0']=dict(training_chain=baseline['training_chain'],source_geometry_sha256=sha(archive/'geometry.npz'))
    for bank,directions in banks.items():
        for name in focus:
            assert directions[name].shape==(3,lines[selected[0]].nvar)
            assert np.isfinite(directions[name]).all()
            np.testing.assert_allclose(directions[name][:,names.index(name)],1.,rtol=0,atol=1e-10)
        np.savez_compressed(args.output/f'geometry_{bank}.npz',**directions)
        out['geometry_banks'][bank]['geometry_sha256']=sha(args.output/f'geometry_{bank}.npz')
    out['stage']='derivative_checks';save()
    for chain,jac in jacs.items():
        line=lines[chain]
        for bank,directions in banks.items():
            for name in focus:
                direction=directions[name][2];predicted=np.asarray(jac@direction).ravel()
                for h in [1e-4,1e-5]:
                    direct=(line.residual(h*direction)-line.residual(-h*direction))/(2*h)
                    error=float(np.linalg.norm(predicted-direct)/max(np.linalg.norm(direct),np.finfo(float).tiny))
                    out['derivative_checks'].append(dict(chain=chain,bank=bank,coordinate=name,step_scaled=h,step_physical=h*scales[name],unit=units[name],relative_vector_error=error,
                        per_term={k:dict(predicted_norm=float(np.linalg.norm(predicted[s])),direct_norm=float(np.linalg.norm(direct[s])),error_norm=float(np.linalg.norm((predicted-direct)[s]))) for k,s in line.rows.items()}))
                save()
    out['derivative_gate_pass']=all(r['relative_vector_error']<.01 for r in out['derivative_checks'])
    out['stage']='exact_probes';save()
    for chain in probes:
        line=lines[chain];zero=np.zeros(line.nvar);r0=line.residual(zero)
        lp0=float(model.logp(states[chain]));norm=line.normalization()
        assert abs(-.5*r0@r0+norm-lp0)<1e-8
        terms0={k:float(-.5*r0[s]@r0[s]) for k,s in line.rows.items()}
        for bank,directions in banks.items():
            for name in focus:
                for mode,idx in [('block_only',0),('globals_landmarks',2)]:
                    for h in [.005,.05]:
                        changes=[];terms=[];failures=[]
                        for sign in [-1,1]:
                            delta=sign*h*directions[name][idx];state=line.expand(delta);exact=float(model.logp(state))
                            if not np.isfinite(exact):
                                changes.append(None);terms.append(None);failures.append(support_details(model,state));continue
                            r=line.residual(delta);assert abs(-.5*r@r+norm-exact)<1e-8
                            change=exact-lp0;term={k:float(-.5*r[s]@r[s])-terms0[k] for k,s in line.rows.items()}
                            assert abs(sum(term.values())-change)<1e-8
                            changes.append(change);terms.append(term);failures.append([])
                        ok=all(c is not None for c in changes)
                        out['probes'].append(dict(chain=chain,bank=bank,coordinate=name,mode=mode,amplitude_scaled=h,amplitude_physical=h*scales[name],unit=units[name],delta_logp=changes,term_delta_logp=terms,support_failures=failures,both_supported=ok,even_penalty=-sum(changes) if ok else None))
                print('tested',chain,bank,name,flush=True);save()
    out['seconds']=time.monotonic()-start;out['stage']='finished';out['status']='complete';save('diagnostic.json')
    (args.output/'README.md').write_text('# Local geometry and transfer diagnostic\n\n'
        'Generated by marjum_mcmc_b21_fine_geometry.py. diagnostic.json records\n'
        'input/code hashes, derivatives at each trained endpoint, and exact signed\n'
        'density penalties and term contributions for every bank and receiving state.\n'
        'Geometry and Jacobian archives retain each trained bank separately;\n'
        'checkpoint.json preserves partial work under the declared time cap.\n'
        'Smoke uses one training endpoint and two coordinates. This is a proposal\n'
        'diagnostic, not posterior sampling or evidence of a physical cause.\n\n'
        '## Recent changes\n\n- 2026-10-05: Compare locally recomputed and archived directions at separated endpoints.\n')
    print('completed_seconds',out['seconds'],'derivative_gate',out['derivative_gate_pass'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--step-multiplier',type=float,default=.001)
    p.add_argument('--log-f-sigma',type=float,default=None,help='override the focal-length prior width (default: the pilot config)')
    p.add_argument('--log-f-sigma-by-camera',type=json.loads,default=None,help='JSON object of fixed per-camera log-f widths')
    p.add_argument('--coordinates',default='cam2223_n,cam2224_n',help='comma-separated directions to check and probe')
    p.add_argument('--source-run',help='completed run under --root: inherit its config and use its final states')
    p.add_argument('--training-chains',help='comma-separated local training endpoints; enables bank comparison')
    p.add_argument('--probe-chains',default='5,6',help='receiving endpoints for local bank comparison')
    p.add_argument('--baseline-geometry',default='fine_geometry_height_angles_20261005')
    p.add_argument('--max-seconds',type=float,default=900,help='local bank comparison wall-time cap')
    p.add_argument('--smoke',action='store_true',help='one local Jacobian and first two coordinates')
    a=p.parse_args();assert 0<a.step_multiplier<=1 and a.max_seconds>0
    if a.training_chains:assert a.source_run,'local comparisons require an unchanged completed-run target'
    if a.smoke:assert a.training_chains,'smoke applies only to local bank comparisons'
    code=Path(__file__).resolve().parent;old=a.root.parent/'v0001';inp=a.root/'inputs'
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    for rel,want in pins.items():assert sha(old/rel)==want,rel
    frozen=json.loads((inp/'input_manifest.json').read_text())
    for item in frozen['files'].values():assert sha(a.root/item['frozen'])==item['sha256']
    for name,item in frozen['feature_cache']['files'].items():assert sha(inp/'cv_features'/name)==item['sha256']
    config=json.loads((old/'pilot_logf_20261002/manifest.json').read_text())['config']
    source_status=None
    if a.source_run:
        source=a.root/a.source_run
        source_status=json.loads((source/'status.json').read_text())
        assert source_status['state']=='complete' and all(w['exit_code']==0 for w in source_status['workers'])
        assert sha(source/'status.json')==pins['../v0002/'+a.source_run+'/status.json']
        config=source_status['signature']['config']
    if a.log_f_sigma is not None:config=dict(config,log_f_sigma=a.log_f_sigma)
    if a.log_f_sigma_by_camera is not None:config=dict(config,log_f_sigma_by_camera=a.log_f_sigma_by_camera)
    a.output.mkdir(exist_ok=False)
    out=dict(provenance=dict(commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        source_hashes={n:sha(code/n) for n in ['marjum_mcmc_b21_fine_geometry.py','marjum_mcmc_b21_coupling.py','marjum_mcmc_b21_dem_diagnostic.py','marjum_mcmc_b21.py','marjum_mcmc.py','marjum_camera.py','marjum_bundle.py']},
        input_pins=pins,input_manifest_sha256=sha(inp/'input_manifest.json'),config=config,step_multiplier=a.step_multiplier,coordinates=a.coordinates),
        endpoint_screen=[],derivative_checks=[],probes=[])
    def save(name='checkpoint.json'):
        tmp=a.output/(name+'.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(a.output/name)
    start=time.monotonic()
    if a.training_chains:
        out['status']='running';out['provenance']['max_seconds']=a.max_seconds
        def interrupted(signum,frame):
            out['status']='time_limit' if signum==signal.SIGALRM else 'interrupted'
            out['seconds']=time.monotonic()-start;save()
            raise SystemExit(124 if signum==signal.SIGALRM else 128+signum)
        signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted)
        signal.setitimer(signal.ITIMER_REAL,a.max_seconds);save()
    m=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),meta_file=inp/'meta.json',exif_file=inp/'marjum_2026_07_exif_joint.npz',feature_dir=inp/'cv_features',config=b.Config(**config))
    out['provenance']['model_input_sha256']={str(f):sha(f) for f in m.input_files}
    out['focal_priors']={k:dict(centre_px=float(f),log_sigma=float(s)) for k,f,s in zip(m.keys,m.focal,m.focal_sigma)}
    if source_status:
        assert config==source_status['signature']['config'],'source-run diagnostics must preserve the target'
        assert out['provenance']['model_input_sha256']==source_status['signature']['model_input_sha256']
        out['provenance']['source_run']=a.source_run
        out['provenance']['source_status_sha256']=sha(source/'status.json')
        out['provenance']['endpoint_sha256']={}
    states={}
    for chain in (source_status['signature']['starts'] if source_status else range(8)):
        path=source/'joint'/f'chain_{chain}.npz' if source_status else old/f'pilot_logf_20261002/chain_{chain}.npz'
        if source_status:
            assert sha(path)==pins['../v0002/'+a.source_run+f'/joint/chain_{chain}.npz']
            out['provenance']['endpoint_sha256'][str(chain)]=sha(path)
        with np.load(path) as z:state=z['final'].copy()
        lp=float(m.logp(state));supported=bool(np.isfinite(lp))
        out['endpoint_screen'].append(dict(chain=chain,supported=supported,logp=lp if supported else None,failures=[] if supported else support_details(m,state)))
        if supported:states[chain]=state
    if a.training_chains:
        try:
            local_transfer(m,states,a,out,save,start,code)
        except Exception as error:
            out['status']='failed';out['error']=repr(error);out['seconds']=time.monotonic()-start;save();raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
        return
    assert 0 in states and len(states)>1
    validation=next(c for c in sorted(states) if c!=0)
    out['training_chain']=0;out['validation_chain']=validation
    out['probe_chains']=sorted(states) if source_status else [0,validation]
    np.savez_compressed(a.output/'supported_endpoints.npz',**{f'chain_{c}':s for c,s in states.items()})
    print('supported endpoints',list(states),'validation',validation,flush=True);save()
    linear=Linearization(m,states[0]);linear.step*=a.step_multiplier
    zero=np.zeros(linear.nvar)
    jac=approx_derivative(linear.residual,zero,method='3-point',abs_step=linear.step,sparsity=linear.sparsity)
    save_npz(a.output/'jacobian.npz',jac)
    out['jacobian_sha256']=sha(a.output/'jacobian.npz');save()
    focus=a.coordinates.split(',')
    directions,widths,checks=geometry(linear,jac=jac,stable=True,focus=focus)
    names=label_coordinates(m.keys,joint=True)[:linear.ng]
    coordinate_scales={name:float(linear.scale[names.index(name)]) for name in focus}
    coordinate_units={name:('rad' if name.endswith(('_th','_ph','_ti')) else 'log_pixels' if name.endswith('_logf') else 'm') for name in focus}
    out['coordinate_scales']=coordinate_scales;out['coordinate_units']=coordinate_units
    out['widths']=widths;out['geometry_checks']=checks
    np.savez_compressed(a.output/'geometry.npz',**directions)
    out['geometry_sha256']=sha(a.output/'geometry.npz');out['supported_endpoints_sha256']=sha(a.output/'supported_endpoints.npz')
    for name in a.coordinates.split(','):
        direction=directions[name][2];predicted=np.asarray(jac@direction).ravel()
        for h in [1e-4,1e-5]:
            direct=(linear.residual(h*direction)-linear.residual(-h*direction))/(2*h)
            out['derivative_checks'].append(dict(coordinate=name,step_scaled=h,step_physical=h*coordinate_scales[name],unit=coordinate_units[name],predicted_norm2=float(predicted@predicted),direct_norm2=float(direct@direct),
                relative_vector_error=float(np.linalg.norm(predicted-direct)/np.linalg.norm(direct)),
                per_term={k:dict(predicted_norm=float(np.linalg.norm(predicted[s])),direct_norm=float(np.linalg.norm(direct[s])),error_norm=float(np.linalg.norm((predicted-direct)[s]))) for k,s in linear.rows.items()}))
    out['derivative_gate_pass']=bool(max(r['relative_vector_error'] for r in out['derivative_checks'])<.01)
    print('derivative gate',out['derivative_gate_pass'],out['derivative_checks'],flush=True);save()
    for chain in out['probe_chains']:
        line=Linearization(m,states[chain]);r0=line.residual(zero);lp0=float(m.logp(states[chain]));norm=line.normalization()
        assert abs(-.5*r0@r0+norm-lp0)<1e-8
        terms0={k:float(-.5*r0[s]@r0[s]) for k,s in line.rows.items()}
        for name in a.coordinates.split(','):
            for mode,idx in [('block_only',0),('globals_landmarks',2)]:
                for h in [.005,.05]:
                    changes=[];terms=[];failures=[]
                    for sign in [-1,1]:
                        delta=sign*h*directions[name][idx];state=line.expand(delta);exact=float(m.logp(state))
                        if not np.isfinite(exact):
                            changes.append(None);terms.append(None);failures.append(support_details(m,state));continue
                        r=line.residual(delta);assert abs(-.5*r@r+norm-exact)<1e-8
                        change=exact-lp0;term={k:float(-.5*r[s]@r[s])-terms0[k] for k,s in line.rows.items()}
                        assert abs(sum(term.values())-change)<1e-8
                        changes.append(change);terms.append(term);failures.append([])
                    ok=all(c is not None for c in changes)
                    out['probes'].append(dict(chain=chain,coordinate=name,mode=mode,amplitude_scaled=h,amplitude_physical=h*coordinate_scales[name],unit=coordinate_units[name],delta_logp=changes,term_delta_logp=terms,
                        support_failures=failures,both_supported=ok,even_penalty=-sum(changes) if ok else None))
            print('tested',chain,name,flush=True);save()
    out['seconds']=time.monotonic()-start;out['status']='complete';save('diagnostic.json')
    (a.output/'README.md').write_text('# Collective-direction transfer diagnostic\n\n'
        'Generated by marjum_mcmc_b21_fine_geometry.py on fix/mcmc-camera-logf.\n'
        'diagnostic.json records target config, input/source hashes, physical units,\n'
        'endpoint support, direct-derivative agreement and signed exact-density probes.\n'
        'Directions train on chain 0; all declared probe chains are retained.\n'
        'This tests local proposals, not posterior convergence or a physical cause.\n\n'
        '## Recent changes\n\n- 2026-10-05: Checked height/orientation directions at corrected-run endpoints.\n')
    print('completed_seconds',out['seconds'],flush=True)

if __name__=='__main__':main()
