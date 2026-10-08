"""Attribute frozen-direction penalties under integer and fractional DEM targets.

This is a deterministic diagnostic, not a sampler. Old-chain endpoints and
old-derived directions are probes only; no posterior uncertainty is estimated.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time
import numpy as np
import marjum_mcmc_b21 as b
from marjum_mcmc_b21_coupling import Linearization, sha


def support_details(model, state):
    """Identify violated support clauses without changing the target or state."""
    from marjum_camera import project
    cam,ant,tx,bias,extra,tx_extra,points=model.unpack(state)
    t=model.terrain; failures=[]
    def add(kind, **details):failures.append(dict(kind=kind,**details))
    if extra<0 or tx_extra<0:add('negative_scatter')
    pp=model.point_logp(z=state)
    if not np.isfinite(pp).all():add('landmark_support',indices=np.flatnonzero(~np.isfinite(pp)).tolist())
    for i,c in enumerate(cam):
        ground=float(t.height(*c[:2]));margin=float(c[2]-ground-.1)
        if not model._camera_support(c):add('camera_support',camera=model.keys[i],height_m=float(c[2]),ground_m=ground,clearance_m=float(c[2]-ground),required_clearance_m=.1,margin_m=margin)
        if model.has_heading[i] and abs(c[4]-model.heading[i])>=np.pi:add('heading_support',camera=model.keys[i])
    for label,x,mask,floor in [('antenna',ant,model.has_ant_label,0.),('transmitter',tx,model.has_tx_label,-5.)]:
        ground=float(t.height(*x[:2]))
        if not(t.e[0]+2<x[0]<t.e[-1]-2 and t.n[0]+2<x[1]<t.n[-1]-2):add(label+'_bounds')
        if not(ground+floor<x[2]<4000.):add(label+'_height',height_m=float(x[2]),ground_m=ground,lower_margin_m=float(x[2]-ground-floor))
        for i in np.flatnonzero(mask):
            _,depth=project(cam[i],model.shapes[i],x,model.distortion[i])
            if depth[0]<=.1:add(label+'_depth',camera=model.keys[i],depth_m=float(depth[0]))
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--continue-from',type=Path,help='Import completed coordinates from a checksum-verified prior code version')
    p.add_argument('--checkpoint-sha256',help='Required checksum for --continue-from')
    p.add_argument('--resume',action='store_true',help='Resume after the last completed coordinate checkpoint')
    a=p.parse_args();code=Path(__file__).resolve().parent
    manifest=json.loads((a.root/'manifest.json').read_text())
    old=Path(manifest['parent_root'])
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    for rel,want in pins.items():assert sha(old/rel)==want,rel
    frozen=json.loads((a.root/'inputs/input_manifest.json').read_text())
    for item in frozen['files'].values():assert sha(a.root/item['frozen'])==item['sha256']
    for name,item in frozen['feature_cache']['files'].items():
        assert sha(a.root/'inputs/cv_features'/name)==item['sha256']
    config=json.loads((old/'pilot_logf_20261002/manifest.json').read_text())['config']
    with np.load(old/'coupling_logf_20261002/geometry.npz') as z:
        directions={n:z[n].copy() for n in ['cam2223_n','cam2224_n']}
    report=dict(provenance=dict(commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        source_hashes={n:sha(code/n) for n in ['marjum_mcmc_b21_dem_diagnostic.py','marjum_mcmc_b21_coupling.py','marjum_mcmc_b21.py','marjum_bundle.py']},
        manifest_sha256=sha(a.root/'manifest.json'),input_manifest_sha256=sha(a.root/'inputs/input_manifest.json'),
        parent_pins=pins,config=config,smoke=a.smoke),baselines=[],probes=[],seconds=None)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if a.continue_from:
        assert not a.resume and a.checkpoint_sha256==sha(a.continue_from)
        previous=json.loads(a.continue_from.read_text())
        for key in ['manifest_sha256','input_manifest_sha256','config','smoke']:
            assert previous['provenance'][key]==report['provenance'][key],key
        for rel,want in previous['provenance']['parent_pins'].items():assert sha(old/rel)==want
        for name,want in previous['provenance']['source_hashes'].items():
            import hashlib
            blob=subprocess.check_output(['git','-C',str(code),'show',previous['provenance']['commit']+':'+name])
            assert hashlib.sha256(blob).hexdigest()==want
        report['provenance']['imported_checkpoint']=dict(path=str(a.continue_from),sha256=a.checkpoint_sha256,provenance=previous['provenance'])
        report['baselines']=previous['baselines'];report['probes']=previous['probes']
    if a.resume:
        previous=json.loads((a.output/'checkpoint.json').read_text())
        for key in ['source_hashes','manifest_sha256','input_manifest_sha256','parent_pins','config','smoke']:
            assert previous['provenance'][key]==report['provenance'][key],key
        report=previous
    def checkpoint():
        temp=a.output/'checkpoint.tmp'
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temp.replace(a.output/'checkpoint.json')
    start=time.monotonic()
    focus=['cam2223_n'] if a.smoke else list(directions)
    chains=[0] if a.smoke else [0,4]
    amplitudes=[.05] if a.smoke else [.005,.01,.05]
    for target,inp in [('int32',old/'inputs'),('float32',a.root/'inputs')]:
        model=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),
            meta_file=inp/'meta.json',exif_file=inp/'marjum_2026_07_exif_joint.npz',
            feature_dir=inp/'cv_features',config=b.Config(**config))
        for chain in chains:
            with np.load(old/f'pilot_logf_20261002/chain_{chain}.npz') as z:state=z['final'].copy()
            linear=Linearization(model,state);zero=np.zeros(linear.nvar)
            r0=linear.residual(zero); lp0=float(model.logp(state)); norm=linear.normalization()
            if not np.isfinite(lp0):
                failures=support_details(model,state)
                assert failures, 'nonfinite density without identified support failure'
                report.setdefault('unsupported_endpoints',[]).append(dict(target=target,chain=chain,failures=failures,untested_coordinates=focus,untested_amplitudes_m=amplitudes))
                print('unsupported endpoint',target,chain,json.dumps(failures),flush=True)
                checkpoint()
                continue
            error=float(-.5*r0@r0+norm-lp0);assert abs(error)<1e-8
            terms0={k:float(-.5*r0[s]@r0[s]) for k,s in linear.rows.items()}
            report['baselines']=[r for r in report['baselines'] if (r['target'],r['chain'])!=(target,chain)]
            report['baselines'].append(dict(target=target,chain=chain,logp=lp0,terms=terms0,normalization=norm,residual_error=error))
            for name in focus:
                saved=[r for r in report['probes'] if (r['target'],r['chain'],r['coordinate'])==(target,chain,name)]
                if len(saved)==2*len(amplitudes):
                    print(target,chain,name,'resumed completed coordinate',flush=True)
                    continue
                assert not saved, 'partial coordinate checkpoint'
                for mode,idx in [('block_only',0),('globals_landmarks',2)]:
                    direction=directions[name][idx]
                    for h in amplitudes:
                        residuals=[];changes=[];termchanges=[];errors=[]
                        for sign in [-1,1]:
                            delta=sign*h*direction;r=linear.residual(delta)
                            exact=float(model.logp(linear.expand(delta)))
                            assert np.isfinite(exact),('probe unsupported',target,chain,name,mode,h,sign)
                            err=float(-.5*r@r+norm-exact);assert abs(err)<1e-8
                            residuals.append(r);changes.append(exact-lp0);errors.append(err)
                            termchanges.append({k:float(-.5*r[s]@r[s])-terms0[k] for k,s in linear.rows.items()})
                        derivative=(residuals[1]-residuals[0])/(2*h)
                        penalties={k:-termchanges[0][k]-termchanges[1][k] for k in linear.rows}
                        even=-sum(changes);assert abs(sum(penalties.values())-even)<1e-8
                        report['probes'].append(dict(target=target,chain=chain,coordinate=name,mode=mode,
                            amplitude_m=h,delta_logp=changes,term_delta_logp=termchanges,even_penalty=even,
                            term_even_penalty=penalties,directional_jacobian_norm2=float(derivative@derivative),
                            residual_errors=errors))
                print(target,chain,name,'complete',flush=True)
                checkpoint()
    report['seconds']=time.monotonic()-start
    report['status']='complete_with_unsupported_endpoints' if report.get('unsupported_endpoints') else 'complete'
    (a.output/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('completed_seconds',report['seconds'],flush=True)

if __name__=='__main__':main()
