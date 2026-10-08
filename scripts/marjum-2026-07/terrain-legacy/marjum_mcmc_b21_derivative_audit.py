"""Audit residual finite differences along saved frozen directions.

By default only image-tie residuals are differentiated. No states, targets or proposals
are changed, and no sampling is performed. A direct directional difference
checks the sparse coordinate Jacobian, including its sparsity coloring.
With --terrain-geometry, audit the two declared terrain failures using saved
local Jacobians, per-landmark cell crossings and a frozen-cell control.
Add --cell-aware to validate all saved banks at both endpoints using two
geometry-selected within-cell steps and an independent step-stability gate.
"""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import marjum_mcmc_b21 as b
from marjum_mcmc_b21_coupling import Linearization,sha
from marjum_mcmc import student_residual
from marjum_camera import project


def terrain_audit(args, code):
    """Audit saved terrain derivatives without changing the likelihood or DEM."""
    import hashlib,signal
    from scipy.sparse import load_npz
    root=args.root;folder=root/args.terrain_geometry;old=root.parent/'v0001';inp=root/'inputs'
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    source=json.loads((folder/'diagnostic.json').read_text())
    for name in ['diagnostic.json','supported_endpoints.npz','geometry_local_5.npz','geometry_local_6.npz','jacobian_5.npz','jacobian_6.npz']:
        assert sha(folder/name)==pins['../v0002/'+args.terrain_geometry+'/'+name]
    assert source['status']=='complete'
    for name,want in source['provenance']['source_hashes'].items():
        blob=subprocess.check_output(['git','-C',str(code),'show',source['provenance']['commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==want
    assert not args.output.exists() and not args.output.with_suffix('.npz').exists()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();arrays={}
    result=dict(status='running',provenance=dict(commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        source_hashes={n:sha(code/n) for n in ['marjum_mcmc_b21_derivative_audit.py','marjum_mcmc_b21_coupling.py','marjum_mcmc_b21.py','marjum_mcmc.py','marjum_camera.py','marjum_bundle.py']},
        geometry_source=str(folder),geometry_diagnostic_sha256=sha(folder/'diagnostic.json'),input_pins=pins,config=source['provenance']['config'],max_seconds=args.max_seconds),
        steps=[1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7],cases=[])
    def save():
        if arrays:
            temporary=args.output.with_suffix('.partial.npz');np.savez_compressed(temporary,**arrays);temporary.replace(args.output.with_suffix('.npz'))
            result['arrays_sha256']=sha(args.output.with_suffix('.npz'))
        result['seconds']=time.monotonic()-started
        temporary=args.output.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    def interrupted(signum,frame):
        result['status']='time_limit' if signum==signal.SIGALRM else 'interrupted';save();raise SystemExit(124 if signum==signal.SIGALRM else 128+signum)
    signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted)
    signal.setitimer(signal.ITIMER_REAL,args.max_seconds);save()
    try:
        m=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),meta_file=inp/'meta.json',exif_file=inp/'marjum_2026_07_exif_joint.npz',feature_dir=inp/'cv_features',config=b.Config(**source['provenance']['config']))
        model_hashes={str(f):sha(f) for f in m.input_files}
        assert model_hashes==source['provenance']['model_input_sha256'];result['provenance']['model_input_sha256']=model_hashes
        terrain=m.terrain;df=m.config.student_df;sigma=m.config.terrain_sigma_m
        result['dem_dtype']=str(terrain.data.dtype)
        def rc(points):return np.c_[(points[:,1]-terrain.n[0])/terrain.res,(points[:,0]-terrain.e[0])/terrain.res]
        def transformed(points):return student_residual((points[:,2]-terrain.height(points[:,0],points[:,1]))/sigma,df)
        def norm(v):return float(np.linalg.norm(v))
        for chain,name in [(5,'cam2159_th'),(6,'cam2159_ph')]:
            key=f'chain_{chain}'
            with np.load(folder/'supported_endpoints.npz') as saved:state=saved[key].copy()
            line=Linearization(m,state);zero=np.zeros(line.nvar);base=line.residual(zero)
            assert abs(-.5*base@base+line.normalization()-m.logp(state))<1e-8
            points=m.unpack(state)[-1];grid=rc(points);cells=np.floor(grid).astype(int);fr=grid-cells
            rows,cols=cells.T
            assert (rows>=0).all() and (cols>=0).all() and (rows+1<terrain.data.shape[0]).all() and (cols+1<terrain.data.shape[1]).all()
            # Promote only the four stored samples for arithmetic; the model and
            # float32 DEM are untouched. This is the exact within-cell polynomial.
            z00=terrain.data[rows,cols].astype(float);z01=terrain.data[rows,cols+1].astype(float)
            z10=terrain.data[rows+1,cols].astype(float);z11=terrain.data[rows+1,cols+1].astype(float)
            def polynomial(p):
                xy=rc(p)-cells;y,x=xy.T
                return (1-y)*((1-x)*z00+x*z01)+y*((1-x)*z10+x*z11)
            def fixed_cell(p):return student_residual((p[:,2]-polynomial(p))/sigma,df)
            height=terrain.height(points[:,0],points[:,1]);assert height.dtype==np.float64
            height_error=float(np.max(abs(height-polynomial(points))));assert height_error<1e-9
            with np.load(folder/f'geometry_local_{chain}.npz') as saved:direction=saved[name][2].copy()
            jac=load_npz(folder/f'jacobian_{chain}.npz');pred=np.asarray(jac@direction).ravel();sl=line.rows['terrain'];pred_t=pred[sl]
            dp=direction[line.ng:].reshape(-1,3);assert dp.shape==points.shape
            y,x=fr.T;de=((1-y)*(z01-z00)+y*(z11-z10))/terrain.res;dn=((1-x)*(z10-z00)+x*(z11-z01))/terrain.res
            raw=(points[:,2]-height)/sigma
            slope=np.full_like(raw,np.sqrt((df+1)/df));nonzero=abs(raw)>1e-10
            slope[nonzero]=(df+1)*abs(raw[nonzero])/((df+raw[nonzero]**2)*np.sqrt((df+1)*np.log1p(raw[nonzero]**2/df)))
            analytic=slope*(dp[:,2]-de*dp[:,0]-dn*dp[:,1])/sigma
            eps=.05*source['provenance']['step_multiplier'];rebuilt=np.zeros(m.npoint);stencil_cross=np.zeros(m.npoint,bool)
            for axis in range(3):
                plus=points.copy();minus=points.copy();plus[:,axis]+=eps;minus[:,axis]-=eps
                rebuilt+=(transformed(plus)-transformed(minus))/(2*eps)*dp[:,axis]
                stencil_cross|=np.any(np.floor(rc(plus)).astype(int)!=cells,axis=1)|np.any(np.floor(rc(minus)).astype(int)!=cells,axis=1)
            entry=dict(chain=chain,coordinate=name,original_coordinate_step_m=eps,npoints=m.npoint,base_height_polynomial_max_error_m=height_error,
                coordinate_stencil_crossing_count=int(sum(stencil_cross)),origin_grid_edge_count=int(sum(np.any(np.minimum(fr,1-fr)<1e-10,axis=1))),
                archived_vs_rebuilt_relative_error=norm(pred_t-rebuilt)/max(norm(pred_t),1e-30),archived_vs_analytic_terrain_error_norm=norm(pred_t-analytic),steps=[])
            assert entry['archived_vs_rebuilt_relative_error']<1e-6
            result['cases'].append(entry)
            arrays.update({key+'_points':points,key+'_grid':grid,key+'_direction_points':dp,key+'_predicted_terrain':pred_t,key+'_analytic_terrain':analytic,key+'_rebuilt_terrain':rebuilt,key+'_coordinate_stencil_cross':stencil_cross})
            for index,h in enumerate(result['steps']):
                plus_state=line.expand(h*direction);minus_state=line.expand(-h*direction)
                plus=m.unpack(plus_state)[-1];minus=m.unpack(minus_state)[-1]
                direct=(line.residual(h*direction)-line.residual(-h*direction))/(2*h);dt=direct[sl]
                fixed=(fixed_cell(plus)-fixed_cell(minus))/(2*h)
                cross=np.any(np.floor(rc(plus)).astype(int)!=cells,axis=1)|np.any(np.floor(rc(minus)).astype(int)!=cells,axis=1)
                relevant=cross|stencil_cross;error=pred_t-dt;power=error**2;den=max(float(power.sum()),1e-300)
                relative=norm(pred-direct)/max(norm(direct),1e-30)
                control=pred-direct;control[sl]=analytic-fixed
                record=dict(step_scaled=h,step_physical_rad=.01*h,relative_full_error=relative,pass_derivative=relative<.01,
                    terrain_error_norm=norm(error),direct_terrain_norm=norm(dt),direct_full_norm=norm(direct),directional_crossing_count=int(sum(cross)),
                    any_stencil_crossing_count=int(sum(relevant)),crossing_error_fraction=float(power[relevant].sum()/den),
                    noncrossing_terrain_error_norm=norm(error[~relevant]),fixed_cell_vs_analytic_error_norm=norm(fixed-analytic),
                    fixed_cell_control_relative_full_error=norm(control)/max(norm(direct),1e-30),
                    actual_vs_fixed_cell_error_norm=norm(dt-fixed),max_abs_terrain_row_error=float(abs(error).max()),worst_landmarks=[])
                for j in np.argsort(-power)[:10]:
                    record['worst_landmarks'].append(dict(point=int(j),fraction=float(power[j]/den),predicted=float(pred_t[j]),direct=float(dt[j]),analytic=float(analytic[j]),fixed_cell=float(fixed[j]),
                        directional_cell_crossing=bool(cross[j]),coordinate_stencil_crossing=bool(stencil_cross[j]),grid_position=grid[j].tolist(),
                        minus_grid=rc(minus[[j]])[0].tolist(),plus_grid=rc(plus[[j]])[0].tolist(),direction_points_m=dp[j].tolist()))
                entry['steps'].append(record)
                arrays[key+f'_direct_{index}']=dt;arrays[key+f'_fixed_{index}']=fixed;arrays[key+f'_cross_{index}']=cross
                if h in [1e-4,1e-5]:
                    old_check=next(r for r in source['derivative_checks'] if (r['chain'],r['bank'],r['coordinate'],r['step_scaled'])==(chain,f'local_{chain}',name,h))
                    np.testing.assert_allclose(relative,old_check['relative_vector_error'],rtol=1e-8,atol=1e-10)
                print('checked',chain,name,h,'error',relative,'crossing rows',sum(relevant),'error fraction',record['crossing_error_fraction'],flush=True);save()
        result['status']='complete';save()
        (args.output.parent/'README.md').write_text('# Terrain derivative audit\n\n'
            'Generated by marjum_mcmc_b21_derivative_audit.py --terrain-geometry.\n'
            'diagnostic.json records provenance, step checks and worst landmarks;\n'
            'diagnostic.npz retains every terrain row for both saved endpoints.\n'
            'A frozen-cell polynomial is a diagnostic control only, not a new target.\n'
            'The likelihood and stored DEM remain unchanged. Partial products are\n'
            'saved after each step under the declared computation time cap.\n\n'
            '## Recent changes\n\n- 2026-10-05: Attribute derivative errors to individual terrain rows and test cell crossings.\n')
        print('completed_seconds',result['seconds'],flush=True)
    except Exception as error:
        result['status']='failed';result['error']=repr(error);save();raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


def cell_check_plan(grid, dgrid):
    """Choose two symmetric steps from geometry alone, without fitting errors."""
    fraction=grid-np.floor(grid);distance=np.minimum(fraction,1-fraction)
    edge_count=int(np.sum(np.any(distance<=1e-10,axis=1)))
    moving=abs(dgrid)>0
    limits=np.full_like(grid,np.inf,dtype=float)
    np.divide(distance,abs(dgrid),out=limits,where=moving)
    index=np.unravel_index(np.argmin(limits),limits.shape);limit=float(limits[index])
    h=min(1e-5,.2*limit)
    reason='origin_near_cell_edge' if edge_count else 'step_below_precision_floor' if h/2<1e-8 else None
    return dict(steps_scaled=[h,h/2],symmetric_cell_limit_scaled=limit if np.isfinite(limit) else None,
        limiting_landmark=int(index[0]) if np.isfinite(limit) else None,
        limiting_grid_axis=int(index[1]) if np.isfinite(limit) else None,
        origin_edge_count=edge_count,geometry_eligible=reason is None,rejection_reason=reason)


def cell_aware_audit(args, code):
    """Validate every archived bank using a preregistered within-cell step rule."""
    import hashlib,signal
    from scipy.sparse import load_npz
    root=args.root;folder=root/args.terrain_geometry;inp=root/'inputs'
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    source=json.loads((folder/'diagnostic.json').read_text())
    files=['diagnostic.json','supported_endpoints.npz','geometry_archived_0.npz','geometry_local_5.npz','geometry_local_6.npz','jacobian_5.npz','jacobian_6.npz']
    for name in files:assert sha(folder/name)==pins['../v0002/'+args.terrain_geometry+'/'+name]
    assert source['status']=='complete' and source['training_chains']==[5,6] and source['probe_chains']==[5,6]
    for name,want in source['provenance']['source_hashes'].items():
        blob=subprocess.check_output(['git','-C',str(code),'show',source['provenance']['commit']+':'+name]);assert hashlib.sha256(blob).hexdigest()==want
    assert not args.output.parent.exists(),'use a fresh diagnostic directory'
    args.output.parent.mkdir(parents=True)
    started=time.monotonic()
    result=dict(status='running',stage='load_model',provenance=dict(commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        source_hashes={n:sha(code/n) for n in ['marjum_mcmc_b21_derivative_audit.py','marjum_mcmc_b21_coupling.py','marjum_mcmc_b21.py','marjum_mcmc.py','marjum_camera.py','marjum_bundle.py']},
        geometry_source=str(folder),geometry_diagnostic_sha256=sha(folder/'diagnostic.json'),input_pins=pins,config=source['provenance']['config'],max_seconds=args.max_seconds),
        rule=dict(max_step_scaled=1e-5,boundary_fraction=.2,second_step_factor=.5,min_step_scaled=1e-8,edge_tolerance_pixels=1e-10,derivative_threshold=.01,stability_threshold=.01),
        step_plan=[],cases=[],original_derivative_checks=source['derivative_checks'])
    def save():
        result['seconds']=time.monotonic()-started
        temporary=args.output.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    def interrupted(signum,frame):
        result['status']='time_limit' if signum==signal.SIGALRM else 'interrupted';save();raise SystemExit(124 if signum==signal.SIGALRM else 128+signum)
    signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted);signal.setitimer(signal.ITIMER_REAL,args.max_seconds);save()
    try:
        m=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),meta_file=inp/'meta.json',exif_file=inp/'marjum_2026_07_exif_joint.npz',feature_dir=inp/'cv_features',config=b.Config(**source['provenance']['config']))
        model_hashes={str(f):sha(f) for f in m.input_files};assert model_hashes==source['provenance']['model_input_sha256']
        result['provenance']['model_input_sha256']=model_hashes
        terrain=m.terrain;result['dem_dtype']=str(terrain.data.dtype);result['terrain_resolution_m']=float(terrain.res)
        def rc(points):return np.c_[(points[:,1]-terrain.n[0])/terrain.res,(points[:,0]-terrain.e[0])/terrain.res]
        banks={}
        for bank in ['archived_0','local_5','local_6']:
            with np.load(folder/f'geometry_{bank}.npz') as saved:banks[bank]={name:saved[name][2].copy() for name in source['coordinates']}
        lines={};jacs={};grids={}
        with np.load(folder/'supported_endpoints.npz') as saved:
            for c in [5,6]:lines[c]=Linearization(m,saved[f'chain_{c}'].copy())
        # Freeze ALL 24 step plans before calculating any derivative errors.
        for c,line in lines.items():
            points=m.unpack(line.origin)[-1];grids[c]=rc(points)
            for bank,directions in banks.items():
                for name,direction in directions.items():
                    dp=direction[line.ng:].reshape(-1,3);dgrid=np.c_[dp[:,1]/terrain.res,dp[:,0]/terrain.res]
                    plan=cell_check_plan(grids[c],dgrid)
                    result['step_plan'].append(dict(chain=c,bank=bank,coordinate=name,scale=source['coordinate_scales'][name],unit=source['coordinate_units'][name],**plan))
        plan_file=args.output.parent/'step_plan.json';plan_file.write_text(json.dumps(dict(rule=result['rule'],plan=result['step_plan']),indent=2,allow_nan=False)+'\n')
        result['step_plan_sha256']=sha(plan_file);result['stage']='derivative_checks';save()
        for c,line in lines.items():
            zero=np.zeros(line.nvar);r0=line.residual(zero)
            assert abs(-.5*r0@r0+line.normalization()-m.logp(line.origin))<1e-8
            jacs[c]=load_npz(folder/f'jacobian_{c}.npz')
        for case_id,plan in enumerate(result['step_plan']):
            c=plan['chain'];bank=plan['bank'];name=plan['coordinate'];line=lines[c];direction=banks[bank][name]
            entry=dict(**plan,checks=[],pass_case=False)
            result['cases'].append(entry)
            if not plan['geometry_eligible']:save();continue
            pred=np.asarray(jacs[c]@direction).ravel();directs=[];arrays=dict(predicted=pred,grid_base=grids[c])
            for j,h in enumerate(plan['steps_scaled']):
                gp=rc(m.unpack(line.expand(h*direction))[-1]);gm=rc(m.unpack(line.expand(-h*direction))[-1])
                cross=np.any(np.floor(gp)!=np.floor(grids[c]),axis=1)|np.any(np.floor(gm)!=np.floor(grids[c]),axis=1)
                direct=(line.residual(h*direction)-line.residual(-h*direction))/(2*h);directs.append(direct)
                denominator=max(float(np.linalg.norm(direct)),1e-30);error=float(np.linalg.norm(pred-direct)/denominator)
                entry['checks'].append(dict(step_scaled=h,step_physical=h*plan['scale'],crossing_count=int(sum(cross)),relative_error=error,
                    pass_check=bool(not cross.any() and np.isfinite(error) and error<.01),direct_norm=denominator,
                    max_abs_row_error=float(abs(pred-direct).max()),per_term={k:dict(error_norm=float(np.linalg.norm((pred-direct)[s])),direct_norm=float(np.linalg.norm(direct[s]))) for k,s in line.rows.items()}))
                arrays.update({f'direct_{j}':direct,f'grid_plus_{j}':gp,f'grid_minus_{j}':gm})
            stability=float(np.linalg.norm(directs[0]-directs[1])/max(np.linalg.norm(directs[0]),np.linalg.norm(directs[1]),1e-30))
            entry['two_step_relative_difference']=stability
            entry['pass_case']=bool(all(v['pass_check'] for v in entry['checks']) and np.isfinite(stability) and stability<.01)
            if not entry['pass_case']:entry['rejection_reason']='derivative_or_cell_or_two_step_failure'
            path=args.output.parent/f'case_{case_id:02d}.npz';temporary=path.with_suffix('.partial.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)
            entry['arrays_file']=path.name;entry['arrays_sha256']=sha(path)
            print('checked',c,bank,name,'steps',plan['steps_scaled'],'errors',[v['relative_error'] for v in entry['checks']],'stable',stability,'pass',entry['pass_case'],flush=True);save()
        result['status']='complete';result['stage']='finished';result['all_pass']=all(r['pass_case'] for r in result['cases']);save()
        (args.output.parent/'README.md').write_text('# Cell-aware derivative validation\n\n'
            'Generated by marjum_mcmc_b21_derivative_audit.py --cell-aware.\n'
            'step_plan.json fixes all steps from geometry before derivative evaluation.\n'
            'diagnostic.json retains all case results, failures, original checks and\n'
            'source/input hashes. Each case NPZ contains the full derivative vectors\n'
            'and actual perturbed DEM coordinates for independent reproduction.\n'
            'Each case requires two checks below 1%, no DEM-cell crossings and\n'
            'two-step agreement below 1%. No proposal or target is changed.\n\n'
            '## Recent changes\n\n- 2026-10-06: Revalidate every saved direction at both endpoints with predetermined cell-aware steps.\n')
        print('completed_seconds',result['seconds'],'all_pass',result['all_pass'],flush=True)
    except Exception as error:
        result['status']='failed';result['error']=repr(error);save();raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--terrain-geometry',help='saved local geometry directory under --root; audit terrain at endpoints 5 and 6')
    p.add_argument('--cell-aware',action='store_true',help='validate every saved bank at both endpoints using geometry-selected steps')
    p.add_argument('--max-seconds',type=float,default=300)
    a=p.parse_args();code=Path(__file__).resolve().parent
    assert a.max_seconds>0
    if a.cell_aware:
        assert a.terrain_geometry,'--cell-aware requires --terrain-geometry'
        cell_aware_audit(a,code);return
    if a.terrain_geometry:
        terrain_audit(a,code);return
    old=a.root.parent/'v0001';inp=a.root/'inputs'
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    for rel,want in pins.items():assert sha(old/rel)==want
    frozen=json.loads((inp/'input_manifest.json').read_text())
    for item in frozen['files'].values():assert sha(a.root/item['frozen'])==item['sha256']
    for name,item in frozen['feature_cache']['files'].items():assert sha(inp/'cv_features'/name)==item['sha256']
    config=json.loads((old/'pilot_logf_20261002/manifest.json').read_text())['config']
    m=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),meta_file=inp/'meta.json',exif_file=inp/'marjum_2026_07_exif_joint.npz',feature_dir=inp/'cv_features',config=b.Config(**config))
    with np.load(old/'coupling_logf_20261002/geometry.npz') as z:directions={n:z[n][2].copy() for n in ['cam2223_n','cam2224_n']}
    result=dict(provenance=dict(commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),source_hashes={n:sha(code/n) for n in ['marjum_mcmc_b21_derivative_audit.py','marjum_mcmc_b21_coupling.py','marjum_mcmc_b21.py','marjum_mcmc.py','marjum_camera.py','marjum_bundle.py']},input_pins=pins,input_manifest_sha256=sha(inp/'input_manifest.json')),states=[])
    start=time.monotonic()
    for chain in [0,4]:
        with np.load(old/f'pilot_logf_20261002/chain_{chain}.npz') as z:state=z['final'].copy()
        linear=Linearization(m,state)
        def ties(delta):
            cam,ant,tx,bias,extra,tx_extra,points=m.unpack(linear.expand(delta));pred=np.empty_like(m.xy)
            for i,idx in enumerate(m.by_camera):pred[idx]=project(cam[i],m.shapes[i],points[m.op[idx]],m.distortion[i])[0]
            return student_residual((pred-m.xy)/m.config.tie_sigma_px,m.config.student_df,2).ravel()
        zero=np.zeros(linear.nvar);base=ties(zero);entry=dict(chain=chain,direct=[],coordinate=[],worst_observations=[])
        direct={}
        for name,direction in directions.items():
            for h in [1e-3,1e-4,1e-5,1e-6]:
                plus=ties(h*direction);minus=ties(-h*direction);der=(plus-minus)/(2*h)
                entry['direct'].append(dict(coordinate=name,step_m=h,norm2=float(der@der),even_penalty=float(.5*(plus@plus+minus@minus)-base@base)))
                if h==1e-6:direct[name]=der
        for factor in [1.,.1,.01,.001]:
            jac=approx_derivative(ties,zero,method='3-point',abs_step=linear.step*factor,sparsity=linear.sparsity[linear.rows['tie']])
            for name,direction in directions.items():
                der=np.asarray(jac@direction).ravel();reference=direct[name]
                entry['coordinate'].append(dict(coordinate=name,step_multiplier=factor,norm2=float(der@der),relative_vector_error=float(np.linalg.norm(der-reference)/np.linalg.norm(reference))))
        cam,_,_,_,_,_,points=m.unpack(state)
        for name,der in direct.items():
            power=np.sum(der.reshape(-1,2)**2,axis=1);indices=np.argsort(power)[-5:][::-1]
            for k in indices:
                ci,pi=int(m.oc[k]),int(m.op[k]);_,depth=project(cam[ci],m.shapes[ci],points[pi],m.distortion[ci])
                entry['worst_observations'].append(dict(coordinate=name,observation=int(k),camera=m.keys[ci],point=pi,power=float(power[k]),fraction=float(power[k]/power.sum()),depth_m=float(depth[0])))
        result['states'].append(entry);print(json.dumps(entry),flush=True)
    result['seconds']=time.monotonic()-start
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
