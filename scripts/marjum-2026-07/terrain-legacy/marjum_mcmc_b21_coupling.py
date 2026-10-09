"""Bounded camera/landmark/target coupling diagnostic, never a sampler.

Use the frozen b21 inputs and two checksum-pinned saved pilot endpoints. A
sparse Gauss-Newton system supplies candidate coordinated directions. Exact
joint-density perturbations at the training endpoint and a second, held-out
chain endpoint test those frozen directions. No posterior widths are reported.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import sys
import time

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix

import marjum_mcmc_b21 as b
from marjum_camera import project
from marjum_mcmc import student_residual
from marjum_mcmc_b21_combine import label_coordinates

FOCUS = ['transmitter_n', 'cam2198_e', 'cam2222_n', 'cam2223_n', 'cam2224_n']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Linearization:
    """Scaled global + landmark increments with the two scatter terms fixed."""

    def __init__(self, model, origin):
        self.m, self.origin = model, origin.copy()
        self.ng = model.ng-2  # joint state: antenna/tx scatter are the last two globals
        assert model.joint and model.ng == 7*model.nc+10
        self.scale = np.r_[np.tile(b.CAMERA_SCALE, model.nc), np.ones(8)]
        self.step = np.r_[np.tile(b.CAMERA_STEP/b.CAMERA_SCALE, model.nc),
                          np.full(8, .05), np.full(3*model.npoint, .05)]
        self.nvar = self.ng+3*model.npoint
        self.rows = {}
        blocks = []
        offset = 0

        def add(name, dependencies):
            nonlocal offset
            dependencies = list(dependencies)
            self.rows[name] = slice(offset, offset+len(dependencies))
            blocks.extend(dependencies)
            offset += len(dependencies)

        cam = lambda i: list(range(7*i, 7*i+7))
        point = lambda j: list(range(self.ng+3*j, self.ng+3*j+3))
        ai, ti, bi = 7*model.nc, 7*model.nc+3, 7*model.nc+6
        add('tie', [cam(i)+point(j) for i,j in zip(model.oc,model.op) for _ in range(2)])
        add('terrain', [point(j) for j in range(model.npoint)])
        add('antenna', [cam(i)+list(range(ai,ai+3)) for i in np.flatnonzero(model.has_ant_label) for _ in range(2)])
        add('transmitter', [cam(i)+list(range(ti,ti+3)) for i in np.flatnonzero(model.has_tx_label) for _ in range(2)])
        add('horizon', [cam(i) for i in range(model.nc) for _ in model.horizon[i]])
        add('gps', [[7*i+j,bi+j] for i in np.flatnonzero(model.has_gps) for j in range(2)])
        add('altitude', [[7*i+2] for i in np.flatnonzero(model.has_alt)])
        add('heading', [[7*i+4] for i in np.flatnonzero(model.has_heading)])
        add('elevation', [[7*i+3] for i in range(model.nc)])
        add('roll', [[7*i+5] for i in range(model.nc)])
        add('logf', [[7*i+6] for i in range(model.nc)])
        add('bias', [[bi+j] for j in range(2)])
        sparsity = lil_matrix((offset,self.nvar), dtype=int)
        for i,cols in enumerate(blocks):
            sparsity[i,cols] = 1
        self.sparsity = sparsity.tocsr()
        self.evaluations = 0

    def expand(self, delta, origin=None):
        z = (self.origin if origin is None else origin).copy()
        z[:self.ng] += delta[:self.ng]*self.scale
        z[self.m.ng:] += delta[self.ng:]
        return z

    def residual(self, delta):
        """Algebraic residuals; exact support is checked by model.logp, not here.

        Fixed-scatter normalizers and scatter priors are an additive constant.
        Ignoring hard support only for differentiation permits derivatives near
        a boundary; every diagnostic proposal is later checked by exact logp.
        """
        self.evaluations += 1
        m = self.m; c = m.config
        cam,ant,tx,bias,extra,tx_extra,points = m.unpack(self.expand(delta))
        pred = np.empty_like(m.xy)
        for i,idx in enumerate(m.by_camera):
            pred[idx] = project(cam[i],m.shapes[i],points[m.op[idx]],m.distortion[i])[0]
        out = [student_residual((pred-m.xy)/c.tie_sigma_px,c.student_df,2).ravel(),
               student_residual((points[:,2]-m.terrain.height(points[:,0],points[:,1]))/c.terrain_sigma_m,c.student_df)]
        for labels,target,xy,scatter in [(m.has_ant_label,ant,m.axy,extra),
                                         (m.has_tx_label,tx,m.txy,tx_extra)]:
            out.append(np.concatenate([(project(cam[i],m.shapes[i],target,m.distortion[i])[0][0]-xy[i])/
                        np.hypot(c.antenna_label_sigma_px,scatter) for i in np.flatnonzero(labels)]))
        out += [np.concatenate([m.horizon_residuals(i,cam[i]) for i in range(m.nc)]),
                ((cam[m.has_gps,:2]+bias-m.gps[m.has_gps])/m.gps_sigma[m.has_gps,None]).ravel(),
                (cam[m.has_alt,2]-m.alt[m.has_alt])/c.altitude_sigma_m,
                (cam[m.has_heading,4]-m.heading[m.has_heading])/c.heading_sigma_rad,
                (cam[:,3]-np.pi/2)/c.elevation_sigma_rad,
                cam[:,5]/c.roll_sigma_rad,
                np.log(cam[:,6]/m.focal)/m.focal_sigma,
                bias/c.gps_common_sigma_m]
        r = np.concatenate(out)
        assert r.shape == (self.sparsity.shape[0],) and np.isfinite(r).all()
        return r

    def normalization(self, origin=None):
        m = self.m; c = m.config
        _,_,_,_,extra,tx_extra,_ = m.unpack(self.origin if origin is None else origin)
        return (-2*m.n_ant_label*np.log(np.hypot(c.antenna_label_sigma_px,extra)/c.antenna_label_sigma_px)
                -2*m.n_tx_label*np.log(np.hypot(c.antenna_label_sigma_px,tx_extra)/c.antenna_label_sigma_px)
                -.5*(extra/c.antenna_extra_prior_px)**2-.5*(tx_extra/c.antenna_extra_prior_px)**2)


def inverse(h):
    h = .5*(h+np.swapaxes(h,-1,-2))
    w,v = np.linalg.eigh(h)
    floor = np.maximum(1e-10, np.max(w,axis=-1,keepdims=True)*1e-12)
    regular = np.maximum(w,floor)
    result = (v/regular[...,None,:]) @ np.swapaxes(v,-1,-2)
    return result,dict(min_eigen=float(w.min()),max_eigen=float(w.max()),
                       clipped=int(np.sum(w<floor)),floor_max=float(np.max(floor)))


def geometry(linear, jac=None, stable=False, focus=None):
    focus = FOCUS if focus is None else list(focus)
    if not focus or len(set(focus)) != len(focus):
        raise ValueError('Geometry coordinates must be nonempty and unique')
    zero = np.zeros(linear.nvar)
    r = linear.residual(zero)
    base = linear.m.logp(linear.origin)
    residual_error = float(-.5*r@r+linear.normalization()-base)
    assert np.isfinite(base) and abs(residual_error)<1e-8
    if jac is None:
        jac = approx_derivative(linear.residual,zero,method='3-point',abs_step=linear.step,
                                sparsity=linear.sparsity)
    jg,jp = jac[:,:linear.ng],jac[:,linear.ng:]
    a = (jg.T@jg).toarray()
    cross = (jp.T@jg).toarray().reshape(linear.m.npoint,3,linear.ng)
    cpp = (jp.T@jp).tocsr()
    coo=cpp.tocoo(); assert np.all(coo.row//3==coo.col//3)
    c=np.empty((linear.m.npoint,3,3))
    for i in range(3):
        for j in range(3):
            c[:,i,j]=np.asarray(cpp[3*np.arange(linear.m.npoint)+i,3*np.arange(linear.m.npoint)+j]).ravel()
    cinv,cinfo = inverse(c)
    response = -cinv@cross
    schur = a+np.einsum('nig,nij->gj',cross,response)
    if stable:
        # Form the actual residual curvature of the landmark response directly.
        # Avoid subtracting nearly equal normal matrices when fine derivatives
        # reveal a very stiff image-tie constraint. If C was regularized, this
        # is the curvature for that approximate response, not an exact profile.
        reduced = jg.toarray()+jp@response.reshape(-1,linear.ng)
        schur = reduced.T@reduced
    afull,ainfo = inverse(a)
    joint,sinfo = inverse(schur)
    names=label_coordinates(linear.m.keys,joint=True)[:linear.ng]
    directions={}; widths={}; checks=[]
    for name in focus:
        j=names.index(name)
        start = (j//7)*7 if j<7*linear.m.nc else 7*linear.m.nc+3
        stop = start+7 if j<7*linear.m.nc else start+3
        local,linfo=inverse(a[start:stop,start:stop]); q=j-start
        local_direction=np.zeros(linear.ng);local_direction[start:stop]=local[:,q]/local[q,q]
        global_direction=afull[:,j]/afull[j,j]
        joint_direction=joint[:,j]/joint[j,j]
        directions[name]=np.array([np.r_[local_direction,np.zeros(3*linear.m.npoint)],
            np.r_[global_direction,np.zeros(3*linear.m.npoint)],
            np.r_[joint_direction,(response@joint_direction).ravel()]])
        widths[name]=dict(local=float(np.sqrt(local[q,q])*linear.scale[j]),
                          globals=float(np.sqrt(afull[j,j])*linear.scale[j]),
                          joint=float(np.sqrt(joint[j,j])*linear.scale[j]),
                          local_regularization=linfo)
        # Verify the predicted reduced cost against the original residual
        # Jacobian, not just a second algebraic use of the reduced matrix.
        predicted=joint_direction@schur@joint_direction
        actual=float(np.sum((jac@directions[name][2])**2))
        checks.append(dict(coordinate=name,schur_cost=float(predicted),jacobian_cost=actual,
                           relative_error=float(abs(actual-predicted)/max(1.,abs(predicted)))))
        assert checks[-1]['relative_error'] < 1e-6, checks[-1]
    # Known invariant: common north translation leaves all tie/target projections unchanged.
    translation=np.zeros(linear.nvar)
    translation[1:7*linear.m.nc:7]=1
    translation[7*linear.m.nc+1]=translation[7*linear.m.nc+4]=1
    translation[linear.ng+1::3]=1
    shifted=linear.residual(.01*translation)
    invariant_error=max(float(np.max(np.abs((shifted-r)[linear.rows[k]]))) for k in ['tie','antenna','transmitter'])
    assert invariant_error<1e-7
    # A numerical check at an unused perturbation of the residual representation.
    displaced=linear.expand(.001*directions[focus[0]][0])
    rr=linear.residual(.001*directions[focus[0]][0])
    exact_error=float(-.5*rr@rr+linear.normalization()-linear.m.logp(displaced))
    assert abs(exact_error)<1e-8
    return directions,widths,dict(residual_error=residual_error,displaced_residual_error=exact_error,
        translation_invariant_error=invariant_error,point_regularization=cinfo,
        global_regularization=ainfo,joint_regularization=sinfo,schur_checks=checks,
        residual_evaluations=linear.evaluations,nresidual=len(r),nvariable=linear.nvar,
        schur_method='residual_projection' if stable else 'normal_matrix_subtraction')


def evaluate(model, linear, directions, widths, endpoints, focus, smoke=False):
    records=[]
    for chain,z in endpoints.items():
        baseline=model.logp(z)
        for name in focus:
            w=widths[name]
            amplitudes=[('conditional_sigma',w['local'])] if smoke else [
                ('conditional_sigma',w['local']),('quarter_joint_sigma',.25*w['joint']),('joint_sigma',w['joint'])]
            for label,amplitude in amplitudes:
                for mode,direction in zip(['block_only','globals','globals_landmarks'],directions[name]):
                    changes=[]
                    for sign in [-1,1]:
                        candidate=linear.expand(sign*amplitude*direction,origin=z)
                        changes.append(float(model.logp(candidate)-baseline))
                    finite=bool(np.isfinite(changes).all())
                    records.append(dict(chain=chain,coordinate=name,mode=mode,amplitude_label=label,
                        amplitude_m=amplitude,delta_logp_minus=changes[0] if np.isfinite(changes[0]) else None,
                        delta_logp_plus=changes[1] if np.isfinite(changes[1]) else None,both_supported=finite,
                        even_penalty=float(-sum(changes)) if finite else None))
            print(f'evaluated chain {chain}, {name}',flush=True)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--geometry',type=Path,help='Reuse this diagnostic\'s saved, hashed geometry')
    args=parser.parse_args()
    root=args.run_root.resolve();out=args.output.resolve();code=Path(__file__).resolve().parent
    out.mkdir(parents=True,exist_ok=False)
    pins=json.loads((code/'marjum_mcmc_b21_review_inputs.json').read_text())
    for rel in ['inputs/input_manifest.json','pilot_logf_20261002/manifest.json']+[
            f'pilot_logf_20261002/chain_{i}.npz' for i in [0,4]]:
        assert sha(root/rel)==pins[rel],rel
    frozen=json.loads((root/'inputs/input_manifest.json').read_text())
    for item in frozen['files'].values(): assert sha(root/item['frozen'])==item['sha256']
    for name,item in frozen['feature_cache']['files'].items(): assert sha(root/'inputs/cv_features'/name)==item['sha256']
    import subprocess
    commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    manifest=json.loads((root/'pilot_logf_20261002/manifest.json').read_text())
    for path,want in manifest['input_sha256'].items():
        if '/inputs/' not in path:
            assert sha(code/Path(path).name)==want, ('sampler source changed',path)
    inp=root/'inputs'
    m=b.Posterior(state_file=inp/'fit_transmitter.npz',dem_file=str(inp/'marjum_dem.npz'),
        meta_file=inp/'meta.json',exif_file=inp/'marjum_2026_07_exif_joint.npz',feature_dir=inp/'cv_features',
        config=b.Config(**manifest['config']))
    endpoints={}
    for i in [0,4]:
        with np.load(root/f'pilot_logf_20261002/chain_{i}.npz') as saved:endpoints[i]=saved['final'].copy()
    linear=Linearization(m,endpoints[0])
    clock=time.monotonic()
    provenance=dict(commit=commit,code_sha256={p.name:sha(p) for p in [Path(__file__),Path(b.__file__)]},
                    chain_sha256={i:pins[f'pilot_logf_20261002/chain_{i}.npz'] for i in endpoints},
                    input_manifest_sha256=pins['inputs/input_manifest.json'],config=asdict(m.config),
                    origin_chain=0,validation_chain=4,focus=FOCUS,smoke=args.smoke,
                    global_scale=linear.scale.tolist(),difference_steps=linear.step[:linear.ng].tolist(),
                    point_difference_step_m=.05)
    if args.geometry:
        source=args.geometry.resolve();previous=json.loads((source.parent/'diagnostic.json').read_text())
        assert sha(source)==previous['geometry_sha256']
        for field in ['code_sha256','chain_sha256','input_manifest_sha256','config']:
            # JSON object keys are strings on reload.
            assert json.loads(json.dumps(provenance[field]))==previous['provenance'][field],field
        with np.load(source) as saved:directions={name:saved[name].copy() for name in FOCUS}
        widths,checks=previous['widths'],previous['checks']
        provenance['reused_geometry_sha256']=sha(source)
    else:
        directions,widths,checks=geometry(linear)
    print('geometry_seconds',time.monotonic()-clock,'widths',json.dumps(widths),flush=True)
    # Store enough to reproduce exact perturbations without recomputing geometry.
    np.savez_compressed(out/'geometry.npz',**directions)
    focus=FOCUS[:1] if args.smoke else FOCUS
    states={0:endpoints[0]} if args.smoke else endpoints
    curves=evaluate(m,linear,directions,widths,states,focus,smoke=args.smoke)
    report=dict(provenance=provenance,widths=widths,checks=checks,curves=curves,
                geometry_sha256=sha(out/'geometry.npz'),seconds=time.monotonic()-clock,
                scope='Local directional diagnostic with fixed scatter; no posterior widths or sampling claims')
    (out/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('completed_seconds',report['seconds'],flush=True)


if __name__=='__main__':main()
