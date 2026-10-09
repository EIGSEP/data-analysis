"""Regenerate the living camera-proposal review from checksum-pinned pilots.

Run in the arp environment. No sampler is launched and no posterior summaries
are released. Python notebook cells execute in-process so no kernel sockets are
needed. The source notebook, HTML, JSON, CSV and figures are generated together.
"""
from pathlib import Path
import argparse
import base64
import contextlib
import io
import json

import nbformat
from nbconvert import HTMLExporter

TITLE = 'Marjum 2026-10 Camera Proposal Repair Review'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path,
                        default=Path('/mnt/data02/eigsep/marjum-2026-07/derived/geometry_posterior/v0001'))
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    nb.cells.append(nbformat.v4.new_markdown_cell('''# Camera proposal repair and convergence review — 2026-10-07

This living review states the current convergence result for the corrected-EXIF, mixed-prior, float32-DEM runs, including the completed paired height/northing, uniform local-angle and state-aware angle-bank pilots and the full v0003 run. The old and corrected runs target different distributions and are never pooled. No calibrated geometry uncertainties are released.

**Question and falsification.** Did the repair remove frozen focal coordinates, and do the joint chains now agree? Convergence is rejected if any global coordinate has nonfinite diagnostics, rank-normalized folded/split R-hat > 1.01, bulk ESS < 400, or tail ESS < 400. The existing repository's 400-draw ESS floor is retained for comparison; the more conservative 100 effective draws per chain (800 total) is also reported. Passing all these marginal checks would be necessary, not sufficient, for full joint convergence.

**DEM scope.** The pilots and original coupling diagnostic use the frozen int32 `v0001/inputs/marjum_dem.npz`. A separate v0002 input set now changes only the DEM to `derived/dem/v0001/marjum_dem.npz`, with fractional float32 elevations. The term-attribution checks below compare these targets using unchanged directions. Both old endpoints are tested; chain 4 lies outside the new target support and is explicitly excluded from its density-difference comparisons. Old chains are diagnostic starting states, not samples from the float32 target.

**Historical pilot scope.** Both early pilots contain eight chains, each 300 discarded warmup plus 200 retained sweeps, on the same frozen 29-camera inputs. All 213 globals are checked without thinning, masking, selecting good chains, or pooling the two runs. Equal-sized complete chains receive equal weight in arithmetic means. The retained halves (sweeps 1–100 and 101–200) are checked separately. They are disjoint portions of these same chains, not independent campaign data.

**Method.** ArviZ rank-normalized R-hat combines split and folded checks; ESS uses bulk and tail methods. The old report's maximum 13.6 was ordinary split R-hat, so both old and repaired runs are recomputed using the same rank-normalized method. Ordinary split R-hat is included separately. See [Stan's diagnostics guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html), [ESS guidance](https://mc-stan.org/rstan/reference/Rhat.html), and [the rank-normalization paper](https://sites.stat.columbia.edu/gelman/research/published/Vehtari_etal_2020_rhat_ess.pdf).

**Important comparison limit.** The old sampler proposed raw focal length without the adjustment needed for its documented log-f prior. The repair samples the intended log-f measure and changes proposal curvature. Thus the runs have different focal measures: movement and convergence diagnostics can be compared descriptively, but neither pooled positions nor changes in their means identify the effect of a single repair.

**Prior implementation checks.** Code commit `a4b4820` passed the correlated position/log-f synthetic target and unit-invariant covariance checks; the 25+50-sweep smoke moved every camera. Commit `8758a06` added checkpointing and passed all six tests, including bit-for-bit interrupted/resumed equivalence. Those tests establish implementation mechanics, not convergence. The repaired pilot uses `8758a06`, verified below against its source hashes.

**Regeneration.** Use the corrected `eigsep_terrain` branch through explicit `PYTHONPATH` (without changing shared editable installs), then run `python marjum_mcmc_b21_review.py --run-root /path/to/geometry_posterior/v0001` in the arp environment. This script regenerates every table, figure, quoted result, notebook output, HTML and machine-readable diagnostic file. Input pins are in `marjum_mcmc_b21_review_inputs.json`. No chains or campaign inputs are modified.
'''))
    setup = f'''from pathlib import Path
import sys, json, hashlib, subprocess, csv
import numpy as np
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
CODE = Path({str(code)!r})
ROOT = Path({str(args.run_root.resolve())!r})
sys.path.insert(0, str(CODE))
from marjum_mcmc_b21_combine import convergence, label_coordinates
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_model_input(path,want,commit):
    if sha(path)==want:return
    assert Path(path).resolve()==CODE/'marjum_mcmc_b21.py',path
    archived=subprocess.check_output(['git','-C',str(CODE),'show',commit+':marjum_mcmc_b21.py'])
    assert hashlib.sha256(archived).hexdigest()==want
    old_prefix=archived.split(b'\\nclass Chain',1)[0]
    current_prefix=(CODE/'marjum_mcmc_b21.py').read_bytes().split(
        b'\\ndef validated_joint_selector',1)[0]
    assert old_prefix==current_prefix,'model target or geometry code changed'
pins = json.loads((CODE/'marjum_mcmc_b21_review_inputs.json').read_text())
for rel, want in pins.items():
    assert sha(ROOT/rel) == want, ('changed review input', rel)
frozen = json.loads((ROOT/'inputs/input_manifest.json').read_text())
for item in frozen['files'].values():
    assert sha(ROOT/item['frozen']) == item['sha256']
for name, item in frozen['feature_cache']['files'].items():
    assert sha(ROOT/'inputs/cv_features'/name) == item['sha256']
status = json.loads((ROOT/'pilot_logf_20261002/status.json').read_text())
assert status['state'] == 'complete'
assert len(status['chains']) == 8 and all(c['exit_code'] == 0 for c in status['chains'])
for name, want in status['signature']['code_sha256'].items():
    blob = subprocess.check_output(['git','-C',str(CODE),'show',status['signature']['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest() == want
runs = {{}}
for run in ['pilot', 'pilot_logf_20261002']:
    folder = ROOT/run
    manifest = json.loads((folder/'manifest.json').read_text())
    for path, want in manifest['input_sha256'].items():
        if '/inputs/' in path:
            assert sha(ROOT/'inputs'/path.split('/inputs/',1)[1]) == want
    files = sorted(folder.glob('chain_*.npz'), key=lambda p: int(p.stem.split('_')[1]))
    assert [p.name for p in files] == [f'chain_{{i}}.npz' for i in range(8)]
    values = []
    for path in files:
        with np.load(path, allow_pickle=False) as s:
            values.append({{k:s[k].copy() for k in s.files}})
    draws = np.stack([v['globals'] for v in values])
    assert draws.shape == (8, 200, 213) and np.isfinite(draws).all()
    keys = manifest['keys']; names = label_coordinates(keys, joint=True)
    assert len(keys) == 29 and len(names) == 213
    if run == 'pilot_logf_20261002':
        for v in values:
            assert v['camera_keys'].tolist() == keys
            a = json.loads(str(v['acceptance']))
            assert a['camera_proposals'] == [200]*29
    runs[run] = dict(draws=draws, values=values, manifest=manifest, keys=keys, names=names)
assert runs['pilot']['keys'] == runs['pilot_logf_20261002']['keys']
assert runs['pilot']['manifest']['config'] == runs['pilot_logf_20261002']['manifest']['config']
for field in ['tune','draws','seed','skyline_samples','shift_every']:
    assert runs['pilot']['manifest']['args'][field] == runs['pilot_logf_20261002']['manifest']['args'][field]
review_commit = subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
provenance = dict(review_commit=review_commit, sampler_commit=status['signature']['code_commit'],
    review_sha256=sha(CODE/'marjum_mcmc_b21_review.py'),
    helper_sha256=sha(CODE/'marjum_mcmc_b21_combine.py'), input_pins=pins,
    arviz_version=az.__version__, numpy_version=np.__version__,
    start_utc=status['started_utc'], finish_utc=status['finished_utc'])
with np.load(ROOT/'inputs/marjum_dem.npz', allow_pickle=False) as dem_file:
    provenance['dem_elevation_dtype'] = str(dem_file['dem'].dtype)
assert provenance['dem_elevation_dtype'] == 'int32'
print('All input, chain and repaired-source hashes verified.')
print('Provenance:',json.dumps(provenance,indent=2))
print('Scope:',[(k,v['draws'].shape) for k,v in runs.items()])
'''
    nb.cells.append(nbformat.v4.new_code_cell(setup))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Full-vector diagnostics and disjoint retained halves

The synthetic check below must distinguish independent draws from deliberately separated chains. It checks the diagnostic path rather than using the field data to confirm themselves. Per-coordinate results, including every failing coordinate, are written to CSV and JSON. No posterior mean or interval is released.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''rng = np.random.default_rng(41)
reference = rng.normal(size=(8, 2000, 2))
iid = convergence(reference)
separated = reference.copy(); separated[:4,:,0] += 3.
bad = convergence(separated)
assert iid[0].max() < 1.01 and bad[0][0] > 1.1
print('Synthetic IID max R-hat:',iid[0].max(),'; shifted-chain R-hat:',bad[0][0])
reports = {}
rows = []
def diagnostics(x):
    r, bulk, tail = convergence(x)
    # Check existing helper against explicit variable/chain/draw dimensions.
    ds = az.from_dict(posterior={'theta':x})
    np.testing.assert_allclose(r, az.rhat(ds,method='rank')['theta'].values)
    split = az.rhat(ds,method='split')['theta'].values
    return dict(rhat=r, ess_bulk=bulk, ess_tail=tail, ordinary_split_rhat=split)
def summary(d):
    r,b,t = d['rhat'],d['ess_bulk'],d['ess_tail']
    finite = np.isfinite(r)&np.isfinite(b)&np.isfinite(t)
    return dict(rhat_max=float(np.max(r)), rhat_above_1_01=int(np.sum(~finite|(r>1.01))),
        rhat_above_1_1=int(np.sum(~finite|(r>1.1))),
        bulk_ess_min=float(np.min(b)), tail_ess_min=float(np.min(t)),
        bulk_below_400=int(np.sum(~finite|(b<400))),tail_below_400=int(np.sum(~finite|(t<400))),
        pass_400=int(np.sum(finite&(r<=1.01)&(b>=400)&(t>=400))),
        pass_800=int(np.sum(finite&(r<=1.01)&(b>=800)&(t>=800))),
        ordinary_split_rhat_max=float(np.max(d['ordinary_split_rhat'])))
for run,v in runs.items():
    x=v['draws']; d=diagnostics(x); s=summary(d)
    s['worst_rhat_at']=v['names'][int(np.argmax(d['rhat']))]
    half=[summary(diagnostics(x[:,sl,:])) for sl in [slice(0,100),slice(100,200)]]
    between=np.std(x.mean(axis=1),axis=0,ddof=1)
    within=np.sqrt(np.mean(np.var(x,axis=1,ddof=1),axis=0))
    # Chain-centered residual drift is descriptive, not a convergence test.
    drift=np.max(np.abs(x[:,100:,:].mean(axis=1)-x[:,:100,:].mean(axis=1)),axis=0)/within
    reports[run]=dict(summary=s,retained_halves=half,diagnostics=d,
                      between_sd=between,within_rms_sd=within,half_mean_drift_in_within_sd=drift)
    for j,name in enumerate(v['names']):
        rows.append(dict(run=run,name=name,**{k:float(a[j]) for k,a in d.items()},
            between_chain_sd=float(between[j]),within_chain_rms_sd=float(within[j]),
            between_within_ratio=float(between[j]/within[j]),
            max_half_mean_drift_in_within_sd=float(drift[j])))
    print(run,json.dumps(s,indent=2))
    print('Disjoint retained halves:',json.dumps(half,indent=2))
    print('Passing coordinates:',[v['names'][i] for i in range(213) if d['rhat'][i]<=1.01 and d['ess_bulk'][i]>=400 and d['ess_tail'][i]>=400])
    print('Worst 15: name, rank R-hat, bulk ESS, tail ESS')
    for j in np.argsort(-d['rhat'])[:15]:
        print(v['names'][j],*[round(float(d[k][j]),3) for k in ['rhat','ess_bulk','ess_tail']])
with (CODE/'marjum_mcmc_b21_review_coordinates.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Every camera: movement and chain separation

A camera accepts a joint seven-coordinate update. Whole-network translations leave its focal length unchanged, so focal changes provide a comparable lower-bound count of camera updates for both runs. Counts below use the 199 transitions between retained draws; the new sampler's exact counters cover all 200 retained proposals and can differ by one. RMS jumps include zero displacement for rejections.

Position separation is `sqrt(sum_xyz Var_chain(mean_draw(position)))`. Within-chain position spread is `sqrt(mean_chain(sum_xyz Var_draw(position)))`. Both use metres in the same working-grid ENU coordinates and the same estimator for both runs. These are finite-run diagnostics, not uncertainty estimates. Their ratio does not have a standalone pass threshold.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''camera_tables={}
for run,v in runs.items():
    poses=v['draws'][:,:,:203].reshape(8,200,29,7)
    df=np.diff(poses[:,:,:,6],axis=1)
    changes=np.count_nonzero(df,axis=1)
    jump=np.sqrt(np.mean(df**2,axis=1))
    position=poses[:,:,:,:3]
    between=np.sqrt(np.sum(np.var(position.mean(axis=1),axis=0,ddof=1),axis=-1))
    within=np.sqrt(np.mean(np.sum(np.var(position,axis=1,ddof=1),axis=-1),axis=0))
    table=[]
    for i,key in enumerate(v['keys']):
        d=reports[run]['diagnostics'];block=slice(7*i,7*i+7)
        table.append(dict(camera=key,rhat_max=float(d['rhat'][block].max()),
            ess_bulk_min=float(d['ess_bulk'][block].min()),focal_rhat=float(d['rhat'][7*i+6]),
            focal_ess_bulk=float(d['ess_bulk'][7*i+6]),
            focal_changes_min=int(changes[:,i].min()),focal_changes_max=int(changes[:,i].max()),
            logf_jump_rms_min=float(jump[:,i].min()),logf_jump_rms_max=float(jump[:,i].max()),
            position_between_m=float(between[i]),position_within_m=float(within[i]),
            position_ratio=float(between[i]/within[i])))
    camera_tables[run]=table
    reports[run]['camera_chain_zero_focal_transitions']=int(np.sum(changes==0))
    print(run,'zero-motion camera/chain pairs',int(np.sum(changes==0)),'of',changes.size)
    print('camera maxRhat minBulkESS focalRhat focalBulkESS changes(min/max) minRMSlogf betweenPos_m withinPos_m ratio')
    for row in table:
        print(f"{row['camera']:>5} {row['rhat_max']:7.3f} {row['ess_bulk_min']:10.1f} {row['focal_rhat']:9.3f} {row['focal_ess_bulk']:12.1f} {row['focal_changes_min']:3}/{row['focal_changes_max']:<3} {row['logf_jump_rms_min']:11.3g} {row['position_between_m']:12.3f} {row['position_within_m']:11.3f} {row['position_ratio']:6.2f}")
new=runs['pilot_logf_20261002']
accept=[json.loads(str(v['acceptance'])) for v in new['values']]
exact=np.array([a['camera_accepts'] for a in accept])
counted=np.count_nonzero(np.diff(new['draws'][:,:,:203].reshape(8,200,29,7)[:,:,:,6],axis=1),axis=1)
assert np.all((exact-counted>=0)&(exact-counted<=1))
geometry=[d for v in new['values'] for h in json.loads(str(v['camera_geometry_history'])) for d in h['cameras']]
mechanics=dict(acceptance_min=float(exact.min()/200),acceptance_max=float(exact.max()/200),
              worst_acceptance_camera=new['keys'][int(np.unravel_index(exact.argmin(),exact.shape)[1])],
              geometry_fallbacks=sum(d['fallback'] for d in geometry),
              curvature_clipped_max=max(d['clipped'] for d in geometry if d['clipped'] is not None),
              pose_support_rejection_max=max(max(a['camera_support_reject_per_camera']) for a in accept))
print('Repaired per-camera mechanics:',json.dumps(mechanics,indent=2))
print('Focal movement: old -> new, worst-chain RMS log-f jump; rank R-hat')
for key in ['2172','2198','2203','2214']:
    a=next(r for r in camera_tables['pilot'] if r['camera']==key)
    b=next(r for r in camera_tables['pilot_logf_20261002'] if r['camera']==key)
    print(key,a['logf_jump_rms_min'],b['logf_jump_rms_min'],a['focal_rhat'],b['focal_rhat'])
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Antenna, transmitter and nuisance parameters

These coordinates are coupled to the cameras. The table gives convergence diagnostics and the ratio of between-chain mean SD to within-chain RMS SD. Coordinate units are metres for target positions and GPS bias, pixels for excess label scatter. Chain-centered half-to-half mean drift is in units of the coordinate's within-chain RMS SD. Neither apparent narrowness nor acceptable camera acceptance establishes convergence of these quantities.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''for run,v in runs.items():
    d=reports[run]['diagnostics']
    print(run)
    print('name Rhat bulkESS tailESS betweenSD withinRMSsd ratio maxHalfDrift/withinSD')
    for j in range(203,213):
        b=reports[run]['between_sd'][j];w=reports[run]['within_rms_sd'][j]
        print(f"{v['names'][j]:>24} {d['rhat'][j]:6.3f} {d['ess_bulk'][j]:8.1f} {d['ess_tail'][j]:8.1f} {b:10.4f} {w:10.4f} {b/w:7.2f} {reports[run]['half_mean_drift_in_within_sd'][j]:8.2f}")
    print('Landmark snapshot shapes:',[v0['landmarks'].shape for v0 in v['values']])
    print('Log-density recorded draw indices:',[v0['logp'][:,0].tolist() for v0 in v['values']])
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''figures=[]
fig,axes=plt.subplots(1,2,figsize=(12,10),constrained_layout=True)
for ax,(run,v) in zip(axes,runs.items()):
    rh=reports[run]['diagnostics']['rhat'][:203].reshape(29,7)
    im=ax.imshow(rh,aspect='auto',norm=LogNorm(vmin=1,vmax=5),cmap='magma')
    ax.set(xticks=np.arange(7),xticklabels=['e','n','u','theta','phi','tilt','log f'],
           yticks=np.arange(29),yticklabels=v['keys'],title=run)
fig.colorbar(im,ax=axes,label='Rank-normalized R-hat (convergence threshold 1.01)')
fig.suptitle('All camera coordinates; same diagnostic applied separately to each pilot')
figures.append(('mcmc_review_camera_rhat.png',fig))
fig,axes=plt.subplots(4,2,figsize=(13,11),constrained_layout=True)
for row,key in enumerate(['2172','2198','2203','2214']):
    for col,(run,v) in enumerate(runs.items()):
        i=v['keys'].index(key);ax=axes[row,col]
        for chain in range(8):
            ax.plot(np.arange(1,201),np.exp(v['draws'][chain,:,7*i+6]),lw=.8,label=str(chain))
        ax.set(title=f'{key} — {run}',ylabel='Focal length [pixels]',xlabel='Retained sweep')
        ax.ticklabel_format(useOffset=False)
axes[0,1].legend(title='Chain',ncol=4,fontsize=8)
fig.suptitle('Focal coordinates move after repair, but chains still disagree; targets differ in focal measure')
figures.append(('mcmc_review_focal_traces.png',fig))
fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
v=runs['pilot_logf_20261002']
for ax,name in zip(axes.flat,['transmitter_n','cam2159_n','cam2224_logf','cam2198_e','antenna_n','transmitter_extra_px']):
    j=v['names'].index(name)
    for chain in range(8):ax.plot(np.arange(1,201),v['draws'][chain,:,j],lw=.9,label=str(chain))
    unit='pixels' if name.endswith('_px') else ('log focal pixels' if name.endswith('_logf') else 'metres')
    ax.set(title=f'{name}, R-hat={reports["pilot_logf_20261002"]["diagnostics"]["rhat"][j]:.2f}',xlabel='Retained sweep',ylabel=unit)
    ax.ticklabel_format(useOffset=False)
axes[0,0].legend(title='Chain',ncol=4,fontsize=8)
fig.suptitle('Repaired pilot: selected failures and target-coordinate traces')
figures.append(('mcmc_review_remaining_failures.png',fig))
fig,ax=plt.subplots(figsize=(9,10),constrained_layout=True)
im=ax.imshow(exact.T/200,aspect='auto',vmin=0,vmax=.5,cmap='viridis')
ax.set(xticks=np.arange(8),xlabel='Chain',yticks=np.arange(29),yticklabels=v['keys'],title='Repaired pilot: each camera has 200 retained proposals')
fig.colorbar(im,ax=ax,label='Acceptance fraction')
figures.append(('mcmc_review_camera_acceptance.png',fig))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Exact coupling diagnostic — two saved endpoints

**Question and falsification.** Does co-motion reduce exact density penalties at the same displacement? A direction fails this local test if its penalty increases relative to the existing block direction, or the benefit fails to transfer to the second endpoint. This tests the precomputed directions, not every possible coordinated move.

Directions were derived at chain 0's final state and transferred unchanged to chain 4's final state. Both use the same frozen campaign data; chain 4 is a held-out state, not an independent physical validation dataset. Scatter stays fixed. The three patterns move (1) the selected camera or transmitter block, (2) all globals with landmarks fixed, or (3) globals plus landmarks. A scaled residual Jacobian defines JᵀJ; eliminating landmark blocks gives the Schur complement (see [Ceres's description](https://ceres-solver.readthedocs.io/latest/nnls_solving.html)). Inverse-column directions have unit displacement in the named coordinate. The widths below are local quadratic profile scales in metres, **not posterior standard deviations**.

The diagnostic evaluates both signs at three amplitudes per coordinate and endpoint. Its dimensionless even penalty is −[Δlogp(−a)+Δlogp(+a)]; it cancels the first-order term, and would equal a²/σ² for a quadratic target. Endpoints are not modes, so this is not an acceptance rate or a general curvature estimate. The table and CSV retain signed changes. Larger exact penalties than predicted can reflect residual curvature, finite differences and nonlinear structure; this run does not isolate the cause.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''coupling = json.loads((ROOT/'coupling_logf_20261002/diagnostic.json').read_text())
cp = coupling['provenance']
assert not cp['smoke'] and cp['origin_chain'] == 0 and cp['validation_chain'] == 4
assert coupling['geometry_sha256'] == sha(ROOT/'coupling_logf_20261002/geometry.npz')
for name, want in cp['code_sha256'].items():
    blob = subprocess.check_output(['git','-C',str(CODE),'show',cp['commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest() == want
for chain, want in cp['chain_sha256'].items():
    assert want == pins[f'pilot_logf_20261002/chain_{chain}.npz']
assert cp['input_manifest_sha256'] == pins['inputs/input_manifest.json']
assert cp['config'] == runs['pilot_logf_20261002']['manifest']['config']
rows = coupling['curves']; modes = ['block_only','globals','globals_landmarks']
labels = ['conditional_sigma','quarter_joint_sigma','joint_sigma']
assert len(rows) == 90
lookup = {(r['chain'],r['coordinate'],r['amplitude_label'],r['mode']): r for r in rows}
assert len(lookup) == 90
assert set(lookup) == {(c,n,a,m) for c in [0,4] for n in cp['focus'] for a in labels for m in modes}
for r in rows:
    if r['both_supported']:
        assert np.isclose(r['even_penalty'], -r['delta_logp_minus']-r['delta_logp_plus'])
checks=coupling['checks']
assert abs(checks['residual_error']) < 1e-8 and abs(checks['displaced_residual_error']) < 1e-8
assert checks['translation_invariant_error'] < 1e-7
assert max(r['relative_error'] for r in checks['schur_checks']) < 1e-6
print('Diagnostic commit:', cp['commit'])
print('Exact evaluations:', 2*len(rows), 'seconds:', coupling['seconds'])
print('Unsupported sign pairs:', sum(not r['both_supported'] for r in rows))
print('Numerical checks:', json.dumps(checks,indent=2))
print('Local quadratic scales (m): coordinate / block / globals / globals+landmarks')
for n,w in coupling['widths'].items():
    print(n, *[f'{w[k]:.6g}' for k in ['local','globals','joint']])
with (CODE/'marjum_mcmc_b21_review_coupling.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('At each block-scale displacement: chain / coordinate / metres / block / globals / joint penalties')
for c in [0,4]:
    for n in cp['focus']:
        selected=[lookup[c,n,'conditional_sigma',m] for m in modes]
        print(c,n,f"{selected[0]['amplitude_m']:.6g}",*[f"{r['even_penalty']:.6g}" for r in selected])
fig,axes=plt.subplots(2,5,figsize=(16,7),constrained_layout=True)
for row,c in enumerate([0,4]):
    for col,n in enumerate(cp['focus']):
        ax=axes[row,col]
        for m in modes:
            chosen=sorted([r for r in rows if r['chain']==c and r['coordinate']==n and r['mode']==m],key=lambda r:r['amplitude_m'])
            ax.plot([r['amplitude_m'] for r in chosen],[r['even_penalty'] for r in chosen],'o-',label=m)
        w=coupling['widths'][n]['joint'];aa=np.array(sorted({r['amplitude_m'] for r in rows if r['coordinate']==n}))
        ax.plot(aa,(aa/w)**2,':',color='black',label='quadratic joint prediction')
        ax.set(xscale='log',yscale='symlog',xlabel='Displacement (m)',title=f'{n}, chain {c}')
        if col==0:ax.set_ylabel('Exact even penalty')
axes[0,0].legend(fontsize=7)
figures=[('mcmc_review_coupling.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Fractional DEM and term-attribution smoke check

This bounded check changes only the DEM while keeping camera 2223's saved chain-0 state, directions and ±0.05 m displacement fixed. It compares the camera block with the precomputed globals-plus-landmarks direction. A terrain explanation would be contradicted if the dominant penalty comes from image ties and persists under the new DEM. Each signed residual-term sum must reproduce the independently evaluated full log density within 1e-8. This is one camera, one endpoint and one amplitude; finite-difference sensitivity and transfer to another endpoint remain untested here. The symmetric residual difference estimates a directional Jacobian at this finite step; it is not an exact Hessian.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''dem_smoke=json.loads((ROOT/'../v0002/dem_smoke_20261003/diagnostic.json').read_text())
assert dem_smoke['provenance']['smoke'] and len(dem_smoke['probes']) == 4
for name,want in dem_smoke['provenance']['source_hashes'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',dem_smoke['provenance']['commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
newroot=ROOT/'../v0002'
new_inputs=json.loads((newroot/'inputs/input_manifest.json').read_text())
for item in new_inputs['files'].values():assert sha(newroot/item['frozen'])==item['sha256']
for name,item in new_inputs['feature_cache']['files'].items():assert sha(newroot/'inputs/cv_features'/name)==item['sha256']
assert sha(newroot/'manifest.json')==dem_smoke['provenance']['manifest_sha256']
assert sha(newroot/'inputs/input_manifest.json')==dem_smoke['provenance']['input_manifest_sha256']
for row in dem_smoke['probes']:
    assert max(abs(x) for x in row['residual_errors'])<1e-8
    assert abs(sum(row['term_even_penalty'].values())-row['even_penalty'])<1e-8
print('Diagnostic source commit:',dem_smoke['provenance']['commit'])
print('Smoke seconds:',dem_smoke['seconds'])
print('Target / direction / exact even penalty / tie / terrain / horizon / directional J norm squared')
for r in dem_smoke['probes']:
    t=r['term_even_penalty']
    print(r['target'],r['mode'],*[f'{v:.6g}' for v in [r['even_penalty'],t['tie'],t['terrain'],t['horizon'],r['directional_jacobian_norm2']]])
print('Signed changes and all terms:',json.dumps(dem_smoke,indent=2))
fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
rr=dem_smoke['probes'];xx=np.arange(len(rr))
for i,term in enumerate(['tie','terrain','horizon']):
    ax.bar(xx+(i-1)*.24,[r['term_even_penalty'][term] for r in rr],width=.24,label=term)
ax.set(xticks=xx,xticklabels=[r['target']+' / '+r['mode'] for r in rr],ylabel='Contribution to even penalty',title='2223 north, chain 0, ±0.05 m; frozen directions')
ax.tick_params(axis='x',labelsize=8);ax.legend()
figures=[('mcmc_review_dem_smoke.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Full DEM diagnostic, support failure, and derivative audit

The full grid requested two DEMs, two endpoints, two camera coordinates, two motion patterns and three displacements (0.005, 0.01, 0.05 m): 48 signed-pair comparisons. The first process halted at an unsupported endpoint. Recovery verifies the original checkpoint checksum and source provenance, retains its completed comparisons, and identifies the failed support clause without moving the state or weakening the constraint. Unsupported comparisons are accounted for, not silently dropped or replaced.

The image-tie derivative audit uses the same frozen directions. It compares a sparse coordinate Jacobian at step multipliers 1, 0.1, 0.01 and 0.001 with independent direct directional differences at displacements from 1e-3 to 1e-6 m. Its reference and coordinate estimates can disagree. Agreement at fine steps, combined with stable direct differences at the smallest steps, tests the finite-difference approximation. It does not test posterior exploration. The image-tie function is independent of the DEM; at chain 4 this is an algebraic derivative audit, not a finite log-density comparison under float32.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''dem_full=json.loads((ROOT/'../v0002/dem_full_20261003_recovered/diagnostic.json').read_text())
audit=json.loads((ROOT/'../v0002/tie_derivative_audit.json').read_text())
for product in [dem_full,audit]:
    pr=product['provenance']
    for name,want in pr['source_hashes'].items():
        blob=subprocess.check_output(['git','-C',str(CODE),'show',pr['commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==want
    assert pr['input_manifest_sha256']==sha(ROOT/'../v0002/inputs/input_manifest.json')
imported=dem_full['provenance']['imported_checkpoint']
assert imported['sha256']==sha(ROOT/'../v0002/dem_full_20261003/checkpoint.json')
prior=json.loads((ROOT/'../v0002/dem_full_20261003/checkpoint.json').read_text())
assert dem_full['probes']==prior['probes'], 'recovery changed completed comparisons'
assert dem_full['status']=='complete_with_unsupported_endpoints'
assert len(dem_full['probes'])==36 and len(dem_full['unsupported_endpoints'])==1
unsupported=dem_full['unsupported_endpoints'][0]
assert (unsupported['target'],unsupported['chain'])==('float32',4)
assert len(unsupported['failures'])==1
support_failure=unsupported['failures'][0]
assert support_failure['camera']=='2232' and support_failure['margin_m']<0
expected={(t,c,n,m,h) for t,c in [('int32',0),('int32',4),('float32',0)] for n in ['cam2223_n','cam2224_n'] for m in ['block_only','globals_landmarks'] for h in [.005,.01,.05]}
actual={(r['target'],r['chain'],r['coordinate'],r['mode'],r['amplitude_m']) for r in dem_full['probes']}
assert actual==expected
for r in dem_full['probes']:
    assert max(abs(v) for v in r['residual_errors'])<1e-8
    assert abs(sum(r['term_even_penalty'].values())-r['even_penalty'])<1e-8
print('Recovered grid: 36 valid comparisons; 12 unavailable because float32 chain 4 baseline is outside support.')
print('Support failure:',json.dumps(support_failure,indent=2))
with (CODE/'marjum_mcmc_b21_review_dem_terms.csv').open('w') as f:
    rows=[{**{k:r[k] for k in ['target','chain','coordinate','mode','amplitude_m','even_penalty','directional_jacobian_norm2']},**r['term_even_penalty']} for r in dem_full['probes']]
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Maximum absolute term-sum mismatch:',max(abs(sum(r['term_even_penalty'].values())-r['even_penalty']) for r in dem_full['probes']))
print('All terms and support records:',json.dumps(dem_full,indent=2))
fine_errors=[];coarse_errors=[];direct_stability=[]
for state in audit['states']:
    print('Derivative audit chain',state['chain'])
    for row in state['coordinate']:
        print(row)
        if row['step_multiplier']==.001:fine_errors.append(row['relative_vector_error'])
        if row['step_multiplier']==1.:coarse_errors.append(row['relative_vector_error'])
    for n in ['cam2223_n','cam2224_n']:
        direct=[r for r in state['direct'] if r['coordinate']==n]
        direct_stability.append(abs(direct[-1]['norm2']/direct[-2]['norm2']-1))
    print('Largest per-observation derivative contributions:',state['worst_observations'])
assert max(fine_errors)<.001
assert max(direct_stability)<1e-4
print('Worst coarse / fine vector relative errors:',max(coarse_errors),max(fine_errors))
print('Worst small-step direct norm-squared relative difference:',max(direct_stability))
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,state in zip(axes,audit['states']):
    for n in ['cam2223_n','cam2224_n']:
        rr=[r for r in state['coordinate'] if r['coordinate']==n]
        ax.loglog([r['step_multiplier'] for r in rr],[r['relative_vector_error'] for r in rr],'o-',label=n)
    ax.set(title=f"Image ties, chain {state['chain']}",xlabel='Coordinate finite-difference step multiplier',ylabel='Relative error against direct derivative')
    ax.legend();ax.grid(alpha=.2)
figures=[('mcmc_review_derivative_audit.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Corrected proposal geometry on the float32 target

The next bounded test recomputes residual derivatives and candidate directions on the fractional DEM. All eight old final states are screened without alteration. Chain 0 supplies the geometry; validation uses the first other supported chain in ascending order, selected before measuring proposal performance. Supported states remain diagnostic seeds from an old-target pilot; they are not samples from the float32 posterior or evidence of dispersed initialization.

The derivative gate requires less than 1% relative vector disagreement with direct differences at both 1e-4 and 1e-5 m displacements, for both corrected joint directions. The 1e-3 coordinate-step multiplier misses this gate, chiefly in horizon residuals, so 1e-4 is tested with the same criterion. Fine derivatives also expose cancellation in subtracting normal matrices: compute the reduced curvature as (Jg+Jp R)ᵀ(Jg+Jp R), where R is the landmark response, and retain the original 1e-6 reduced-cost check. Report any eigenvalue clipping; if landmark blocks were regularized this would describe that approximate response, not an exact profile.

Exact density tests compare corrected camera-only and corrected globals-plus-landmarks moves for 2223/2224, at ±0.005 and ±0.05 m, on training and validation states. The directions are transferred unchanged. Unsupported candidates, signed changes and negative even penalties are retained. The even penalty can be negative at these non-mode states and is not a posterior width or acceptance rate.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''fine_runs={}
for label,folder in [('initial','fine_geometry_stable_20261003'),('refined','fine_geometry_refined_20261003')]:
    path=ROOT/'../v0002'/folder
    product=json.loads((path/'diagnostic.json').read_text());pr=product['provenance']
    assert product['status']=='complete'
    for name,want in pr['source_hashes'].items():
        blob=subprocess.check_output(['git','-C',str(CODE),'show',pr['commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==want
    assert pr['input_manifest_sha256']==sha(ROOT/'../v0002/inputs/input_manifest.json')
    for name in ['geometry','supported_endpoints','jacobian']:
        assert product[name+'_sha256']==sha(path/(name+'.npz'))
    fine_runs[label]=product
fine=fine_runs['refined'];initial_fine=fine_runs['initial']
supported=[r['chain'] for r in fine['endpoint_screen'] if r['supported']]
assert fine['training_chain']==0 and fine['validation_chain']==next(c for c in supported if c!=0)
assert fine['endpoint_screen']==initial_fine['endpoint_screen']
assert len(fine['probes'])==16
assert len({(r['chain'],r['coordinate'],r['mode'],r['amplitude_m']) for r in fine['probes']})==16
fine_error=max(r['relative_vector_error'] for r in fine['derivative_checks'])
assert fine['derivative_gate_pass']==(fine_error<.01)
for r in fine['probes']:
    if r['both_supported']:
        for delta,terms in zip(r['delta_logp'],r['term_delta_logp']):assert abs(delta-sum(terms.values()))<1e-8
print('Endpoint support screen:',json.dumps(fine['endpoint_screen'],indent=2))
print('Training / validation:',fine['training_chain'],fine['validation_chain'])
print('Step multipliers and worst derivative errors:',[(v['provenance']['step_multiplier'],max(r['relative_vector_error'] for r in v['derivative_checks'])) for v in fine_runs.values()])
print('Refined geometry checks:',json.dumps(fine['geometry_checks'],indent=2))
print('Refined directional derivative checks:',json.dumps(fine['derivative_checks'],indent=2))
print('Exact signed density changes:',json.dumps(fine['probes'],indent=2))
with (CODE/'marjum_mcmc_b21_review_fine_geometry.csv').open('w') as f:
    rows=[{k:r[k] for k in ['chain','coordinate','mode','amplitude_m','both_supported','even_penalty']}|{'delta_minus':r['delta_logp'][0],'delta_plus':r['delta_logp'][1]} for r in fine['probes']]
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fine_lookup={(r['chain'],r['coordinate'],r['mode'],r['amplitude_m']):r for r in fine['probes']}
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,n in zip(axes,['cam2223_n','cam2224_n']):
    for c in [fine['training_chain'],fine['validation_chain']]:
        for mode,marker in [('block_only','o'),('globals_landmarks','s')]:
            rr=[fine_lookup[c,n,mode,h] for h in [.005,.05]]
            ax.plot([r['amplitude_m'] for r in rr],[r['even_penalty'] if r['both_supported'] else np.nan for r in rr],marker=marker,label=f'{mode}, chain {c}')
    ax.set(xscale='log',xlabel='Displacement (m)',ylabel='Exact even penalty',title=n);ax.axhline(0,color='gray',lw=.5);ax.legend(fontsize=7)
figures=[('mcmc_review_fine_geometry.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Implemented joint kernel and paired-driver smoke

The sampler now offers optional frozen joint directions. Each scheduled joint update selects one of the two directions with fixed equal probability, draws a zero-mean Gaussian scalar, and uses the exact target-density ratio in stored log-f coordinates. An accepted move refreshes every cached horizon. The two scatter coordinates are unchanged by these directions and remain updated by the existing block kernel. Directions are fixed; only scalar scales adapt during discarded warmup. Both random streams and all adaptation state are checkpointed.

The paired comparison uses the same fractional DEM, supported states, fine camera/landmark derivatives, and block schedules in both arms; only the added joint kernel differs. This separates its effect from correcting finite differences. A separate joint RNG avoids consuming the block RNG stream, though state-dependent support decisions can subsequently change random-number alignment. It is not a claim of perfectly coupled trajectories.

Before the pilot, the actual kernel is tested on a correlated Gaussian target, and a real float32 chain is interrupted and resumed with bit-for-bit comparison of saved outputs. The four-worker driver smoke runs two warmup and three retained sweeps in each worker. It checks finite states, recorded proposals, source/input provenance and completed worker outputs; it is too short for any mixing or uncertainty claim.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''tests=json.loads((ROOT/'../v0002/joint_kernel_tests_20261003.json').read_text())
assert tests['log_sha256']==sha(ROOT/'../v0002/joint_kernel_tests_20261003.log')
test_log=(ROOT/'../v0002/joint_kernel_tests_20261003.log').read_text()
assert '7 passed' in test_log and 'failed' not in test_log and 'skipped' not in test_log
for name,want in tests['source_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',tests['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
smoke_root=ROOT/'../v0002/joint_pilot_smoke_20261003'
pilot_smoke=json.loads((smoke_root/'status.json').read_text())
assert pilot_smoke['state']=='complete' and len(pilot_smoke['workers'])==4
assert all(w['exit_code']==0 for w in pilot_smoke['workers'])
sg=pilot_smoke['signature']
assert (sg['tune'],sg['draws'])==(2,3)
for name,want in sg['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',sg['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
smoke_workers=[]
for arm in ['blocks','joint']:
    for chain in [0,1]:
        path=smoke_root/arm
        done=json.loads((path/f'completion_{chain}.json').read_text())
        assert done['result_sha256']==sha(path/f'chain_{chain}.npz')
        with np.load(path/f'chain_{chain}.npz') as saved:
            assert saved['globals'].shape==(3,213) and np.isfinite(saved['globals']).all()
            assert np.isfinite(saved['logp']).all()
            joint=json.loads(str(saved['joint_acceptance']))
            assert sum(joint['proposals'])==(3 if arm=='joint' else 0)
        smoke_workers.append(done)
print('Kernel tests:',test_log)
print('Smoke source commit:',sg['code_commit'])
for r in smoke_workers:
    j=r['joint_acceptance']
    print(r['arm'],r['chain'],'seconds',r['seconds'],'joint proposals',j['proposals'],'accepts',j['accepts'],'support rejects',j['support_rejections'])
print('All smoke worker diagnostics:',json.dumps(smoke_workers,indent=2))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## EXIF focal conversion: independent definition and 29-camera audit

[CIPA's specification](https://www.cipa.jp/std/documents/e/DCG-X001-2018_E.pdf) defines 35 mm equivalence through the ratio of image diagonals. Thus f_pixels = f_35mm × hypot(width,height) / hypot(36,24), assuming the supplied pixel dimensions retain the EXIF field of view. The previous width-only conversion changes the result under portrait rotation and assumes the full-frame aspect ratio. This is a definitional defect, independently of fitted focal values.

Both posterior implementations now call a shared `eigsep_terrain.exif.focal_length_pixels` helper, also used by EXIF pose initialization. The model's horizon residual uses the same corrected focal scale. This changes both focal-prior centres and horizon weighting. Camera/landmark feature extraction maps resized detections back to native full-image dimensions; all frozen shapes are checked below. Additional crops or anisotropic resizes would require their own intrinsics transform.

The numerical audit uses all retained focal draws from the completed loose-prior run2 and tight-prior run3: five equal-length chains (0,1,5,6,7), no masking or thinning. First take arithmetic means of exp(log f) within each chain, then equal-weight chain and camera means. These describe nonconverged old-target runs; they are not calibrated lens measurements or evidence for choosing a prior width. Every camera and the maximum absolute discrepancy are reported. The ultrawide offsets cannot identify distortion separately from metadata, processing or geometry errors.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''from eigsep_terrain import exif as exif_module
package_file=Path(exif_module.__file__).resolve()
package_root=package_file.parents[2]
package_commit=subprocess.check_output(['git','-C',str(package_root),'rev-parse','HEAD'],text=True).strip()
package_blob=subprocess.check_output(['git','-C',str(package_root),'show',package_commit+':src/eigsep_terrain/exif.py'])
assert hashlib.sha256(package_blob).hexdigest()==sha(package_file)
provenance['exif_package']=dict(commit=package_commit,source_sha256=sha(package_file),path=str(package_file))
with np.load(ROOT/'../v0002/inputs/fit_transmitter.npz') as saved:
    focal_keys=saved['keys'].astype(str).tolist();shapes=saved['shapes'].copy()
with np.load(ROOT/'../v0002/inputs/marjum_2026_07_exif_joint.npz') as ex:
    indices=[ex['keys'].astype(str).tolist().index(k) for k in focal_keys]
    f35=ex['focal_35mm'][indices]
old_centres=shapes[:,1]*f35/36
new_centres=exif_module.focal_length_pixels(f35,shapes[:,1],shapes[:,0])
np.testing.assert_allclose(new_centres,exif_module.focal_length_pixels(f35,shapes[:,0],shapes[:,1]))
np.testing.assert_allclose(new_centres/2,exif_module.focal_length_pixels(f35,shapes[:,1]/2,shapes[:,0]/2))
assert np.isfinite(new_centres).all() and len(focal_keys)==29
portrait=shapes[:,0]>shapes[:,1];ultrawide=f35==14
assert sorted(np.unique(shapes,axis=0).tolist())==[[3024,4032],[4032,3024]]
focal_audit={};focal_rows=[]
for run in ['joint_run2_20261004','joint_run3_20261004']:
    folder=ROOT/'../v0002'/run
    status=json.loads((folder/'status.json').read_text());assert status['state']=='complete'
    chain_means=[]
    for chain in [0,1,5,6,7]:
        with np.load(folder/'joint'/f'chain_{chain}.npz') as saved:
            assert saved['camera_keys'].astype(str).tolist()==focal_keys
            draws=saved['globals'];assert draws.shape==(2000,213) and np.isfinite(draws).all()
            chain_means.append(np.exp(draws[:,6:203:7]).mean(axis=0))
    means=np.mean(chain_means,axis=0);delta=100*(means/new_centres-1)
    groups={}
    for label,mask in [('portrait',portrait),('landscape',~portrait),('non_ultrawide',~ultrawide)]:
        groups[label]=dict(n=int(mask.sum()),old_mean_pct=float(np.mean(100*(means[mask]/old_centres[mask]-1))),
                          diagonal_mean_pct=float(delta[mask].mean()),diagonal_max_abs_pct=float(abs(delta[mask]).max()))
    focal_audit[run]=dict(groups=groups,mean_focal_px=means.tolist(),diagonal_offset_pct=delta.tolist(),signature=status['signature'])
    print(run, json.dumps(groups,indent=2))
    for i,k in enumerate(focal_keys):
        focal_rows.append(dict(run=run,camera=k,height=int(shapes[i,0]),width=int(shapes[i,1]),f35_mm=float(f35[i]),
                              old_centre_px=float(old_centres[i]),diagonal_centre_px=float(new_centres[i]),mean_focal_px=float(means[i]),diagonal_offset_pct=float(delta[i])))
with (CODE/'marjum_mcmc_b21_review_focal_audit.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(focal_rows[0]));writer.writeheader();writer.writerows(focal_rows)
print('Per-camera audit:',json.dumps(focal_rows,indent=2))
print('Conversion package:',provenance['exif_package'])
for name,count in [('exif_conversion_tests_20261004.log',11),('exif_model_tests_20261004.log',7)]:
    log=(ROOT/'../v0002'/name).read_text()
    assert f'{count} passed' in log and 'failed' not in log and 'skipped' not in log
    print(name,log)
fig,ax=plt.subplots(figsize=(13,4),constrained_layout=True)
for name,entry in focal_audit.items():ax.plot(focal_keys,entry['diagonal_offset_pct'],'o-',label=name)
ax.axhline(0,color='black',lw=.7);ax.tick_params(axis='x',rotation=90)
ax.set(ylabel='Arithmetic mean f offset from diagonal EXIF (%)',xlabel='Camera',title='Old-target runs; not calibrated focal estimates');ax.legend()
figures=[('mcmc_review_focal_conversion.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Approved corrected-target diagnostic preparation

Aaron approved continuing with 3% log-f width for the 27 main/telephoto images and 25% for the two 14 mm ultrawides, 2198 and 2203. These are fixed sensitivity-model priors, not estimates calibrated from the old chains. The model retains the frozen fractional float32 DEM input. The direction calculation is repeated for the corrected diagonal focal centres, corrected horizon scale and these mixed widths.

The derivative check could fail if the full direct directional derivative disagreed with the sparse Jacobian prediction by 1% or more. Exact-density probes test both signs of 5 mm and 5 cm displacements at training endpoint 0 and validation endpoint 1, selected by support before performance testing. The kernel smoke tests five supported starts with two warmup and three retained sweeps each. It establishes launch mechanics and finite support, not convergence.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''mixed_root=ROOT/'../v0002'
mixed_geometry=json.loads((mixed_root/'fine_geometry_diagonal_mixed_20261004/diagnostic.json').read_text())
assert mixed_geometry['status']=='complete' and mixed_geometry['derivative_gate_pass']
mg=mixed_geometry['provenance']
for name,want in mg['source_hashes'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',mg['commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
assert mg['model_input_sha256'][str(package_file)]==sha(package_file)
assert mg['config']['log_f_sigma']==.03 and mg['config']['log_f_sigma_by_camera']=={'2198':.25,'2203':.25}
for key,prior in mixed_geometry['focal_priors'].items():
    assert prior['log_sigma']==(.25 if key in ['2198','2203'] else .03)
    np.testing.assert_allclose(prior['centre_px'],new_centres[focal_keys.index(key)],rtol=1e-14)
assert [r['chain'] for r in mixed_geometry['endpoint_screen'] if r['supported']]==[0,1,5,6,7]
mixed_error=max(r['relative_vector_error'] for r in mixed_geometry['derivative_checks'])
assert mixed_error<.01
mixed_rows=[{k:r[k] for k in ['chain','coordinate','mode','amplitude_m','both_supported','even_penalty']} for r in mixed_geometry['probes']]
with (CODE/'marjum_mcmc_b21_review_mixed_geometry.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(mixed_rows[0]));writer.writeheader();writer.writerows(mixed_rows)
print('Corrected geometry:',json.dumps(mixed_geometry,indent=2))
log=(mixed_root/'mixed_focal_tests_20261004.log').read_text()
assert '8 passed' in log and 'failed' not in log and 'skipped' not in log
print('Mixed-prior tests:',log)
mixed_smoke_root=mixed_root/'joint_diagonal_mixed_smoke_20261004'
mixed_smoke=json.loads((mixed_smoke_root/'status.json').read_text())
assert mixed_smoke['state']=='complete'
assert (mixed_smoke['signature']['tune'],mixed_smoke['signature']['draws'])==(2,3)
assert mixed_smoke['signature']['config']==mg['config']
mixed_smoke_workers=[]
for c in [0,1,5,6,7]:
    done=json.loads((mixed_smoke_root/'joint'/f'completion_{c}.json').read_text())
    assert done['result_sha256']==sha(mixed_smoke_root/'joint'/f'chain_{c}.npz')
    with np.load(mixed_smoke_root/'joint'/f'chain_{c}.npz') as saved:
        assert saved['globals'].shape==(3,213) and np.isfinite(saved['globals']).all()
        assert np.isfinite(saved['logp']).all()
    mixed_smoke_workers.append(done)
print('Five-start smoke:',json.dumps(mixed_smoke_workers,indent=2))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Corrected diagonal-EXIF run: completed convergence review

This is the current result. Five chains (0, 1, 5, 6, 7) each contain 300 discarded warmup sweeps and 2,000 retained sweeps, using the frozen float32 DEM, diagonal EXIF centres, 0.03 log-f widths for 27 main/telephoto cameras and 0.25 for ultrawides 2198/2203. The two frozen collective directions were recomputed for this target. All 213 globals are checked with rank-normalized split/folded R-hat <= 1.01 and bulk/tail ESS >= 400; 500 total ESS (100 per chain) is also reported. Reject convergence if any coordinate fails or is nonfinite. Both disjoint 1,000-draw halves are evaluated separately, without selecting chains or discarding more draws. Passing marginal checks would still be insufficient to establish full joint convergence.

The CSV reports every coordinate for this run and both same-length predecessor runs. The targets differ: the correction changed prior centres and horizon weighting, and the new width policy differs from both predecessors. Comparisons are descriptive, with no pooling or attribution to a single change. Chain means, their spans and half-window drift describe chain separation; they are not released as calibrated geometry estimates. The log density is checked on its 200 saved samples per chain, at stride 10. Landmark coordinates are not assessed here: the global failures already reject convergence of the joint chain. Camera acceptance, proposal regularization and support rejections are reported separately from convergence. No goodness-of-fit or physical distortion conclusion follows from these diagnostics.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''corrected_id='joint_diagonal_mixed_20261004'
corrected_root=ROOT/'../v0002'/corrected_id
corrected_status=json.loads((corrected_root/'status.json').read_text())
assert corrected_status['state']=='complete' and all(w['exit_code']==0 for w in corrected_status['workers'])
cs=corrected_status['signature']
assert (cs['tune'],cs['draws'],cs['starts'])==(300,2000,[0,1,5,6,7])
assert cs['config']['log_f_sigma']==.03 and cs['config']['log_f_sigma_by_camera']=={'2198':.25,'2203':.25}
for name,want in cs['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',cs['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
for path,want in cs['model_input_sha256'].items():verify_model_input(path,want,cs['code_commit'])
corrected_reports={};corrected_arrays={};corrected_rows=[];corrected_values=[]
def corrected_diagnostics(x):
    ds=az.from_dict(posterior={'theta':x})
    return {k:getattr(az,fun)(ds,method=method)['theta'].values for k,fun,method in
            [('rhat','rhat','rank'),('ess_bulk','ess','bulk'),('ess_tail','ess','tail')]}
def corrected_summary(d):
    r,b,t=(d[k] for k in ['rhat','ess_bulk','ess_tail'])
    finite=np.isfinite(r)&np.isfinite(b)&np.isfinite(t)
    return dict(pass_count=int(np.sum(finite&(r<=1.01)&(b>=400)&(t>=400))),
        pass_500=int(np.sum(finite&(r<=1.01)&(b>=500)&(t>=500))),
        rhat_max=float(np.max(r)),rhat_fail=int(np.sum(~finite|(r>1.01))),rhat_over_1_1=int(np.sum(~finite|(r>1.1))),
        bulk_min=float(np.min(b)),bulk_fail=int(np.sum(~finite|(b<400))),tail_min=float(np.min(t)),tail_fail=int(np.sum(~finite|(t<400))),nonfinite=int((~finite).sum()))
for run in ['joint_run2_20261004','joint_run3_20261004',corrected_id]:
    folder=ROOT/'../v0002'/run
    arrays=[]
    for chain in [0,1,5,6,7]:
        with np.load(folder/'joint'/f'chain_{chain}.npz',allow_pickle=False) as saved:
            assert saved['camera_keys'].astype(str).tolist()==focal_keys
            arrays.append(saved['globals'].copy())
            if run==corrected_id:
                done=json.loads((folder/'joint'/f'completion_{chain}.json').read_text())
                manifest=json.loads((folder/'joint'/f'manifest_{chain}.json').read_text())
                assert done['result_sha256']==sha(folder/'joint'/f'chain_{chain}.npz')
                assert manifest['signature']==cs
                assert manifest['input_sha256']==cs['model_input_sha256']
                corrected_values.append(dict(chain=chain,completion=done,logp=saved['logp'].copy(),
                    acceptance=json.loads(str(saved['acceptance'])),history=json.loads(str(saved['camera_geometry_history'])),final=saved['final'].copy()))
    x=np.stack(arrays);assert x.shape==(5,2000,213) and np.isfinite(x).all()
    names=label_coordinates(focal_keys,joint=True)
    d=corrected_diagnostics(x);s=corrected_summary(d);s['worst_name']=names[int(np.argmax(d['rhat']))]
    halves=[corrected_summary(corrected_diagnostics(x[:,sl,:])) for sl in [slice(0,1000),slice(1000,2000)]]
    within=np.sqrt(np.mean(np.var(x,axis=1,ddof=1),axis=0))
    means=x.mean(axis=1);between=means.std(axis=0,ddof=1)
    drift=(x[:,1000:,:].mean(axis=1)-x[:,:1000,:].mean(axis=1))/within
    corrected_reports[run]=dict(summary=s,halves=halves,diagnostics={k:v.tolist() for k,v in d.items()},
        chain_means=means.tolist(),chain_mean_span=np.ptp(means,axis=0).tolist(),within_rms_sd=within.tolist(),
        max_abs_half_drift_within_sd=np.max(abs(drift),axis=0).tolist())
    corrected_arrays[run]=x
    for j,name in enumerate(names):
        corrected_rows.append(dict(run=run,name=name,**{k:float(v[j]) for k,v in d.items()},
            passes=bool(d['rhat'][j]<=1.01 and d['ess_bulk'][j]>=400 and d['ess_tail'][j]>=400),
            between_sd=float(between[j]),within_sd=float(within[j]),chain_mean_span=float(np.ptp(means[:,j])),
            max_abs_half_drift_within_sd=float(np.max(abs(drift[:,j])))))
    print(run,json.dumps(s), 'halves',json.dumps(halves))
    for j in np.argsort(-d['rhat'])[:15]:print(names[j],*[float(d[k][j]) for k in d])
corrected_result=corrected_reports[corrected_id]
corrected_x=corrected_arrays[corrected_id]
corrected_mechanics={}
for v in corrected_values:
    a=v['acceptance'];joint=v['completion']['joint_acceptance']
    history=[c for h in v['history'] for c in h['cameras']]
    corrected_mechanics[str(v['chain'])]=dict(joint=joint,camera_min=float(min(a['camera_per_camera'])),camera_max=float(max(a['camera_per_camera'])),
        camera_per_camera=a['camera_per_camera'],camera_zero_accepts=int(np.sum(np.array(a['camera_accepts'])==0)),
        geometry_fallbacks=sum(c['fallback'] for c in history),geometry_clipped=sum(c['clipped'] for c in history),
        logp_first_half_mean=float(v['logp'][:100,1].mean()),logp_second_half_mean=float(v['logp'][100:,1].mean()),
        seconds=v['completion']['seconds'],cpu_seconds=v['completion']['cpu_seconds'])
    assert np.isfinite(v['logp']).all() and np.isfinite(v['final']).all()
corrected_logp=corrected_diagnostics(np.stack([v['logp'][:,1] for v in corrected_values])[:,:,None])
print('Mechanics',json.dumps(corrected_mechanics,indent=2))
print('Log-density diagnostics', {k:v.tolist() for k,v in corrected_logp.items()})
print('Targets and nuisance coordinates')
for row in corrected_rows:
    if row['run']==corrected_id and not row['name'].startswith('cam'): print(row)
with (CODE/'marjum_mcmc_b21_review_corrected_coordinates.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(corrected_rows[0]));writer.writeheader();writer.writerows(corrected_rows)
# Focus traces on the worst coordinates plus antenna/transmitter and ultrawides.
worst=np.argsort(-np.array(corrected_result['diagnostics']['rhat']))[:3].tolist()
focus=list(dict.fromkeys(worst+[names.index(k) for k in ['antenna_n','transmitter_e','transmitter_n','transmitter_u','cam2198_logf','cam2203_logf']]))
fig,axes=plt.subplots(len(focus),1,figsize=(12,2.0*len(focus)),constrained_layout=True)
for ax,j in zip(np.atleast_1d(axes),focus):
    values=corrected_x[:,:,j]
    if names[j].endswith('logf'):values=np.exp(values);unit='focal pixels'
    elif names[j].endswith(('th','ph','ti')):unit='radians'
    else:unit='metres'
    for c,row in zip([0,1,5,6,7],values):ax.plot(np.arange(1,2001),row,lw=.6,alpha=.8,label=f'chain {c}')
    ax.set(title=f"{names[j]}: rank R-hat {corrected_result['diagnostics']['rhat'][j]:.3f}",ylabel=unit)
axes[0].legend(ncol=5);axes[-1].set_xlabel('Retained sweep')
figures=[('mcmc_review_corrected_traces.png',fig)]
fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
for run,rep in corrected_reports.items():
    axes[0].plot(np.arange(213),rep['diagnostics']['rhat'],'.',label=run)
    axes[1].plot(np.arange(213),rep['diagnostics']['ess_bulk'],'.',label=run)
axes[0].axhline(1.01,color='black',lw=.8);axes[0].set(ylabel='Rank R-hat',yscale='log',title='Same length; different targets — descriptive comparison only')
axes[1].axhline(400,color='black',lw=.8);axes[1].set(ylabel='Bulk ESS',yscale='log',xlabel='Global coordinate index (camera 0–202, target/nuisance 203–212)')
axes[0].legend();figures.append(('mcmc_review_corrected_diagnostics.png',fig))
fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
for v in corrected_values:ax.plot(v['logp'][:,0],v['logp'][:,1],label=f"chain {v['chain']}",lw=1)
ax.set(xlabel='Saved iteration',ylabel='Log density',title='Corrected target: density traces saved every 10 sweeps');ax.legend(ncol=5)
figures.append(('mcmc_review_corrected_logp.png',fig))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Height/orientation collective directions at current endpoints

This bounded follow-up keeps the corrected target unchanged. It screens all five final states from the completed corrected run, trains a sparse fine-step Jacobian and reduced residual curvature at chain 0, and tests all five endpoints (0, 1, 5, 6, 7). Four candidates address the observed slow coordinates: transmitter height, camera 2159 heading/elevation and camera 2222 roll. Recomputed camera-2223 and transmitter northing directions are controls. These controls are newly trained at the current chain-0 endpoint; they are not the exact frozen vectors used in the completed sampler run.

A candidate passes numerical validation only when both direct directional derivative checks agree with the sparse prediction within 1%. Those derivative checks are at the training endpoint. Transfer is a separate check: compare both signs of matched primary-coordinate displacements with block-only and collective camera/landmark moves, and require supported states and a lower collective even penalty at every endpoint. The spatial probes are 0.005 and 0.05 m; angular probes are 0.00005 and 0.0005 rad. The direction arrays use dimensionless scaled coordinates, whose normalization and physical units are verified below. All 120 signed-pair comparisons are retained, including support failures. Every supported exact-density change agrees with the residual decomposition to 1e-8.

The even penalty cancels the first-order term and can be negative away from a mode; it is not an acceptance probability. The CSV also records the equal-weight mean Metropolis acceptance for these two fixed signed steps, counting unsupported steps as zero. That is a finite two-point check, not the sampler's Gaussian-proposal acceptance rate. Derivative agreement and local transfer do not establish convergence, a physical cause, or global validity of a frozen direction.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''height_folder=ROOT/'../v0002/fine_geometry_height_angles_20261005'
height_geo=json.loads((height_folder/'diagnostic.json').read_text())
hp=height_geo['provenance']
assert height_geo['status']=='complete'
assert hp['source_run']==corrected_id and hp['config']==corrected_status['signature']['config']
assert hp['model_input_sha256']==corrected_status['signature']['model_input_sha256']
assert height_geo['probe_chains']==[0,1,5,6,7] and height_geo['training_chain']==0
for name,want in hp['source_hashes'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',hp['commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
for c,want in hp['endpoint_sha256'].items():assert sha(corrected_root/'joint'/f'chain_{c}.npz')==want
assert sha(height_folder/'geometry.npz')==height_geo['geometry_sha256']
assert sha(height_folder/'supported_endpoints.npz')==height_geo['supported_endpoints_sha256']
height_focus=hp['coordinates'].split(',')
with np.load(height_folder/'geometry.npz') as saved:
    for name in height_focus:
        j=names.index(name)
        np.testing.assert_allclose(saved[name][:,j],np.ones(3),rtol=0,atol=1e-14)
        expected_scale=.01 if name.endswith(('_th','_ph','_ti','_logf')) else 1.
        assert height_geo['coordinate_scales'][name]==expected_scale
height_derivatives={name:max(r['relative_vector_error'] for r in height_geo['derivative_checks'] if r['coordinate']==name) for name in height_focus}
assert height_geo['derivative_gate_pass']==bool(max(height_derivatives.values())<.01)
height_lookup={(r['chain'],r['coordinate'],r['mode'],r['amplitude_scaled']):r for r in height_geo['probes']}
assert len(height_lookup)==5*6*2*2
height_rows=[];height_summary=[]
for name in height_focus:
    for h in [.005,.05]:
        grouped=[]
        for c in height_geo['probe_chains']:
            block=height_lookup[c,name,'block_only',h];joint=height_lookup[c,name,'globals_landmarks',h]
            both=block['both_supported'] and joint['both_supported']
            def pair_accept(r):return float(np.mean([0. if d is None else np.exp(min(0.,d)) for d in r['delta_logp']]))
            row=dict(chain=c,coordinate=name,amplitude_physical=joint['amplitude_physical'],unit=joint['unit'],
                derivative_relative_error=height_derivatives[name],both_supported=both,
                block_penalty=block['even_penalty'],joint_penalty=joint['even_penalty'],
                joint_gain=block['even_penalty']-joint['even_penalty'] if both else None,
                block_pair_accept=pair_accept(block),joint_pair_accept=pair_accept(joint),
                improves=bool(both and joint['even_penalty']<block['even_penalty']))
            height_rows.append(row);grouped.append(row)
        supported=[r for r in grouped if r['both_supported']]
        height_summary.append(dict(coordinate=name,amplitude_physical=h*height_geo['coordinate_scales'][name],unit=height_geo['coordinate_units'][name],
            derivative_pass=height_derivatives[name]<.01,supported=len(supported),improved=sum(r['improves'] for r in grouped),
            min_gain=min((r['joint_gain'] for r in supported),default=None),
            max_joint_penalty=max((r['joint_penalty'] for r in supported),default=None),
            joint_pair_accept_min=min(r['joint_pair_accept'] for r in grouped)))
with (CODE/'marjum_mcmc_b21_review_height_angles.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(height_rows[0]));writer.writeheader();writer.writerows(height_rows)
print('Derivative agreement by direction:',json.dumps(height_derivatives,indent=2))
print('Transfer summary:',json.dumps(height_summary,indent=2))
print('Every chain and amplitude:',json.dumps(height_rows,indent=2))
height_angle_terms=[]
for c in height_geo['probe_chains']:
    for mode in ['block_only','globals_landmarks']:
        r=height_lookup[c,'cam2159_ph',mode,.005]
        if r['both_supported']:
            terms={k:-sum(t[k] for t in r['term_delta_logp']) for k in r['term_delta_logp'][0]}
            np.testing.assert_allclose(sum(terms.values()),r['even_penalty'],rtol=0,atol=1e-8)
            height_angle_terms.append(dict(chain=c,mode=mode,terms=terms))
print('Camera 2159 heading small-step penalty terms:',json.dumps(height_angle_terms,indent=2))
print('Full geometry provenance and exact signed terms:',json.dumps(height_geo,indent=2))
height_small_pass=[r['coordinate'] for r in height_summary if r['amplitude_physical']==.005*height_geo['coordinate_scales'][r['coordinate']] and r['derivative_pass'] and r['supported']==5 and r['improved']==5]
height_large_pass=[r['coordinate'] for r in height_summary if r['amplitude_physical']==.05*height_geo['coordinate_scales'][r['coordinate']] and r['derivative_pass'] and r['supported']==5 and r['improved']==5]
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for ax,name in zip(axes.ravel(),height_focus):
    for h,mark in [(.005,'o'),(.05,'s')]:
        rows=[r for r in height_rows if r['coordinate']==name and r['amplitude_physical']==h*height_geo['coordinate_scales'][name]]
        ax.plot([r['chain'] for r in rows],[r['joint_gain'] if r['both_supported'] else np.nan for r in rows],mark+'-',label=f'{h*height_geo["coordinate_scales"][name]:g} {height_geo["coordinate_units"][name]}')
    ax.axhline(0,color='black',lw=.7);ax.set(title=name,xlabel='Endpoint chain',ylabel='Block penalty − collective penalty',);ax.set_yscale('symlog',linthresh=.01);ax.legend()
fig.suptitle('Positive gain means lower collective even penalty; missing point = unsupported sign pair')
figures=[('mcmc_review_height_angles_transfer.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Approved paired height/northing pilot: launch checks

Aaron authorized the next short pilot after the endpoint-transfer review. Compare the actual two frozen directions used in the completed corrected run (`cam2223_n`, `transmitter_n`, archived geometry `fine_geometry_diagonal_mixed_20261004`) with the newly validated `transmitter_u`, `transmitter_n` pair from `fine_geometry_height_angles_20261005`. The arms share the same current endpoints, target, random seeds, block steps, adaptation schedule, initial joint scale and number of joint attempts. Because the northing vector is also recomputed, this compares whole frozen-direction kernels; it does not identify height's separate causal contribution.

Select endpoints 5 and 6 before pilot outcomes: they have the highest and lowest transmitter-height chain means in the completed corrected run. This deliberately tests the difficult separated cases. The source states themselves are their final saved states, unmodified. Use seed 20261005, four workers, 100 discarded warmup and 200 retained sweeps each. One equally weighted choice from each arm's two directions per sweep, scalar scale initially 0.005 m, adapted only in warmup. The candidate must pass all recorded endpoint-transfer checks; the archived baseline is retained as a valid symmetric control even if inefficient.

Primary comparisons are transmitter-height movement and separation between chains, alongside per-coordinate bulk/tail ESS, rank R-hat and CPU/wall cost. Compare movement and effective sampling per compute time; acceptance alone is insufficient. A practical improvement is contradicted if increased cost outweighs movement/effective-sampling gains or if the worst-coordinate behavior deteriorates. With only two short chains per arm and one paired seed set, ESS remains a screening diagnostic, not calibrated precision or proof of convergence. Check all 213 globals, not just the height coordinate. No long run or uncertainty release follows automatically.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''height_pilot_smoke_root=ROOT/'../v0002/height_north_paired_smoke_20261005'
height_pilot_smoke=json.loads((height_pilot_smoke_root/'status.json').read_text())
assert height_pilot_smoke['state']=='complete' and len(height_pilot_smoke['workers'])==4
assert all(w['exit_code']==0 for w in height_pilot_smoke['workers'])
hps=height_pilot_smoke['signature']
assert hps['starts']==[5,6] and hps['arms']==['previous','joint']
assert (hps['tune'],hps['draws'],hps['seed'])==(2,3,20261005)
assert hps['config']==corrected_status['signature']['config']
assert hps['model_input_sha256']==corrected_status['signature']['model_input_sha256']
assert hps['arm_geometry_sha256']['previous']==corrected_status['signature']['geometry_sha256']
assert hps['arm_names']['previous']==corrected_status['signature']['joint_names']
assert hps['arm_names']['joint']==['transmitter_u','transmitter_n']
assert hps['arm_geometry_sha256']['joint']==height_geo['geometry_sha256']
assert hps['starts_sha256']==height_geo['supported_endpoints_sha256']
assert set(hps['arm_names']['joint']) <= set(height_small_pass)&set(height_large_pass)
height_means=np.array(corrected_result['chain_means'])[:,names.index('transmitter_u')]
labels=np.array([0,1,5,6,7])
assert labels[np.argmax(height_means)]==5 and labels[np.argmin(height_means)]==6
for name,want in hps['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',hps['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
height_pilot_workers=[];height_pilot_manifest={};height_pilot_starts={}
for arm in ['previous','joint']:
    for c in [5,6]:
        folder=height_pilot_smoke_root/arm
        manifest=json.loads((folder/f'manifest_{c}.json').read_text())
        done=json.loads((folder/f'completion_{c}.json').read_text())
        assert manifest['signature']==hps and manifest['input_sha256']==hps['model_input_sha256']
        assert done['result_sha256']==sha(folder/f'chain_{c}.npz')
        with np.load(folder/f'chain_{c}.npz') as saved:
            assert saved['globals'].shape==(3,213) and np.isfinite(saved['globals']).all()
            assert np.isfinite(saved['final']).all()
            assert sum(json.loads(str(saved['joint_acceptance']))['proposals'])==3
            height_pilot_starts[arm,c]=saved['start'].copy()
        height_pilot_manifest[arm,c]=manifest; height_pilot_workers.append(done)
for c in [5,6]:
    np.testing.assert_array_equal(height_pilot_starts['previous',c],height_pilot_starts['joint',c])
    assert height_pilot_manifest['previous',c]['initial_logp']==height_pilot_manifest['joint',c]['initial_logp']
print('Paired pilot smoke:',json.dumps(height_pilot_workers,indent=2))
print('Start-selection means:',dict(zip(labels.tolist(),height_means.tolist())))
print('All four smoke workers completed; identical paired starts/targets and correct archived/candidate directions verified.')
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Completed paired pilot: measured comparison

The following cell checks the completed outputs against their launch signature, source commit and model-input hashes, then compares the same 200 retained sweeps from the two preselected starts in each arm. It verifies identical paired initial global states and log densities. All 213 coordinates and both disjoint retained halves are included. The run is `v0002/height_north_paired_20261005`; the archived directions, starts, target and adaptation protocol are those declared above.

Full-sweep movement is the sum of squared differences between consecutive retained global states (199 transitions per chain). Normalize it and bulk ESS by the sum of both workers' sampler CPU seconds, including initialization of proposal geometry, warmup and retained sampling; model loading and supervisor overhead are outside that timer. This cost denominator is identical in scope across arms. Joint-only height movement projects each accepted scalar displacement onto its actual height component, so different direction bases are compared in metres. Joint support-rejection counts cover the 200 retained attempts per worker.

Short nonconverged chains give screening ESS values, not reliable posterior precision. The two halves are disjoint windows of the same paths, not independent replicates. A lower worst R-hat alone cannot establish a gain, particularly when the primary height separation increases. This experiment compares entire frozen kernels, including a recomputed northing direction, at one seed set and two deliberately difficult endpoints.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''paired_root=ROOT/'../v0002/height_north_paired_20261005'
paired_status=json.loads((paired_root/'status.json').read_text());ps=paired_status['signature']
assert paired_status['state']=='complete' and all(w['exit_code']==0 for w in paired_status['workers'])
assert ps['starts']==[5,6] and ps['arms']==['previous','joint']
assert (ps['tune'],ps['draws'],ps['seed'])==(100,200,20261005)
for name,want in ps['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',ps['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
for path,want in ps['model_input_sha256'].items():verify_model_input(path,want,ps['code_commit'])
paired_arrays={};paired_report={};paired_rows=[];paired_chain_rows=[];paired_manifests={};paired_starts={}
height_j=names.index('transmitter_u')
for arm in ps['arms']:
    arrays=[];completed=[];saved_logp=[]
    folder=paired_root/arm
    for c in ps['starts']:
        manifest=json.loads((folder/f'manifest_{c}.json').read_text())
        done=json.loads((folder/f'completion_{c}.json').read_text())
        assert manifest['signature']==ps and manifest['input_sha256']==ps['model_input_sha256']
        assert done['result_sha256']==sha(folder/f'chain_{c}.npz')
        paired_manifests[arm,c]=manifest
        with np.load(folder/f'chain_{c}.npz') as saved:
            assert saved['camera_keys'].astype(str).tolist()==focal_keys
            arrays.append(saved['globals'].copy());paired_starts[arm,c]=saved['start'].copy()
            assert np.isfinite(saved['final']).all() and np.isfinite(saved['logp']).all()
            saved_logp.append(saved['logp'].copy())
            assert np.array_equal(saved['globals'][-1],saved['final'][:213])
        completed.append(done)
    x=np.stack(arrays);assert x.shape==(2,200,213) and np.isfinite(x).all()
    paired_arrays[arm]=x
    d=corrected_diagnostics(x);summary_pair=corrected_summary(d)
    summary_pair['worst_name']=names[int(np.argmax(d['rhat']))]
    halves=[corrected_summary(corrected_diagnostics(x[:,sl,:])) for sl in [slice(0,100),slice(100,200)]]
    cpu=sum(v['cpu_seconds'] for v in completed);wall=max(v['seconds'] for v in completed)
    jump=np.diff(x,axis=1);sq=np.sum(jump**2,axis=(0,1));msd=np.mean(jump**2,axis=(0,1))
    means=x.mean(axis=1);within=np.sqrt(np.mean(np.var(x,axis=1,ddof=1),axis=0))
    spans=abs(means[0]-means[1]);first_gap=abs(x[0,:100].mean(axis=0)-x[1,:100].mean(axis=0));last_gap=abs(x[0,100:].mean(axis=0)-x[1,100:].mean(axis=0))
    geo=ROOT/'../v0002'/ps['arm_geometry'][arm]/'geometry.npz'
    assert sha(geo)==ps['arm_geometry_sha256'][arm]
    # Target ENU coordinates have scale one in Linearization. The saved
    # amplitude sums therefore give each direction's exact height jump sum.
    with np.load(geo) as saved:height_response=np.array([saved[n][2,height_j] for n in ps['arm_names'][arm]])
    joint_height_sq=[]
    for k,(c,done) in enumerate(zip(ps['starts'],completed)):
        j=done['joint_acceptance'];assert sum(j['proposals'])==200
        joint_height_sq.append(float(np.dot(j['accepted_sq'],height_response**2)))
        row=dict(arm=arm,chain=c,cpu_seconds=done['cpu_seconds'],wall_seconds=done['seconds'],
            start_height=float(paired_starts[arm,c][height_j]),first_retained_height=float(x[k,0,height_j]),
            mean_height=float(means[k,height_j]),last_height=float(x[k,-1,height_j]),
            first_half_mean_height=float(x[k,:100,height_j].mean()),last_half_mean_height=float(x[k,100:,height_j].mean()),
            full_sweep_height_rms=float(np.sqrt(np.mean(jump[k,:,height_j]**2))),joint_height_squared_m=joint_height_sq[-1],
            joint_proposals=sum(j['proposals']),joint_accepts=sum(j['accepts']),joint_support_rejects=sum(j['support_rejections']),
            camera_accept_min=float(min(done['acceptance']['camera_per_camera'])),camera_accept_max=float(max(done['acceptance']['camera_per_camera'])))
        paired_chain_rows.append(row)
    paired_report[arm]=dict(summary=summary_pair,halves=halves,diagnostics={k:v.tolist() for k,v in d.items()},
        cpu_seconds=cpu,worker_wall_seconds=wall,chain_mean_span=spans.tolist(),within_rms_sd=within.tolist(),
        first_half_span=first_gap.tolist(),last_half_span=last_gap.tolist(),full_sweep_rms=np.sqrt(msd).tolist(),
        squared_movement_per_cpu_second=(sq/cpu).tolist(),bulk_ess_per_cpu_second=(d['ess_bulk']/cpu).tolist(),
        height_joint_squared_m=sum(joint_height_sq),height_joint_rms_per_attempt=float(np.sqrt(sum(joint_height_sq)/400)),
        height_joint_response=height_response.tolist(),completion=completed,
        logp={k:v.tolist() for k,v in corrected_diagnostics(np.stack([v[:,1] for v in saved_logp])[:,:,None]).items()})
    for j,name in enumerate(names):
        paired_rows.append(dict(arm=arm,name=name,**{k:float(v[j]) for k,v in d.items()},
            pass_all=bool(d['rhat'][j]<=1.01 and d['ess_bulk'][j]>=400 and d['ess_tail'][j]>=400),
            bulk_ess_per_cpu_second=float(d['ess_bulk'][j]/cpu),full_sweep_rms=float(np.sqrt(msd[j])),
            squared_movement_per_cpu_second=float(sq[j]/cpu),chain_mean_span=float(spans[j]),
            first_half_span=float(first_gap[j]),last_half_span=float(last_gap[j]),within_rms_sd=float(within[j])))
    print(arm,json.dumps(summary_pair),'halves',json.dumps(halves),'CPU seconds',cpu)
    print('Worst coordinates:',[(names[j],float(d['rhat'][j]),float(d['ess_bulk'][j])) for j in np.argsort(-d['rhat'])[:12]])
for c in ps['starts']:
    np.testing.assert_array_equal(paired_starts['previous',c],paired_starts['joint',c])
    assert paired_manifests['previous',c]['initial_logp']==paired_manifests['joint',c]['initial_logp']
    assert not np.array_equal(paired_arrays['previous'][ps['starts'].index(c)],paired_arrays['joint'][ps['starts'].index(c)])
paired_ratios=[]
oldp=paired_report['previous'];newp=paired_report['joint']
for j,name in enumerate(names):
    paired_ratios.append(dict(name=name,bulk_ess_per_cpu_ratio=newp['bulk_ess_per_cpu_second'][j]/oldp['bulk_ess_per_cpu_second'][j],
        movement_per_cpu_ratio=newp['squared_movement_per_cpu_second'][j]/oldp['squared_movement_per_cpu_second'][j],
        mean_span_ratio=newp['chain_mean_span'][j]/oldp['chain_mean_span'][j]))
for filename,rows in [('marjum_mcmc_b21_review_paired_coordinates.csv',paired_rows),('marjum_mcmc_b21_review_paired_chains.csv',paired_chain_rows),('marjum_mcmc_b21_review_paired_ratios.csv',paired_ratios)]:
    with (CODE/filename).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Per-chain mechanics:',json.dumps(paired_chain_rows,indent=2))
print('Target and worst-orientation results:')
for row in paired_rows:
    if row['name'] in ['transmitter_u','transmitter_n','transmitter_e','antenna_u','cam2159_ph','cam2159_th','cam2222_ti']:print(row)
paired_cost_ratio=newp['cpu_seconds']/oldp['cpu_seconds']
paired_ratio_summary={}
for field in ['bulk_ess_per_cpu_ratio','movement_per_cpu_ratio','mean_span_ratio']:
    values=np.array([r[field] for r in paired_ratios])
    paired_ratio_summary[field]=dict(above_one=int(sum(values>1)),below_one=int(sum(values<1)),minimum=float(values.min()),maximum=float(values.max()),minimum_name=names[int(values.argmin())],maximum_name=names[int(values.argmax())])
print('Height ratios:',paired_ratios[height_j])
print('All-coordinate ratios (candidate / control):',json.dumps(paired_ratio_summary,indent=2))
print('Sampler CPU cost ratio:',paired_cost_ratio)

fig,axes=plt.subplots(4,2,figsize=(13,10),sharey="row",constrained_layout=True)
for col,arm in enumerate(ps['arms']):
    for row,name in enumerate(['transmitter_u','transmitter_n','cam2159_ph','cam2222_ti']):
        ax=axes[row,col];j=names.index(name)
        for c,values in zip(ps['starts'],paired_arrays[arm][:,:,j]):ax.plot(np.arange(1,201),values,label=f'chain {c}',lw=1)
        ax.set(title=f"{arm}: {name}, R-hat {paired_report[arm]['diagnostics']['rhat'][j]:.2f}",ylabel='rad' if name.endswith(('_ph','_ti')) else 'm')
        if row==0:ax.legend()
        if row==3:ax.set_xlabel('Retained sweep')
figures=[('mcmc_review_height_north_paired_traces.png',fig)]
fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
for arm in ps['arms']:
    axes[0].plot(paired_report[arm]['diagnostics']['rhat'],'.',label=arm)
    axes[1].plot(paired_report[arm]['bulk_ess_per_cpu_second'],'.',label=arm)
axes[0].axhline(1.01,color='black',lw=.7);axes[0].set(ylabel='Rank R-hat',yscale='log');axes[0].legend()
axes[1].set(ylabel='Bulk ESS / sampler CPU second',yscale='log',xlabel='Global coordinate index')
fig.suptitle('Two short chains per arm: screening diagnostics, not calibrated precision')
figures.append(('mcmc_review_height_north_paired_diagnostics.png',fig))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Local geometry at the separated endpoints: completed diagnostic

Aaron authorized recomputing geometry at the two previously selected full-run endpoints, 5 and 6. The target, float32 DEM, EXIF centres and mixed focal priors remain unchanged. One new Jacobian is built at each endpoint, using the validated residual formulation and coordinate-step multiplier 1e-4. The new direction banks and the archived chain-0 bank each supply transmitter height, camera-2159 heading/elevation, and camera-2222 roll moves. Every bank is tested at both receiving endpoints, without altering those states.

The full diagnostic comprises 48 direct derivative comparisons (two receiving endpoints, three banks, four coordinates, two derivative steps) and 96 exact signed pairs (the same endpoints/banks/coordinates, block and collective moves, two amplitudes). Independent directional central differences use scaled steps 1e-4 and 1e-5; density probes use 0.005 and 0.05 m for height, and 0.00005 and 0.0005 rad for angles. Derivative discrepancies must stay below 1%. Each supported exact log-density change is checked against the residual representation and its term sum to 1e-8. All signed changes, failures and term contributions are retained.

For a like-for-like penalty comparison, all banks are compared with the **same locally recomputed block-only direction at the receiving endpoint**. The CSV also retains each bank's original block penalty. Positive gain is lower collective even penalty, where even penalty is minus the sum of positive/negative log-density changes. Negative even penalty is possible away from a local maximum; it is not negative variance. These finite perturbations do not measure posterior precision or mixing. Locally trained checks test local validity; transfer to the other endpoint supplies the distinct-state check.

The reduced smoke used one Jacobian, two coordinates and both endpoints. The full diagnostic ran under the approved host supervisor with an internal 840-second limit and a 900-second host limit. Input products, saved endpoints, geometry/Jacobian archives, source commits and the launch record are checksum-pinned below.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''local_root=ROOT/'../v0002/local_geometry_transfer_20261005'
local_geo=json.loads((local_root/'diagnostic.json').read_text())
local_smoke=json.loads((ROOT/'../v0002/local_geometry_transfer_smoke_20261005/diagnostic.json').read_text())
assert local_geo['status']=='complete' and local_geo['training_chains']==[5,6] and local_geo['probe_chains']==[5,6]
assert len(local_geo['derivative_checks'])==48 and len(local_geo['probes'])==96
assert local_smoke['status']=='complete' and local_smoke['smoke'] and local_smoke['derivative_gate_pass']
assert len(local_smoke['derivative_checks'])==8 and len(local_smoke['probes'])==32
for product,folder in [(local_geo,local_root),(local_smoke,ROOT/'../v0002/local_geometry_transfer_smoke_20261005')]:
    provenance_local=product['provenance']
    for name,want in provenance_local['source_hashes'].items():
        blob=subprocess.check_output(['git','-C',str(CODE),'show',provenance_local['commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==want
    assert provenance_local['config']==ps['config']
    assert provenance_local['model_input_sha256']==ps['model_input_sha256']
    for path,want in provenance_local['model_input_sha256'].items():
        verify_model_input(path,want,provenance_local['commit'])
    assert sha(folder/'supported_endpoints.npz')==product['supported_endpoints_sha256']
    with np.load(folder/'supported_endpoints.npz') as saved:
        for c in [5,6]:
            with np.load(ROOT/f'../v0002/joint_diagonal_mixed_20261004/joint/chain_{c}.npz') as source:
                np.testing.assert_array_equal(saved[f'chain_{c}'],source['final'])
    for bank,meta in product['geometry_banks'].items():
        assert sha(folder/f'geometry_{bank}.npz')==meta['geometry_sha256']
        if 'jacobian_sha256' in meta:assert sha(folder/f"jacobian_{meta['training_chain']}.npz")==meta['jacobian_sha256']
        with np.load(folder/f'geometry_{bank}.npz') as saved:
            assert set(saved.files)==set(product['coordinates'])
            assert all(np.isfinite(saved[n]).all() for n in saved.files)
    for row in product['probes']:
        if row['both_supported']:
            assert np.isfinite(row['delta_logp']).all()
            assert abs(sum(row['delta_logp'])+row['even_penalty'])<1e-8
            for d,terms in zip(row['delta_logp'],row['term_delta_logp']):assert abs(sum(terms.values())-d)<1e-8
local_focus=local_geo['coordinates'];local_banks=['archived_0','local_5','local_6']
local_lookup={(r['chain'],r['bank'],r['coordinate'],r['mode'],r['amplitude_scaled']):r for r in local_geo['probes']}
local_derivative_rows=[]
for row in local_geo['derivative_checks']:
    terms=row['per_term'];error_sq=sum(v['error_norm']**2 for v in terms.values())
    local_derivative_rows.append(dict(**{k:v for k,v in row.items() if k!='per_term'},pass_derivative=row['relative_vector_error']<.01,
        terrain_error_fraction=terms['terrain']['error_norm']**2/error_sq,
        largest_error_term=max(terms,key=lambda k:terms[k]['error_norm'])))
local_failures=[r for r in local_derivative_rows if not r['pass_derivative']]
assert local_geo['derivative_gate_pass']==(not local_failures)
local_rows=[]
for chain in [5,6]:
    for bank in local_banks:
        for name in local_focus:
            errors=[r['relative_vector_error'] for r in local_derivative_rows if (r['chain'],r['bank'],r['coordinate'])==(chain,bank,name)]
            assert len(errors)==2
            for h in [.005,.05]:
                # Every bank is compared with the SAME locally recomputed camera
                # block at this receiving endpoint. Retain its own block too.
                base=local_lookup[chain,f'local_{chain}',name,'block_only',h]
                own=local_lookup[chain,bank,name,'block_only',h]
                joint=local_lookup[chain,bank,name,'globals_landmarks',h]
                supported=base['both_supported'] and joint['both_supported']
                terms=({k:-sum(t[k] for t in joint['term_delta_logp']) for k in joint['term_delta_logp'][0]} if joint['both_supported'] else {})
                local_rows.append(dict(chain=chain,bank=bank,coordinate=name,amplitude_scaled=h,amplitude_physical=joint['amplitude_physical'],unit=joint['unit'],
                    own_endpoint=bank==f'local_{chain}',max_derivative_error=max(errors),pass_derivative=max(errors)<.01,both_supported=supported,
                    common_block_penalty=base['even_penalty'],bank_block_penalty=own['even_penalty'],joint_penalty=joint['even_penalty'],
                    gain=base['even_penalty']-joint['even_penalty'] if supported else None,
                    improves=supported and joint['even_penalty']<base['even_penalty'],
                    joint_delta_minus=joint['delta_logp'][0],joint_delta_plus=joint['delta_logp'][1],
                    tie_penalty=terms.get('tie'),terrain_penalty=terms.get('terrain'),horizon_penalty=terms.get('horizon')))
local_summary=[]
for name in local_focus:
    for h in [.005,.05]:
        for selection in ['own','transferred','archived']:
            rows=[r for r in local_rows if r['coordinate']==name and r['amplitude_scaled']==h and
                (r['own_endpoint'] if selection=='own' else r['bank']=='archived_0' if selection=='archived' else r['bank']!='archived_0' and not r['own_endpoint'])]
            assert len(rows)==2
            local_summary.append(dict(coordinate=name,amplitude_scaled=h,selection=selection,supported=sum(r['both_supported'] for r in rows),improved=sum(r['improves'] for r in rows),validated_improvement=sum(r['improves'] and r['pass_derivative'] for r in rows),worst_gain=min(r['gain'] for r in rows)))
for filename,rows in [('marjum_mcmc_b21_review_local_derivatives.csv',local_derivative_rows),('marjum_mcmc_b21_review_local_transfer.csv',local_rows)]:
    with (CODE/filename).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Local geometry derivative failures:',json.dumps(local_failures,indent=2))
print('Penalty/support summary (same receiving-state block baseline):',json.dumps(local_summary,indent=2))
print('Every comparison:',json.dumps(local_rows,indent=2))
local_worst=max(local_derivative_rows,key=lambda r:r['relative_vector_error'])
print('Worst derivative term attribution:',json.dumps(next(r for r in local_geo['derivative_checks'] if all(r[k]==local_worst[k] for k in ['chain','bank','coordinate','step_scaled'])),indent=2))
labels=[f'{c} ← {b.split("_")[-1]}' for c in [5,6] for b in local_banks]
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,name in zip(axes.ravel(),local_focus):
    for h,mark in [(.005,'o'),(.05,'s')]:
        rows=[r for r in local_rows if r['coordinate']==name and r['amplitude_scaled']==h]
        ax.plot(range(6),[r['gain'] for r in rows],mark+'-',label=f'{h*local_geo["coordinate_scales"][name]:g} {local_geo["coordinate_units"][name]}')
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(6),labels);ax.set(title=name,ylabel='Common block penalty − collective penalty',xlabel='Receiving endpoint ← training endpoint',yscale='symlog');ax.legend()
fig.suptitle('Exact penalty reduction: positive helps; local gains do not guarantee transfer')
figures=[('mcmc_review_local_geometry_transfer.png',fig)]
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,name in zip(axes.ravel(),local_focus):
    for h,mark in [(1e-4,'o'),(1e-5,'s')]:
        values=[next(r['relative_vector_error'] for r in local_derivative_rows if (r['chain'],r['bank'],r['coordinate'],r['step_scaled'])==(c,b,name,h)) for c in [5,6] for b in local_banks]
        ax.plot(range(6),np.array(values)*100,mark+'-',label=f'scaled h = {h:g}')
    ax.axhline(1,color='black',ls='--',lw=.8);ax.set_xticks(range(6),labels);ax.set(title=name,ylabel='Relative derivative discrepancy (%)',xlabel='Receiving endpoint ← training endpoint',yscale='log');ax.legend()
fig.suptitle(f'{len(local_failures)} checks exceed the predeclared 1% gate; failures are dominated by terrain residuals')
figures.append(('mcmc_review_local_geometry_derivatives.png',fig))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Targeted terrain derivative audit: cell crossings explain the selected failures

The approved audit reuses endpoint-5 camera-2159 elevation and endpoint-6 camera-2159 heading, their saved local Jacobians and frozen directions. It changes neither the model nor the DEM. Both cases include all terrain landmarks. The predeclared scaled check steps are 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7 and 1e-7; multiply by 0.01 for the selected angle in radians. Each coordinated move also displaces landmarks, whose cell membership is checked explicitly in both signs. The original point-coordinate Jacobian stencil is checked separately.

An independent bilinear polynomial uses the four stored DEM samples surrounding each landmark. Its value must match the actual interpolator at the original state to 1e-9 m. Differentiating that polynomial and the Student-t residual transform gives an analytic within-cell directional derivative. Reconstructing the original coordinate finite differences is a reproducibility check; it is not the independent validation. The analytic derivative and direct differences inside the cell supply that validation.

For attribution, the same cell polynomial is extended over the finite perturbation as a diagnostic control. This control removes changes of interpolation cell while retaining the original corner values and residual transform; it is never substituted into the posterior. The reported full-residual control discrepancy replaces only the terrain comparison with analytic versus fixed-cell differences and retains the measured errors from all other terms. It is a diagnostic decomposition, not a new residual model.

The cell-crossing explanation would fail if substantial error remained on landmarks whose original and directional stencils stay inside one cell, or if within-cell direct differences disagreed with the independent analytic derivative. Every terrain row is saved in diagnostic.npz, including rows with negligible errors. No row is masked or removed from the reported norms. The 1% full-residual criterion is unchanged; all original failing checks are reproduced before smaller steps are assessed.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''terrain_audit_root=ROOT/'../v0002/terrain_derivative_audit_20261005'
terrain_audit=json.loads((terrain_audit_root/'diagnostic.json').read_text())
assert terrain_audit['status']=='complete' and terrain_audit['seconds']<terrain_audit['provenance']['max_seconds']
assert terrain_audit['dem_dtype']=='float32'
assert [(r['chain'],r['coordinate']) for r in terrain_audit['cases']]==[(5,'cam2159_th'),(6,'cam2159_ph')]
assert terrain_audit['steps']==[1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7]
for name,want in terrain_audit['provenance']['source_hashes'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',terrain_audit['provenance']['commit']+':'+name]);assert hashlib.sha256(blob).hexdigest()==want
assert terrain_audit['provenance']['model_input_sha256']==local_geo['provenance']['model_input_sha256']
assert terrain_audit['provenance']['config']==local_geo['provenance']['config']
assert terrain_audit['provenance']['geometry_diagnostic_sha256']==sha(local_root/'diagnostic.json')
assert terrain_audit['arrays_sha256']==sha(terrain_audit_root/'diagnostic.npz')
with np.load(terrain_audit_root/'diagnostic.npz') as saved:terrain_arrays={k:saved[k].copy() for k in saved.files}
terrain_audit_rows=[];terrain_case_summary=[]
for case in terrain_audit['cases']:
    key=f"chain_{case['chain']}";pred=terrain_arrays[key+'_predicted_terrain'];analytic=terrain_arrays[key+'_analytic_terrain']
    assert len(pred)==case['npoints'] and np.isfinite(pred).all() and np.isfinite(analytic).all()
    assert case['base_height_polynomial_max_error_m']<1e-9
    assert case['archived_vs_rebuilt_relative_error']<1e-6
    np.testing.assert_allclose(np.linalg.norm(pred-analytic),case['archived_vs_analytic_terrain_error_norm'],rtol=1e-12)
    assert len(case['steps'])==len(terrain_audit['steps'])
    for j,row in enumerate(case['steps']):
        cross=terrain_arrays[key+f'_cross_{j}']|terrain_arrays[key+'_coordinate_stencil_cross']
        error=pred-terrain_arrays[key+f'_direct_{j}'];power=error**2
        assert int(sum(cross))==row['any_stencil_crossing_count']
        np.testing.assert_allclose(power[cross].sum()/power.sum(),row['crossing_error_fraction'],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(error[~cross]),row['noncrossing_terrain_error_norm'],rtol=1e-12)
        assert row['pass_derivative']==(row['relative_full_error']<.01)
        if row['step_scaled'] in [1e-4,1e-5]:
            original=next(r for r in local_geo['derivative_checks'] if (r['chain'],r['bank'],r['coordinate'],r['step_scaled'])==(case['chain'],f"local_{case['chain']}",case['coordinate'],row['step_scaled']))
            np.testing.assert_allclose(row['relative_full_error'],original['relative_vector_error'],rtol=1e-8,atol=1e-10)
        terrain_audit_rows.append(dict(chain=case['chain'],coordinate=case['coordinate'],**{k:v for k,v in row.items() if k!='worst_landmarks'}))
    uncut=[r for r in case['steps'] if r['any_stencil_crossing_count']==0]
    failures=[r for r in case['steps'] if not r['pass_derivative']]
    worst=case['steps'][0]['worst_landmarks'][0]
    terrain_case_summary.append(dict(chain=case['chain'],coordinate=case['coordinate'],points=case['npoints'],failures=len(failures),
        first_no_crossing_step=uncut[0]['step_scaled'] if uncut else None,
        first_no_crossing_error=uncut[0]['relative_full_error'] if uncut else None,
        maximum_no_crossing_error=max((r['relative_full_error'] for r in uncut),default=None),
        minimum_failure_crossing_fraction=min((r['crossing_error_fraction'] for r in failures),default=None),
        max_fixed_cell_control_error=max(r['fixed_cell_control_relative_full_error'] for r in case['steps']),
        dominant_point_at_largest_step=worst['point'],dominant_fraction_at_largest_step=worst['fraction'],
        original_stencil_crossing_count=case['coordinate_stencil_crossing_count'],origin_edge_count=case['origin_grid_edge_count']))
with (CODE/'marjum_mcmc_b21_review_terrain_derivatives.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(terrain_audit_rows[0]));writer.writeheader();writer.writerows(terrain_audit_rows)
print('Terrain derivative audit summary:',json.dumps(terrain_case_summary,indent=2))
print('All steps:',json.dumps(terrain_audit_rows,indent=2))
print('Worst landmarks for failed checks:',json.dumps([dict(chain=c['chain'],coordinate=c['coordinate'],steps=[r for r in c['steps'] if not r['pass_derivative']]) for c in terrain_audit['cases']],indent=2))
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for col,case in enumerate(terrain_audit['cases']):
    ax=axes[0,col];rows=case['steps'];h=np.array([r['step_scaled'] for r in rows]);err=np.array([r['relative_full_error'] for r in rows])*100
    ax.loglog(h,err,'o-',label='Actual directional check')
    ax.loglog(h,[100*r['fixed_cell_control_relative_full_error'] for r in rows],'s--',label='Within-cell analytic/control comparison')
    crossing=np.array([r['any_stencil_crossing_count']>0 for r in rows]);ax.scatter(h[crossing],err[crossing],facecolors='none',edgecolors='red',s=110,label='DEM-cell crossing',zorder=4)
    ax.axhline(1,color='black',lw=.8);ax.set(title=f"Endpoint {case['chain']}: {case['coordinate']}",xlabel='Scaled check step h (angle = 0.01 h rad)',ylabel='Full residual derivative discrepancy (%)');ax.legend(fontsize=8)
    ax=axes[1,col];key=f"chain_{case['chain']}";error=terrain_arrays[key+'_predicted_terrain']-terrain_arrays[key+'_direct_0'];power=error**2;fraction=power/power.sum()
    crossing=terrain_arrays[key+'_cross_0'];ax.semilogy(np.arange(len(error)),fraction,'.',ms=2,label='All terrain rows')
    ax.scatter(np.flatnonzero(crossing),fraction[crossing],color='red',s=25,label='Rows crossing DEM cells')
    for j in np.argsort(-power)[:2]:ax.annotate(str(j),(j,fraction[j]),xytext=(-40,-15),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Landmark index',ylabel='Fraction of squared terrain derivative error',title='Largest check step: h = 0.0001');ax.legend(fontsize=8)
fig.suptitle('Selected derivative failures are caused by finite check steps crossing DEM cells')
figures=[('mcmc_review_terrain_derivative_cells.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## All saved direction banks: cell-aware validation

This approved audit covers all four coordinates (transmitter height, camera-2159 heading/elevation, camera-2222 roll), all three saved banks (archived endpoint 0 and local endpoints 5/6), and both receiving endpoints 5/6. It reuses the saved Jacobians and directions, with the same target and unchanged 1% derivative threshold. No direction is selected by its observed error.

For each case, compute the symmetric distance along the direction to the first DEM-cell boundary over every landmark and both horizontal axes. Choose h = min(1e-5, 0.2 times that distance), then h/2. Reject a state with any landmark within 1e-10 grid pixels of an edge or a smaller step below 1e-8 scaled units. All 24 plans are written before any derivative errors are evaluated. Every case requires no actual cell crossings, both full-residual derivative discrepancies below 1%, and the relative difference between the two direct estimates below 1%. The latter is normalized by the larger direct-derivative norm and guards against an unreliable finite-difference scale. These checks do not claim coverage of every possible floating-point error.

Original fixed-step results remain in the input product and comparison figure. Checkpointed JSON and a per-case NPZ retain the full predicted/direct derivative vectors and actual perturbed grid coordinates; the executed cell below independently recalculates the reported norms and crossing counts from those arrays. The geometry step-selection helper was checked on a known cell distance, exact-edge rejection, the precision floor and zero horizontal motion before the data run. The maximum computation allowance is five minutes.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''cell_root=ROOT/'../v0002/cell_aware_derivatives_20261006'
cell_audit=json.loads((cell_root/'diagnostic.json').read_text())
cell_plan=json.loads((cell_root/'step_plan.json').read_text())
assert cell_audit['status']=='complete' and cell_audit['seconds']<cell_audit['provenance']['max_seconds']
assert len(cell_audit['cases'])==24 and len(cell_audit['step_plan'])==24
assert cell_audit['rule']==dict(max_step_scaled=1e-5,boundary_fraction=.2,second_step_factor=.5,min_step_scaled=1e-8,edge_tolerance_pixels=1e-10,derivative_threshold=.01,stability_threshold=.01)
assert cell_plan==dict(rule=cell_audit['rule'],plan=cell_audit['step_plan'])
assert sha(cell_root/'step_plan.json')==cell_audit['step_plan_sha256']
assert cell_audit['original_derivative_checks']==local_geo['derivative_checks']
assert cell_audit['provenance']['model_input_sha256']==local_geo['provenance']['model_input_sha256']
assert cell_audit['provenance']['config']==local_geo['provenance']['config']
assert cell_audit['provenance']['geometry_diagnostic_sha256']==sha(local_root/'diagnostic.json')
for name,want in cell_audit['provenance']['source_hashes'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',cell_audit['provenance']['commit']+':'+name]);assert hashlib.sha256(blob).hexdigest()==want
cell_rows=[];cell_cases=[]
for plan,case in zip(cell_audit['step_plan'],cell_audit['cases']):
    assert all(case[k]==v for k,v in plan.items())
    assert plan['steps_scaled'][1]==.5*plan['steps_scaled'][0]
    assert plan['steps_scaled'][0]==min(1e-5,.2*plan['symmetric_cell_limit_scaled'])
    if not case['geometry_eligible']:
        assert not case['pass_case'] and case['rejection_reason'];cell_cases.append(case);continue
    assert len(case['checks'])==2 and min(plan['steps_scaled'])>=1e-8
    assert sha(cell_root/case['arrays_file'])==case['arrays_sha256']
    with np.load(cell_root/case['arrays_file']) as saved:
        pred=saved['predicted'];grid=saved['grid_base'];directs=[]
        for j,row in enumerate(case['checks']):
            direct=saved[f'direct_{j}'];directs.append(direct.copy())
            assert np.isfinite(pred).all() and np.isfinite(direct).all()
            crosses=np.any(np.floor(saved[f'grid_plus_{j}'])!=np.floor(grid),axis=1)|np.any(np.floor(saved[f'grid_minus_{j}'])!=np.floor(grid),axis=1)
            assert sum(crosses)==row['crossing_count']
            relative=float(np.linalg.norm(pred-direct)/max(np.linalg.norm(direct),1e-30))
            np.testing.assert_allclose(relative,row['relative_error'],rtol=1e-12)
            cell_rows.append(dict(chain=case['chain'],bank=case['bank'],coordinate=case['coordinate'],unit=case['unit'],**{k:v for k,v in row.items() if k!='per_term'}))
        stability=float(np.linalg.norm(directs[0]-directs[1])/max(np.linalg.norm(directs[0]),np.linalg.norm(directs[1]),1e-30))
        np.testing.assert_allclose(stability,case['two_step_relative_difference'],rtol=1e-12)
    assert case['pass_case']==(all(r['pass_check'] for r in case['checks']) and stability<.01)
    original=[r['relative_vector_error'] for r in local_geo['derivative_checks'] if (r['chain'],r['bank'],r['coordinate'])==(case['chain'],case['bank'],case['coordinate'])]
    cell_cases.append(dict(chain=case['chain'],bank=case['bank'],coordinate=case['coordinate'],geometry_eligible=True,pass_case=case['pass_case'],
        original_max_error=max(original),new_max_error=max(r['relative_error'] for r in case['checks']),two_step_relative_difference=stability,
        larger_step_scaled=plan['steps_scaled'][0],smaller_step_scaled=plan['steps_scaled'][1],limiting_landmark=plan['limiting_landmark']))
assert cell_audit['all_pass']==all(r['pass_case'] for r in cell_cases)
with (CODE/'marjum_mcmc_b21_review_cell_aware.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(cell_rows[0]));writer.writeheader();writer.writerows(cell_rows)
cell_worst=max(cell_rows,key=lambda r:r['relative_error'])
cell_summary=dict(cases=len(cell_cases),checks=len(cell_rows),case_failures=sum(not r['pass_case'] for r in cell_cases),
    check_failures=sum(not r['pass_check'] for r in cell_rows),geometry_rejections=sum(not r['geometry_eligible'] for r in cell_cases),
    crossings=sum(r['crossing_count'] for r in cell_rows),worst_check=cell_worst,
    max_stability=max(r.get('two_step_relative_difference',0) for r in cell_cases),
    min_step=min(r['step_scaled'] for r in cell_rows),max_step=max(r['step_scaled'] for r in cell_rows),
    original_failures=sum(r['relative_vector_error']>=.01 for r in local_geo['derivative_checks']))
print('All-bank within-cell validation:',json.dumps(cell_summary,indent=2))
print('Each case, including original checks:',json.dumps(cell_cases,indent=2))
fig,axes=plt.subplots(2,1,figsize=(14,9),constrained_layout=True)
x=np.arange(len(cell_cases));labels=[f"{r['chain']}←{r['bank'].split('_')[-1]} {r['coordinate']}" for r in cell_cases]
axes[0].semilogy(x,[100*r['original_max_error'] for r in cell_cases],'x',label='Original fixed-step maximum')
axes[0].semilogy(x,[100*r['new_max_error'] for r in cell_cases],'o',label='Within-cell two-step maximum')
axes[0].axhline(1,color='black',ls='--',lw=.8);axes[0].set(ylabel='Derivative discrepancy (%)');axes[0].legend()
axes[1].semilogy(x,[100*r['two_step_relative_difference'] for r in cell_cases],'s',label='Difference between two direct estimates')
axes[1].axhline(1,color='black',ls='--',lw=.8);axes[1].set(ylabel='Two-step difference (%)');axes[1].legend()
for ax in axes:ax.set_xticks(x,labels,rotation=75,ha='right',fontsize=7)
fig.suptitle('All saved banks: geometry-selected steps pass without changing the 1% thresholds')
figures=[('mcmc_review_cell_aware_all_banks.png',fig)]
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Fixed local-angle mixture: frozen packet and paired smoke

The candidate selects uniformly among eight fixed vectors: the exact two height/northing control vectors and six camera-angle vectors trained at starts 5 and 6. Selection is independent of current state, so the existing symmetric Gaussian amplitude and exact-density Metropolis test apply. The signed finite moves have valid support at both starts, although angle penalties can worsen away from their training state. This smoke uses two warmup and three retained sweeps per worker; it tests launch mechanics, paired starts, source/target identity and finite output, not mixing.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''mixture_root=ROOT/'../v0002/local_angle_mixture_geometry_20261006'
mixture=json.loads((mixture_root/'diagnostic.json').read_text())
assert mixture['status']=='complete' and mixture['proposal_validation']=='fixed_local_bank_mixture'
assert mixture['derivative_gate_pass'] and mixture['probe_chains']==[5,6]
assert sha(mixture_root/'geometry.npz')==mixture['geometry_sha256']
assert sha(mixture_root/'supported_endpoints.npz')==mixture['supported_endpoints_sha256']==local_geo['supported_endpoints_sha256']
for folder,files in mixture['source_products'].items():
    for filename,want in files.items():assert sha(ROOT/'../v0002'/folder/filename)==want
for name,want in mixture['provenance']['source_hashes'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',mixture['provenance']['commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
assert mixture['provenance']['config']==local_geo['provenance']['config']==cell_audit['provenance']['config']
assert mixture['provenance']['model_input_sha256']==local_geo['provenance']['model_input_sha256']==cell_audit['provenance']['model_input_sha256']
expected_names=['transmitter_u','transmitter_n']+[f'{n}_local_{c}' for c in [5,6] for n in ['cam2159_ph','cam2159_th','cam2222_ti']]
with np.load(mixture_root/'geometry.npz') as mixed, np.load(height_folder/'geometry.npz') as control:
    assert mixed.files==expected_names
    for n in expected_names[:2]:np.testing.assert_array_equal(mixed[n],control[n])
    for c in [5,6]:
        with np.load(local_root/f'geometry_local_{c}.npz') as source:
            for n in ['cam2159_ph','cam2159_th','cam2222_ti']:
                np.testing.assert_array_equal(mixed[f'{n}_local_{c}'],source[n])
probe_lookup={(r['chain'],r['coordinate'],r['mode'],r['amplitude_scaled']):r for r in mixture['probes']}
assert len(probe_lookup)==64 and all(r['both_supported'] for r in probe_lookup.values())
assert len(mixture['derivative_checks'])==14
for r in mixture['derivative_checks']:
    if 'checks' in r:assert len(r['checks'])==2 and all(x['pass_check'] and x['crossing_count']==0 for x in r['checks'])
mixture_smoke_root=ROOT/'../v0002/local_angle_mixture_paired_smoke_20261006'
mixture_smoke=json.loads((mixture_smoke_root/'status.json').read_text())
assert mixture_smoke['state']=='complete' and len(mixture_smoke['workers'])==4
assert all(w['exit_code']==0 for w in mixture_smoke['workers'])
ms=mixture_smoke['signature']
assert (ms['tune'],ms['draws'],ms['seed'],ms['starts'])==(2,3,20261006,[5,6])
assert ms['arms']==['previous','joint'] and ms['arm_names']=={'previous':expected_names[:2],'joint':expected_names}
assert ms['geometry_sha256']==mixture['geometry_sha256'] and ms['starts_sha256']==mixture['supported_endpoints_sha256']
assert ms['config']==mixture['provenance']['config'] and ms['model_input_sha256']==mixture['provenance']['model_input_sha256']
assert (ms['joint_scale'],ms['joint_every'],ms['difference_step_factor'])==(.005,1,1e-4)
for name,want in ms['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',ms['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
mixture_smoke_workers=[]
for arm in ['previous','joint']:
    for chain in [5,6]:
        folder=mixture_smoke_root/arm
        manifest=json.loads((folder/f'manifest_{chain}.json').read_text())
        done=json.loads((folder/f'completion_{chain}.json').read_text())
        assert manifest['signature']==ms and done['result_sha256']==sha(folder/f'chain_{chain}.npz')
        with np.load(folder/f'chain_{chain}.npz') as saved:
            assert saved['globals'].shape==(3,213) and np.isfinite(saved['globals']).all()
        mixture_smoke_workers.append(dict(arm=arm,chain=chain,initial_logp=manifest['initial_logp'],
                                  cpu_seconds=done['cpu_seconds'],result_sha256=done['result_sha256']))
for chain in [5,6]:
    a,b=[r['initial_logp'] for r in mixture_smoke_workers if r['chain']==chain]
    assert a==b
print('Fixed mixture geometry:',mixture['geometry_sha256'])
print('All 64 signed support checks and twelve two-step angle validations passed.')
print('Paired smoke workers:',json.dumps(mixture_smoke_workers,indent=2))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Fixed local-angle mixture: completed paired pilot

The pilot compares the same two saved starts and 200 retained sweeps in each arm. Both arms retain the exact corrected target and fine block updates; the only intervention is the fixed set of joint directions. All 213 global coordinates are checked. Full-sweep squared movement and bulk ESS are normalized by summed sampler CPU time across the two workers per arm. The first and second retained halves are reported separately. Joint-only movement projects accepted direction amplitudes into transmitter height and the three selected camera angles in their physical units. These two short chains screen for a mixing improvement but cannot establish calibrated posterior uncertainties.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''angle_root=ROOT/'../v0002/local_angle_mixture_paired_20261006'
angle_status=json.loads((angle_root/'status.json').read_text());angle_sig=angle_status['signature']
assert angle_status['state']=='complete' and len(angle_status['workers'])==4
assert all(w['exit_code']==0 for w in angle_status['workers'])
assert (angle_sig['starts'],angle_sig['tune'],angle_sig['draws'],angle_sig['seed'])==([5,6],100,200,20261006)
assert angle_sig['arms']==['previous','joint'] and angle_sig['arm_names']=={'previous':expected_names[:2],'joint':expected_names}
assert angle_sig['geometry_sha256']==mixture['geometry_sha256'] and angle_sig['starts_sha256']==mixture['supported_endpoints_sha256']
assert angle_sig['config']==mixture['provenance']['config'] and angle_sig['model_input_sha256']==mixture['provenance']['model_input_sha256']
assert (angle_sig['joint_scale'],angle_sig['joint_every'],angle_sig['difference_step_factor'])==(.005,1,1e-4)
for name,want in angle_sig['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',angle_sig['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
for path,want in angle_sig['model_input_sha256'].items():verify_model_input(path,want,angle_sig['code_commit'])
angle_arrays={};angle_report={};angle_rows=[];angle_chain_rows=[];angle_manifests={};angle_starts={}
focus=['transmitter_u','transmitter_n','cam2159_ph','cam2159_th','cam2222_ti']
for arm in angle_sig['arms']:
    folder=angle_root/arm;arrays=[];completed=[]
    for c in angle_sig['starts']:
        manifest=json.loads((folder/f'manifest_{c}.json').read_text())
        done=json.loads((folder/f'completion_{c}.json').read_text())
        assert manifest['signature']==angle_sig and manifest['input_sha256']==angle_sig['model_input_sha256']
        assert done['result_sha256']==sha(folder/f'chain_{c}.npz')
        angle_manifests[arm,c]=manifest
        with np.load(folder/f'chain_{c}.npz') as saved:
            assert saved['camera_keys'].astype(str).tolist()==focal_keys
            arrays.append(saved['globals'].copy());angle_starts[arm,c]=saved['start'].copy()
            assert np.isfinite(saved['final']).all() and np.isfinite(saved['logp']).all()
            assert np.array_equal(saved['globals'][-1],saved['final'][:213])
        completed.append(done)
    x=np.stack(arrays);assert x.shape==(2,200,213) and np.isfinite(x).all()
    angle_arrays[arm]=x;d=corrected_diagnostics(x);summary=corrected_summary(d)
    summary['worst_name']=names[int(np.argmax(d['rhat']))]
    halves=[corrected_summary(corrected_diagnostics(x[:,sl,:])) for sl in [slice(0,100),slice(100,200)]]
    cpu=sum(v['cpu_seconds'] for v in completed)
    jump=np.diff(x,axis=1);sq=np.sum(jump**2,axis=(0,1));msd=np.mean(jump**2,axis=(0,1))
    means=x.mean(axis=1);spans=abs(means[0]-means[1])
    first_gap=abs(x[0,:100].mean(axis=0)-x[1,:100].mean(axis=0))
    last_gap=abs(x[0,100:].mean(axis=0)-x[1,100:].mean(axis=0))
    geo=ROOT/'../v0002'/angle_sig['arm_geometry'][arm]/'geometry.npz'
    assert sha(geo)==angle_sig['arm_geometry_sha256'][arm]
    with np.load(geo) as saved:
        # Saved global vectors use Linearization's scaled coordinates; the
        # worker multiplies them by scale before proposing physical radians.
        response={name:np.array([saved[n][2,names.index(name)]*height_geo['coordinate_scales'][name]
                                 for n in angle_sig['arm_names'][arm]]) for name in focus}
    joint_sq={name:0.0 for name in focus}
    direction_stats={name:dict(proposals=0,accepts=0,support_rejections=0) for name in angle_sig['arm_names'][arm]}
    for k,(c,done) in enumerate(zip(angle_sig['starts'],completed)):
        j=done['joint_acceptance'];assert sum(j['proposals'])==200
        assert len(j['proposals'])==len(angle_sig['arm_names'][arm])
        for i,name in enumerate(angle_sig['arm_names'][arm]):
            for key in ['proposals','accepts','support_rejections']:
                direction_stats[name][key]+=int(j[key][i])
        for name in focus:joint_sq[name]+=float(np.dot(j['accepted_sq'],response[name]**2))
        angle_chain_rows.append(dict(arm=arm,chain=c,cpu_seconds=done['cpu_seconds'],wall_seconds=done['seconds'],
            joint_proposals=sum(j['proposals']),joint_accepts=sum(j['accepts']),joint_support_rejects=sum(j['support_rejections']),
            **{'start_'+name:float(angle_starts[arm,c][names.index(name)]) for name in focus},
            **{'mean_'+name:float(means[k,names.index(name)]) for name in focus}))
    angle_report[arm]=dict(summary=summary,halves=halves,diagnostics={k:v.tolist() for k,v in d.items()},
        cpu_seconds=cpu,worker_wall_seconds=max(v['seconds'] for v in completed),
        chain_mean_span=spans.tolist(),first_half_span=first_gap.tolist(),last_half_span=last_gap.tolist(),
        full_sweep_rms=np.sqrt(msd).tolist(),squared_movement_per_cpu_second=(sq/cpu).tolist(),
        bulk_ess_per_cpu_second=(d['ess_bulk']/cpu).tolist(),joint_squared_movement=joint_sq,
        direction_stats=direction_stats,completion=completed)
    for j,name in enumerate(names):
        angle_rows.append(dict(arm=arm,name=name,**{field:float(values[j]) for field,values in d.items()},
            pass_all=bool(d['rhat'][j]<=1.01 and d['ess_bulk'][j]>=400 and d['ess_tail'][j]>=400),
            bulk_ess_per_cpu_second=float(d['ess_bulk'][j]/cpu),
            squared_movement_per_cpu_second=float(sq[j]/cpu),chain_mean_span=float(spans[j]),
            first_half_span=float(first_gap[j]),last_half_span=float(last_gap[j]),
            full_sweep_rms=float(np.sqrt(msd[j])),gap_over_rms_step=float(spans[j]/np.sqrt(msd[j]))))
    print(arm,'all-coordinate summary',json.dumps(summary),'halves',json.dumps(halves),'sampler CPU seconds',cpu)
    print(arm,'worst coordinates',[(names[j],float(d['rhat'][j]),float(d['ess_bulk'][j])) for j in np.argsort(-d['rhat'])[:12]])
for c in angle_sig['starts']:
    np.testing.assert_array_equal(angle_starts['previous',c],angle_starts['joint',c])
    assert angle_manifests['previous',c]['initial_logp']==angle_manifests['joint',c]['initial_logp']
angle_ratios=[];old_angle=angle_report['previous'];new_angle=angle_report['joint']
for j,name in enumerate(names):
    angle_ratios.append(dict(name=name,
        bulk_ess_per_cpu_ratio=new_angle['bulk_ess_per_cpu_second'][j]/old_angle['bulk_ess_per_cpu_second'][j],
        movement_per_cpu_ratio=new_angle['squared_movement_per_cpu_second'][j]/old_angle['squared_movement_per_cpu_second'][j],
        mean_span_ratio=new_angle['chain_mean_span'][j]/old_angle['chain_mean_span'][j]))
for filename,rows in [('marjum_mcmc_b21_review_angle_coordinates.csv',angle_rows),
                      ('marjum_mcmc_b21_review_angle_chains.csv',angle_chain_rows),
                      ('marjum_mcmc_b21_review_angle_ratios.csv',angle_ratios)]:
    with (CODE/filename).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Selected coordinate comparisons:')
for name in focus:
    j=names.index(name)
    print(name,json.dumps(dict(control_span=old_angle['chain_mean_span'][j],candidate_span=new_angle['chain_mean_span'][j],
        control_rhat=old_angle['diagnostics']['rhat'][j],candidate_rhat=new_angle['diagnostics']['rhat'][j],
        movement_per_cpu_ratio=angle_ratios[j]['movement_per_cpu_ratio'],
        bulk_ess_per_cpu_ratio=angle_ratios[j]['bulk_ess_per_cpu_ratio'],
        control_joint_squared=old_angle['joint_squared_movement'][name],
        candidate_joint_squared=new_angle['joint_squared_movement'][name])))
angle_ratio_summary={}
for field in ['bulk_ess_per_cpu_ratio','movement_per_cpu_ratio','mean_span_ratio']:
    values=np.array([r[field] for r in angle_ratios])
    angle_ratio_summary[field]=dict(above_one=int(sum(values>1)),below_one=int(sum(values<1)),
        minimum=float(values.min()),maximum=float(values.max()),
        minimum_name=names[int(values.argmin())],maximum_name=names[int(values.argmax())])
print('All-coordinate ratios:',json.dumps(angle_ratio_summary,indent=2))
print('Per-direction attempts, acceptances and support rejections:',
      json.dumps({arm:angle_report[arm]['direction_stats'] for arm in angle_sig['arms']},indent=2))
print('Per-worker mechanics:',json.dumps(angle_chain_rows,indent=2))
fig,axes=plt.subplots(len(focus),2,figsize=(13,12),sharey='row',constrained_layout=True)
for col,arm in enumerate(angle_sig['arms']):
    for row,name in enumerate(focus):
        ax=axes[row,col];j=names.index(name)
        for c,values in zip(angle_sig['starts'],angle_arrays[arm][:,:,j]):
            ax.plot(np.arange(1,201),values,label=f'chain {c}',lw=1)
        ax.set(title=f"{arm}: {name}, R-hat {angle_report[arm]['diagnostics']['rhat'][j]:.2f}",
               ylabel='rad' if name.endswith(('_ph','_th','_ti')) else 'm')
        if row==0:ax.legend()
        if row==len(focus)-1:ax.set_xlabel('Retained sweep')
figures=[('mcmc_review_local_angle_paired_traces.png',fig)]
fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
for arm in angle_sig['arms']:
    axes[0].plot(angle_report[arm]['diagnostics']['rhat'],'.',label=arm)
    axes[1].plot(angle_report[arm]['bulk_ess_per_cpu_second'],'.',label=arm)
axes[0].axhline(1.01,color='black',lw=.7);axes[0].set(ylabel='Rank R-hat',yscale='log');axes[0].legend()
axes[1].set(ylabel='Bulk ESS / sampler CPU second',yscale='log',xlabel='Global coordinate index')
fig.suptitle('Two short chains per arm: screening diagnostics, not calibrated precision')
figures.append(('mcmc_review_local_angle_paired_diagnostics.png',fig))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## State-aware angle-bank selection: target check and smoke

The eight vectors remain fixed. At each joint attempt, selection favors the archived bank nearest the current three camera angles; height/northing retain 25% total probability and both angle banks retain positive probability everywhere. For selected branch `j`, the scalar Gaussian move is symmetric. The log acceptance ratio adds `log w_j(proposal) − log w_j(current)` to the exact target difference, as required by [Hastings' general proposal rule](https://academic.oup.com/biomet/article-abstract/57/1/97/284580). The archived target/geometry source prefix is byte-identical to the current source; only the sampler section changed. A synthetic correlated-target test and bit-for-bit real-input checkpoint-resume tests cover mechanics. The 2+3 smoke checks the frozen Marjum target, paired starts, finite outputs and unchanged control path, not convergence.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''import marjum_mcmc_b21 as b21
aware_smoke_root=ROOT/'../v0002/state_aware_angle_paired_smoke_20261006'
aware_smoke=json.loads((aware_smoke_root/'status.json').read_text());ass=aware_smoke['signature']
assert aware_smoke['state']=='complete' and len(aware_smoke['workers'])==4
assert all(w['exit_code']==0 for w in aware_smoke['workers'])
assert (ass['starts'],ass['tune'],ass['draws'],ass['seed'])==([5,6],2,3,20261006)
assert ass['arm_names']=={'previous':expected_names[:2],'joint':expected_names}
assert ass['archived_model_input_sha256']==mixture['provenance']['model_input_sha256']
changed=[p for p in ass['model_input_sha256'] if ass['model_input_sha256'][p]!=ass['archived_model_input_sha256'][p]]
assert changed==[str(CODE/'marjum_mcmc_b21.py')]
for p,want in ass['model_input_sha256'].items():assert sha(p)==want,p
for name,want in ass['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',ass['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
current_prefix=(CODE/'marjum_mcmc_b21.py').read_bytes().split(b'\\ndef validated_joint_selector',1)[0]
assert hashlib.sha256(current_prefix).hexdigest()==ass['target_prefix_sha256']
for geometry in [mixture,height_geo]:
    archived=subprocess.check_output(['git','-C',str(CODE),'show',
        geometry['provenance']['commit']+':marjum_mcmc_b21.py'])
    assert archived.split(b'\\nclass Chain',1)[0]==current_prefix
selector=ass['joint_selector'];assert selector['groups']==[0,0,1,1,1,2,2,2]
assert selector['neutral_mass']==.25 and selector['bank_floor']==.05
np.testing.assert_allclose(selector['bandwidth'],np.linalg.norm(np.diff(selector['centers'],axis=0))/2)
with np.load(mixture_root/'supported_endpoints.npz') as endpoints:
    for c in [5,6]:
        probabilities=b21.joint_selection_probabilities(endpoints[f'chain_{c}'],selector)
        assert np.isfinite(probabilities).all() and np.all(probabilities>0)
        np.testing.assert_allclose(probabilities.sum(),1.)
        local_bank=1 if c==5 else 2
        assert probabilities[np.array(selector['groups'])==local_bank].sum()>.5
aware_smoke_rows=[]
for c in [5,6]:
    before=[]
    for arm in ['previous','joint']:
        folder=aware_smoke_root/arm
        manifest=json.loads((folder/f'manifest_{c}.json').read_text())
        done=json.loads((folder/f'completion_{c}.json').read_text())
        assert manifest['signature']==ass and done['result_sha256']==sha(folder/f'chain_{c}.npz')
        with np.load(folder/f'chain_{c}.npz') as saved:
            assert saved['globals'].shape==(3,213) and np.isfinite(saved['globals']).all()
            if arm=='previous':
                with np.load(mixture_smoke_root/'previous'/f'chain_{c}.npz') as archived:
                    assert saved.files==archived.files
                    for key in saved.files:np.testing.assert_array_equal(saved[key],archived[key])
        before.append(manifest['initial_logp'])
        aware_smoke_rows.append(dict(arm=arm,chain=c,cpu_seconds=done['cpu_seconds'],
                                     joint=done['joint_acceptance']))
    assert before[0]==before[1]
print('Target prefix',ass['target_prefix_sha256'])
print('State-aware selector',json.dumps(selector,indent=2))
print('Paired smoke and bit-for-bit control checks:',json.dumps(aware_smoke_rows,indent=2))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Completed state-aware paired pilot

The control must be bit-for-bit identical to the prior uniform-mixture control. The candidate differs only in branch selection and its exact reverse/forward correction; the vectors, starts, target and sampler schedule remain fixed. The next cell checks all source/input hashes and all four outputs, then compares all 213 globals, both retained halves, selected angles and transmitter height, branch use, support rejections and sampler CPU. Before examining the result, the proposed benefit would be more angle movement per CPU together with smaller between-chain gaps or better effective-sample diagnostics than the uniform mixture. More movement without cross-chain improvement would contradict that benefit. Passing the same all-coordinate R-hat and ESS thresholds would be needed to call the short pilot converged; even then, it could not by itself certify posterior precision.
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''aware_root=ROOT/'../v0002/state_aware_angle_paired_20261006'
aware_status=json.loads((aware_root/'status.json').read_text());aware_sig=aware_status['signature']
assert aware_status['state']=='complete' and len(aware_status['workers'])==4
assert all(w['exit_code']==0 for w in aware_status['workers'])
assert (aware_sig['starts'],aware_sig['tune'],aware_sig['draws'],aware_sig['seed'])==([5,6],100,200,20261006)
assert aware_sig['joint_selector']==selector and aware_sig['target_prefix_sha256']==ass['target_prefix_sha256']
assert aware_sig['arm_names']==ass['arm_names'] and aware_sig['arm_geometry_sha256']==ass['arm_geometry_sha256']
assert aware_sig['config']==ass['config'] and aware_sig['model_input_sha256']==ass['model_input_sha256']
for name,want in aware_sig['code_sha256'].items():
    blob=subprocess.check_output(['git','-C',str(CODE),'show',aware_sig['code_commit']+':'+name])
    assert hashlib.sha256(blob).hexdigest()==want
for p,want in aware_sig['model_input_sha256'].items():assert sha(p)==want,p
aware_arrays={};aware_report={};aware_chain_rows=[];aware_rows=[];aware_manifests={}
for arm in ['previous','joint']:
    folder=aware_root/arm;arrays=[];done_rows=[];direction_stats={n:dict(proposals=0,accepts=0,support_rejections=0)
        for n in aware_sig['arm_names'][arm]}
    for c in [5,6]:
        manifest=json.loads((folder/f'manifest_{c}.json').read_text())
        done=json.loads((folder/f'completion_{c}.json').read_text())
        assert manifest['signature']==aware_sig and manifest['input_sha256']==aware_sig['model_input_sha256']
        assert done['result_sha256']==sha(folder/f'chain_{c}.npz')
        aware_manifests[arm,c]=manifest
        with np.load(folder/f'chain_{c}.npz') as saved:
            assert saved['camera_keys'].astype(str).tolist()==focal_keys
            arrays.append(saved['globals'].copy())
            assert np.isfinite(saved['globals']).all() and np.isfinite(saved['final']).all()
            assert np.array_equal(saved['globals'][-1],saved['final'][:213])
            if arm=='previous':
                with np.load(angle_root/'previous'/f'chain_{c}.npz') as archived:
                    assert saved.files==archived.files
                    for key in saved.files:np.testing.assert_array_equal(saved[key],archived[key])
        j=done['joint_acceptance'];assert sum(j['proposals'])==200
        for i,n in enumerate(aware_sig['arm_names'][arm]):
            for k in ['proposals','accepts','support_rejections']:
                direction_stats[n][k]+=int(j[k][i])
        groups=np.asarray(selector['groups']) if arm=='joint' else np.zeros(len(j['proposals']),int)
        local_group=1 if c==5 else 2
        aware_chain_rows.append(dict(arm=arm,chain=c,cpu_seconds=done['cpu_seconds'],
            wall_seconds=done['seconds'],joint_proposals=sum(j['proposals']),
            joint_accepts=sum(j['accepts']),joint_support_rejections=sum(j['support_rejections']),
            local_bank_proposals=int(np.sum(np.asarray(j['proposals'])[groups==local_group])),
            remote_bank_proposals=int(np.sum(np.asarray(j['proposals'])[groups==3-local_group]))))
        done_rows.append(done)
    x=np.stack(arrays);assert x.shape==(2,200,213)
    aware_arrays[arm]=x;diagnostics=corrected_diagnostics(x);summary=corrected_summary(diagnostics)
    summary['worst_name']=names[int(np.argmax(diagnostics['rhat']))]
    halves=[corrected_summary(corrected_diagnostics(x[:,sl,:])) for sl in [slice(0,100),slice(100,200)]]
    cpu=sum(d['cpu_seconds'] for d in done_rows)
    jump=np.diff(x,axis=1);sq=np.sum(jump**2,axis=(0,1));msd=np.mean(jump**2,axis=(0,1))
    means=x.mean(axis=1);span=abs(means[0]-means[1])
    aware_report[arm]=dict(summary=summary,halves=halves,
        diagnostics={k:v.tolist() for k,v in diagnostics.items()},cpu_seconds=cpu,
        chain_mean_span=span.tolist(),full_sweep_rms=np.sqrt(msd).tolist(),
        squared_movement_per_cpu_second=(sq/cpu).tolist(),
        bulk_ess_per_cpu_second=(diagnostics['ess_bulk']/cpu).tolist(),
        direction_stats=direction_stats,completion=done_rows)
    for j,n in enumerate(names):
        aware_rows.append(dict(arm=arm,name=n,
            **{k:float(v[j]) for k,v in diagnostics.items()},
            chain_mean_span=float(span[j]),full_sweep_rms=float(np.sqrt(msd[j])),
            squared_movement_per_cpu_second=float(sq[j]/cpu),
            bulk_ess_per_cpu_second=float(diagnostics['ess_bulk'][j]/cpu)))
    print(arm,'summary',json.dumps(summary),'halves',json.dumps(halves),'CPU',cpu)
    print('Worst coordinates',[(names[j],float(diagnostics['rhat'][j])) for j in np.argsort(-diagnostics['rhat'])[:12]])
for c in [5,6]:
    assert aware_manifests['previous',c]['initial_logp']==aware_manifests['joint',c]['initial_logp']
aware_ratios=[]
for j,n in enumerate(names):
    aware_ratios.append(dict(name=n,
        movement_per_cpu_vs_uniform=aware_report['joint']['squared_movement_per_cpu_second'][j]/angle_report['joint']['squared_movement_per_cpu_second'][j],
        bulk_ess_per_cpu_vs_uniform=aware_report['joint']['bulk_ess_per_cpu_second'][j]/angle_report['joint']['bulk_ess_per_cpu_second'][j],
        span_vs_uniform=aware_report['joint']['chain_mean_span'][j]/angle_report['joint']['chain_mean_span'][j],
        movement_per_cpu_vs_control=aware_report['joint']['squared_movement_per_cpu_second'][j]/angle_report['previous']['squared_movement_per_cpu_second'][j],
        span_vs_control=aware_report['joint']['chain_mean_span'][j]/angle_report['previous']['chain_mean_span'][j],
        paired_movement_uplift_vs_uniform=(aware_report['joint']['squared_movement_per_cpu_second'][j]/aware_report['previous']['squared_movement_per_cpu_second'][j])/
            (angle_report['joint']['squared_movement_per_cpu_second'][j]/angle_report['previous']['squared_movement_per_cpu_second'][j]),
        paired_bulk_ess_uplift_vs_uniform=(aware_report['joint']['bulk_ess_per_cpu_second'][j]/aware_report['previous']['bulk_ess_per_cpu_second'][j])/
            (angle_report['joint']['bulk_ess_per_cpu_second'][j]/angle_report['previous']['bulk_ess_per_cpu_second'][j])))
for filename,rows in [('marjum_mcmc_b21_review_state_aware_coordinates.csv',aware_rows),
                      ('marjum_mcmc_b21_review_state_aware_chains.csv',aware_chain_rows),
                      ('marjum_mcmc_b21_review_state_aware_ratios.csv',aware_ratios)]:
    with (CODE/filename).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Selected coordinate comparisons:')
for n in focus:
    j=names.index(n)
    print(n,json.dumps(dict(control_rhat=aware_report['previous']['diagnostics']['rhat'][j],
        uniform_rhat=angle_report['joint']['diagnostics']['rhat'][j],
        state_aware_rhat=aware_report['joint']['diagnostics']['rhat'][j],
        control_span=aware_report['previous']['chain_mean_span'][j],
        uniform_span=angle_report['joint']['chain_mean_span'][j],
        state_aware_span=aware_report['joint']['chain_mean_span'][j],**aware_ratios[j])))
print('Direction counts:',json.dumps({arm:aware_report[arm]['direction_stats'] for arm in ['previous','joint']},indent=2))
fig,axes=plt.subplots(len(focus),3,figsize=(16,12),sharey='row',constrained_layout=True)
for col,(label,draws,report) in enumerate([('height/north control',aware_arrays['previous'],aware_report['previous']),
                                            ('uniform mixture',angle_arrays['joint'],angle_report['joint']),
                                            ('state-aware mixture',aware_arrays['joint'],aware_report['joint'])]):
    for row,n in enumerate(focus):
        ax=axes[row,col];j=names.index(n)
        for c,values in zip([5,6],draws[:,:,j]):ax.plot(np.arange(1,201),values,label=f'chain {c}',lw=1)
        ax.set(title=f"{label}: {n}, R-hat {report['diagnostics']['rhat'][j]:.2f}",
               ylabel='rad' if n.endswith(('_ph','_th','_ti')) else 'm')
        if row==0:ax.legend()
        if row==len(focus)-1:ax.set_xlabel('Retained sweep')
figures=[('mcmc_review_state_aware_paired_traces.png',fig)]
fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
for label,report in [('control',aware_report['previous']),('uniform',angle_report['joint']),
                     ('state-aware',aware_report['joint'])]:
    axes[0].plot(report['diagnostics']['rhat'],'.',label=label)
    axes[1].plot(report['bulk_ess_per_cpu_second'],'.',label=label)
axes[0].axhline(1.01,color='black',lw=.7);axes[0].set(ylabel='Rank R-hat',yscale='log');axes[0].legend()
axes[1].set(ylabel='Bulk ESS / sampler CPU second',yscale='log',xlabel='Global coordinate index')
figures.append(('mcmc_review_state_aware_paired_diagnostics.png',fig))
'''))
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Completed provisional five-chain geometry update

The frozen v0003 run uses the same corrected float32 DEM, diagonal EXIF and mixed focal-prior target as the preceding corrected five-chain run. It starts from the latest saved complete state for each of chains 0, 1, 5, 6 and 7, runs 300 warmup plus 2,000 retained sweeps each, and uses the exact state-aware angle-bank kernel. Before interpreting it, reject a convergence claim if any of the 213 globals has rank R-hat above 1.01 or bulk/tail ESS below 400. Check both disjoint retained halves as a drift screen. Recompute the diagnostics from all complete chains, independently of the run finalizer; compare the published point with the actual thinned complete state and its recorded exact density. A scalar density comparison with the predecessor can be negative, so it can contradict the claim that this is the best fit seen. Both runs share the target, but the selected states are not posterior summaries.'''))
    nb.cells.append(nbformat.v4.new_code_cell('''v3_root=ROOT/'../v0003'
v3_manifest=json.loads((v3_root/'manifest.json').read_text())
v3_report=json.loads((v3_root/'full_20261006/convergence_report.json').read_text())
v3_status=json.loads((v3_root/'full_20261006/status.json').read_text())
v3_launch=json.loads((v3_root/'full_20261006/launch.json').read_text())
assert v3_status['state']=='complete' and len(v3_status['workers'])==5
assert all(w['state']=='complete' and w['exit_code']==0 for w in v3_status['workers'])
assert v3_launch['signature']['config']==corrected_status['signature']['config']
assert v3_launch['signature']['input_manifest_sha256']==corrected_status['signature']['input_manifest_sha256']
target_prefix=(CODE/'marjum_mcmc_b21.py').read_bytes().split(b'\\ndef validated_joint_selector',1)[0]
assert hashlib.sha256(target_prefix).hexdigest()==v3_launch['signature']['target_prefix_sha256']
assert v3_manifest['geometry_sha256']==sha(v3_root/'provisional_geometry.npz')==v3_report['geometry_sha256']
assert v3_report['input_manifest_sha256']==v3_launch['signature']['input_manifest_sha256']
assert v3_report['start_states_sha256']==v3_launch['signature']['starts_sha256']
v3_traces=[]
for chain in [0,1,5,6,7]:
    path=v3_root/'full_20261006/joint'/f'chain_{chain}.npz'
    assert sha(path)==v3_report['chain_sha256'][str(chain)]
    with np.load(path,allow_pickle=False) as saved:
        assert saved['globals'].shape==(2000,213)
        assert saved['camera_keys'].astype(str).tolist()==focal_keys
        v3_traces.append(saved['globals'].copy())
        if chain==v3_report['selected']['chain']:
            sweep=v3_report['selected']['retained_sweep']
            assert sweep%10==0
            selected_record=np.r_[saved['globals'][sweep],saved['landmarks'][sweep//10].ravel()]
            selected_saved_logp=float(saved['logp'][sweep//10,1])
v3_x=np.stack(v3_traces)
v3_summary=corrected_summary(corrected_diagnostics(v3_x))
v3_halves=[corrected_summary(corrected_diagnostics(v3_x[:,sl,:])) for sl in [slice(0,1000),slice(1000,2000)]]
for actual,want in [(v3_summary,v3_report['full']),*zip(v3_halves,v3_report['retained_halves'])]:
    for key in ['pass_count','rhat_fail','bulk_fail','tail_fail']:
        assert actual[key]==want[key],(key,actual[key],want[key])
    assert np.isclose(actual['rhat_max'],want['rhat_max'],rtol=0,atol=1e-10)
with np.load(v3_root/'provisional_geometry.npz',allow_pickle=False) as saved:
    assert np.array_equal(saved['state'],selected_record)
    assert saved['camera'].shape==(29,7) and saved['landmarks'].shape[1]==3
assert np.isclose(selected_saved_logp,v3_report['selected']['exact_log_density'],rtol=0,atol=1e-10)
v3_previous_best=max(float(np.max(v['logp'][:,1])) for v in corrected_values)
v3_density_delta=float(selected_saved_logp-v3_previous_best)
print('v0003 full:',v3_summary,'halves:',v3_halves)
print('selected chain/sweep/logp:',v3_report['selected']['chain'],sweep,selected_saved_logp)
print('previous corrected-run highest saved logp:',v3_previous_best,'v0003 minus predecessor:',v3_density_delta)
'''))
    ns = {}
    execution = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        execution += 1
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(cell.source, f'<review-cell-{execution}>', 'exec'), ns)
        cell.execution_count = execution
        cell.outputs = []
        if stream.getvalue():
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=stream.getvalue()))
        for name, fig in ns.pop('figures', []):
            fig.savefig(code/name, dpi=140)
            cell.outputs.append(nbformat.v4.new_output('display_data', data={
                'image/png': base64.b64encode((code/name).read_bytes()).decode()}, metadata={}))
            ns['plt'].close(fig)
    old = ns['reports']['pilot']['summary']
    new = ns['reports']['pilot_logf_20261002']['summary']
    total_pairs = len(ns['runs']['pilot']['keys'])*ns['runs']['pilot']['draws'].shape[0]
    old_frozen = ns['reports']['pilot']['camera_chain_zero_focal_transitions']
    landmark_coordinates = 3*ns['runs']['pilot']['values'][0]['landmarks'].shape[1]
    coupling = ns['coupling']
    lookup = ns['lookup']
    def penalty(c, n, m, a='conditional_sigma'):
        return lookup[c,n,a,m]['even_penalty']
    def pair(n,m,a='conditional_sigma'):
        return ' / '.join(f'{penalty(c,n,m,a):.3g}' for c in [0,4])
    def fine_cost(chain,name,mode,h):
        r=ns['fine_lookup'][chain,name,mode,h]
        return f"{r['even_penalty']:.5f}" if r['both_supported'] else 'unsupported'
    corrected=ns['corrected_result']['summary']
    corrected_diag=ns['corrected_result']['diagnostics']
    def corr_value(name,field):return ns['corrected_result'][field][ns['names'].index(name)]
    paired=ns['paired_report'];previous=paired['previous'];candidate=paired['joint']
    def pv(arm,field,name='transmitter_u'):
        return paired[arm][field][ns['names'].index(name)]
    def pd(arm,field,name='transmitter_u'):
        return paired[arm]['diagnostics'][field][ns['names'].index(name)]
    def pr(field,name='transmitter_u'):
        return ns['paired_ratios'][ns['names'].index(name)][field]
    def lv(chain,bank,name,h,field):
        return next(r[field] for r in ns['local_rows'] if (r['chain'],r['bank'],r['coordinate'],r['amplitude_scaled'])==(chain,bank,name,h))
    def ld(chain,bank,name,h):
        return next(r['relative_vector_error'] for r in ns['local_derivative_rows'] if (r['chain'],r['bank'],r['coordinate'],r['step_scaled'])==(chain,bank,name,h))
    def ta(chain):return next(r for r in ns['terrain_audit']['cases'] if r['chain']==chain)
    def ts(chain):return next(r for r in ns['terrain_case_summary'] if r['chain']==chain)
    def tr(chain,h):return next(r for r in ta(chain)['steps'] if r['step_scaled']==h)
    angle_previous=ns['angle_report']['previous'];angle_candidate=ns['angle_report']['joint']
    aware_candidate=ns['aware_report']['joint']
    def angle_value(arm,field,name):
        return ns['angle_report'][arm][field][ns['names'].index(name)]
    def angle_rhat(arm,name):
        return ns['angle_report'][arm]['diagnostics']['rhat'][ns['names'].index(name)]
    def angle_ratio(field,name):
        return ns['angle_ratios'][ns['names'].index(name)][field]
    def aware_value(field,name):
        return aware_candidate[field][ns['names'].index(name)]
    def aware_ratio(field,name):
        return ns['aware_ratios'][ns['names'].index(name)][field]
    decision = f'''## Current corrected-target conclusion

**NOT CONVERGED.** The corrected five-chain run has {corrected['rhat_fail']} of 213 globals above R-hat 1.01, {corrected['bulk_fail']} below bulk ESS 400, and {corrected['tail_fail']} below tail ESS 400. Only {corrected['pass_count']} coordinate passes all three criteria: GPS bias northing. Maximum R-hat is {corrected['rhat_max']:.3f} at `{corrected['worst_name']}`; minimum bulk/tail ESS are {corrected['bulk_min']:.2f}/{corrected['tail_min']:.2f}, from 10,000 retained global draws. The second retained half still has {ns['corrected_result']['halves'][1]['rhat_fail']} R-hat failures and a maximum of {ns['corrected_result']['halves'][1]['rhat_max']:.3f}; discarding the first half does not rescue this run.

Transmitter chain means span {corr_value('transmitter_e','chain_mean_span'):.3f}/{corr_value('transmitter_n','chain_mean_span'):.3f}/{corr_value('transmitter_u','chain_mean_span'):.3f} m in east/north/up. Within-chain RMS standard deviations are {corr_value('transmitter_e','within_rms_sd'):.3f}/{corr_value('transmitter_n','within_rms_sd'):.3f}/{corr_value('transmitter_u','within_rms_sd'):.3f} m. These are descriptive chain spreads, not position error bars. Antenna coordinate R-hat ranges from {min(corrected_diag['rhat'][203:206]):.3f} to {max(corrected_diag['rhat'][203:206]):.3f}; none passes. The strongest failures also involve camera 2159 heading/elevation and multiple camera rolls. This is broader than the two ultrawide focal lengths.

All 145 camera/chain combinations accepted updates; camera acceptance ranges {100*min(v['camera_min'] for v in ns['corrected_mechanics'].values()):.2f}%–{100*max(v['camera_max'] for v in ns['corrected_mechanics'].values()):.2f}%. Joint-direction acceptance ranges {100*min(a for v in ns['corrected_mechanics'].values() for a in v['joint']['acceptance']):.2f}%–{100*max(a for v in ns['corrected_mechanics'].values() for a in v['joint']['acceptance']):.2f}%; support rejections are retained in the mechanics table. There were no geometry fallbacks, but {sum(v['geometry_clipped'] for v in ns['corrected_mechanics'].values())} camera-curvature eigenvalue floor applications across recorded warmup refreshes. Those regularization events are not support violations. Movement and acceptance do not imply mixing. Log-density R-hat is {ns['corrected_logp']['rhat'][0]:.4f}, while its bulk ESS is {ns['corrected_logp']['ess_bulk'][0]:.1f}; a nearly agreeing scalar log-density trace conceals badly separated coordinates.

The focal conversion correction is required by the definition, but it has not resolved mixing in this bounded run. The observed height/orientation separation motivates testing collective directions for those coordinates; it does not prove a curved ridge, disconnected modes, distortion, or another physical cause.

## Historical proposal-repair conclusion

**The proposal repair restored movement, but this pilot is not converged.** All {total_pairs} camera/chain pairs changed focal length during the retained window, compared with {old_frozen} entirely frozen pairs in the old pilot. The repaired pilot still has **{new['rhat_above_1_01']} of 213 globals above R-hat 1.01**, **{new['rhat_above_1_1']} above 1.1**, and only **{new['pass_400']} coordinate passing the joint R-hat/bulk-ESS/tail-ESS gate**. The worst rank-normalized R-hat is **{new['rhat_max']:.3f} at {new['worst_rhat_at']}**; minimum bulk and tail ESS are **{new['bulk_ess_min']:.2f} and {new['tail_ess_min']:.2f}**, out of 1,600 retained draws.

The comparable old maximum is {old['rhat_max']:.3f} (rank-normalized), not its previously quoted ordinary split value of {old['ordinary_split_rhat_max']:.2f}. Focal R-hat improved for the four highlighted cameras, but neither those focal parameters nor the joint target positions pass. The failures also include terrain-camera coordinates, so the remaining problem is not confined to the original close-camera focal lengths.

This supports the limited claim that the repaired proposal no longer freezes those focal coordinates in this pilot. It does not establish exploration of the joint camera–landmark–target posterior. The two half-window checks and trace plots remain diagnostics of the same short, nonconverged run. The pilot cannot distinguish insufficient warmup, slow correlated joint directions, disconnected modes or combinations of these causes.

## What the coupling diagnostic establishes

All {2*len(coupling['curves'])} exact evaluations completed in {coupling['seconds']:.1f} seconds; all tested candidates remained within hard support. Residual-density equivalence and translation checks passed, the largest Schur/Jacobian discrepancy was {max(r['relative_error'] for r in coupling['checks']['schur_checks']):.2g}, and no eigenvalues were clipped. Those checks establish the computation's internal consistency, not the accuracy of the quadratic approximation.

- **Transmitter northing:** at ±{coupling['widths']['transmitter_n']['local']:.5f} m, block penalties are {pair('transmitter_n','block_only')} (chains 0 / 4), versus {pair('transmitter_n','globals')} for global co-motion. At ±{.25*coupling['widths']['transmitter_n']['joint']:.4f} m they are {pair('transmitter_n','block_only','quarter_joint_sigma')} versus {pair('transmitter_n','globals','quarter_joint_sigma')}. This supports a transferable local coupling bottleneck. Adding landmark motion is worse than global-only motion at every tested amplitude and endpoint.
- **Camera 2222 northing:** at ±{coupling['widths']['cam2222_n']['local']:.5f} m, landmark co-motion reduces penalties from {pair('cam2222_n','block_only')} to {pair('cam2222_n','globals_landmarks')}. The benefit holds at all tested amplitudes and both endpoints, but at the nominal joint scale exact penalties are {pair('cam2222_n','globals_landmarks','joint_sigma')}, compared with a chain-0 quadratic prediction of 1. The direction helps at the tested points; its nominal scale is too optimistic. The later derivative audit invalidates trusting this shared coarse-difference construction without fresh checks.
- **Cameras 2223 and 2224 northing:** at block-scale displacements, landmark co-motion increases penalties from {pair('cam2223_n','block_only')} to {pair('cam2223_n','globals_landmarks')} for 2223, and from {pair('cam2224_n','block_only')} to {pair('cam2224_n','globals_landmarks')} for 2224. This reversal occurs at every tested amplitude and both endpoints. The later derivative audit below shows that these directions were derived from inaccurate coarse finite differences. Their failure does not reject accurately computed joint directions. At the nominal joint scales, the worst penalty is {max(penalty(c,n,'globals_landmarks','joint_sigma') for c in [0,4] for n in ['cam2223_n','cam2224_n']):.3g}, despite a quadratic prediction of 1 at chain 0.
- **Camera 2198 easting:** allowing global/landmark motion widens the local quadratic scale only from {coupling['widths']['cam2198_e']['local']:.3f} to {coupling['widths']['cam2198_e']['joint']:.3f} m. Global-only penalties barely change; landmark motion worsens them throughout the tested grid. This diagnostic does not explain its slow mixing through the tested coupling.

These are local directional findings at two nonconverged endpoints. They do not establish a common physical degeneracy, a mode count, posterior widths, or improved MCMC mixing.

## Current diagnosis from the full grid

**The float32 target invalidates one saved endpoint.** Camera {ns['support_failure']['camera']} at chain 4 is at height {ns['support_failure']['height_m']:.6f} m, while the new DEM gives {ns['support_failure']['ground_m']:.6f} m there. Its clearance is {ns['support_failure']['clearance_m']:.6f} m; the required clearance is {ns['support_failure']['required_clearance_m']:.3f} m. The state misses support by {-ns['support_failure']['margin_m']:.6f} m. This explains the halt. The completed {len(ns['dem_full']['probes'])} comparisons are preserved byte-for-value; the remaining 12 comparisons are undefined relative to an unsupported baseline. No state was nudged, no support constraint relaxed, and no substitute endpoint selected.

**The failed 2223/2224 directions are dominated by image ties, and the coarse numerical Jacobian is inaccurate.** This attribution holds for both coordinates at both old-target endpoints and at the one supported new-target endpoint. The DEM changes the terrain and horizon terms, but cannot change the identical image-tie terms at fixed states. The full CSV reports every term, displacement and valid endpoint; the float32 chain-4 density comparison is unavailable.

For chain 0's 2223 direction, the original-step image-tie Jacobian predicts squared derivative norm {ns['audit']['states'][0]['coordinate'][0]['norm2']:.3f}; direct small-step differences give {ns['audit']['states'][0]['direct'][3]['norm2']:,.1f}. For 2224 the corresponding values are {ns['audit']['states'][0]['coordinate'][1]['norm2']:.3f} and {ns['audit']['states'][0]['direct'][7]['norm2']:,.1f}. The original calculation therefore manufactured apparently weak directions through inaccurate finite differences. Reducing all coordinate difference steps by 1,000 makes the Jacobian-vector product agree with independent directional differences: worst relative vector error {100*max(ns['fine_errors']):.4f}% across both directions and both endpoints, versus {100*max(ns['coarse_errors']):.2f}% at the original steps. The direct derivative squared norm changes by at most {100*max(ns['direct_stability']):.5f}% between 1e-5 and 1e-6 m steps.

The highest derivative contributions concentrate in observations of landmark 1947, at short projected depths in these saved states (individual worst cases listed above). This establishes sensitivity of the numerical calculation, not a physical identification or justification for deleting that landmark. A Gauss–Newton derivative norm is also not an exact Hessian: accurate derivatives alone do not guarantee a quadratic approximation over a finite proposal.

The earlier Schur algebra checks still pass because they compare reductions of the same inaccurate Jacobian. They did not independently validate differentiation. The new direct-direction test supplies that missing check. The earlier quadratic widths and proposed directions should not be used to select a sampler. The following corrected-geometry test recomputes and validates derivatives on the float32 target.

## Corrected-geometry result

Five of the eight old final states are supported under float32: {ns['supported']}. The comparison uses unchanged chains {ns['fine']['training_chain']} and {ns['fine']['validation_chain']}; the other supported states have not had directional performance tested here.

The initial 1e-3 coordinate-step multiplier has worst full-residual derivative error {100*max(r['relative_vector_error'] for r in ns['initial_fine']['derivative_checks']):.3f}%, failing the 1% gate. At the refined multiplier {ns['fine']['provenance']['step_multiplier']}, the worst error is {100*ns['fine_error']:.4f}% and the gate result is **{ns['fine']['derivative_gate_pass']}**. Reduced-cost checks retain their original tolerance; the maximum discrepancy is {max(r['relative_error'] for r in ns['fine']['geometry_checks']['schur_checks']):.3g}. The reported clipping counts must be considered with the widths; none of these widths are posterior uncertainties.

At ±5 mm, corrected joint penalties for 2223 are {fine_cost(0,'cam2223_n','globals_landmarks',.005)} / {fine_cost(1,'cam2223_n','globals_landmarks',.005)} (chains 0 / 1), versus corrected camera-only penalties {fine_cost(0,'cam2223_n','block_only',.005)} / {fine_cost(1,'cam2223_n','block_only',.005)}. For 2224, joint penalties are {fine_cost(0,'cam2224_n','globals_landmarks',.005)} / {fine_cost(1,'cam2224_n','globals_landmarks',.005)}, versus camera-only {fine_cost(0,'cam2224_n','block_only',.005)} / {fine_cost(1,'cam2224_n','block_only',.005)}.

At ±5 cm on the validation endpoint, joint penalties are {fine_cost(1,'cam2223_n','globals_landmarks',.05)} and {fine_cost(1,'cam2224_n','globals_landmarks',.05)}, versus camera-only {fine_cost(1,'cam2223_n','block_only',.05)} and {fine_cost(1,'cam2224_n','block_only',.05)}. This tests finite-step transfer separately from derivative accuracy; the table must govern any initial proposal scale. All signed exact-density results are above and in the generated CSV. Corrected directions are evidence about local proposal geometry only. Lower finite-displacement penalties do not establish acceptance rates or improved mixing, and transfer to one supported validation endpoint does not establish campaign-wide behavior.

## Current model correction

The EXIF centre must use the diagonal. For these image dimensions the correction raises portrait centres by {100*(ns['new_centres'][ns['portrait']]/ns['old_centres'][ns['portrait']]-1).mean():.3f}% and landscape centres by {100*(ns['new_centres'][~ns['portrait']]/ns['old_centres'][~ns['portrait']]-1).mean():.3f}%. It also changes the horizon angular scale. The corrected helper passes rotation, isotropic-resize, full-frame reference and invalid-metadata checks; both posterior models use it. All 11 conversion tests and seven sampler/target tests pass with explicit branch imports.

In the loose-prior run, the arithmetic mean offset from diagonal EXIF is {ns['focal_audit']['joint_run2_20261004']['groups']['portrait']['diagonal_mean_pct']:.3f}% for portrait and {ns['focal_audit']['joint_run2_20261004']['groups']['landscape']['diagonal_mean_pct']:.3f}% for landscape; the maximum absolute offsets are {ns['focal_audit']['joint_run2_20261004']['groups']['portrait']['diagonal_max_abs_pct']:.3f}% and {ns['focal_audit']['joint_run2_20261004']['groups']['landscape']['diagonal_max_abs_pct']:.3f}%. Camera 2198 is {ns['focal_audit']['joint_run2_20261004']['diagonal_offset_pct'][ns['focal_keys'].index('2198')]:.3f}% and 2203 is {ns['focal_audit']['joint_run2_20261004']['diagonal_offset_pct'][ns['focal_keys'].index('2203')]:.3f}%. These old-target offsets do not establish a distortion defect or validate a 3% prior.

The earlier uniform-3% run used the wrong centres and horizon scale. It does not answer whether a correctly centred tighter prior improves mixing. Those old-target directions were superseded by the independently checked corrected-target directions used in the completed run. Existing draws must not be pooled with corrected-target draws.

## Limitations and deferred findings

- The early pilots have only 200 retained draws per chain; the corrected run has 2,000. ESS estimates in a nonconverged chain are warnings, not calibrated error bars; passing one global coordinate does not validate any derived target uncertainty.
- The early pilots saved only two landmark snapshots per chain and two log-density samples (retained indices 0 and 100); the corrected run saves 200 of each. The early snapshots cannot establish stationarity. Corrected-run log-density diagnostics are reported above; landmark convergence has not been assessed. No convergence is claimed for the {landmark_coordinates:,} landmark coordinates.
- No new goodness-of-fit claim is made. The tabulated between/within spreads and half-window drift are chain residual diagnostics; they do not validate the image likelihood, DEM, fixed lens distortion, or physical identifiability.
- The suspected line-of-sight/focal ridge remains untested. The present review does not identify the physical source of any residual pattern.
- Code, covariance regularization and the focal target measure changed together. Their separate contributions are not identified by this comparison.

## What the height/orientation diagnostic establishes

All {len(ns['height_geo']['derivative_checks'])} direct derivative checks pass the 1% criterion; the worst discrepancy is {100*max(ns['height_derivatives'].values()):.3f}%. The four angular/spatial candidates and two controls were then tested in {len(ns['height_geo']['probes'])} signed pairs at all five current endpoints. There are {sum(not r['both_supported'] for r in ns['height_geo']['probes'])} unsupported pairs; none was omitted or repaired. Runtime was {ns['height_geo']['seconds']/60:.2f} minutes.

**Only transmitter height and transmitter northing pass the all-endpoint penalty/support criterion at both tested sizes.** The passing sets are {', '.join(ns['height_small_pass'])} at the smaller size and {', '.join(ns['height_large_pass'])} at the larger size. The angular candidates do not pass: camera-2159 heading improves {next(r['improved'] for r in ns['height_summary'] if r['coordinate']=='cam2159_ph' and r['amplitude_physical']==.00005)}/5 endpoints at the smaller angular step, camera-2159 elevation improves {next(r['improved'] for r in ns['height_summary'] if r['coordinate']=='cam2159_th' and r['amplitude_physical']==.00005)}/5, and camera-2222 roll improves {next(r['improved'] for r in ns['height_summary'] if r['coordinate']=='cam2222_ti' and r['amplitude_physical']==.00005)}/5. The newly recomputed camera-2223 northing control also fails transfer; its +0.05 m collective step at chain 0 puts camera 2235 below its required ground clearance.

The angle transfer failure is quantitatively visible in the image ties. For camera-2159 heading at chain 1 and ±0.00005 rad, the collective even penalty is {ns['height_lookup'][1,'cam2159_ph','globals_landmarks',.005]['even_penalty']:.6f}, versus {ns['height_lookup'][1,'cam2159_ph','block_only',.005]['even_penalty']:.6f} for the camera block. The collective image-tie contribution alone is {next(r['terms']['tie'] for r in ns['height_angle_terms'] if r['chain']==1 and r['mode']=='globals_landmarks'):.6f}; it is {next(r['terms']['tie'] for r in ns['height_angle_terms'] if r['chain']==0 and r['mode']=='globals_landmarks'):.6f} at the training endpoint. This shows failure of the frozen compensating direction to transfer between these states. It does not identify optical distortion or prove a particular manifold shape. The independent derivative check at chain 0 prevents conflating this result with the earlier coarse-difference defect there; the subsequent local-bank and cell-aware audits below now check derivatives at endpoints 5 and 6.

## Completed paired pilot: no demonstrated mixing benefit

All four workers completed 100 warmup and 200 retained sweeps. The candidate consumed {candidate['cpu_seconds']/60:.2f} sampler CPU-minutes across two workers, versus {previous['cpu_seconds']/60:.2f} for the control ({100*(1-ns['paired_cost_ratio']):.1f}% less in this run). This is comparable measured cost, not a repeatable speed benchmark. The actual launch, command, host supervisor and completion records are checksum-pinned with the chains.

| Metric, same two starts and retained window | Previous frozen kernel | Height/northing kernel |
|---|---:|---:|
| Height RMS net full-sweep increment (m) | {pv('previous','full_sweep_rms'):.5f} | {pv('joint','full_sweep_rms'):.5f} |
| Height squared movement / sampler CPU second (m²/s) | {pv('previous','squared_movement_per_cpu_second'):.3g} | {pv('joint','squared_movement_per_cpu_second'):.3g} |
| Height joint-only RMS displacement per attempted joint move (m) | {previous['height_joint_rms_per_attempt']:.5f} | {candidate['height_joint_rms_per_attempt']:.5f} |
| Absolute difference of height chain means (m) | {pv('previous','chain_mean_span'):.3f} | {pv('joint','chain_mean_span'):.3f} |
| Height R-hat | {pd('previous','rhat'):.3f} | {pd('joint','rhat'):.3f} |
| Height bulk ESS | {pd('previous','ess_bulk'):.2f} | {pd('joint','ess_bulk'):.2f} |
| Worst R-hat over all 213 globals | {previous['summary']['rhat_max']:.3f} | {candidate['summary']['rhat_max']:.3f} |
| Coordinates passing all three criteria | {previous['summary']['pass_count']}/213 | {candidate['summary']['pass_count']}/213 |

**The candidate increases local height movement but does not demonstrate better height mixing.** Height squared movement per CPU second increases by {pr('movement_per_cpu_ratio'):.2f} times, while the height mean gap is {pr('mean_span_ratio'):.2f} times larger and height bulk ESS per CPU is {pr('bulk_ess_per_cpu_ratio'):.3f} times the control. The first/last retained-half mean gaps are {pv('previous','first_half_span'):.3f}/{pv('previous','last_half_span'):.3f} m for the control and {pv('joint','first_half_span'):.3f}/{pv('joint','last_half_span'):.3f} m for the candidate. Both halves retain the height separation; a modest late reduction in the candidate does not establish reunion. These gaps describe chain disagreement, not position estimates or uncertainties.

Northing improves descriptively: mean separation falls from {pv('previous','chain_mean_span','transmitter_n'):.3f} to {pv('joint','chain_mean_span','transmitter_n'):.3f} m, R-hat from {pd('previous','rhat','transmitter_n'):.3f} to {pd('joint','rhat','transmitter_n'):.3f}, and bulk ESS per CPU rises {pr('bulk_ess_per_cpu_ratio','transmitter_n'):.2f} times. Across all coordinates, {ns['paired_ratio_summary']['bulk_ess_per_cpu_ratio']['above_one']}/213 have higher bulk ESS per CPU, while {ns['paired_ratio_summary']['bulk_ess_per_cpu_ratio']['below_one']}/213 have lower values. The worst ratio is {ns['paired_ratio_summary']['bulk_ess_per_cpu_ratio']['minimum']:.3f} at `{ns['paired_ratio_summary']['bulk_ess_per_cpu_ratio']['minimum_name']}`. Antenna-height R-hat worsens from {pd('previous','rhat','antenna_u'):.3f} to {pd('joint','rhat','antenna_u'):.3f}; its chain-mean gap grows from {pv('previous','chain_mean_span','antenna_u'):.3f} to {pv('joint','chain_mean_span','antenna_u'):.3f} m. No broad efficiency improvement is established.

The control/candidate have {previous['summary']['rhat_fail']}/{candidate['summary']['rhat_fail']} R-hat failures, {previous['summary']['bulk_fail']}/{candidate['summary']['bulk_fail']} bulk-ESS failures, and {previous['summary']['tail_fail']}/{candidate['summary']['tail_fail']} tail-ESS failures. Worst R-hat occurs at `{previous['summary']['worst_name']}` / `{candidate['summary']['worst_name']}`. Last-half maximum R-hat remains {previous['halves'][1]['rhat_max']:.3f}/{candidate['halves'][1]['rhat_max']:.3f}. Camera-2159 heading still has R-hat {pd('joint','rhat','cam2159_ph'):.3f}. Retained joint support rejections total {sum(r['joint_support_rejects'] for r in ns['paired_chain_rows'] if r['arm']=='previous'):.0f}/{sum(r['joint_support_rejects'] for r in ns['paired_chain_rows'] if r['arm']=='joint'):.0f} out of 400 attempts per arm. Every worker remains finite, and every camera accepts updates; this does not certify mixing.

The primary mixing objective is unmet despite increased movement at comparable cost. Do not promote this pair to a longer five-chain run on the basis of this experiment. The two-chain pilot does not prove that the candidate is intrinsically worse, identify disconnected modes, or isolate the effect of height from the changed northing vector. Nor can its R-hat be compared directly with the longer five-chain run as an improvement estimate.

## Local recomputation and angle transfer

The full diagnostic completed in {ns['local_geo']['seconds']/60:.2f} minutes with all {len(ns['local_geo']['probes'])} signed pairs and {len(ns['local_geo']['derivative_checks'])} derivative checks. There were {sum(not r['both_supported'] for r in ns['local_geo']['probes'])} unsupported pairs. The smoke passed all its smaller set of checks, but the expanded diagnostic fails {len(ns['local_failures'])}/48 derivative checks; no failures were dropped.

**Recomputing locally improves most tested exact penalties, but it is not yet a validated replacement.** At their own training endpoints, the collective directions beat the common block baseline in {sum(r['improves'] for r in ns['local_rows'] if r['own_endpoint'] and r['amplitude_scaled']==.005)}/8 small-amplitude and {sum(r['improves'] for r in ns['local_rows'] if r['own_endpoint'] and r['amplitude_scaled']==.05)}/8 larger-amplitude cases. The exception is camera-2159 elevation at endpoint 6: small-step even penalty {lv(6,'local_6','cam2159_th',.005,'joint_penalty'):.5f}, versus {lv(6,'local_6','cam2159_th',.005,'common_block_penalty'):.5f} for its block. At the larger amplitude it does improve. The actual signed log-density changes establish these finite-step facts independently of trusting a Jacobian approximation.

**The camera-2159 angular directions fail transfer in both directions at both amplitudes.** For heading, endpoint 5's local direction has a larger-amplitude penalty of {lv(5,'local_5','cam2159_ph',.05,'joint_penalty'):.3f} at home but {lv(6,'local_5','cam2159_ph',.05,'joint_penalty'):.3f} at endpoint 6, against a receiving-state block penalty of {lv(6,'local_5','cam2159_ph',.05,'common_block_penalty'):.3f}. Endpoint 6's heading direction has a home penalty of {lv(6,'local_6','cam2159_ph',.05,'joint_penalty'):.3f} but {lv(5,'local_6','cam2159_ph',.05,'joint_penalty'):.3f} at endpoint 5, against its block penalty of {lv(5,'local_6','cam2159_ph',.05,'common_block_penalty'):.3f}. Image ties contribute {lv(6,'local_5','cam2159_ph',.05,'tie_penalty'):.3f} and {lv(5,'local_6','cam2159_ph',.05,'tie_penalty'):.3f} to those transferred penalties. Transmitter-height moves improve at both endpoints for both new banks and sizes; camera-2222 roll transfers beneficially from 5 to 6 but fails from 6 to 5. This is evidence about these finite moves and states, not proof of posterior topology or an optical cause.

**The original fixed-step derivative checks fail primarily in terrain residuals; the targeted audit below explains the two selected worst cases.** All {len(ns['local_failures'])} failures occur along a new bank at its own training endpoint. Worst discrepancy is {100*ns['local_worst']['relative_vector_error']:.2f}% for `{ns['local_worst']['coordinate']}` at endpoint {ns['local_worst']['chain']}, scaled step {ns['local_worst']['step_scaled']:g} (physical angle {ns['local_worst']['step_physical']:g} rad). Terrain accounts for {100*min(r['terrain_error_fraction'] for r in ns['local_failures']):.2f}%–{100*max(r['terrain_error_fraction'] for r in ns['local_failures']):.2f}% of the squared derivative discrepancy in failing checks. Reducing the directional check step helps several cases, but camera-2159 elevation at endpoint 5 still fails at the smaller step: {100*ld(5,'local_5','cam2159_th',1e-4):.2f}% → {100*ld(5,'local_5','cam2159_th',1e-5):.2f}%. It would be incorrect to say that every locally recomputed direction is now validated, or that simply reducing the check step resolves the problem. The previously passing height/heading-only smoke did not cover this elevation failure or training at endpoint 6.

The exact transfer penalties remain meaningful; their residual-density identities and support checks passed. The targeted audit below establishes local derivative agreement for its two selected directions, and the subsequent all-bank audit checks every saved direction at both endpoints using cell-aware steps. Conversely, a local exact-penalty improvement does not validate the derivative or demonstrate convergence. No sampler or target change follows from this diagnostic. The completed MCMC runs remain nonconverged.

## Terrain audit: the selected local Jacobians agree within their DEM cells

The targeted audit completed in {ns['terrain_audit']['seconds']:.1f} seconds, retaining all {ta(5)['npoints']} terrain landmarks at each of two endpoints and {len(ns['terrain_audit_rows'])} step comparisons. The stored DEM is float32 and the interpolator returns float64 values. The independently reconstructed bilinear height matches the original interpolator within {max(c['base_height_polynomial_max_error_m'] for c in ns['terrain_audit']['cases']):.2g} m. Reconstructing the saved coordinate-difference prediction reproduces it with relative error {max(c['archived_vs_rebuilt_relative_error'] for c in ns['terrain_audit']['cases']):.2g}; independent analytic terrain-derivative errors have norms {ta(5)['archived_vs_analytic_terrain_error_norm']:.3g} / {ta(6)['archived_vs_analytic_terrain_error_norm']:.3g} at endpoints 5 / 6. Neither original coordinate stencil crosses a DEM cell, and neither state has a landmark exactly on a cell edge within the declared numerical tolerance.

**The large directional checks cross cell boundaries and explain the selected failures.** At scaled h = 1e-4, only {tr(5,1e-4)['directional_crossing_count']} / {tr(6,1e-4)['directional_crossing_count']} terrain rows cross cells at endpoints 5 / 6. Across every failed check in these two cases, those rows account for at least {100*min(v['minimum_failure_crossing_fraction'] for v in ns['terrain_case_summary']):.9f}% of squared terrain derivative error. At the largest step, landmark {ts(5)['dominant_point_at_largest_step']} contributes {100*ts(5)['dominant_fraction_at_largest_step']:.2f}% at endpoint 5; landmark {ts(6)['dominant_point_at_largest_step']} contributes {100*ts(6)['dominant_fraction_at_largest_step']:.2f}% at endpoint 6. These are landmark indices in the frozen state, not camera identifiers. The independent fixed-cell control falls below the unchanged 1% full-residual threshold at every tested step in both cases, with maximum discrepancy {100*max(v['max_fixed_cell_control_error'] for v in ns['terrain_case_summary']):.3f}%.

| Selected direction | Original h = 1e-4 discrepancy | Largest tested step with no cell crossing | Discrepancy at that step |
|---|---:|---:|---:|
| Endpoint 5, camera-2159 elevation | {100*tr(5,1e-4)['relative_full_error']:.2f}% | {ts(5)['first_no_crossing_step']:g} | {100*ts(5)['first_no_crossing_error']:.4f}% |
| Endpoint 6, camera-2159 heading | {100*tr(6,1e-4)['relative_full_error']:.2f}% | {ts(6)['first_no_crossing_step']:g} | {100*ts(6)['first_no_crossing_error']:.6f}% |

Endpoint 5's previously failing smaller step, h = 1e-5, still crosses {tr(5,1e-5)['directional_crossing_count']} cell boundary; its {100*tr(5,1e-5)['relative_full_error']:.2f}% discrepancy therefore was not a within-cell check. At h = 3e-6 and 1e-6 its full errors are {100*tr(5,3e-6)['relative_full_error']:.4f}% and {100*tr(5,1e-6)['relative_full_error']:.4f}%. Endpoint 6 is already below the threshold at h = 3e-5 and 1e-5 ({100*tr(6,3e-5)['relative_full_error']:.6f}% / {100*tr(6,1e-5)['relative_full_error']:.6f}%). All tested no-crossing steps pass for both cases, including the smallest steps where numerical discrepancy begins to grow. This supports local derivative accuracy for these two directions rather than a DEM precision change or a new Jacobian construction.

**This resolves the two audited derivative failures, not posterior mixing.** It explains why the previous fixed check sizes were too large for a local derivative test along these coordinated directions. It does not make the finite-step target globally smooth, improve the failed transfer of camera-2159 angle moves, establish convergence, or by itself validate the remaining saved directions; the all-bank validation below addresses that last question. Finite exact-density proposals must still be tested at the intended proposal sizes even after a within-cell derivative check passes.

## All-bank derivative validation passes

The complete saved set passes: **{ns['cell_summary']['checks']-ns['cell_summary']['check_failures']}/{ns['cell_summary']['checks']} derivative checks and {ns['cell_summary']['cases']-ns['cell_summary']['case_failures']}/{ns['cell_summary']['cases']} two-step cases**, in {ns['cell_audit']['seconds']:.1f} seconds. There are {ns['cell_summary']['crossings']} actual DEM-cell crossings, {ns['cell_summary']['geometry_rejections']} edge/precision rejections, and no excluded coordinates or endpoints. Selected scaled steps range from {ns['cell_summary']['min_step']:.3g} to {ns['cell_summary']['max_step']:.3g}; physical units and every selected step are recorded in the CSV and frozen plan.

The worst derivative discrepancy is **{100*ns['cell_worst']['relative_error']:.4f}%**, at receiving endpoint {ns['cell_worst']['chain']}, bank `{ns['cell_worst']['bank']}`, coordinate `{ns['cell_worst']['coordinate']}`. The largest difference between two direct derivative estimates is **{100*ns['cell_summary']['max_stability']:.5f}%**. Both remain below the unchanged 1% thresholds. The {ns['cell_summary']['original_failures']} original fixed-step failures are retained, not relabelled as passes; their smaller geometry-selected checks pass. Derivative accuracy is therefore cleared for these 12 saved directions at these two endpoints, under the stated step rule. This is a local validation, not a claim about every point along a proposed move or the campaign as a whole.

**The remaining obstacle is finite-step transfer and actual mixing, not a demonstrated failure of these saved local derivatives.** The previous exact-density comparisons still show that the camera-angle directions are useful principally near their training states and can be expensive at the other endpoint. The corrected MCMC and paired pilot remain nonconverged. Further narrowing the derivative steps is not supported as the next mixing intervention by these results.

The fixed mixture packet has SHA-256 `{ns['mixture']['geometry_sha256']}` and copies the two control vectors exactly. All 64 signed support checks pass at the two starts and both amplitudes. The twelve angle/endpoint two-step derivative cases are covered by the validated all-bank audit. Four paired smoke workers completed 2 warmup and 3 retained sweeps with finite 213-coordinate outputs and identical initial log density within each paired start. The smoke consumed {sum(r['cpu_seconds'] for r in ns['mixture_smoke_workers']):.1f} CPU-seconds across four workers. These draws are too few to measure mixing or convergence.

The completed paired mixture pilot used the planned 100 warmup and 200 retained sweeps at starts 5/6 in each arm; all four workers completed after {sum(v['cpu_seconds'] for arm in ns['angle_report'].values() for v in arm['completion'])/60:.1f} sampler CPU-minutes. **The mixture did not achieve convergence.** It has {angle_candidate['summary']['rhat_fail']}/213 R-hat failures, {angle_candidate['summary']['bulk_fail']}/213 bulk-ESS failures, {angle_candidate['summary']['tail_fail']}/213 tail-ESS failures and zero coordinates passing all three thresholds. The maximum R-hat is {angle_candidate['summary']['rhat_max']:.3f} at `{angle_candidate['summary']['worst_name']}`, versus {angle_previous['summary']['rhat_max']:.3f} in the height/northing control. The candidate's second retained half still has {angle_candidate['halves'][1]['rhat_fail']} R-hat failures and all 213 bulk/tail ESS failures. This is not an early-warmup artifact that the second half resolves.

The three targeted angle coordinates move {angle_ratio('movement_per_cpu_ratio','cam2159_ph'):.2f}, {angle_ratio('movement_per_cpu_ratio','cam2159_th'):.2f}, and {angle_ratio('movement_per_cpu_ratio','cam2222_ti'):.2f} times farther in squared full-sweep movement per CPU than the control, respectively. Their bulk ESS per CPU ratios are {angle_ratio('bulk_ess_per_cpu_ratio','cam2159_ph'):.2f}, {angle_ratio('bulk_ess_per_cpu_ratio','cam2159_th'):.2f}, and {angle_ratio('bulk_ess_per_cpu_ratio','cam2222_ti'):.2f}; each angle's R-hat worsens. Extra local movement therefore does not demonstrate travel between the separated chains. In the candidate, the chain-mean gaps are {next(r['gap_over_rms_step'] for r in ns['angle_rows'] if r['arm']=='joint' and r['name']=='cam2159_ph'):.1f}, {next(r['gap_over_rms_step'] for r in ns['angle_rows'] if r['arm']=='joint' and r['name']=='cam2159_th'):.1f}, and {next(r['gap_over_rms_step'] for r in ns['angle_rows'] if r['arm']=='joint' and r['name']=='cam2222_ti'):.1f} times the respective root-mean-square full-sweep step; this is a scale comparison, not a diffusion-time estimate. Transmitter height separation narrows from {angle_value('previous','chain_mean_span','transmitter_u'):.3f} to {angle_value('joint','chain_mean_span','transmitter_u'):.3f} m, but its R-hat rises from {angle_rhat('previous','transmitter_u'):.3f} to {angle_rhat('joint','transmitter_u'):.3f}. Northing separation narrows from {angle_value('previous','chain_mean_span','transmitter_n'):.3f} to {angle_value('joint','chain_mean_span','transmitter_n'):.3f} m and its R-hat improves, but still exceeds 1.01. The candidate spends only {angle_candidate['direction_stats']['transmitter_u']['proposals']} and {angle_candidate['direction_stats']['transmitter_n']['proposals']} of 400 joint attempts on the original height and northing vectors, versus {angle_previous['direction_stats']['transmitter_u']['proposals']} and {angle_previous['direction_stats']['transmitter_n']['proposals']} in the control. This is an entire-kernel comparison, not a causal estimate of an individual direction's benefit.

The state-aware selector passed its synthetic correlated-target and exact real-input resume tests. Its paired smoke and full control chains are bit-for-bit identical to the uniform experiment's controls. The archived target code prefix is byte-identical; the only changed model file section is the sampler. In the full run, chain 5 chose its local angle bank {next(r['local_bank_proposals'] for r in ns['aware_chain_rows'] if r['arm']=='joint' and r['chain']==5)} times versus {next(r['remote_bank_proposals'] for r in ns['aware_chain_rows'] if r['arm']=='joint' and r['chain']==5)} remote selections; chain 6 chose its local bank {next(r['local_bank_proposals'] for r in ns['aware_chain_rows'] if r['arm']=='joint' and r['chain']==6)} times versus {next(r['remote_bank_proposals'] for r in ns['aware_chain_rows'] if r['arm']=='joint' and r['chain']==6)} remote selections. The selector therefore did change proposal frequency as intended.

**State-aware selection still does not converge.** It has {aware_candidate['summary']['rhat_fail']}/213 R-hat failures, {aware_candidate['summary']['bulk_fail']}/213 bulk-ESS failures, {aware_candidate['summary']['tail_fail']}/213 tail-ESS failures and zero coordinates passing all three. Maximum R-hat is {aware_candidate['summary']['rhat_max']:.3f} at `{aware_candidate['summary']['worst_name']}`; the second half has {aware_candidate['halves'][1]['rhat_fail']} R-hat failures and all 213 bulk/tail ESS failures. Relative to the uniform mixture, squared full-sweep movement per CPU rises by {aware_ratio('movement_per_cpu_vs_uniform','cam2159_ph'):.2f}, {aware_ratio('movement_per_cpu_vs_uniform','cam2159_th'):.2f}, and {aware_ratio('movement_per_cpu_vs_uniform','cam2222_ti'):.2f} times for the three targeted angles. Their between-chain mean gaps are {aware_value('chain_mean_span','cam2159_ph'):.6f}, {aware_value('chain_mean_span','cam2159_th'):.6f}, and {aware_value('chain_mean_span','cam2222_ti'):.6f} rad, versus {angle_value('joint','chain_mean_span','cam2159_ph'):.6f}, {angle_value('joint','chain_mean_span','cam2159_th'):.6f}, and {angle_value('joint','chain_mean_span','cam2222_ti'):.6f} rad under uniform selection. None closes; the latter two widen. Northing separation reopens from {angle_value('joint','chain_mean_span','transmitter_n'):.3f} to {aware_value('chain_mean_span','transmitter_n'):.3f} m. The better angle R-hat and screening ESS per CPU do not outweigh persistent chain disagreement and all-coordinate failures. CPU-normalized comparisons to the uniform run are interpreted alongside the within-run control because wall-time throughput differed between runs.

## Decision requested

Do not extend these two-chain pilots or release posterior geometry uncertainty. The exact state-aware branch selector improved local angle movement but did not bring the chains together; proposal frequency alone is not the missing fix. The next bounded diagnostic should isolate the camera-2159 angle/transmitter-height subproblem and test an exact-density bridge between the separated starts while allowing coupled landmarks to relax. A bridge whose profiled exact density remains sharply suppressed would support a target-geometry barrier; a supported, low-penalty bridge would instead point toward insufficient or misaligned proposals. Report image-tie, terrain, horizon and prior terms separately, and do not infer a physical cause from the trace pattern alone. This is a diagnostic proposal, not an approved posterior or production result.

## Revision history

- 2026-10-06: Exact state-aware bank selection favored each start's local directions and raised targeted angle movement, but no coordinate converged and chain gaps persisted; proposal frequency alone is insufficient.
- 2026-10-06: The paired local-angle mixture raises angle movement but not effective cross-chain mixing; zero coordinates pass and the second half still fails, so a longer run is not justified.
- 2026-10-06: Froze and smoke-tested the fixed eight-direction local-angle mixture, with paired starts, signed support and source/target identity verified; requests the bounded pilot.
- 2026-10-06: All 48 cell-aware checks and all 24 two-step cases pass; local derivative accuracy is cleared for the saved banks while finite-step transfer and mixing remain unresolved.
- 2026-10-05: Independent bilinear derivatives and landmark-level controls explain both selected failures as check steps crossing DEM cells; smaller within-cell checks pass without altering the target.
- 2026-10-05: Local recomputation lowers most home-endpoint penalties but fails angular transfer and five derivative checks; terrain residuals dominate those mismatches.
- 2026-10-05: Completed the paired comparison: height movement per CPU improves, but height-chain separation grows and neither arm passes convergence; no longer run is supported.
- 2026-10-05: Prepared and smoke-tested the approved paired previous-kernel versus height/northing pilot at the two height-separated current endpoints.
- 2026-10-05: Current-endpoint transfer checks support transmitter height/northing moves but reject the frozen camera-angle candidates; image ties explain the tested angle penalty increase.
- 2026-10-05: Completed corrected-target convergence review rejects all geometry coordinates despite active proposals; transmitter height and camera orientations remain separated across chains.
- 2026-10-04: Approved mixed-width corrected-target preparation passes analytic target checks, recomputed derivative/transfer checks and all five smoke starts.
- 2026-10-04: Independently confirmed the diagonal EXIF definition, corrected all three implementations, and audited all cameras in the loose/tight five-chain runs using arithmetic means and worst cases; replaces the pending old-target pilot launch decision.
- 2026-10-03: Implemented the frozen-direction kernel, passed seven tests including real exact resume, and completed the paired-driver smoke before a bounded float32 pilot.
- 2026-10-03: Screened float32 endpoint support and recomputed candidate geometry using stable residual projection, refined differences, independent full-residual checks and exact-density transfer tests.
- 2026-10-03: Recovered the interrupted grid, identified camera 2232 outside float32 support, and independently established severe finite-difference error in the 2223/2224 image-tie Jacobian; supersedes treating their failed directions as evidence against joint moves.
- 2026-10-03: Froze float32 DEM inputs and attributed the 2223 chain-0 smoke penalty to image ties; the DEM replacement does not rescue this frozen direction.
- 2026-10-03: Exact coupling checks at two endpoints support transmitter global co-motion and 2222 landmark co-motion, but overturn the generic joint-direction hypothesis for 2223/2224; updated next diagnostic request.
- 2026-10-02: The complete eight-chain repaired pilot shows movement restored but joint convergence still fails; replaced the earlier smoke-only checkpoint and request to launch this now-completed pilot.
- 2026-10-01–02: Identified the raw-f/log-f inconsistency, implemented scaled residual-based proposals, and validated mechanics with synthetic tests and the frozen-input smoke run.
'''
    # One compact ledger keeps the pilot sequence and its changing targets visible.
    pilot_ledger = [
        ('2026-10-01', 'Original raw-f pilot', 'int32 DEM, width EXIF; 8 × (300+200)', ns['reports']['pilot']['summary']['pass_400'], ns['reports']['pilot']['summary']['rhat_above_1_01'], 'Focal coordinates froze; raw-f proposal and log-f target measure disagreed.'),
        ('2026-10-02', 'Log-f repair pilot', 'int32 DEM, width EXIF; 8 × (300+200)', ns['reports']['pilot_logf_20261002']['summary']['pass_400'], ns['reports']['pilot_logf_20261002']['summary']['rhat_above_1_01'], 'All cameras moved; transmitter and coupled geometry stayed separated.'),
        ('2026-10-03', 'Float32 paired collective pilot', 'float32 DEM, width EXIF; 2 starts × 2 arms, 200 retained', 0, None, 'One collective network direction reduced camera separation; neither arm converged.'),
        ('2026-10-03', 'Float32 five-chain run', 'float32 DEM, width EXIF; 5 × (300+2000)', 3, None, 'Transmitter position and two transmitter-era cameras remained separated.'),
        ('2026-10-04', 'Loose focal-prior five-chain run', 'float32 DEM, width EXIF; 5 × (300+2000)', ns['corrected_reports']['joint_run2_20261004']['summary']['pass_count'], ns['corrected_reports']['joint_run2_20261004']['summary']['rhat_fail'], 'No global passed; it exposed the width-only EXIF conversion error.'),
        ('2026-10-04', '3% focal-prior five-chain run', 'float32 DEM, width EXIF; 5 × (300+2000)', ns['corrected_reports']['joint_run3_20261004']['summary']['pass_count'], ns['corrected_reports']['joint_run3_20261004']['summary']['rhat_fail'], 'Narrow prior tested the wrong focal centre; no lens-physics inference.'),
        ('2026-10-04', 'Corrected EXIF mixed-prior run', 'float32 DEM, diagonal EXIF; 5 × (300+2000)', corrected['pass_count'], corrected['rhat_fail'], 'Only GPS bias northing passed; height and orientations stayed separated.'),
        ('2026-10-05', 'Paired height/northing pilot', 'corrected target; 2 starts × 2 arms, 100+200', paired['joint']['summary']['pass_count'], paired['joint']['summary']['rhat_fail'], 'Height movement per CPU rose, but height-chain separation grew.'),
        ('2026-10-06', 'Uniform eight-direction pilot', 'corrected target; 2 starts × 2 arms, 100+200', ns['angle_report']['joint']['summary']['pass_count'], ns['angle_report']['joint']['summary']['rhat_fail'], 'Angle movement rose without effective cross-chain mixing.'),
        ('2026-10-06', 'State-aware bank pilot', 'corrected target; 2 starts × 2 arms, 100+200', ns['aware_report']['joint']['summary']['pass_count'], ns['aware_report']['joint']['summary']['rhat_fail'], 'Local bank use rose as intended; all-coordinate convergence still failed.'),
    ]
    ledger_lines = ['## Pilot-study ledger and disposition', '',
        'Each row is a separate target or kernel study. Counts are globals passing all three marginal criteria and globals failing rank R-hat 1.01, out of 213. A dash means the older report did not retain that exact count. Short paired studies screen proposals; they cannot certify a posterior. The sections above give the input checksums, methods, full coordinate diagnostics and limits.', '',
        '| Date | Study | Scope | Pass / 213 | R-hat fail / 213 | Result |',
        '|---|---|---|---:|---:|---|']
    for date, name, scope, passed, failed, finding in pilot_ledger:
        ledger_lines.append(f'| {date} | {name} | {scope} | {passed} | {failed if failed is not None else "—"} | {finding} |')
    ledger_lines += ['', 'Smoke runs, derivative audits and finite-step support probes were implementation or geometry checks rather than sampling pilots; their outcomes and rejected cases remain documented above.', '',
        '## Current disposition', '',
        f'The full v0003 run completed with {ns["v3_summary"]["rhat_fail"]}/213 R-hat failures, {ns["v3_summary"]["bulk_fail"]} bulk-ESS failures and {ns["v3_summary"]["tail_fail"]} tail-ESS failures. Its second retained half is also nonconverged. The highest-density complete state saved by this run is chain {ns["v3_report"]["selected"]["chain"]}, retained sweep {ns["v3_report"]["selected"]["retained_sweep"]}, at exact log density {ns["v3_report"]["selected"]["exact_log_density"]:.3f}. That is {ns["v3_density_delta"]:.3f} below the best saved complete state in the preceding corrected-target run. Version v0003 is therefore the latest **provisional point geometry**, not a demonstrably better fit and not a posterior estimate. Do not report chain means or posterior widths as geometry uncertainty. The model and proposal-mixing problems remain open; stop proposal-tuning pilots for now.']
    nb.cells.append(nbformat.v4.new_markdown_cell('\n'.join(ledger_lines)))
    decision = decision.replace('## Decision requested\n\n'+decision.split('## Decision requested\n\n',1)[1].split('\n\n## Revision history',1)[0], '')
    decision = decision.replace('## Revision history\n\n', '## Revision history\n\n- 2026-10-07: The five-chain v0003 run completed without convergence; its saved point fit is provisional and has lower exact log density than the best saved predecessor state on the same target.\n- 2026-10-06: Closed proposal-tuning pilots, consolidated all sampling studies, and authorized a provisional full point-fit run from the latest saved states despite expected convergence failure.\n')
    nb.cells.append(nbformat.v4.new_markdown_cell(decision))
    nb.cells.insert(1, nbformat.v4.new_markdown_cell(
        f'**Current result: NOT CONVERGED.** The completed v0003 corrected-target run has '
        f'{ns["v3_summary"]["rhat_fail"]}/213 globals above R-hat 1.01, a maximum of {ns["v3_summary"]["rhat_max"]:.3f} '
        f'at `{ns["v3_report"]["full"]["worst_name"]}`, and minimum bulk ESS {ns["v3_summary"]["bulk_min"]:.2f}. '
        f'The v0003 geometry is a provisional point fit only. Its selected log density is {ns["v3_density_delta"]:.3f} below the best saved state in the preceding corrected-target run; neither result supports posterior uncertainty. The pilot-study ledger and complete diagnostics follow.'))
    def serializable(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(type(value).__name__)
    report = dict(provenance=ns['provenance'], thresholds=dict(rhat_max=1.01, ess_min=400, secondary_ess_min=800),
                  state_aware_angle_result=dict(status=ns['aware_status'],smoke=ns['aware_smoke'],names=ns['names'],arms=ns['aware_report'],chains=ns['aware_chain_rows'],ratios=ns['aware_ratios'],conclusion='Local bank selected more often and angle movement improves, but no coordinate passes convergence and chain gaps persist'),
                  local_angle_mixture_result=dict(status=ns['angle_status'],names=ns['names'],arms=ns['angle_report'],chains=ns['angle_chain_rows'],ratios=ns['angle_ratios'],ratio_summary=ns['angle_ratio_summary'],conclusion='Angle movement rises but no coordinate passes convergence; second half does not rescue the result'),
                  local_angle_mixture_preparation=dict(geometry=ns['mixture'],smoke=ns['mixture_smoke'],workers=ns['mixture_smoke_workers'],conclusion='Four paired smoke workers complete; no mixing inference from three retained draws'),
                  cell_aware_derivatives=dict(diagnostic=ns['cell_audit'],summary=ns['cell_summary'],cases=ns['cell_cases'],rows=ns['cell_rows'],conclusion='All saved directions pass local derivative and two-step checks at both endpoints; finite-step transfer and mixing remain unresolved'),
                  terrain_derivative_audit=dict(diagnostic=ns['terrain_audit'],summary=ns['terrain_case_summary'],rows=ns['terrain_audit_rows'],conclusion='Both selected failures are explained by directional check steps crossing DEM cells; independent within-cell derivatives pass without changing the target'),
                  local_geometry_transfer=dict(diagnostic=ns['local_geo'],smoke=ns['local_smoke'],comparisons=ns['local_rows'],summary=ns['local_summary'],derivative_rows=ns['local_derivative_rows'],derivative_failures=ns['local_failures'],conclusion='Angular transfer fails despite local finite-step gains; five original fixed-step checks fail, with the targeted terrain audit explaining its two selected cases'),
                  paired_height_north_result=dict(status=ns['paired_status'],names=ns['names'],arms=paired,chains=ns['paired_chain_rows'],ratios=ns['paired_ratios'],ratio_summary=ns['paired_ratio_summary'],cost_scope='Sampler CPU summed across workers, includes proposal initialization and warmup; excludes model loading and supervisor overhead',conclusion='Local height movement improves, but height mixing benefit is not demonstrated; no longer run supported'),
                  height_north_pilot_preparation=dict(status=ns['height_pilot_smoke'],workers=ns['height_pilot_workers']), height_orientation_diagnostic=dict(geometry=ns['height_geo'],summary=ns['height_summary'],derivatives=ns['height_derivatives'],small_pass=ns['height_small_pass'],large_pass=ns['height_large_pass'],angle_terms=ns['height_angle_terms']), corrected_run=dict(status=ns['corrected_status'],names=ns['names'],comparisons=ns['corrected_reports'],mechanics=ns['corrected_mechanics'],logp=ns['corrected_logp']), corrected_preparation=dict(geometry=ns['mixed_geometry'],smoke=ns['mixed_smoke'],workers=ns['mixed_smoke_workers']), focal_conversion_audit=ns['focal_audit'], joint_kernel_tests=ns['tests'], joint_pilot_smoke=dict(status=ns['pilot_smoke'],workers=ns['smoke_workers']), fine_geometry=ns['fine_runs'], dem_full=ns['dem_full'], derivative_audit=ns['audit'], dem_smoke=ns['dem_smoke'], coupling=coupling, reports=ns['reports'], camera_tables=ns['camera_tables'], mechanics=ns['mechanics'],
                  pilot_ledger=[dict(date=date,name=name,scope=scope,pass_count=passed,rhat_fail=failed,result=finding) for date,name,scope,passed,failed,finding in pilot_ledger],
                  v0003_full_run=dict(manifest=ns['v3_manifest'],report=ns['v3_report'],status=ns['v3_status'],recomputed=ns['v3_summary'],retained_halves=ns['v3_halves'],previous_best_saved_logp=ns['v3_previous_best'],density_delta_from_previous=ns['v3_density_delta']),
                  verdict='NOT CONVERGED; v0003 is a provisional point fit, not a posterior estimate or demonstrably better exact-density fit')
    (code/'marjum_mcmc_b21_review.json').write_text(json.dumps(report,indent=2,default=serializable,allow_nan=False)+'\n')
    nbformat.validate(nb)
    nbformat.write(nb, code/(TITLE+'.ipynb'))
    html, _ = HTMLExporter().from_notebook_node(nb)
    (code/(TITLE+'.html')).write_text(html)
    print(decision)
    print('Wrote notebook, HTML, JSON, CSVs and figures in',code)


if __name__ == '__main__':
    main()
