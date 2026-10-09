"""Check a joint transmitter candidate before making it an inspector default."""
import argparse,json
from pathlib import Path
import numpy as np
from marjum_add_views_validate import validate
from marjum_transmitter_joint_polish import FREE
from marjum_mcmc import digest

def assess(output='cv_transmitter_joint_v3'):
    out=Path(output);report=json.loads((out/'report.json').read_text())
    source=Path(report['source']);old=dict(np.load(source));new=dict(np.load(out/'fit_transmitter.npz'))
    checks={}
    checks['source_unmodified']=digest(source)==report['input_sha256'][str(source)]
    checks['transmitter_matches_report']=np.array_equal(new['transmitter'],report['transmitter'])
    mask=np.array([str(k) not in FREE for k in old['keys']])
    checks['fixed_cameras_unchanged']=all(old[k][mask].tobytes()==new[k][mask].tobytes() for k in ['cameras','distortion'])
    checks['other_geometry_unchanged']=all(np.array_equal(old[k],new[k]) for k in ['keys','shapes','groups','antenna','points','obs_cam','obs_point','obs_xy','obs_fid'])
    full=validate(output,str(source),targets=list(FREE),overwrite=True)
    baseline=json.loads(Path('cv_transmitter_joint_v2/baseline/validation/report.json').read_text())
    before={r['key']:r for r in baseline['images']}
    comparison={}
    for row in full['images']:
        key=row['key'];b=before[key]
        checks[key+'_horizon_preserved']=row['raster_horizon_rms_px']<=b['raster_horizon_rms_px']+5
        checks[key+'_sky_probability_preserved']=row['tree_adjusted_sky_nll']<=b['tree_adjusted_sky_nll']*1.15+.5
        checks[key+'_above_dem']=row['camera_clearance_m']>=0
        checks[key+'_monotonic_distortion']=row['radial_min_derivative']>=.29
        comparison[key]=dict(before_rms_px=b['raster_horizon_rms_px'],after_rms_px=row['raster_horizon_rms_px'],
                             before_sky_nll=b['tree_adjusted_sky_nll'],after_sky_nll=row['tree_adjusted_sky_nll'])
    for key,row in report['transmitter_residuals'].items():
        checks[key+'_transmitter_reprojection']=row['depth_m']>0 and row['error_px']<(10 if key in FREE else 70)
    for pair in report['pairs']:
        checks[pair['a']+'_'+pair['b']+'_heldout_match']=pair['heldout_median_px']<5
    checks['optimizer_converged']=report['optimizer']['success']
    result=dict(accepted=all(checks.values()),checks=checks,comparison=comparison,
                note='Acceptance checks are diagnostics, not a posterior uncertainty estimate.')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='cv_transmitter_joint_v3');a=p.parse_args();assess(a.output)
