"""Read-only model checks; write a small diagnostic report into a pilot folder."""
import argparse
import json
from pathlib import Path
import numpy as np
from marjum_mcmc import Config, Posterior, digest
from marjum_bundle import project, rays


def validate(output):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    changed = [k for k,v in manifest['input_sha256'].items()
               if not Path(k).exists() or digest(k) != v]
    if changed:
        raise ValueError(f'Changed posterior inputs: {changed}')
    model = Posterior(config=Config(**manifest['config']))
    # Read one coherent joint sample, including its own nuisance landmarks.
    candidates = []
    for file in sorted(output.glob('chain_[0-9]*.npz')):
        with np.load(file) as data:
            i = int(np.argmax(data['stats'][:,0]))
            candidates.append((float(data['stats'][i,0]),
                               np.r_[data['global_samples'][i],data['landmarks'][i].ravel()]))
    value = max(candidates,key=lambda item:item[0])[1]
    z = (value-model.origin)/model.scale
    cam,ant,bias,extra,points = model.unpack(z)
    report = dict(logp=model.logp(z),antenna_enu=ant.tolist(),
                  antenna_clearance_m=float(ant[2]-model.terrain.height(*ant[:2])),
                  antenna_extra_px=float(extra),images={},
                  note='Single diagnostic sample, not a converged estimate or uncertainty.')
    deltas = []
    for i,key in enumerate(model.keys):
        d = rays(cam[i],model.shapes[i],model.horizon[i])
        azimuth = np.arctan2(d[:,1],d[:,0])
        low = model.terrain.skyline(cam[i,:3],azimuth,model.config.skyline_samples)
        high = model.terrain.skyline(cam[i,:3],azimuth,3072)
        delta = (high-low)*model.focal[i]
        deltas.extend(delta)
        item = dict(camera_clearance_m=float(cam[i,2]-model.terrain.height(*cam[i,:2])),
                    skyline_resolution_rms_equiv_px=float(np.sqrt(np.mean(delta**2))),
                    skyline_resolution_max_equiv_px=float(np.max(abs(delta))))
        if i in model.ai:
            measured = model.axy[np.flatnonzero(model.ai==i)[0]]
            predicted = project(cam[i],model.shapes[i],ant)[0][0]
            item['antenna_residual_px'] = float(np.linalg.norm(predicted-measured))
        report['images'][key] = item
    report['skyline_resolution_rms_equiv_px'] = float(np.sqrt(np.mean(np.array(deltas)**2)))
    report['skyline_resolution_max_equiv_px'] = float(np.max(abs(np.array(deltas))))
    near = []
    for j,point in enumerate(points):
        obs = np.unique(model.oc[model.op==j])
        distance = np.linalg.norm(point-cam[obs,:3],axis=1)
        if distance.min() < 5:
            near.append(dict(landmark=j,observing_images=[model.keys[i] for i in obs],
                             distances_m=distance.tolist()))
    report['landmarks_with_observing_camera_within_5m'] = near
    path = output/'model_checks.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='mcmc_parallax_pilot_v2')
    validate(**vars(parser.parse_args()))
