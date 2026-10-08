import json
import numpy as np
import pytest
from marjum_camera import project
from marjum_transmitter import fit_transmitter


def test_conditioned_rays_cannot_move_transmitter(tmp_path):
    cameras=np.array([[0,-4,2,np.pi/2,0,0,1000],
                      [0,4,2,np.pi/2,0,0,1000],
                      [8,-1,2,np.pi/2,0,0,1000]],float)
    keys=['2210','2211','2159'];shapes=np.tile([2000,2000],(3,1))
    truth=np.array([10.,0,2]);distortion=np.zeros((3,2))
    source=tmp_path/'camera.npz';meta=tmp_path/'meta.json';out=tmp_path/'fit.npz'
    np.savez(source,keys=keys,cameras=cameras,shapes=shapes,distortion=distortion)
    picks={k:{'transmitter_px':project(p,s,truth)[0][0].tolist()} for k,p,s in zip(keys,cameras,shapes)}
    picks['2159']['transmitter_px']=[50.,50.]
    meta.write_text(json.dumps(picks))
    result=fit_transmitter(source,out,meta)
    np.testing.assert_allclose(result['transmitter'],truth,atol=1e-7)
    with np.load(out) as saved:
        np.testing.assert_array_equal(saved['cameras'],cameras)
        np.testing.assert_array_equal(saved['transmitter_fit_keys'],keys[:2])
    assert not result['residuals'][2]['used_in_fit']
    with pytest.raises(ValueError,match='At least two'):
        fit_transmitter(source,out,meta,independent_keys=('2210',))
