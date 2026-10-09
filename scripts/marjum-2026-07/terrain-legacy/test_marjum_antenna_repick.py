import numpy as np

from marjum_antenna_repick import solve_position
from marjum_camera import project


def test_camera_fixed_triangulation_recovers_position():
    cameras=np.array([
        [-10.,0.,0.,0.,0.,0.,1000.],
        [10.,0.,0.,0.,0.,0.,1000.],
        [0.,-10.,0.,0.,0.,0.,1000.],
        [0.,10.,0.,0.,0.,0.,1000.],
    ])
    shape=np.array([[1000,1200]]*len(cameras));distortion=np.zeros((len(cameras),2))
    truth=np.array([1.,2.,100.])
    observed=np.array([project(p,s,truth,k)[0][0]
                       for p,s,k in zip(cameras,shape,distortion)])
    state=dict(keys=np.array(["a","b","c","d"]),cameras=cameras,
               shapes=shape,distortion=distortion)
    fit=solve_position(state,observed,truth+[5.,-4.,8.])
    np.testing.assert_allclose(fit.x,truth,atol=1e-7)
