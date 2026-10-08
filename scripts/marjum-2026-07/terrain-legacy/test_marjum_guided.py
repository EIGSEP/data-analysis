import numpy as np
from scipy.optimize import least_squares
from marjum_bundle import project as pinhole_project, rays as pinhole_rays
from marjum_camera import project, rays, radial_support, epipolar_error
from marjum_guided import Fit, reserve, triangulate, graph, patch_score


def test_zero_distortion_agrees_with_library_convention():
    p=np.array([20.,30.,10.,1.4,.2,.1,1500.]);shape=(3024,4032)
    xy=np.array([[100.,200.],[2000.,1500.],[3900.,2900.]])
    np.testing.assert_allclose(rays(p,shape,xy),pinhole_rays(p,shape,xy),atol=1e-14)
    xyz=p[:3]+rays(p,shape,xy)*300
    np.testing.assert_allclose(project(p,shape,xyz)[0],pinhole_project(p,shape,xyz)[0],atol=1e-10)


def test_distorted_round_trip_and_crop_invariance():
    p=np.array([20.,30.,10.,1.4,.2,.1,1400.]);shape=(3024,4032);k=np.array([.04,-.003])
    rng=np.random.default_rng(7);xy=rng.uniform([0,0],[4031,3023],(100,2))
    direction=rays(p,shape,xy,k)
    np.testing.assert_allclose(project(p,shape,p[:3]+direction*300,k)[0],xy,atol=1e-8)
    assert radial_support(p,shape,k).min()>.2
    cropped=np.array([2000,3000]);zoom=1.3;p2=p.copy();p2[6]*=zoom
    xy2=(xy-np.array(shape[::-1])//2)*zoom+cropped[::-1]//2
    np.testing.assert_allclose(rays(p2,cropped,xy2,k),direction,atol=1e-10)


def test_distortion_recovery_and_epipolar_error():
    p=np.array([0.,0.,0.,np.pi/2,0.,0.,1800.]);q=p.copy();q[1]=20
    shape=(3024,4032);rng=np.random.default_rng(3)
    xy=rng.uniform([200,200],[3800,2800],(80,2));k=np.array([.045,-.006])
    xyz=p[:3]+rays(p,shape,xy,k)*rng.uniform(80,300,(80,1))
    measured=project(p,shape,xyz,k)[0]
    result=least_squares(lambda d:(project(p,shape,xyz,d)[0]-measured).ravel(),[0.,0.])
    np.testing.assert_allclose(result.x,k,atol=1e-8)
    other=project(q,shape,xyz,k)[0]
    assert np.max(epipolar_error(p,shape,k,q,shape,k,measured,other))<1e-8
    assert np.median(epipolar_error(p,shape,[0,0],q,shape,[0,0],measured,other))>.1
    point=triangulate(np.array([p,q]),[shape,shape],np.array([k,k]),[0,1],[measured[0],other[0]])
    np.testing.assert_allclose(point,xyz[0],atol=1e-8)


class FlatTerrain:
    e=np.array([-10000.,10000.]);n=e
    def height(self,e,n):return np.zeros(np.broadcast(e,n).shape)+1700.


def test_horizon_free_fit_sparse_dependencies():
    keys=['2213','2216'];shape=(3024,4032)
    cams=np.array([[1750.,2070.,1750.,np.pi/2,0,0,2000.],[1750.,2090.,1750.,np.pi/2,0,0,2000.]])
    points=np.array([[2000.,2070.,1730.],[1900.,2120.,1710.],[2100.,2040.,1705.]])
    oc=np.repeat([0,1],3);op=np.tile(np.arange(3),2)
    xy=np.concatenate([project(cams[i],shape,points)[0] for i in range(2)])
    obs=dict(points=points,oc=oc,op=op,xy=xy)
    # No horizon or antenna fields are supplied: the fit must not need them.
    model=Fit(keys,{k:dict(shape=shape) for k in keys},cams,np.zeros((2,2)),[0,1],obs,FlatTerrain())
    z=model.x0.copy();r=model.residuals(z)
    assert model.sparsity.shape==(len(r),len(z))
    for j in [0,4,6,14,16,18,len(z)-1]:
        q=z.copy();q[j]+=.0001;delta=model.residuals(q)-r
        mask=model.sparsity[:,j].toarray().ravel().astype(bool)
        np.testing.assert_allclose(delta[~mask],0,atol=1e-10)
    assert graph(keys,obs)==[keys]


def test_patch_score_identity():
    p=np.array([1750.,2070.,1750.,np.pi/2,0.,0.,100.]);q=p.copy();q[1]+=10
    # Texture on a plane normal to the view; use a front-facing flat terrain normal
    # by orienting cameras downward so DEM tangent-plane intersections are valid.
    p[3]=q[3]=np.pi
    shape=(200,200);rng=np.random.default_rng(4)
    gray=rng.uniform(0,255,shape).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    gray=gaussian_filter(gray,1.)
    xyz=np.array([1750.,2070.,1700.]);a=project(p,shape,xyz)[0][0];b=project(q,shape,xyz)[0][0]
    # Camera translation here shifts the projected texture by a known pixel offset.
    shift=b-a
    yy,xx=np.indices(shape,dtype=float)
    from scipy.ndimage import map_coordinates
    gray2=map_coordinates(gray,[yy-shift[1],xx-shift[0]],order=1,mode='nearest')
    score,delta=patch_score(p,shape,[0,0],q,shape,[0,0],a,b,gray,gray2,FlatTerrain(),True)
    assert score>.99
    np.testing.assert_allclose(delta,0,atol=.01)


def test_saved_experiment_has_no_reserved_training_features():
    from pathlib import Path
    import pytest
    root=Path('cv_distortion_guided_v2')
    if not (root/'refined.npz').exists():pytest.skip('Local experiment not run')
    with np.load(root/'reservations.npz') as reserved, np.load(root/'holdout.npz') as held:
        for stage in ['initial','pinhole','radial','guided','refined']:
            with np.load(root/f'{stage}.npz') as state:
                for i,key in enumerate(state['keys']):
                    fid=state['obs_fid'][state['obs_cam']==i]
                    assert reserved[f'train_{key}'][fid].all()
                    assert not reserved[f'test_{key}'][fid].any()
        for pair in held.files:
            a,b=pair.split('_');ids=held[pair]
            assert reserved[f'test_{a}'][ids[:,0]].all()
            assert reserved[f'test_{b}'][ids[:,1]].all()
