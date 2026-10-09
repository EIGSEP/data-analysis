import numpy as np
from types import SimpleNamespace
from marjum_bundle import rays, project, Terrain, build_tracks, boundary_pixels


def test_camera_matches_library():
    from eigsep_terrain.img import HorizonImage, PRM_ORDER
    from eigsep_terrain.tiepoints import project_world_point
    p = np.array([1700, 2050, 1750, 1.4, -2.8, .07, 3500.])
    shape = (4032, 3024)
    xy = np.array([[10., 100.], [2012., 3300.], [1500., 2000.]])
    img = HorizonImage.__new__(HorizonImage)
    img.img = np.empty((*shape, 0))
    img.prms = dict(zip(PRM_ORDER, p))
    direction = rays(p, shape, xy)
    np.testing.assert_allclose(direction, img.get_rays(xy[:, ::-1].T, dtype=np.float64).T, atol=1e-12)
    xyz = p[:3] + 100 * direction
    recovered, depth = project(p, shape, xyz)
    np.testing.assert_allclose(recovered, xy, atol=1e-9)
    for point, q in zip(xyz, xy):
        np.testing.assert_allclose(project_world_point(point, img)[::-1], q, atol=1e-9)
    assert np.all(depth > 0)
    _, z = project(p, shape, p[:3] - 100*direction)
    assert np.all(z < 0)


def test_terrain_interpolation_is_continuous_and_origin_correct():
    e, n = np.arange(-10., 11.), np.arange(20., 41.)
    data = 100 + 2*e[None] + 3*n[:, None]
    t = Terrain(SimpleNamespace(data=data.astype(np.int32), res=1., get_en=lambda:(e,n)))
    np.testing.assert_allclose(t.height([.25, 1.5], [22.25, 23.5]), [167.25, 173.5])
    derivative = (t.height(.25+1e-6, 22.25)-t.height(.25, 22.25))/1e-6
    np.testing.assert_allclose(derivative, 2., rtol=1e-6)
    assert np.isnan(t.height(-11, 30))


def test_tracks_have_unique_observation_per_image():
    pairs = [('a','b',np.array([[0,0],[1,1]])), ('b','c',np.array([[0,0]])),
             ('a','c',np.array([[1,0]]))]
    tracks, conflicts = build_tracks(pairs)
    assert conflicts == 1
    assert len(tracks[0]) == 3
    assert all(len({k for k,_ in t}) == len(t) for t in tracks)


def test_boundary_ignores_missing_and_clipped_horizons():
    sky = np.array([[False,False,True], [False,True,True], [False,True,True]])
    np.testing.assert_allclose(boundary_pixels(sky, spacing=1), [[1,.5]])


def test_boundary_excludes_occluded_terrain_without_inventing_skyline():
    import pytest
    sky = np.array([[False, False, False], [False, True, True], [True, True, True]])
    excluded = np.zeros_like(sky)
    excluded[0, 1] = True
    # A masked boundary is omitted, not moved to the next rock/sky transition.
    np.testing.assert_allclose(boundary_pixels(sky, spacing=1, exclude=excluded),
                               [[0, 1.5], [2, .5]])
    with pytest.raises(ValueError, match='shape mismatch'):
        boundary_pixels(sky, spacing=1, exclude=np.zeros((2, 2)))


def test_pair_vegetation_masks_preserve_adjacent_rock():
    import cv2
    import json
    from pathlib import Path
    masks = json.loads(Path(__file__).with_name('horizon_masks.json').read_text())
    for key, rock, tree in [('2159', [(2362, 775), (2452, 733)], (2800, 500)),
                            ('2199', [(2362, 470), (2452, 440)], (2800, 160))]:
        polygon = np.array(masks[key], np.int32)
        assert all(cv2.pointPolygonTest(polygon, p, False) < 0 for p in rock)
        assert cv2.pointPolygonTest(polygon, tree, False) > 0


def test_joint_fit_uses_translation_and_sparse_dependencies():
    from marjum_bundle import Bundle, PRM_ORDER
    e, n = np.arange(-100., 301.), np.arange(-100., 101.)
    terrain = Terrain(SimpleNamespace(data=np.full((len(n),len(e)),100.), res=1., get_en=lambda:(e,n)))
    def intersect(p, shape, xy):
        d = rays(p, shape, xy)
        return p[:3] + ((100-p[2])/d[:,2])[:,None]*d
    terrain.intersect = intersect
    points = np.array([[x,y,100.] for x in (65,85,105) for y in (-15,0,15)])
    antenna = np.array([80., 5., 110.])
    poses, features, meta, gps = {}, {}, {}, {}
    for i, key in enumerate(['a','b','c']):
        p = np.array([i*10., i*3., 130., 2.1, 0., 0., 800.])
        xy, _ = project(p, (1000,1000), points)
        features[key] = dict(xy=xy, shape=(1000,1000), horizon=np.empty((0,2)))
        meta[key] = dict(ant_px=project(p, (1000,1000), antenna)[0][0])
        gps[key] = np.r_[p[:2], .5]
        if i == 1:
            p[0] += 2.
            p[4] += .01
        poses[key] = dict(zip(PRM_ORDER,p))
    tracks = [[(k,j) for k in poses] for j in range(len(points))]
    b = Bundle(poses, antenna+np.array([1.,1.,1.]), features, tracks, terrain, meta, gps)
    baseline = b.residuals(b.x0)
    perturbed = b.x0.copy()
    perturbed[7] = 1.
    assert np.linalg.norm(b.residuals(perturbed)[:2*len(b.obs_xy)]-baseline[:2*len(b.obs_xy)]) > 1
    # Every numerically changing residual must be represented in the sparse Jacobian.
    for col in range(len(b.x0)):
        x = b.x0.copy()
        x[col] += 1e-4
        changed = abs(b.residuals(x)-baseline) > 1e-8
        assert np.all(b.sparsity[:,col].toarray().ravel()[changed])
    result = b.solve(max_nfev=100)
    assert np.linalg.norm(b.residuals(result.x)) < np.linalg.norm(baseline)*.05
    assert np.max(np.linalg.norm(b.components(result.x)['antenna_px'],axis=1)) < .1


def test_epipolar_validation_on_known_geometry():
    from marjum_cv_holdout import epipolar_error
    p = np.array([0.,0.,130.,2.1,0.,0.,800.])
    q = p.copy()
    q[:3] += [10.,3.,1.]
    xyz = np.array([[70.,-10.,100.],[80.,10.,103.],[100.,5.,98.]])
    x,_ = project(p,(1000,1000),xyz)
    y,_ = project(q,(1000,1000),xyz)
    np.testing.assert_allclose(epipolar_error(p,(1000,1000),q,(1000,1000),x,y),0.,atol=1e-10)
    y[:,1] += 10.
    assert np.max(epipolar_error(p,(1000,1000),q,(1000,1000),x,y)) > 1.


def test_mcmc_initvals_refuse_below_ground():
    import pytest
    from marjum_bundle import mcmc_initvals, PRM_ORDER
    p = dict(zip(PRM_ORDER,[1.,2.,103.,1.,1.,0.,1000.]))
    dem = SimpleNamespace(interp_alt=lambda e,n:100.)
    v = mcmc_initvals({'a':p},[10.,20.,110.],dem)
    np.testing.assert_allclose(np.exp(v['a_log_h']),3.)
    np.testing.assert_allclose(np.exp(v['ant_log_h']),10.)
    p['u'] = 99.
    with pytest.raises(ValueError,match='above'):
        mcmc_initvals({'a':p},[10.,20.,110.],dem)
