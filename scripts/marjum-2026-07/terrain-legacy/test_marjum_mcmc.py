"""Small target/kernel tests; the DEM integration test uses the local data."""
import numpy as np
import pytest
from marjum_mcmc import Posterior, student_residual, metropolis_sweep


def test_student_target():
    x = np.array([[0., 0.], [2., -3.], [100., 1.]])
    r = student_residual(x, 4, 2)
    np.testing.assert_allclose(np.sum(r*r, axis=1), 6*np.log1p(np.sum(x*x, axis=1)/4))


def test_focal_prior_widths():
    from marjum_mcmc import Config, focal_prior_widths
    keys=['main','ultrawide']
    np.testing.assert_array_equal(focal_prior_widths(Config(),keys),[.25,.25])
    np.testing.assert_array_equal(focal_prior_widths(Config(log_f_sigma=.03,
        log_f_sigma_by_camera={'ultrawide':.25}),keys),[.03,.25])
    for widths in [{'typo':.25},{'main':0},{'main':float('nan')}]:
        with pytest.raises(ValueError):focal_prior_widths(Config(log_f_sigma_by_camera=widths),keys)


class Gaussian:
    ng, np = 1, 1

    def point_logp(self, z):
        return np.array([-.5*np.sum((z[1:]-np.array([.8, -.4, .2])*z[0])**2)])

    def logp(self, z):
        return -.5*z[0]**2+self.point_logp(z).sum()


def test_block_kernel_correlated_gaussian():
    model = Gaussian()
    response = np.array([[[.8], [-.4], [.2]]])
    geometry = np.ones((1, 1)), np.eye(3)[None], response
    rng = np.random.default_rng(37)
    z = np.zeros(4)
    lp = model.logp(z)
    samples = []
    for i in range(14000):
        z, lp, _, _ = metropolis_sweep(model, z, lp, geometry, rng, 2., 1.3)
        if i % 100 == 0:
            assert lp == pytest.approx(model.logp(z), abs=1e-10)
        if i >= 2000:
            samples.append(z.copy())
    samples = np.array(samples)
    residual = samples[:, 1:]-samples[:, :1]*response[0, :, 0]
    whitened = np.c_[samples[:, 0], residual]
    np.testing.assert_allclose(whitened.mean(axis=0), 0, atol=.08)
    np.testing.assert_allclose(np.cov(whitened.T), np.eye(4), atol=.12)


def test_b21_camera_kernel_log_f_target():
    """Exercise the actual camera update on a correlated position/log-f target.

    A raw-f proposal using the log-f density either freezes or samples the
    wrong measure; checking both moments and covariance detects that regression.
    """
    from types import SimpleNamespace
    from marjum_mcmc_b21 import Chain

    mu = np.log(1500.)
    covariance = np.array([[1., .85*.25], [.85*.25, .25**2]])
    precision = np.linalg.inv(covariance)

    def logp(i, cam, *args, **kwargs):
        q = np.array([cam[0], np.log(cam[6])-mu])
        return -.5*q @ precision @ q

    chain = Chain.__new__(Chain)
    chain.model = SimpleNamespace(nc=1, camera_logp=logp,
                                 horizon_logp=lambda i, cam: 0.,
                                 _camera_support=lambda cam: np.isfinite(cam).all() and cam[6] > 0)
    chain.rng = np.random.default_rng(71)
    chain.cam = np.array([[0., 0., 0., 1., 0., 0., 1500.]])
    chain.ant = chain.points = chain.bias = chain.tx = None
    chain.extra = chain.tx_extra = 0.
    chain.horizon = np.zeros(1)
    chain.camera_factor = np.zeros((1, 7, 7))
    chain.camera_factor[0][np.ix_([0, 6], [0, 6])] = np.linalg.cholesky(covariance)
    chain.camera_scale = np.array([1.7])
    chain.counts = {k: np.zeros(1) for k in ['camera', 'camera_n',
                    'camera_proposed_logf_sq', 'camera_jump_logf_sq', 'camera_support_reject']}
    samples = []
    for step in range(18000):
        chain.update_cameras()
        if step >= 2000:
            samples.append([chain.cam[0, 0], (np.log(chain.cam[0, 6])-mu)/.25])
    samples = np.asarray(samples)
    np.testing.assert_allclose(samples.mean(axis=0), 0., atol=.08)
    np.testing.assert_allclose(np.cov(samples.T), [[1., .85], [.85, 1.]], atol=.10)
    assert chain.counts['camera_n'][0] == 18000
    assert chain.counts['camera_jump_logf_sq'][0] > 0


def test_b21_camera_curvature_units_and_weak_directions():
    from marjum_mcmc_b21 import _scaled_camera_factor, CAMERA_SCALE

    rng = np.random.default_rng(4)
    # Rank deficiency deliberately exercises the regularization, not just the
    # easy positive-definite case. Expressing positions in cm must not alter it.
    jacobian = rng.normal(size=(5, 7))/CAMERA_SCALE
    factor, info = _scaled_camera_factor(jacobian)
    units = np.array([100., 100., 100., 1., 1., 1., 1.])
    changed, _ = _scaled_camera_factor(jacobian/units, CAMERA_SCALE*units)
    np.testing.assert_allclose((changed/units[:, None]) @ (changed/units[:, None]).T,
                               factor @ factor.T, atol=1e-12)
    assert info['clipped'] >= 2
    assert np.isfinite(factor).all()
    assert np.max(np.linalg.eigvalsh((factor/CAMERA_SCALE[:, None]) @
                                    (factor/CAMERA_SCALE[:, None]).T)) <= 1.+1e-12


@pytest.mark.parametrize('state_aware',[False,True])
def test_b21_checkpoint_exact_resume(tmp_path, monkeypatch, state_aware):
    """An interrupted real-data chain must resume bit-for-bit, including RNG."""
    from pathlib import Path
    import marjum_mcmc_b21 as b

    inp = Path.cwd().parent/'marjum-2026-07/derived/geometry_posterior/v0001/inputs'
    if not (inp/'input_manifest.json').exists():
        pytest.skip('run from the data-bearing terrain directory for frozen-input test')
    model = b.Posterior(state_file=inp/'fit_transmitter.npz',
                        dem_file=str(inp/'marjum_dem.npz'), meta_file=inp/'meta.json',
                        exif_file=inp/'marjum_2026_07_exif_joint.npz',
                        feature_dir=inp/'cv_features')
    fine=inp.parent.parent/'v0002/fine_geometry_refined_20261003'
    extra_args={}
    if (fine/'geometry.npz').exists():
        from marjum_mcmc_b21_coupling import Linearization
        inp2=inp.parent.parent/'v0002/inputs'
        model=b.Posterior(state_file=inp2/'fit_transmitter.npz',dem_file=str(inp2/'marjum_dem.npz'),
            meta_file=inp2/'meta.json',exif_file=inp2/'marjum_2026_07_exif_joint.npz',feature_dir=inp2/'cv_features',
            config=b.Config(log_f_sigma=.03,log_f_sigma_by_camera={'2198':.25,'2203':.25}))
        with np.load(fine/'supported_endpoints.npz') as saved:start=saved['chain_0'].copy()
        # Independently predict the target change from fixed Gaussian widths.
        # The camera, full-target and linearized-residual paths must all agree.
        desired=np.array([.25 if k in ('2198','2203') else .03 for k in model.keys])
        np.testing.assert_array_equal(model.focal_sigma,desired)
        cam,ant,tx,bias,extra,tx_extra,points=model.unpack(start)
        u=np.log(cam[:,6]/model.focal)
        expected=-.5*np.sum(u*u*(desired**-2-.25**-2))
        tight=model.logp(start)
        camera_tight=np.array([model.camera_logp(i,c,ant,points,bias,extra,tx=tx,tx_extra=tx_extra)
                               for i,c in enumerate(cam)])
        model.focal_sigma=np.full(model.nc,.25)
        loose=model.logp(start)
        camera_loose=np.array([model.camera_logp(i,c,ant,points,bias,extra,tx=tx,tx_extra=tx_extra)
                               for i,c in enumerate(cam)])
        model.focal_sigma=desired
        assert tight-loose==pytest.approx(expected,abs=1e-8)
        np.testing.assert_allclose(camera_tight-camera_loose,-.5*u*u*(desired**-2-.25**-2),atol=1e-8)
        linear=Linearization(model,start)
        residual=linear.residual(np.zeros(linear.nvar))
        assert -.5*residual@residual+linear.normalization()==pytest.approx(tight,abs=1e-8)
        with np.load(fine/'geometry.npz') as saved:
            directions=np.array([np.r_[saved[n][2,:linear.ng]*linear.scale,np.zeros(2),saved[n][2,linear.ng:]] for n in ['cam2223_n','cam2224_n']])
        extra_args=dict(start=start,difference_step_factor=1e-4,joint_directions=directions)
    if state_aware:
        packet=inp.parent.parent/'v0002/local_angle_mixture_geometry_20261006'
        if not (fine/'geometry.npz').exists() or not (packet/'geometry.npz').exists():
            pytest.skip('state-aware exact resume requires the frozen local-angle packet')
        from marjum_mcmc_b21_coupling import Linearization
        from marjum_mcmc_b21_combine import label_coordinates
        with np.load(packet/'supported_endpoints.npz') as saved:
            start=saved['chain_5'].copy()
            names=label_coordinates(model.keys,joint=True)
            indices=[names.index(n) for n in ['cam2159_ph','cam2159_th','cam2222_ti']]
            centers=np.stack([saved[f'chain_{c}'][indices] for c in [5,6]])
        linear=Linearization(model,start)
        direction_names=['transmitter_u','transmitter_n']+[
            f'{n}_local_{c}' for c in [5,6] for n in ['cam2159_ph','cam2159_th','cam2222_ti']]
        with np.load(packet/'geometry.npz') as saved:
            directions=np.array([np.r_[saved[n][2,:linear.ng]*linear.scale,np.zeros(2),saved[n][2,linear.ng:]]
                                 for n in direction_names])
        selector=dict(indices=indices,centers=centers.tolist(),groups=[0,0,1,1,1,2,2,2],
            bandwidth=float(np.linalg.norm(centers[1]-centers[0])/2),
            neutral_mass=.25,bank_floor=.05)
        extra_args=dict(start=start,difference_step_factor=1e-4,joint_directions=directions,
                        joint_selector=selector)
    whole, split = tmp_path/'whole', tmp_path/'split'
    whole.mkdir(); split.mkdir()
    args = dict(model=model, index=0, tune=2, draws=3, seed=71,
                thin_points=1, shift_every=2, **extra_args)
    b.run_chain(**args, out=whole, checkpoint_every=0)
    original = b.Chain.sweep
    calls = 0

    def interrupted(self, *a, **kw):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError('simulated interruption')
        return original(self, *a, **kw)

    monkeypatch.setattr(b.Chain, 'sweep', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        b.run_chain(**args, out=split, checkpoint_every=2)
    monkeypatch.setattr(b.Chain, 'sweep', original)
    with pytest.raises(ValueError, match='configuration changed'):
        b.run_chain(**{**args, 'seed': 72}, out=split, resume=True)
    b.run_chain(**args, out=split, checkpoint_every=2, resume=True)
    with np.load(whole/'chain_0.npz') as expected, np.load(split/'chain_0.npz') as actual:
        assert expected.files == actual.files
        for name in expected.files:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        if extra_args:
            import json
            assert sum(json.loads(str(actual['joint_acceptance']))['proposals'])==3


def test_b21_frozen_joint_kernel_correlated_target():
    """The actual frozen-direction kernel must preserve a correlated target."""
    from types import SimpleNamespace
    from marjum_mcmc_b21 import Chain
    covariance=np.array([[1.,.95],[.95,1.]])
    precision=np.linalg.inv(covariance)
    def unpack(z):return np.empty((0,7)),z.copy(),None,np.empty(0),0.,None,np.empty((0,3))
    chain=Chain.__new__(Chain)
    chain.model=SimpleNamespace(nc=0,logp=lambda z:-.5*z@precision@z,
        unpack=unpack,pack=lambda cam,ant,*args:ant.copy())
    (chain.cam,chain.ant,chain.tx,chain.bias,chain.extra,chain.tx_extra,chain.points)=unpack(np.zeros(2))
    chain.horizon=np.empty(0)
    chain.joint_rng=np.random.default_rng(551)
    chain.joint_directions=np.linalg.cholesky(covariance).T
    chain.joint_scale=np.full(2,2.4)
    chain.joint_counts={k:np.zeros(2) for k in ['proposals','accepts','support_rejections','proposed_sq','accepted_sq']}
    samples=[]
    for iteration in range(18000):
        chain.update_joint()
        if iteration>=2000:samples.append(chain.state())
    samples=np.array(samples)
    np.testing.assert_allclose(samples.mean(axis=0),0,atol=.07)
    np.testing.assert_allclose(np.cov(samples.T),covariance,atol=.08)
    assert chain.joint_counts['proposals'].sum()==18000
    assert np.all(chain.joint_counts['accepted_sq']>0)


def test_b21_state_dependent_selector_correlated_target():
    """State-dependent direction weights retain the correlated target."""
    from types import SimpleNamespace
    from marjum_mcmc_b21 import Chain, validated_joint_selector, joint_selection_probabilities
    covariance=np.array([[1.,.6],[.6,1.]])
    precision=np.linalg.inv(covariance)
    def unpack(z):return np.empty((0,7)),z.copy(),None,np.empty(0),0.,None,np.empty((0,3))
    chain=Chain.__new__(Chain)
    chain.model=SimpleNamespace(nc=0,logp=lambda z:-.5*z@precision@z,
        unpack=unpack,pack=lambda cam,ant,*args:ant.copy())
    (chain.cam,chain.ant,chain.tx,chain.bias,chain.extra,chain.tx_extra,chain.points)=unpack(np.zeros(2))
    chain.horizon=np.empty(0)
    chain.joint_rng=np.random.default_rng(9201)
    chain.joint_directions=np.array([[1.,0.],[0.,1.],[1.,1.]])
    chain.joint_scale=np.full(3,2.)
    chain.joint_selector=validated_joint_selector(dict(indices=[0,1],
        centers=[[-1.,0.],[1.,0.]],groups=[0,1,2],bandwidth=.7,
        neutral_mass=.25,bank_floor=.05),2,3)
    chain.joint_counts={k:np.zeros(3) for k in ['proposals','accepts','support_rejections','proposed_sq','accepted_sq']}
    near_left=joint_selection_probabilities(np.array([-1.,0.]),chain.joint_selector)
    near_right=joint_selection_probabilities(np.array([1.,0.]),chain.joint_selector)
    assert near_left[1]>near_left[2] and near_right[2]>near_right[1]
    np.testing.assert_allclose(near_left.sum(),1.)
    assert min(near_left)>0 and min(near_right)>0
    samples=[]
    for iteration in range(30000):
        chain.update_joint()
        if iteration>=3000:samples.append(chain.state())
    samples=np.array(samples)
    np.testing.assert_allclose(samples.mean(axis=0),0,atol=.08)
    np.testing.assert_allclose(np.cov(samples.T),covariance,atol=.12)
    assert chain.joint_counts['proposals'].sum()==30000
    assert np.all(chain.joint_counts['accepted_sq']>0)


def test_real_target_and_sparse_dependencies():
    model = Posterior()
    z = model.starts[0].copy()
    r = model.residuals(z)
    assert model.valid(z)
    assert model.sparsity.shape == (len(r), len(z))
    assert model.logp(z) == pytest.approx(-.5*r@r)
    assert model.logp(z) == model.logp(z)
    # The 2D Gaussian label likelihood must retain its unknown-scale normalizer.
    from marjum_bundle import project
    cam, ant, _, extra, _ = model.unpack(z)
    pred = np.array([project(cam[i], model.shapes[i], ant)[0][0] for i in model.ai])
    sigma = np.hypot(model.config.antenna_label_sigma_px, extra)
    expected = -.5*np.sum(((pred-model.axy)/sigma)**2)-2*len(model.ai)*np.log(sigma/model.config.antenna_label_sigma_px)
    antenna_terms = model.global_residuals(z)[:3*len(model.ai)]
    assert -.5*antenna_terms@antenna_terms == pytest.approx(expected)
    invalid = z.copy()
    invalid[model.ng-1] = (-1-model.origin[model.ng-1])/model.scale[model.ng-1]
    assert model.logp(invalid) == -np.inf
    for j in [0, 4, 6, 7*model.nc, model.ng-1, model.ng, len(z)-1]:
        changed = z.copy()
        changed[j] += 1e-4
        delta = model.residuals(changed)-r
        allowed = model.sparsity[:, j].toarray().ravel().astype(bool)
        np.testing.assert_allclose(delta[~allowed], 0, atol=1e-10)
    # An affine numerical origin must not change the statistical target.
    before = model.logp(z)
    shift = np.full_like(z, .01)
    model.origin = model.origin+shift*model.scale
    assert model.logp(z-shift) == pytest.approx(before, abs=1e-8)
    # Independent point acceptances must preserve the full target accounting.
    geometry = np.eye(model.ng)*.001, np.tile(np.eye(3)*.001,(model.np,1,1)), np.zeros((model.np,3,model.ng))
    z = z-shift
    for _ in range(3):
        z, before, _, _ = metropolis_sweep(model,z,before,geometry,np.random.default_rng(2),1.,1.)
        assert before == pytest.approx(model.logp(z), abs=1e-7)
