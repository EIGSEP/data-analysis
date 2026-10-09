"""Geometry checks for registering additional distortion-aware views."""
import unittest
import numpy as np

from marjum_add_views import joint_refine,pnp_pose,spatial_split
from marjum_bundle import Terrain
from marjum_camera import project,rays


class AddedViewTests(unittest.TestCase):
    def test_pnp_bottom_up_roundtrip(self):
        rng=np.random.default_rng(1);shape=(4032,3024)
        original=np.array([1700.,2050.,1720.,1.25,-2.4,.03,2600.])
        distortion=np.array([-.0215,.0122])
        xy=rng.uniform([100,100],[shape[1]-100,shape[0]-100],(300,2))
        xyz=original[:3]+rays(original,shape,xy,distortion)*rng.uniform(50,500,(300,1))
        fitted,inliers=pnp_pose(xyz,xy,shape,original[6],distortion)
        self.assertEqual(len(inliers),len(xy))
        error=np.linalg.norm(project(fitted,shape,xyz,distortion)[0]-xy,axis=1)
        self.assertLess(np.median(error),1e-3)

    def test_spatial_holdout_is_deterministic_and_disjoint(self):
        xy=np.array(np.meshgrid(np.arange(20)*120+10,np.arange(12)*120+10)).reshape(2,-1).T
        train,test=spatial_split(xy)
        np.testing.assert_array_equal(train,spatial_split(xy)[0])
        self.assertFalse(np.any(train&test))
        self.assertTrue(np.all(train|test))
        self.assertGreater(train.sum(),12);self.assertGreater(test.sum(),8)

    def test_joint_refine_can_use_fixed_antenna_pixel(self):
        class FlatTerrain:
            e=np.array([-1000.,1000.]);n=e;data=np.array([[0.,0.],[0.,0.]])
            def height(self,e,n):return np.zeros_like(np.asarray(e,float))
            def skyline(self,cam,azimuth):return np.full(len(azimuth),-.5)
        shape=(1000,1200);k=np.zeros(2)
        truth=np.array([0.,0.,10.,np.pi/2,0.,0.,1000.])
        antenna=np.array([100.,0.,10.])
        antenna_xy=project(truth,shape,antenna,k)[0][0]
        # Empty feature and horizon arrays isolate the antenna term. A small
        # heading error should be reduced while the physical antenna is fixed.
        start=truth.copy();start[4]=.02
        fitted,_=joint_refine(start,np.empty((0,3)),np.empty((0,2)),shape,k,
                              truth[6],FlatTerrain(),np.empty((0,2)),
                              np.empty(0,bool),antenna=antenna,
                              antenna_xy=antenna_xy,max_nfev=30)
        before=np.linalg.norm(project(start,shape,antenna,k)[0][0]-antenna_xy)
        after=np.linalg.norm(project(fitted,shape,antenna,k)[0][0]-antenna_xy)
        self.assertLess(after,before)


if __name__=='__main__':unittest.main()
