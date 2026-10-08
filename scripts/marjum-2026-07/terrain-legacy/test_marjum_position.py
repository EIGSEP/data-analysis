"""Regression checks for the DEM-native 2026 position model."""
import unittest
import numpy as np
from marjum_position import RefinedTerrain, PositionFit, load_inputs
from marjum_camera import project
from marjum_position_validate import sky_nll


class SkyProbabilityTests(unittest.TestCase):
    def test_neutral_tree_pixels_do_not_constrain_geometry(self):
        self.assertEqual(sky_nll([.5,.5],[True,True],2),sky_nll([.5,.5],[False,False],2))

    def test_repeating_correlated_samples_does_not_add_weight(self):
        self.assertAlmostEqual(sky_nll([.9,.1],[True,False],2),sky_nll([.9,.1]*10,[True,False]*10,2))

    def test_matching_classification_scores_better(self):
        self.assertLess(sky_nll([.9,.1],[True,False],2),sky_nll([.9,.1],[False,True],2))


class PositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from eigsep_terrain.marjum_dem import MarjumDEM
        s,k,f,o,h=load_inputs()
        cls.model=PositionFit(k,f,s['cameras'],s['distortion'],s['groups'],o,
            RefinedTerrain(MarjumDEM(cache_file='marjum_dem_sw.npz')),False,antenna=s['antenna'])

    def test_grid_roundtrip(self):
        g=self.model.grid
        p=np.array([1655.,2030.])
        np.testing.assert_allclose(g.from_lonlat(*g.to_lonlat(*p)),p,atol=1e-6)
        np.testing.assert_allclose(g.origin,[291000.25,4345000.25])

    def test_residuals_and_bias(self):
        m=self.model;r=m.residuals(m.x0)
        self.assertEqual(m.sparsity.shape,(len(r),len(m.x0)))
        self.assertTrue(np.isfinite(r).all())
        np.testing.assert_array_equal(r,m.residuals(m.x0))
        for j in range(3):
            x=m.x0.copy();x[m.base_size+j]=1.
            changed=np.flatnonzero(abs(m.residuals(x)-r)>1e-9)
            declared=m.sparsity[:,m.base_size+j].toarray().ravel().astype(bool)
            self.assertTrue(declared[changed].all())
            self.assertEqual(len(changed),m.nc+1)

    def test_translation_preserves_antenna_projection(self):
        m=self.model;shift=np.array([8.,-5.,2.]);x=m.shifted(shift)
        cams,ks,pts=m.unpack(x)
        ant=m.ant0+x[m.geometry_size:m.geometry_size+3]
        for i in m.ai:
            np.testing.assert_allclose(project(cams[i],m.shapes[i],ant,ks[i])[0],
                project(m.base[i],m.shapes[i],m.ant0,m.ks[i])[0],atol=1e-8)

    def test_modes_and_horizon_split(self):
        m=self.model
        positions=set(m.active('positions'))
        for i in range(m.nc):
            self.assertTrue(all(7*i+j in positions for j in range(3)))
            self.assertTrue(all(7*i+j not in positions for j in range(3,7)))
        for mask in m.htrain:
            self.assertTrue(mask.any() and (~mask).any())


if __name__=='__main__':unittest.main()
