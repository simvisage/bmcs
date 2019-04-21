#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Aug 19, 2009 by: rch

from ibvpy.mats.mats3D.__test__ import TestMATS3D

from .vmats3D_elastic import \
    MATS3DElastic


class TestMATS3DElastic(TestMATS3D):
    '''Test the elementary load cases of 3D material models.
    '''

    def _mats_eval_default(self):
        mats_eval = MATS3DElastic(E=34000, nu=0.20)
        return mats_eval

    def test_symmetry_tension(self):
        '''
        Symmetry of the response ... tensile loading.
        '''
        self.assert_symmetry_on_cube_with_clamped_face(load_dirs=[0, 1, 2],
                                                       load=0.001)

    def test_symmetry_bending(self):
        '''
        Symmetry of the response ... out of plane load.
        '''
        self.assert_symmetry_on_cube_with_clamped_face(load_dirs=[1, 2, 0],
                                                       load=0.00016)
        return

    def test_stress_value(self):
        '''Assert a correct value of resulting stress a material point.
        '''
        self.assert_stress_value([340, 0, 0, 0, 0, 0, 0, 0, 0],
                                 n_steps=1, load=0.01)


if __name__ == "__main__":
    import unittest
    #import sys;sys.argv = ['', 'TestMATS3DElastic.test_stress_value']
    unittest.main()
