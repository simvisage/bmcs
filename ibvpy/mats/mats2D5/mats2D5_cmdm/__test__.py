#-------------------------------------------------------------------------------
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
    
from .mats2D5_cmdm import \
    MATS2D5MicroplaneDamage, PhiFnStrainHardening

from ibvpy.mats.mats2D5.__test__ import \
    TestMATS2D5

class TestMATS2D5( TestMATS2D5 ):
    '''Test the elementary load cases of 2D5 material models.
    '''

    def _mats_eval_default(self):
        return MATS2D5MicroplaneDamage( E = 34000, nu = 0.25,
                                        phi_fn = PhiFnStrainHardening() )
        
    def test_symmetry_for_tension(self):
        '''
        Symmetry of the response ... inplane tensile loading.
        '''
        self.assert_2D_symmetry_clamped_cube( load_dirs = [0,1], 
                                              load = 0.0001 )        
        
    def test_symmetry_for_inplane_shear(self):
        '''
        Symmetry of the response ... inplane shear loading.
        '''
        self.assert_2D_symmetry_clamped_cube( load_dirs = [1,0], 
                                              load = 0.0002 )        

    def test_symmetry_for_bending(self):
        '''
        Symmetry of the response ... out of plane load.ing
        '''
        self.assert_2D_symmetry_clamped_cube( load_dirs = [2,2], 
                                              load = 0.001 )        

    def test_stress_value(self):
        '''Assert a correct value of resulting stress a material point.
        '''
        # @todo - response tracer should return 3D stress state - the current
        # value is an intermediate value that does not have transversal 
        # component equal to zero. This should be fixed by the 2D5 implementation
        # of CMDM.
        self.assert_stress_value( [ 3.0117268131744122, 0, 0, 0.35733911053546097 ],
                                    n_steps = 1, load = 0.0001 )
        
if __name__ == "__main__":
    import unittest
    import sys;sys.argv = ['', 'TestMATS2D5.test_symmetry_for_inplane_shear']
    unittest.main()