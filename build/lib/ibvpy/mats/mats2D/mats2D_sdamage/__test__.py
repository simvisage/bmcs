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
# Created on Aug 18, 2009 by: rch

from .mats2D_sdamage import \
    MATS2DScalarDamage

from ibvpy.mats.mats2D.__test__ import \
    TestMATS2D

class TestMATS2DScalarDamage( TestMATS2D ):

    def _mats_eval_default(self):
        return MATS2DScalarDamage( E = 34000, nu = 0.25 ) 

    def test_symmetry_tension(self):
        '''Symmetry of response for tensile and shear loading.
        '''
        self.assert_symmetry( load_dirs = [0,1] )
    
    def test_symmetry_shear(self):
        '''Symmetry of response for tensile and shear loading.
        '''
        self.assert_symmetry( load_dirs = [1,0] )

    def test_stress_value(self):
        '''Assert a correct value of resulting stress a material point.
        '''
        # @todo - check this, it does not seem to be correct.
        self.assert_stress_value( [ 1.9378035241758689, 0, 0, 0 ],
                                    n_steps = 1, load = 0.0001 )
        
if __name__ == "__main__":
    import unittest
    import sys;sys.argv = ['', 'TestMATS2DScalarDamage.test_stress_value'] 
    unittest.main()