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

from .mats2D_plastic import \
    MATS2DPlastic

from ibvpy.mats.mats2D.__test__ import TestMATS2D

class TestMATS2DMicroplaneDamage( TestMATS2D ):

    def _mats_eval_default(self):
        return MATS2DPlastic( E = 34000, nu = 0.25 ) 

    def test_symmetry_tension(self):
        '''Symmetry of response for tensile and shear loading.
        '''
        self.assert_symmetry( load_dirs = [0,1] )
    
    def test_symmetry_shear(self):
        '''Symmetry of response for tensile and shear loading.
        '''
        self.assert_symmetry( load_dirs = [1,0] )

if __name__ == "__main__":
    import unittest
    unittest.main()