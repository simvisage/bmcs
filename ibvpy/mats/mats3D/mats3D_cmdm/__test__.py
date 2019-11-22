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

from .mats3D_cmdm import \
    MATS3DMicroplaneDamage, PhiFnStrainSoftening, PhiFnStrainHardening

from ibvpy.mats.mats3D.__test__ import TestMATS3D

class TestMATS3DMicroplaneDamageCompliance( TestMATS3D ):
    '''Test the elementary load cases of 3D material models.
    '''

    def _mats_eval_default(self):
        return MATS3DMicroplaneDamage( E = 34000, nu = 0.25,
                                       model_version = 'compliance',
                                       phi_fn = PhiFnStrainHardening() )
     
    def test_symmetry_tension(self):
        '''
        Symmetry of the response ... tensile loading.
        '''
        self.assert_symmetry_on_cube_with_clamped_face( load_dirs = [0,1,2], 
                                                        load = 0.001 )
                        
    def test_symmetry_bending(self):
        '''
        Symmetry of the response ... out of plane load.
        '''
        self.assert_symmetry_on_cube_with_clamped_face( load_dirs = [1,2,0],
                                                        load = 0.00016 )
        return

    def test_stress_value(self):
        '''Assert a correct value of resulting stress a material point.
        '''
        self.assert_stress_value( [ 2.9990935740536142, 0, 0, 0, 0, 0, 0, 0, 0 ],
                                    n_steps = 1, load = 0.0001 )

    def test_total_fracture_energy_value(self):
        '''Assert a correct value of resulting elastic energy in the unit domain
        '''
        self.assert_total_energy_value( 0.00078899038265171733,
                                        ivar = 'fracture_energy',
                                        n_steps = 5, load = 0.01 )
                
class TestMATS3DMicroplaneDamageStiffness( TestMATS3D ):
    '''Test the elementary load cases of 3D material models.
    '''

    def _mats_eval_default(self):
        return MATS3DMicroplaneDamage( E = 34000, nu = 0.25,
                                       model_version = 'stiffness',
                                       phi_fn = PhiFnStrainSoftening(
                                                                  G_f = 0.001117,
                                                                  f_t = 2.8968,
                                                                  md = 0.0,
                                                                  h = 1.0 )
                                        )
     
    def test_symmetry_tension(self):
        '''
        Symmetry of the response ... tensile loading.
        '''
        self.assert_symmetry_on_cube_with_clamped_face( load_dirs = [0,1,2], 
                                                        load = 0.001 )
        
                
    def test_symmetry_bending(self):
        '''
        Symmetry of the response ... out of plane load.
        '''
        self.assert_symmetry_on_cube_with_clamped_face( load_dirs = [1,2,0],
                                                        load = 0.00016 )
        return

    def test_stress_value(self):
        '''Assert a correct value of resulting stress a material point.
        '''
        self.assert_stress_value( [ 2.7904899530421789, 0, 0, 0, 0, 0, 0, 0, 0 ],
                                    n_steps = 1, load = 0.0001 )
        
    def test_total_fracture_energy_value(self):
        '''Assert a correct value of resulting fracture energy in the unit domain
        '''
        self.assert_total_energy_value( 0.0011078180050320837,
                                        ivar = 'fracture_energy',
                                        n_steps = 5, load = 0.01 )
    
if __name__ == "__main__":
    import unittest
    import sys;sys.argv = ['', 'TestMATS3DMicroplaneDamageCompliance.test_total_fracture_energy_value']         
    unittest.main()
    