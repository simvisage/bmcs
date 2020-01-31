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

from .mats2D_cmdm import \
    MATS2DMicroplaneDamage, PhiFnStrainSoftening

from ibvpy.mats.mats2D.__test__ import TestMATS2D

class TestMATS2DMicroplaneDamageCompliance( TestMATS2D ):
    '''Instantiation of the material tests in 2D for the compliance version.
    '''
    def _mats_eval_default(self):
        return MATS2DMicroplaneDamage( E = 34000, nu = 0.25, 
                                       stress_state  = "plane_strain",
                                       model_version = 'compliance',
                                       phi_fn = PhiFnStrainSoftening(
                                                                  G_f = 0.001117,
                                                                  f_t = 2.8968,
                                                                  md = 0.0,
                                                                  h = 1.0 )                                       
                                       )

    # Tests for the stiffness version
    #
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
        self.assert_stress_value( [2.43263348, 0, 0, 0],
                                    n_steps = 3, load = 0.0001 )

class TestMATS2DMicroplaneDamageStiffness( TestMATS2D ):
    '''Instantiation of the material tests in 2D for the stiffness version.
    '''
    def _mats_eval_default(self):
        return MATS2DMicroplaneDamage( E = 34000, nu = 0.25, 
                                       stress_state  = "plane_strain",
                                       model_version = 'stiffness',
                                       phi_fn = PhiFnStrainSoftening(
                                                                  G_f = 0.001117,
                                                                  f_t = 2.8968,
                                                                  md = 0.0,
                                                                  h = 1.0 )
                                       )

    # Tests for the stiffness version
    #
    def test_symmetry_tension_stiffness(self):
        '''Symmetry of response for tensile and shear loading.
        '''
        self.assert_symmetry( load_dirs = [0,1] )
    
    def test_symmetry_shear_stiffness(self):
        '''Symmetry of response for tensile and shear loading.
        '''
        self.assert_symmetry( load_dirs = [1,0] )

    def test_stress_value(self):
        '''Assert a correct value of resulting stress a material point.
        '''
        self.assert_stress_value( [2.541103649207904, 0, 0, 0],
                                  n_steps = 3, load = 0.0001 )

    def test_total_strain_energy_value(self):
        '''Assert a correct value of resulting elastic energy in the unit domain
        '''
        self.assert_total_energy_value( 2.541103649207904/2.*0.0001,
                                               n_steps = 3, load = 0.0001 )

    def test_total_fracture_energy_value(self):
        '''Assert a correct value of resulting elastic energy in the unit domain
        '''
        self.assert_total_energy_value( 0.00081182249919380263,
                                        ivar = 'fracture_energy',
                                        n_steps = 3, load = 0.01 )

if __name__ == "__main__":
    import unittest
    import sys;sys.argv = ['', 'TestMATS2DMicroplaneDamageStiffness.test_symmetry_shear_stiffness']    
    unittest.main()