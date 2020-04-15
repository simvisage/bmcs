'''
Created on Dec 17, 2018

@author: rch
'''
import unittest

import numpy as np
from simulator.sim_controler import Model
import traits.api as tr


class SinusModel(Model):

    #=========================================================================
    # Model implementation - F and Fprime
    #=========================================================================

    def _get_F(self):
        return np.sin(self.U_k)

    def _get_d_F_U(self):
        return [np.cos(self.U_k)]

    #=========================================================================
    # Derived variables
    #=========================================================================
    G = tr.Property

    def _get_G(self):
        return np.arctan(self.U_k)


class Test(unittest.TestCase):

    def setUp(self):
        self.m = SinusModel()

    def test_model_init(self):
        m = self.m
        self.assertTrue(m == m.tstep.model, 'Model initialization success')
        self.assertTrue(m.tstep == m.tloop.tstep,
                        'Model initialization success')
        self.assertTrue(m.hist == m.sim.hist,
                        'Model initialization success')

    def test_tloop_eval(self):
        '''Run the model evaluation.

        For convenience, the model should incorporate
        the boundary conditions as well.
        '''
        m = self.m
        m.tloop.init()
        m.tloop.eval()
        self.assertAlmostEqual(m.U_k[0], 1.55702096)

    def test_bc(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
