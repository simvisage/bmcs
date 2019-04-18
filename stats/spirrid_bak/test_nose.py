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
# Created on Oct 5, 2011 by: rch

'''Unit tests for spirrid.

 1) limit case (no random variable changed)
 2) state changes
 3) add a test with varying n_ints in individual directions.
  
  @todo: add multidimensional control variable.
  and further types of randomization configurations (only one variable)
  
  @todo: add C-code tests.
'''

import unittest

import nose
from stats.spirrid_bak import SPIRRIDLAB, Heaviside, SPIRRID, RV, RF, IRF

from etsproxy.traits.api import Str
import numpy as np


#===========================================================================
# Response function
#===========================================================================
class fiber_tt_2p(RF):
    '''Linear elastic, brittle filament.
    '''
    implements(IRF)

    title = Str('brittle filament')

    def __call__(self, e, la, xi):
        ''' Response function of a single fiber '''
        return la * e * Heaviside(xi - e)

    C_code = '''
            // Computation of the q( ... ) function
            if ( eps < 0 || eps > xi ){
                q = 0.0;
            }else{
                  q = la * eps;
            }
        '''


class SPIRRIDAlgTest(unittest.TestCase):
    '''
    Test functionality connected with the application of
    boundary conditions.
    '''

    @classmethod
    def setup_class(cls):

        np.random.seed(2356)

        #===========================================================================
        # Control variable
        #===========================================================================
        e_arr = np.linspace(0, 0.012, 80)

        cls.m_la, cls.std_la = 10., 1.0
        cls.m_xi, cls.std_xi = 1.0, 0.1

        #===========================================================================
        # Randomization
        #===========================================================================

        cls.s = SPIRRID(q=fiber_tt_2p(),
                         evars={'e':e_arr},
                         codegen_type='numpy',
                         n_int=10,
                         tvars=dict(la=RV('norm', cls.m_la, cls.std_la),
                                      xi=RV('norm', cls.m_xi, cls.std_xi)
                                      ),
                         )

    #===========================================================================
    # Test all combinations of randomization: TGrid, PGrid, MCS, LHS
    #===========================================================================
    def test_numpy_tgrid(self):
        '''Check the result of the computation for TGrid'''
        self.s.codegen_type = 'numpy'
        self.s.sampling_type = 'TGrid'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999724956278124, 10)

    def test_numpy_pgrid(self):
        '''Check the result of the computation for PGrid'''
        self.s.codegen_type = 'numpy'
        self.s.sampling_type = 'PGrid'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999999999999997, 10)

    def test_numpy_monte_carlo(self):
        '''Check the result of the computation for Monte Carlo'''
        self.s.codegen_type = 'numpy'
        self.s.sampling_type = 'MCS'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.12056771344542541, 10)

    def test_numpy_lhs(self):
        '''Check the result of the computation for LHS'''
        self.s.codegen_type = 'numpy'
        self.s.sampling_type = 'LHS'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999999999999995, 10)

    def test_numpy_variance_tgrid(self):
        '''Check the variance of the computation for TGrid'''
        self.s.codegen_type = 'numpy'
        self.s.sampling_type = 'TGrid'
        max_var_q = np.max(self.s.var_q_arr)
        self.assertAlmostEqual(max_var_q, 0.00014429189095519629, 10)

    def test_numpy_variance_lhs(self):
        '''Check the mean and variance of the computation for LHS'''
        self.s.codegen_type = 'numpy'
        self.s.sampling_type = 'LHS'
        max_var_q = np.max(self.s.var_q_arr)
        self.assertAlmostEqual(max_var_q, 0.00014217258709777608, 10)
        # check cached values
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999999999999995, 10)

    #===========================================================================
    # Test C-code implementation
    #===========================================================================
    def test_c_tgrid01(self):
        '''Check the C-implementation for TGrid'''
        self.s.codegen_type = 'c'
        self.s.codegen.set(cached_dG=False, compiled_eps=False)
        self.s.sampling_type = 'TGrid'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999724956278124, 10)

    def test_c_tgrid02(self):
        '''Check the C-implementation for TGrid'''
        self.s.codegen_type = 'c'
        self.s.codegen.set(cached_dG=True, compiled_eps=False)
        self.s.sampling_type = 'TGrid'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999724956278124, 10)

    def test_c_tgrid03(self):
        '''Check the C-implementation for TGrid'''
        self.s.codegen_type = 'c'
        self.s.codegen.set(cached_dG=False, compiled_eps=True)
        self.s.sampling_type = 'TGrid'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999724956278124, 10)

    def test_c_tgrid04(self):
        '''Check the C-implementation for TGrid'''
        self.s.codegen_type = 'c'
        self.s.codegen.set(cached_dG=True, compiled_eps=True)
        self.s.sampling_type = 'TGrid'
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.11999724956278124, 10)

    def test_randomization_change(self):
        '''Check the C-implementation for TGrid'''
        self.s.codegen_type = 'numpy'
        self.s.sampling_type = 'PGrid'
        self.s.tvars['xi'] = self.m_xi
        max_mu_q = np.max(self.s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.12000000000000001, 10)


class SPIRRIDChangeTest(unittest.TestCase):
    '''
    Test functionality connected with the application of
    boundary conditions.
    '''

    def setUp(self):

        np.random.seed(2356)

    def test_e_arr(self):
        '''Check the convenience constructor for evars
        '''
        #===========================================================================
        # Control variable
        #===========================================================================
        e_arr = np.linspace(0, 0.012, 2)

        m_la, std_la = 10., 1.0
        m_xi, std_xi = 1.0, 0.1

        #===========================================================================
        # Randomization
        #===========================================================================

        s = SPIRRID(q=fiber_tt_2p(),
                         e_arr=e_arr,
                         codegen_type='numpy',
                         n_int=10,
                         tvars=dict(la=m_la,
                                      xi=m_xi
                                      ),
                         )

        max_mu_q = np.max(s.mu_q_arr)
        self.assertAlmostEqual(max_mu_q, 0.12000000000000001, 10)

