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
# Created on Sep 8, 2011 by: rch

from enthought.traits.api import implements, Str
from scipy.special import erf
from stats.spirrid import SPIRRID, Heaviside, RV, RF, IRF
from stats.spirrid.extras import SPIRRIDLAB
import math
import numpy as np

#===========================================================================
# Response function
#===========================================================================
class fiber_tt_2p(RF):
    ur'''
Response Function with two-parameters.
======================================
    
The function describes a linear dependency with a coefficient :math:`\lambda` 
up to the threshold :math:`\xi` and zero value beyond the threshold: 
    
..    math::
       q( \varepsilon; \theta, \lambda ) = \lambda \varepsilon H( \xi - \varepsilon )

where the variables :math:`\lambda=` stiffness parameter and :math:`\xi=` 
breaking strain are considered random and normally distributed. The function 
:math:`H(\eta)` represents the Heaviside function with values 0 for 
:math:`\eta < 0` and 1 for :math:`\eta > 0`.
   
    '''
    implements(IRF)

    title = Str('brittle filament')

    def __call__(self, e, la, xi):
        ''' Response function of a single fiber '''
        return la * e * Heaviside(xi - e)

    cython_code = '''
            # Computation of the q( ... ) function
            if eps < 0 or eps > xi:
                q = 0.0
            else:
                q = la * eps
            '''

    c_code = '''
            // Computation of the q( ... ) function
            if ( eps < 0 || eps > xi ){
                q = 0.0;
            }else{
                  q = la * eps;
            }
            '''

def create_demo_object():

    m_la, std_la = 10., 1.0
    m_xi, std_xi = 1.0, 0.1

    # discretize the control variable (x-axis)
    e_arr = np.linspace(0, 2.0, 80)

    # n_int range for sampling efficiency test
    powers = np.linspace(1, math.log(500, 10), 50)
    n_int_range = np.array(np.power(10, powers), dtype = int)

    #===========================================================================
    # Randomization
    #===========================================================================
    s = SPIRRID(q = fiber_tt_2p(),
                e_arr = e_arr,
                n_int = 10,
                tvars = dict(la = RV('norm', m_la, std_la),
                             xi = RV('norm', m_xi, std_xi)
                             ),
                )

    #===========================================================================
    # Exact solution
    #===========================================================================
    def mu_q_ex(e, m_xi, std_xi, m_la):
        return e * (0.5 - 0.5 *
                    erf(0.5 * math.sqrt(2) * (e - m_xi) / std_xi)) * m_la

    #===========================================================================
    # Lab
    #===========================================================================
    slab = SPIRRIDLAB(s = s, save_output = False, show_output = True,
                      dpi = 300,
                      exact_arr = mu_q_ex(e_arr, m_xi, std_xi, m_la),
                      plot_mode = 'subplots',
                      n_int_range = n_int_range,
                      extra_compiler_args = True,
                      le_sampling_lst = ['LHS', 'PGrid'],
                      le_n_int_lst = [440, 5000])

    return slab

import types

if __name__ == '__main__':

    slab = create_demo_object()

    slab.configure_traits()

    #===============================================================================
    # RUN SPIRRID_LAB TESTS
    #===============================================================================
    #===========================================================================
    # Compare efficiency of sampling types 
    #===========================================================================
#    powers = np.linspace(1, math.log(1000, 10), 100)
#    n_int_range = np.array(np.power(10, powers), dtype = int)
#
#    slab.sampling_efficiency()

    #===========================================================================
    # Compare the structure of sampling
    #===========================================================================

    #slab.sampling_structure(ylim = 18.0, xlim = 1.2,)

    #===========================================================================
    # Compare the language efficiency
    #===========================================================================
#    e_arr = np.linspace(0, 2.0, 80)
#    s.set(e_arr = e_arr, n_int = 400)
#    slab.set(n_recalc = 2, exact_arr = mu_q_ex(e_arr, m_xi, std_xi, m_la))
#    slab.codegen_language_efficiency(extra_compiler_args = False)
