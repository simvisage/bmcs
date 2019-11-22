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

import math
import os

from scipy.interpolate import interp1d

from etsproxy.traits.api import Str
import numpy as np
from stats.spirrid import  SPIRRID, RV, RF, IRF, Heaviside
from stats.spirrid.extras import SPIRRIDLAB

file_dir = os.path.dirname(os.path.abspath(__file__))


#===========================================================================
# Response function
#===========================================================================
class fiber_tt_5p(RF):
    r'''
Response function of a single fiber 
===================================

Response of a fiber loaded in tension can be described by a linear function
with the domain bounded from left and right by the Heaviside terms.

..    math::
    q\left(\varepsilon; E,A,\theta,\lambda,\xi\right) = E A \frac{{\varepsilon
    - \theta \left( {1 + \lambda } \right)}}{{\left( {1 + \theta } \right)
    \left( {1 + \lambda } \right)}}
    H\left[ {e - \theta \left( {1 + \lambda } \right)} \right]
    H\left[ {\xi  - \frac{{e - \theta \left( {1 + \lambda } \right)}}
    {{\left( {1 + \theta } \right)\left( {1 + \lambda } \right)}}} \right]

where the variables :math:`A=` cross-sectional area, :math:`E=` Young's modulus,
:math:`\theta=` filament activation strain, :math:`\lambda=` ratio of extra
(stretched) filament length to the nominal length and :math:`\xi=` breaking strain
are considered random and normally distributed. The function :math:`H(\eta)`
represents the Heaviside function with values 0 for :math:`\eta < 0`
and 1 for :math:`\eta > 0`.

'''
    implements(IRF)

    title = Str('brittle filament')

    def __call__(self, eps, lambd, xi, E_mod, theta, A):
        '''
        Implements the response function with arrays as variables.
        first extract the variable discretizations from the orthogonal grid.
        '''
        # NOTE: as each variable is an array oriented in different direction
        # the algebraic expressions (-+*/) perform broadcasting,. i.e. performing
        # the operation for all combinations of values. Thus, the resulgin eps
        # is contains the value of local strain for any combination of 
        # global strain, xi, theta and lambda 
        #

        eps_ = (eps - theta * (1 + lambd)) / ((1 + theta) * (1 + lambd))

        # cut off all the negative strains due to delayed activation
        # 
        eps_ *= Heaviside(eps_)

        # broadcast eps also in the xi - dimension 
        # (by multiplying with array containing ones with the same shape as xi )
        #
        eps_grid = eps_ * Heaviside(xi - eps_)

        # cut off all the realizations with strain greater than the critical one.
        # 
        # eps_grid[ eps_grid >= xi ] = 0

        # transform it to the force
        # 
        q_grid = E_mod * A * eps_grid

        return q_grid

    cython_code = '''
            eps_ = ( eps - theta * ( 1 + lambd ) ) / ( ( 1 + theta ) * ( 1 + lambd ) )
            # Computation of the q( ... ) function
            if eps_ < 0 or eps_ > xi:
                q = 0.0
            else:
                q = E_mod * A * eps_
        '''

    c_code = '''
            double eps_ = ( eps - theta * ( 1 + lambd ) ) /
                             ( ( 1 + theta ) * ( 1 + lambd ) );
            // Computation of the q( ... ) function
            if ( eps_ < 0 || eps_ > xi ){
                q = 0.0;
            }else{
                q = E_mod * A * eps_;
            }
        '''


def create_demo_object():

    D = 26 * 1.0e-6  # m
    A = (D / 2.0) ** 2 * math.pi

    # set the mean and standard deviation of the two random variables
    la_mean, la_stdev = 0.0, 0.2
    xi_mean, xi_stdev = 0.019027, 0.0022891
    E_mean, E_stdev = 70.0e+9, 15.0e+9
    th_mean, th_stdev = 0.0, 0.01
    A_mean, A_stdev = A * 0.3, 0.7 * A

    do = 'norm'

    if do == 'general':

        # set the mean and standard deviation of the two random variables
        la_mean, la_stdev = 0.0, 0.2
        xi_mean, xi_stdev = 0.019027, 0.0022891
        E_mean, E_stdev = 70.0e+9, 15.0e+9
        th_mean, th_stdev = 0.0, 0.01
        A_mean, A_stdev = A * 0.3, 0.7 * A

        # construct the normal distributions and get the methods
        # for the evaluation of the probability density functions
        g_la = RV('uniform', la_mean, la_stdev)
        g_xi = RV('norm', xi_mean, xi_stdev)
        g_E = RV('uniform', E_mean, E_stdev)
        g_th = RV('uniform', th_mean, th_stdev)
        g_A = RV('uniform', A_mean, A_stdev)

        mu_ex_file = 'fiber_tt_5p_30.txt'
        delimiter = ','

    elif do == 'uniform':

        # set the mean and standard deviation of the two random variables
        la_mean, la_stdev = 0.0, 0.2
        xi_mean, xi_stdev = 0.01, 0.02
        E_mean, E_stdev = 70.0e+9, 15.0e+9
        th_mean, th_stdev = 0.0, 0.01
        A_mean, A_stdev = A * 0.3, 0.7 * A

        # construct the uniform distributions and get the methods
        # for the evaluation of the probability density functions
        g_la = RV('uniform', la_mean, la_stdev)
        g_xi = RV('uniform', xi_mean, xi_stdev)
        g_E = RV('uniform', E_mean, E_stdev)
        g_th = RV('uniform', th_mean, th_stdev)
        g_A = RV('uniform', A_mean, A_stdev)

        mu_ex_file = 'fiber_tt_5p_40_unif.txt'
        delimiter = ' '

    elif do == 'norm':

        # set the mean and standard deviation of the two random variables
        la_mean, la_stdev = 0.1, 0.02
        xi_mean, xi_stdev = 0.019027, 0.0022891
        E_mean, E_stdev = 70.0e+9, 15.0e+9
        th_mean, th_stdev = 0.005, 0.001
        A_mean, A_stdev = 5.3e-10, 1.0e-11

        # construct the normal distributions and get the methods
        # for the evaluation of the probability density functions
        g_la = RV('norm', la_mean, la_stdev)
        g_xi = RV('norm', xi_mean, xi_stdev)
        g_E = RV('norm', E_mean, E_stdev)
        g_th = RV('norm', th_mean, th_stdev)
        g_A = RV('norm', A_mean, A_stdev)

        mu_ex_file = os.path.join(file_dir, 'fiber_tt_5p_n_int_40_norm_exact.txt')
        delimiter = ' '

    # discretize the control variable (x-axis)
    e_arr = np.linspace(0, 0.04, 40)

    # n_int range for sampling efficiency test
    powers = np.linspace(1, math.log(20, 10), 15)
    n_int_range = np.array(np.power(10, powers), dtype=int)

    #===========================================================================
    # Randomization
    #===========================================================================
    s = SPIRRID(q=fiber_tt_5p(),
                e_arr=e_arr,
                n_int=10,
                tvars=dict(lambd=g_la, xi=g_xi, E_mod=g_E, theta=g_th, A=g_A),
                )

    # Exact solution
    def mu_q_ex(e):
        data = np.loadtxt(mu_ex_file, delimiter=delimiter)
        x, y = data[:, 0], data[:, 1]
        f = interp1d(x, y, kind='linear')
        return f(e)

    #===========================================================================
    # Lab
    #===========================================================================
    slab = SPIRRIDLAB(s=s, save_output=False, show_output=True, dpi=300,
                      exact_arr=mu_q_ex(e_arr),
                      plot_mode='subplots',
                      n_int_range=n_int_range,
                      extra_compiler_args=True,
                      le_sampling_lst=['LHS', 'PGrid'],
                      le_n_int_lst=[25, 30])

    return slab


if __name__ == '__main__':

    slab = create_demo_object()
    slab.configure_traits()

    #===========================================================================
    # RUN SPIRRID_LAB TESTS
    #===========================================================================
    #===========================================================================
    # Compare efficiency of sampling types 
    #===========================================================================
#    powers = np.linspace(1, math.log(20, 10), 15)
#    n_int_range = np.array(np.power(10, powers), dtype = int)

    # slab.sampling_efficiency(n_int_range = n_int_range)

    #===========================================================================
    # Compare the structure of sampling
    #===========================================================================

    # slab.sampling_structure( ylim = 1.1, xlim = 0.04 )

    #===========================================================================
    # Compare the code efficiency
    #===========================================================================

    # s.set(e_arr = np.linspace(0, 0.04, 20), n_int = 40)
    # s.sampling_type = 'PGrid'
    # slab.codegen_efficiency()

    #===========================================================================
    # Compare the language efficiency
    #===========================================================================
#    e_arr = np.linspace(0, 0.04, 40)
#    s.set(e_arr = e_arr)
#    print 'n_rv', s.sampling.n_rand_vars
#    slab.set(n_recalc = 2, exact_arr = mu_q_ex(e_arr))
#    slab.codegen_language_efficiency(extra_compiler_args = False,
#                                     sampling_list = ['LHS'], # 'PGrid'],
#                                     n_int_list = [25]) # , 30])

