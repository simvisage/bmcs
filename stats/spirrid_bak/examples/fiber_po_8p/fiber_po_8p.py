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

from stats.spirrid import SPIRRID, RV, Heaviside
from stats.spirrid.extras import SPIRRIDLAB
import math
import numpy as np
from etsproxy.traits.api import Float, Str, implements, Range
from math import pi, e
from numpy import  cos, sqrt, exp
from stats.spirrid import IRF, RF

class ConstantFrictionFiniteFiber(RF):
    r'''
Fiber with constant friction
============================

Function describing the pullout of a fiber decomposed into the stage of 
debonding and pullout as:

..    math:: 
    P(w; f_\mathrm{u}, q_\mathrm{f}, L, A, E, z, \varphi, f) = P_\mathrm{deb} + P_\mathrm{pull}\\

During the debonding process, the force is obtained as 

..    math::
    P_\mathrm{deb, full} = \sqrt{2EAq_\mathrm{f}w} e^{f\varphi }

Once the debonding reaches the end of the fiber, the pullout force is calculated as

..    math::
    P_\mathrm{pull, full} = \left[ \ell_\mathrm{e}q_\mathrm{f} - \frac{\ell _\mathrm{e}q_\mathrm{f}}{\ell _\mathrm{e} - w_\mathrm{deb}} \left( w - w_\mathrm{deb} \right) \right] e^{f\varphi }

..    math::
    w_\mathrm{deb} = \frac{\ell_\mathrm{e}^2 q_\mathrm{f}}{2 E A}

The resulting equation defines the ranges for debbonding and pullout

..    math::
    P_\mathrm{full} = P_\mathrm{deb, full} \cdot
    H\left( w_\mathrm{deb} - w \right) \cdot H\left( w \right)\\
    + P_\mathrm{pull, full} \cdot H\left( w - w_\mathrm{deb} \right)

and includes the breaking strain

..    math::
    P = P_\mathrm{full} \cdot H\left( f_\mathrm{u}A - P_\mathrm{full} \right)

The effective fiber length is given as

..    math::
    \ell _\mathrm{e} = \frac{L}{2} - \frac{z}{\cos \left( \varphi  \right)}

where 
 * :math:`w=` crack width, 
 * :math:`L=` fiber length,
 * :math:`\ell_\mathrm{e}=` embedded length, 
 * :math:`\varphi=` orientation angle of the fiber, 
 * :math:`E=` Young's modulus, 
 * :math:`A=` cross-sectional area of fiber, 
 * :math:`z=` distance of fibercentroid from crack plane, 
 * :math:`f=` snubbing coefficient,
 * :math:`w_\mathrm{deb}=` crack width in debonding stage, 
 * :math:`q_\mathrm{f}=` frictional stress, 
 * :math:`f_\mathrm{u}=` strength, 
 * :math:`P_\mathrm{deb}=` force in debonding stage, and 
 * :math:`P_\mathrm{pull}=` force in pullout stage.
 '''

    implements(IRF)

    title = Str('pull-out with constant friction')

    fu = Float(1200e6, auto_set = False, enter_set = True,
                distr = ['weibull_min'])

    qf = Float(1500, auto_set = False, enter_set = True,
                distr = ['uniform', 'norm'])

    L = Float(0.02, auto_set = False, enter_set = True,
               distr = ['uniform'])

    A = Float(5.30929158457e-10, auto_set = False, enter_set = True,
               distr = ['uniform', 'weibull_min'])

    E_mod = Float(70.0e9, auto_set = False, enter_set = True,
                   distr = ['uniform'])

    phi = Range(0, pi, auto_set = False, enter_set = True,
                 distr = ['sin_distr'])

    z = Float(0, auto_set = False, enter_set = True,
               distr = ['uniform'])

    f = Float(0.01, auto_set = False, enter_set = True,
               distr = ['uniform'])

    w = Float(ctrl_range = (0, 0.016, 20), auto_set = False, enter_set = True)

    c_code = '''
            double w = eps;
            double Le = L / 2. - z / cos( phi );
            double w_deb = exp( f * phi ) * qf * pow(Le,2.0) / E_mod / A;
            double P_deb_full = sqrt( 2. * w / 2. * E_mod * A * qf ) * exp( f * phi );
            double P_deb;
            
            // Heaviside
            if ( Le < 0 || P_deb_full > fu * A || w > w_deb ){
                P_deb = 0;
            }else{
                P_deb =P_deb_full;
            }
            
            double P_pull_x = ( Le * qf - Le * qf / ( Le - w_deb ) * ( w - w_deb ) ) * exp( f * phi );
            double P_pull;
            
            // Heaviside 
            if ( P_pull_x < 0 || w_deb > w ){
                P_pull = 0;
            }else{
                P_pull = P_pull_x;
            }
            
            // Computation of the q( ... ) function
            q = P_deb + P_pull;
        '''

    def __callx__(self, w, fu, qf, L, A, E_mod, z, phi, f):
        '''Intial vectorized implementation - without regarding
        the lexical structure of the expression.
        '''
        Le = L / 2. - z / cos(phi)
        w_deb = e ** (f * phi) * qf * Le ** 2.0 / E_mod / A
        P_deb_full = sqrt(2. * w / 2. * E_mod * A * qf) * e ** (f * phi)
        P_deb = P_deb_full * Heaviside(fu * A - P_deb_full) * Heaviside(w_deb - w) * Heaviside(Le)
        P_pull_x = (Le * qf - Le * qf / (Le - w_deb) * (w - w_deb)) * e ** (f * phi)
        P_pull = P_pull_x * Heaviside(P_pull_x) * Heaviside(w - w_deb)
        return P_deb + P_pull

    def __call__(self, w, fu, qf, L, A, E_mod, z, phi, f):
        '''Lexically optimized expresseion - each result is 
        calculated only once. 
        Further optimization possible by printing out the shape
        and doing inplace operation where possible.
        However, this does not seem to have a significant effect.
        '''
        t4 = sqrt(w * E_mod * A * qf)
        t5 = f * phi
        t6 = exp(t5)
        t7 = t4 * t6
        t11 = Heaviside(fu * A - 0.1000000000e1 * t7)
        t12 = exp(t5);
        t14 = 0.5000000000e0 * L
        t15 = cos(phi)
        t17 = z / t15
        t18 = t14 - t17
        t19 = pow(t18, 0.20e1)
        t24 = t12 * qf * t19 / E_mod / A
        t25 = t24 - w
        t26 = Heaviside(t25)
        t28 = Heaviside(t18)
        t32 = t18 * qf
        t38 = (t32 + t32 / (t14 - t17 - t24) * t25) * t6
        t39 = Heaviside(t38)
        t40 = Heaviside(-t25)
        res = 0.1000000000e1 * t7 * t11 * t26 * t28 + t38 * t39 * t40
        return res

def create_demo_object():

    #===========================================================================
    # Control variable
    #===========================================================================
    e_arr = np.linspace(0, 0.012, 80)

    powers = np.linspace(1, math.log(20, 10), 6)
    n_int_range = np.array(np.power(10, powers), dtype = int)

    #===========================================================================
    # Randomization
    #===========================================================================
    tvars = dict(fu = RV('weibull_min', 1200.0e6, 200.),
                  qf = 1500.0,
                  # qf = RV('uniform', 1500., 100.),
                  L = 0.02, # 
                  # L = RV('uniform', 0.02, 0.02 / 2.),
                  A = RV('norm', 5.30929158457e-10, .03 * 5.30929158457e-10),
                  E_mod = RV('uniform', 70.e9, 250.e9),
                  z = RV('uniform', 0.0, 0.03),
                  phi = 0.0, # 
                  # phi = RV('cos_distr', 0.0, 1.0),
                  # phi = RV('uniform', 0.0, 1.0),
                  f = RV('uniform', 0.0, 0.03))

    #===========================================================================
    # Integrator object
    #===========================================================================
    s = SPIRRID(q = ConstantFrictionFiniteFiber(),
                e_arr = e_arr,
                n_int = 10,
                tvars = tvars,
                )

    #===========================================================================
    # Lab
    #===========================================================================
    slab = SPIRRIDLAB(s = s, save_output = False, show_output = True, dpi = 300,
                      qname = 'fiber_po_8p',
                      plot_mode = 'subplots',
                      n_int_range = n_int_range,
                      extra_compiler_args = True,
                      le_sampling_lst = ['LHS', 'PGrid'],
                      le_n_int_lst = [10, 10],
                      plot_sampling_idx = [0, 3, ]
                      )

    return slab

if __name__ == '__main__':

    slab = create_demo_object()
    slab.configure_traits()

    #===========================================================================
    # Compare efficiency of sampling types 
    #===========================================================================
#    powers = np.linspace(1, math.log(20, 10), 6)
#    n_int_range = np.array(np.power(10, powers), dtype = int)
    #slab.sampling_efficiency(n_int_range = n_int_range)

    #===========================================================================
    # Compare the structure of sampling
    #===========================================================================

    #slab.sampling_structure(ylim = 10.0, xlim = 0.012, plot_idx = [0, 3])

    #===========================================================================
    # Compare the code efficiency
    #===========================================================================

    slab.codegen_language_efficiency()

