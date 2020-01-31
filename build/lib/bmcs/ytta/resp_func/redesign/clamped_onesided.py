#-------------------------------------------------------------------------
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
# Created on Jun 14, 2010 by: rch

from math import pi

from matplotlib import pyplot as plt
from numpy import \
    sign, sqrt, linspace
from stats.spirrid_bak.i_rf import \
    IRF
from stats.spirrid_bak.rf import \
    RF
from traits.api import \
    HasTraits, Float, Str, Range, Property, cached_property


def H(x):
    return (sign(x) + 1.0) / 2.0


class PullOutWithConstantFriction(RF):
    '''
    @todo:
    1) merge the model for constant friction for calculating
        - P( u ) diagram 
        - sigma_m( x, u ) diagram
        Note that taking u as a uniform distribution in the studied range 
        delivers directly the data points of the sigma_m( x ) curve. 
        It would be enough to reshape them - (no loop over u necessary)
        Would it be possible to say spirrid that it should not integrate 
        over a certain dimension?

    2) check the units - take material parameters from the dissertation thesis of Konrad

    3) The crack bridge object contains two spirrid solvers and delivers 
       the delta( P ) curve needed by the SCM module.

    '''

    #implements( IRF )

    title = Str(
        'stress profile in a clamped fiber with constant friction to a stiff matrix')

    E_f = Float(70.0e6, input=True, unit='MPa', distr=['uniform'],
                desc='fiber modulus of elasticity')

    D_f = Float(26e-3, input=True, unit='mm', distr=['uniform', 'norm'],
                desc='fiber diameter')

    tau = Float(3.4, unit='MPa', auto_set=False, enter_set=True, rf_change=True,
                distr=['uniform', 'norm'], desc='interface friction')

    cs = Float(30., unit='mm', auto_set=False, enter_set=True, rf_change=True,
               distr=['uniform', 'norm'], desc='crack spacing')

    l = Float(2.0, auto_set=False, enter_set=True,
              distr=['uniform', 'norm'], desc='free length')

    A_f = Property(depends_on='D_f', unit='mm^2')

    @cached_property
    def _get_A_f(self):
        return pi * (self.D_f / 2.0) ** 2

    P_f = Property(depends_on='D_f', unit='mm^2')

    @cached_property
    def _get_P_f(self):
        return pi * self.D_f

    T = Property(depends_on='tau, D_f', unit='N/mm')

    @cached_property
    def _get_T(self):
        return self.tau * self.P_f

    u = Float(auto_set=False, enter_set=True,
              ctrl_range=(0.0, 1.0, 10))

    x_label = Str('displacement [mm]')
    y_label = Str('force [N]')

    C_code = Str('')

    #
    def __call__(self, u, tau, l, E_f, D_f, cs):
        '''Get the pullout force
        '''

        A_f = pi * (D_f / 2.0) ** 2
        P_f = pi * D_f
        T = tau * P_f
        # P-u with infinite fiber embedded in matrix
        q = -l * T + sqrt((l * T) ** 2 + 2 * E_f * A_f * T * u)

        # P-u with clamped fiber end
        d = self.cs / 2.
        u0 = T * (d ** 2 - l ** 2) / (2 * E_f * A_f)
        q = q * H(T * (d - l) - q) + (E_f * A_f * (u - u0) /
                                      d + T * (d - l)) * H(q - (d - l) * T)

        return q


class StressInFiberWithConstantFriction(PullOutWithConstantFriction):

    #implements( IRF )

    # @todo - define them as a range
    # - define embedded loops for compiled eps_loop
    # - visualization of nd response? - mayavi, cutting, slicing in spirrid_view?
    #
    x = Float(auto_set=False, enter_set=True,
              ctrl_range=(0.0, 200.0, 100))

    u = Float(auto_set=False, enter_set=True,
              ctrl_range=(0.0, 0.05, 100))

    def __call__(self, x, u, tau, l, E_f, D_f):
        '''Calculate the stress transfer 
        length associated with the current displacement.
        '''
        P_f = pi * self.D_f
        T = tau * P_f
        q = super(StressInFiberWithConstantFriction,
                  self).__call__(u, tau, l, E_f, D_f)
        xi = x - l
        return (q - (T * xi * H(xi))) * H(q - T * xi)


if __name__ == '__main__':

    u = linspace(0, 0.003, 100)
    P = PullOutWithConstantFriction()
    plt.plot(u, P(u, 3.4, 3.0, 70e6, 26e-3))
    plt.show()

    rf = StressInFiberWithConstantFriction(u=0.01)

    # print rf.ctrl_keys

#    rf.configure_traits()

    from stats.spirrid_bak.spirrid_nd import SPIRRID

#    X = linspace( 0, 1.5, 100 )
#    Y = rf( X, 0.01, 0.8, 0.02, 7e10, 1e-9 )
#    plt.plot( X, Y, linewidth = 2 )

    s = SPIRRID(rf=rf,
                cached_dG=True,
                compiled_QdG_loop=False,
                compiled_eps_loop=False
                )

    # construct the random variables

    n_int = 20

    s.add_rv('tau', distribution='uniform', loc=0.1, scale=4.0, n_int=n_int)
    #s.add_rv( 'l', distribution = 'uniform', loc = 0.0, scale = 5.0, n_int = n_int )
    #s.add_rv( 'D_f', distribution = 'uniform', loc = 25.0e-3, scale = 5.0e-3, n_int = n_int )

    #
    eps_list = s.eps_list
    mu_q = s.mu_q_grid

    # extract values from the grid - interpolate? search criteria - get the nonzero domain
    #
    #s.plot( x_axis = 'x' )

    for i, x in enumerate(eps_list[0]):
        plt.plot(eps_list[0], mu_q[:, i], label=str(x))

#    print s.mean_curve.ydata #.plot( plt )
#
    # plt.legend()

    # plt.show()

#    from numpy import linspace, ones
#    from scipy.interpolate import RectBivariateSpline
#
#    def f( x, y ):
#        spline = RectBivariateSpline( eps_list[0], eps_list[1], mu_q )
#        return spline.ev( x, y )
#
#    e = linspace( 0, 120, 100 )
#    plt.plot( e, f( e, ones( len( e ) ) * 0.009 ), color = 'black', lw = 3, ls = '--' )
#    plt.show()
