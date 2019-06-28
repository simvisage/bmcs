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
# Created on Jun 2, 2010 by: rch


import os

from matplotlib import pyplot as plt
from numpy import sign, linspace, array
from stats.spirrid_bak.i_rf import \
    IRF
from stats.spirrid_bak.rf import \
    RF

from traits.api import \
    Float, Str


def Heaviside(x):
    return (sign(x) + 1.0) / 2.0


class Filament(RF):
    '''Response of an elastic brittle filament with
    slack and delayed activation.
    '''

    #implements( IRF )

    title = Str('brittle filament')

    xi = Float(0.017857, auto_set=False, enter_set=True,
               distr=['weibull_min', 'uniform'],
               scale=0.0178, shape=4.0)

    theta = Float(0.01, auto_set=False, enter_set=True,
                  distr=['uniform', 'norm'],
                  loc=0.01, scale=0.001)

    lambd = Float(0.2, auto_set=False, enter_set=True,
                  distr=['uniform'],
                  loc=0.0, scale=0.1)

    A = Float(5.30929158457e-10, auto_set=False, enter_set=True,
              distr=['weibull_min', 'uniform', 'norm'],
              scale=5.3e-10, shape=8)

    E_mod = Float(70.0e9, auto_set=False, enter_set=True,
                  distr=['weibull_min', 'uniform', 'norm'],
                  scale=70e9, shape=8)

    eps = Float(ctrl_range=(0, 0.2, 20), auto_set=False, enter_set=True)

    x_label = Str('force [N]', enter_set=True, auto_set=False)
    y_label = Str('sigma', enter_set=True, auto_set=False)

    C_code = '''
            double eps_ = ( eps - theta * ( 1 + lambd ) ) /
                             ( ( 1 + theta ) * ( 1 + lambd ) );
            // Computation of the q( ... ) function
            if ( eps_ < 0 || eps_ > xi ){
                q = 0.0;
            }else{
                  q = E_mod * A * eps_;
            }
        '''

    def __call__(self, eps, xi, theta, lambd, A, E_mod):
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


if __name__ == '__main__':
    bf = Filament()
    print('keys', bf.param_keys)
    print('values', bf.param_values)

    print('uniform', bf.traits(distr=lambda x: x != None and 'uniform' in x))

    X = linspace(0, 0.05, 100)
    Y = []
    for eps in X:
        Y.append(bf(eps, .017, .01, .2, 5.30929158457e-10, 70.e9))
    plt.plot(X, Y, linewidth=2, color='navy')
    plt.show()
