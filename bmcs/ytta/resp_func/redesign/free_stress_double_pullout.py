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

from math import e, pi

from matplotlib import pyplot as plt
from numpy import sqrt, linspace, sign, max, abs
from stats.spirrid_bak.i_rf import IRF
from stats.spirrid_bak.rf import RF

from enthought.traits.api import \
    Float, Int, Str
from enthought.traits.ui.api import \
    View, Item
from enthought.traits.ui.menu import OKButton, CancelButton


def H(x):
    return sign(sign(x) + 1.)


class ConstantFrictionFiniteFiber(RF):
    '''Pullout of fiber from a matrix; stress criterion for debonding, free fiber end'''

    # implements( IRF )

    title = Str('double sided pull-out - short fiber with constant friction')

    E_f = Float(200e+3 , auto_set=False, enter_set=True,
                desc='filament stiffness [N/mm2]',
                distr=['uniform', 'norm'],
                scale=210e3, shape=0)

    d = Float(0.3, auto_set=False, enter_set=True,
                desc='filament diameter [mm]',
                distr=['uniform', 'norm'],
                scale=0.5, shape=0)

    z = Float(0.0, auto_set=False, enter_set=True,
                desc='fiber centroid distance from crack [mm]',
                distr=['uniform'],
                scale=8.5, shape=0)

    L_f = Float(17.0, auto_set=False, enter_set=True,
                desc='fiber length [mm]',
                distr=['uniform', 'norm'],
                scale=30, shape=0)

    tau_fr = Float(1.76, auto_set=False, enter_set=True,
                desc='bond shear stress [N/mm2]',
                distr=['norm', 'uniform'],
                scale=1.76, shape=0.5)

    f = Float(0.03, auto_set=False, enter_set=True,
            desc='snubbing coefficient',
            distr=['uniform', 'norm'],
                scale=0.05, shape=0)

    phi = Float(0.0, auto_set=False, enter_set=True,
       desc='inclination angle',
       distr=['cos_distr'],
                scale=1.0, shape=0)

    w = Float(ctrl_range=(0, 0.016, 20), auto_set=False, enter_set=True)

    x_label = Str('crack_opening [mm]', enter_set=True, auto_set=False)
    y_label = Str('force [N]', enter_set=True, auto_set=False)

    C_code = ''

    def __call__(self, w, tau_fr, L_f, d, E_f, z, phi, f):
        le = L_f / 2. - abs(z)
        le = le * H(le)
        tau = tau_fr * pi * d
        E = E_f
        A = d ** 2 / 4. * pi

        P_deb = sqrt(E * A * tau * w) * e ** (f * phi)
        P_pull = le * tau * e ** (f * phi)
        P = P_deb * H(le * tau - P_deb) + P_pull * H(P_deb - le * tau)

        return P

    traits_view = View(Item('E_f'),
                        Item('d'),
                        Item('f'),
                        Item('phi'),
                        Item('z'),
                        Item('tau_fr'),
                        resizable=True,
                        scrollable=True,
                        height=0.8, width=0.8,
                        buttons=[OKButton, CancelButton]
                        )


if __name__ == '__main__':
    po = ConstantFrictionFiniteFiber()
    w = linspace(0.0, 0.016, 100)
    P = po(w, 1.76, 17.0, 0.3, 200e3, 0.0, 0.0, 0.03)
    plt.plot(w, P)
    plt.show()
    # po.configure_traits()

