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
from numpy import sqrt, linspace, sign, hstack, argmax, where, array, tanh, cos
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from stats.spirrid_bak import IRF, RF

from enthought.traits.api import Float, Int, Str, Property, Bool
from enthought.traits.ui.api import View, Item
from enthought.traits.ui.menu import OKButton, CancelButton


def H(x):
    return sign(sign(x) + 1.)


class FreeStressPullout(RF):
    '''Pullout of fiber from a matrix; stress criterion for debonding, free fiber end'''

    # implements( IRF )

    title = Str('one sided pull-out - short fiber with trilinear bond law')

    E_f = Float(200e+3 , auto_set=False, enter_set=True,
                desc='filament stiffness [N/mm2]',
                distr=['uniform', 'norm'])

    d = Float(0.3, auto_set=False, enter_set=True,
                desc='filament diameter [mm]',
                distr=['uniform', 'norm'])

    z = Float(0.0, auto_set=False, enter_set=True,
                desc='fiber centroid distance from crack [mm]',
                distr=['uniform'])

    L_f = Float(17.0, auto_set=False, enter_set=True,
                desc='fiber length [mm]',
                distr=['uniform', 'norm'])

    k = Float(1.76, auto_set=False, enter_set=True,
            desc='bond shear stiffness [N/mm]',
            distr=['uniform', 'norm'])

    qf = Float(3.76, auto_set=False, enter_set=True,
                desc='bond shear stress [N/mm2]',
                distr=['uniform', 'norm'])

    qy = Float(19.76, auto_set=False, enter_set=True,
                desc='debbonding stress [N/mm2]',
                distr=['uniform', 'norm'])

    fu = Float(500, auto_set=False, enter_set=True,
                desc='fiber breaking stress [N/mm2]',
                distr=['uniform', 'norm'])

    l = Float(0.001, auto_set=False, enter_set=True,
                desc='free length',
                distr=['uniform', 'norm'])

    f = Float(0.03, auto_set=False, enter_set=True,
            desc='snubbing coefficient',
            distr=['uniform', 'norm'])

    phi = Float(0.0, auto_set=False, enter_set=True,
       desc='inclination angle',
       distr=['cos'])

    accuracy = Int(50, auto_set=False, enter_set=True)

    include_fu = Bool(False)

    u = Float(ctrl_range=(0, 0.016, 20), auto_set=False, enter_set=True)

    x_label = Str('displacement [mm]', enter_set=True, auto_set=False)
    y_label = Str('force [N]', enter_set=True, auto_set=False)

    tau = Property(Float, depends_on='qf', label='tau')

    def _get_tau(self):
        return self.qf / (self.p)

    Pu = Property(Float, depends_on='fu, rf, phi', label='Pu')

    def _get_Pu(self):
        return self.fu * self.Af * cos(self.phi)

    w = Property(Float, depends_on='rf, k, Ef', label='w')

    def _get_w(self):
        return sqrt(self.k / self.Ef / self.Af)

    Af = Property(Float, depends_on='d')

    def _get_Af(self):
        return pi * self.d ** 2 / 4.

    def get_P(self, a, qf, qy, L):
        return (self.qf * a + qy / self.w * tanh(self.w * (L - a)))

    def get_u(self, P, a):
        ''' takes a- and P-array and returns u-array '''
        Ef = self.Ef
        A = self.Af
        w = self.w

        u = (P - self.qf * a) / Ef / A / w / self.get_clamp(a) + \
            (P - .5 * self.qf * a) / Ef / A * a + P * self.l / A / Ef
        return u

    def u_L0_residuum(self, L0, qf, L, Ef, A, l):

        a = linspace(0, L - L / 1e10, self.accuracy)
        P_deb = self.get_P(a)
        u_deb = self.get_u(P_deb, a, Ef, A)
        idxmax = argmax(u_deb)
        u_max = u_deb[idxmax]

        P = qf * L0 * (1 + self.beta * (L - L0) / (2 * self.rf))
        delta_u = P * L0 / (2. * Ef * A)
        delta_free_l = (l + L - L0) * P / (Ef * A)
        delta_l = L - L0
        u = delta_u + delta_free_l + delta_l
        return u_max - u

    def continuous_function(self, u, E_f, L_f, d, qy, qf, k, z, phi, f):
        # returns the u and P array for a fiber with infinite strength      

        L = L_f
        Ef = E_f
        A = self.Af

        a = linspace(0, L - L / 1e10, self.accuracy)
        # P-u diagram including snap back
        P_deb_full = self.get_P(a, qf, qy, L)
        u_deb_full = self.get_u(P_deb_full, a)
        idxmax = argmax(u_deb_full)
        # P-u diagram snap back cutted
        u_deb = u_deb_full[0:idxmax + 1]
        P_deb = P_deb_full[0:idxmax + 1]

        # pull-out stage
        # L0 is the embedded length of a pure frictional pull-out that
        # corresponds to the displacement at the end of the debonding stage
        L0 = brentq(self.u_L0_residuum, 1e-12, 2 * L)
        # if L0 is not in interval (0,L), the load drops to zero
        if round(L, 7) >= round(L0, 7) >= 0:
            lp = linspace(L0, 0, 100)
            P_pull = qf * lp * (1 + self.beta * (L - lp) / (2 * self.rf))
            # displacement corresponding to the actual embedded length
            delta_u = P_pull * lp / (2. * Ef * A)
            # displacement corresponding to the actual free length
            delta_free_l = (self.l + L - lp) * P_pull / (Ef * A)
            # displacement corresponding to the free length increment
            delta_l = L - lp
            u_pull = delta_u + delta_free_l + delta_l
            return u_deb, u_pull, P_deb, P_pull
        else:
            u_pull = u_deb[-1]
            P_pull = 0
            return u_deb, u_pull, P_deb, P_pull

    def value_finite(self, u, E_f, L_f, d, qy, qf, k, z, phi, f):
        ''' returns the final x and y arrays for finite embedded length '''

        Pu = self.Pu
        A = self.Af
        w = self.w
        Ef = E_f

        values = self.continuous_finite(u, E_f, L_f, d, qy, qf, k, z, phi, f)

        u_deb = values[0]
        u_pull = values[1]
        P_deb = values[2] * e ** (self.f * self.phi)
        P_pull = values[3] * e ** (self.f * self.phi)

        # if the pull-out force is lower than the breaking force
        if all(hstack((P_deb, P_pull)) < Pu):
            xdata = hstack((0, u_deb, u_pull))
            ydata = hstack((0, P_deb, P_pull))
            # if the breaking force is reached
        else:
            # max force reached in the pull-out stage
            if all(P_deb < Pu):
                max = where(P_pull > Pu)[0][0]
                xdata = hstack((0, u_deb, u_pull[:max + 1]))
                ydata = hstack((0, P_deb, P_pull[:max + 1]))
            else:
                # max force reached during debonding
                if P_deb[1] < Pu:
                    # max force reached after the debonding has started
                    a_lim = brentq(self.P_a_residuum, 1e-12, 1e3)
                    a = linspace(0, a_lim, 50)
                    P_deb = self.get_P(a)
                    u_deb = self.get_u(P_deb, a)
                    xdata = hstack((0, u_deb, u_deb[-1]))
                    ydata = hstack((0, P_deb, 0))
                else:
                    # max force reached before the debonding has started
                    u_max = Pu / (tanh(w * self.L) * Ef * A * w) + Pu * self.l / A / Ef
                    xdata = array([0, u_max, u_max])
                    ydata = array([0, Pu, 0])
        return xdata, ydata

    def __call__(self, u, E_f, L_f, d, qy, qf, k, z, phi, f):
        if self.include_fu == True:
            return self.value_finite(u, E_f, L_f, d, qy, qf, k, z, phi, f)
        else:
            values = self.continuous_finite(u, E_f, L_f, d, qy, qf, k, z, phi, f)
            u_deb = values[0]
            u_pull = values[1]
            P_deb = values[2] * e ** (self.f * self.phi)
            P_pull = values[3] * e ** (self.f * self.phi)
            xdata = hstack((0, u_deb, u_pull))
            ydata = hstack((0, P_deb, P_pull))
            interp_func = interp1d(xdata, ydata)
            return interp_func(u)

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
    po = FreeStressPullout()
    u = linspace(0.0, 0.016, 100)
    P = po(u, 1.76, 17.0, 0.3, 200e3, 0.0, 0.0, 0.03)
    plt.plot(u, P)
    plt.show()
    # po.configure_traits()

