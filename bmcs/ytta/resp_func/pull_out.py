'''
Created on Mar 17, 2010

@author: rostislav
'''

""" defining the pullout model as a traited class """

from math import pi, e

from numpy import hstack, linspace, infty, array, linspace, tanh, sign, sqrt, \
    argmax, where, sign, all, cos
from scipy.optimize import fsolve, brentq
from traits.api import Float, Instance, DelegatesTo, \
    Property, Interface, Str, List, HasTraits, Bool
from traitsui.api import View, Item, VGroup
from util.traits.either_type import EitherType

from .parameters import Geometry, Material, Plot


def Heaviside(x):
    return (sign(x) + 1) / 0.5


class PullOut(HasTraits):

    param_names = List

    material = Instance(Material)

    def _material_default(self):
        return Material()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    plot = Instance(Plot)

    def _plot_default(self):
        return Plot()

    material_choice = DelegatesTo('material')

    Ef = DelegatesTo('material')

    k = DelegatesTo('material')

    qf = DelegatesTo('material')

    G = DelegatesTo('material')

    f = DelegatesTo('material')

    fu = DelegatesTo('material')

    include_fu = DelegatesTo('material')

    qy = DelegatesTo('material')

    beta = DelegatesTo('material')

    rf = DelegatesTo('geometry')

    phi = DelegatesTo('geometry')

    l = DelegatesTo('geometry')

    u_plot = DelegatesTo('plot')

    w_plot = DelegatesTo('plot')

    yvalues = DelegatesTo('plot')

    z = DelegatesTo('geometry')

    Lf = DelegatesTo('geometry')

    Lef = DelegatesTo('geometry')

    L = DelegatesTo('geometry')

    Af = DelegatesTo('geometry')

    p = DelegatesTo('geometry')

    clamp = Bool(True)

    tau = Property(Float, depends_on='qf', label='tau')

    def _get_tau(self):
        return self.qf / (self.p)

    Pu = Property(Float, depends_on='fu, rf, phi', label='Pu')

    def _get_Pu(self):
        return self.fu * self.Af * cos(self.phi)

    w = Property(Float, depends_on='rf, k, Ef', label='w')

    def _get_w(self):
        return sqrt(self.k / self.Ef / self.Af)

    def get_clamp(self, a):
        if self.clamp == False:
            return tanh(self.w * (self.L - a))
        else:
            return 1 / tanh(self.w * (self.L - a))

    def get_P(self, a):
        ''' debonding force - default as having the same value as the
        frictional resistance in the debonded region
        get_P() is overwritten by stress or energy criterion for debonding'''
        return a * self.qf * e ** (self.phi * self.f)

    def u_a_residuum(self, a):
        ''' for computing the maximum a for infinite embedding '''
        Ef = self.Ef
        A = self.Af
        w = self.w
        P_deb = self.get_P(a)
        u = (P_deb - self.qf * a) / Ef / A / w / self.get_clamp(a) + \
            (P_deb - .5 * self.qf * a) / Ef / A * a + P_deb * self.l / A / Ef
        return self.u_plot - u

    def P_a_residuum(self, a):
        P = self.get_P(a)
        return P - self.Pu

    def get_u(self, P, a):
        ''' takes a- and P-array and returns u-array '''
        Ef = self.Ef
        A = self.Af
        w = self.w

        u = (P - self.qf * a) / Ef / A / w / self.get_clamp(a) + \
            (P - .5 * self.qf * a) / Ef / A * a + P * self.l / A / Ef
        return u

    def u_L0_residuum(self, L0):
        qf = self.qf
        L = self.L
        Ef = self.Ef
        A = self.Af
        l = self.l

        a = linspace(0, L - L / 1e10, 100)
        P_deb = self.get_P(a)
        u_deb = self.get_u(P_deb, a)
        idxmax = argmax(u_deb)
        u_max = u_deb[idxmax]

        P = qf * L0 * (1 + self.beta * (L - L0) / (2 * self.rf))
        delta_u = P * L0 / (2. * Ef * A)
        delta_free_l = (l + L - L0) * P / (Ef * A)
        delta_l = L - L0
        u = delta_u + delta_free_l + delta_l
        return u_max - u

#################################################
# class evaluating x,y (P,u) for infinite fibre #
#################################################

    def continuous_infinite(self):
        ''' returns the u and P array for infinite embedded length '''

        # L is infinite
        ''' returns the u and P array without considering the breaking stress '''
        L = self.L
        Ef = self.Ef
        A = self.Af
        l = self.l
        w = self.w

        try:
            # tries if P for u_plot is higher than load that initiates
            # debonding (Py)
            a_lim = brentq(self.u_a_residuum, 1e-12, 1e15)
            # debonded length 'a' linearly sliced (used as control variable)
            a = linspace(0, a_lim, 100)
            P_infty = self.get_P(a)
            u_infty = self.get_u(P_infty, a)
        except:
            u_infty = self.u_plot
            P_infty = Ef * A * w * u_infty / (w * l + 1 / tanh(w * L))
        _xdata = hstack((0, u_infty))
        _ydata = hstack((0, P_infty)) * e ** (self.f * self.phi)
        return _xdata, _ydata

    def value_infinite(self):
        ''' returns the final x and y arrays for infinite embedded length '''
        Pu = self.Pu
        A = self.Af
        w = self.w
        Ef = self.Ef
        L = self.L
        l = self.l

        if Pu < self.continuous_infinite()[1][1]:
            u_max = Pu / (tanh(w * L) * Ef * A * w) + Pu * l / A / Ef
            xdata = array([0, u_max, u_max])
            ydata = array([0, Pu, 0])
        else:
            a_lim = brentq(self.P_a_residuum, 1e-12, 1e3)
            a = linspace(0, a_lim, 50)
            P_deb = self.get_P(a)
            u_deb = self.get_u(P_deb, a)
            xdata = hstack((0, u_deb, u_deb[-1]))
            ydata = hstack((0, P_deb, 0))
        return xdata, ydata

    def prepare_infinite(self):
        if self.include_fu == True:
            return self.value_infinite()
        else:
            return self.continuous_infinite()

###############################################################
# class evaluating x,y (P,u) for finite embedded fibre length #
###############################################################

    def continuous_finite(self):
        ''' returns the u and P array for finite embedded length '''

        L = self.L
        Ef = self.Ef
        A = self.Af
        qf = self.qf

        a = linspace(0, L - L / 1e10, 100)
        # P-u diagram including snap back
        P_deb_full = self.get_P(a)
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

    def value_finite(self):
        ''' returns the final x and y arrays for finite embedded length '''

        Pu = self.Pu
        A = self.Af
        w = self.w
        Ef = self.Ef

        values = self.continuous_finite()

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
                    u_max = Pu / (tanh(w * self.L) * Ef * A * 
                                  w) + Pu * self.l / A / Ef
                    xdata = array([0, u_max, u_max])
                    ydata = array([0, Pu, 0])
        return xdata, ydata

    def prepare_finite(self):
        if self.include_fu == True:
            return self.value_finite()
        else:
            values = self.continuous_finite()
            u_deb = values[0]
            u_pull = values[1]
            P_deb = values[2] * e ** (self.f * self.phi)
            P_pull = values[3] * e ** (self.f * self.phi)
            xdata = hstack((0, u_deb, u_pull))
            ydata = hstack((0, P_deb, P_pull))
            return xdata, ydata

#################################################
#    evaluating x,y (P,u) for clamped fibre end #
#################################################

    def continuous_clamp(self):
        ''' returns the u and P array for finite embedded length '''

        L = self.L
        Ef = self.Ef
        A = self.Af
        l = self.l
        w = self.w

        # for clamped fibre end
        try:
            # tries if P for u_plot is higher than load that initiates
            # debonding (Py)
            a_lim = brentq(self.u_a_residuum, 0, self.L - self.L / 1e15)
            a = linspace(0, a_lim, 100)
            P_clamp = self.get_P(a)
            u_clamp = self.get_u(P_clamp, a)
        except:
            u_clamp = self.u_plot
            P_clamp = Ef * A * w * u_clamp / (w * l + tanh(w * L))
        _xdata = hstack((0, u_clamp))
        _ydata = hstack((0, P_clamp)) * e ** (self.f * self.phi)
        return _xdata, _ydata

    def value_clamp(self):
        ''' returns the final x and y arrays for clamped fibre end '''

        Pu = self.Pu
        A = self.Af
        w = self.w
        Ef = self.Ef
        L = self.L
        l = self.l

        if Pu < self.continuous_clamp()[1][1]:
            u_max = Pu / (1 / tanh(w * L) * Ef * A * w) + Pu * l / A / Ef
            xdata = array([0, u_max, u_max])
            ydata = array([0, Pu, 0])
        else:
            a_lim = brentq(self.P_a_residuum, 1e-12, self.L - self.L / 1e15)
            a = linspace(0, a_lim, 50)
            P_deb = self.get_P(a)
            u_deb = self.get_u(P_deb, a)
            xdata = hstack((0, u_deb, u_deb[-1]))
            ydata = hstack((0, P_deb, 0))
        return xdata, ydata

    def prepare_clamp(self):
        if self.include_fu == True:
            return self.value_clamp()
        else:
            return self.continuous_clamp()


if __name__ == "__main__":
    pullout = PullOut()
    pullout.prepare_clamp()
