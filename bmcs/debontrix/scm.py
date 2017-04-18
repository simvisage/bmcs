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
# Created on Sep 21, 2009 by: rch

from math import exp

from traits.api import HasTraits, Float, Property, cached_property, Instance
from traitsui.api import View, Item, VGroup, HGroup
from util.traits.either_type import EitherType
from view.plot2d import Viz2D, Vis2D
from view.window import BMCSModel

import numpy as np


class Viz2DSCMSigEps(Viz2D):

    def plot(self, ax, vot, *args, **kw):
        sig, eps = self.vis2d.get_sig_eps()
        ax.plot(sig, eps, *args, **kw)


class SCM(BMCSModel, Vis2D):
    '''
    Stochastic Cracking Theory
    '''
    E_f = Float(70e+3, auto_set=False, enter_set=True,  # [N/mm^2]
                MAT=True,
                desc='E-Modulus of the fiber')

    E_m = Float(34.e+3, auto_set=False, enter_set=True,  # [N/mm^2]
                MAT=True,
                desc='E-Modulus of the matrix')

    tau = Float(8.0, auto_set=False, enter_set=True,  # [N/mm^2]
                MAT=True,
                desc='Frictional stress')

    r = Float(0.5, auto_set=False, enter_set=True,  # [mm]
              MAT=True,
              desc='Radius')

    rho = Float(0.03, auto_set=False, enter_set=True,  # [-]
                MAT=True,
                desc='Reinforcement ratio')

#    reinf_cs = EitherType(klasses=[SimplyRatio, GridReinforcement])

    m = Float(4.0, auto_set=False, enter_set=True,  # [-]
              MAT=True,
              desc='Weibull modulus')

    sigma_mu = Float(3.0, auto_set=False, enter_set=True,  # [N/mm^2]
                     MAT=True,
                     desc='Matrix tensional strength')

    sigma_fu = Float(800.0, auto_set=False, enter_set=True,  # [N/mm^2]
                     MAT=True,
                     desc='Fiber tensional strength')

    V_f = Property(Float,
                   depends_on='rho')

    @cached_property
    def _get_V_f(self):
        return self.rho

    V_m = Property(Float, depends_on='MAT')

    @cached_property
    def _get_V_m(self):
        return 1 - self.rho

    alpha = Property(Float, depends_on='MAT')

    @cached_property
    def _get_alpha(self):
        return (self.E_m * self.V_m) / (self.E_f * self.V_f)

    E_c1 = Property(Float, depends_on='MAT')

    @cached_property
    def _get_E_c1(self):
        return self.E_f * self.V_f + self.E_m * self.V_m

    delta_final = Property(Float, depends_on='MAT')

    @cached_property
    def _get_delta_final(self):
        return self.sigma_mu * (self.V_m * self.r) / (self.V_f * 2 * self.tau)

    cs_final = Property(Float)

    def _get_cs_final(self):
        return 1.337 * self.delta_final

    def _get_delta(self, sigma_c):
        return (sigma_c * (self.V_m * self.r * self.E_m) /
                (self.V_f * 2 * self.tau * self.E_c1))

    def _get_cs(self, sigma_c):
        '''Get crack spacing for current composite stress.
        '''
        Pf = (
            1 - exp(-((sigma_c * self.E_m) / (self.sigma_mu *
                                              self.E_c1))**self.m))
        if Pf == 0:
            Pf = 1e-15
        return self.cs_final * 1.0 / Pf

    def _get_epsilon_c(self, sigma_c):
        '''Get composite strain for current composite stress.
        '''
        cs = self._get_cs(sigma_c)
        delta = self._get_delta(sigma_c)
        if cs > 2 * delta:
            return sigma_c / self.E_c1 * (1 + self.alpha * delta / cs)
        else:
            return sigma_c * (1 / (self.E_f * self.V_f) -
                              (self.alpha * cs) / (4 * delta * self.E_c1))

    def eval(self):
        self.tline.val = self.tline.max

    def get_sig_eps(self):
        n_points = 100
        sigma_max = self.sigma_fu * self.rho

        sigma_arr = np.linspace(0, sigma_max, n_points)

        get_epsilon_f = np.frompyfunc(lambda sigma: sigma / self.E_f, 1, 1)
        epsilon_f_arr = get_epsilon_f(sigma_arr)

        get_epsilon_c = np.frompyfunc(self._get_epsilon_c, 1, 1)
        epsilon_c_arr = get_epsilon_c(sigma_arr)

        return epsilon_c_arr, sigma_arr

    traits_view = View(
        VGroup(
            Item('E_f'),
            Item('E_m'),
            Item('tau'),
            Item('r'),
            Item('rho'),
            Item('m'),
            Item('sigma_mu'),
            Item('sigma_fu'),
            label='Material parameters',
            id='scm.params',
        ),
        id='scm',
        dock='horizontal',
        resizable=True,
        height=0.8, width=0.8
    )

    tree_view = traits_view
    viz2d_classes = {'sig-eps': Viz2DSCMSigEps}


if __name__ == '__main__':
    from view.window import BMCSWindow
    scm = SCM()
    w = BMCSWindow(model=scm)
    scm.add_viz2d('sig-eps')
    w.configure_traits()
