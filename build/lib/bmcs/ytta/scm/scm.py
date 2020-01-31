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

from math import cos, pi as Pi

from numpy import linspace, frompyfunc,\
    hstack
from scipy.special import gamma
from scipy.stats import weibull_min
from traits.api import HasTraits, Float, Property, \
    cached_property, Range
from traitsui.api import View, Item, Tabbed, VGroup, \
    Group
from traitsui.menu import OKButton
from util.traits.either_type import EitherType

from .reinf_cross_section import SimplyRatio, GridReinforcement


class SCM(HasTraits):

    '''
    Stochastic Cracking Theory
    '''
    E_f = Float(120e+3, auto_set=False, enter_set=True,  # [N/mm^2]
                desc='the E-Modulus of the fiber [MPa]',
                modified=True)

    E_m = Float(34.e+3, auto_set=False, enter_set=True,  # [N/mm^2]
                desc='E-Modulus of the matrix [MPa]',
                modified=True)

    tau = Float(8.0, auto_set=False, enter_set=True,  # [N/mm^2]
                desc='the frictional stress between fiber and matrix [MPa]',
                modified=True)

    r = Float(0.5, auto_set=False, enter_set=True,  # [mm]
              desc='the radius of the fiber',
              modified=True)

    m = Float(2.3, auto_set=False, enter_set=True,  # [-]
              desc='the Weibull modulus defining the scatter of the matrix strength [-]',
              modified=True)

    orientation = Range(low=0.0, high=Pi / 2, value=0.0,
                        auto_set=False, enter_set=True,  # [-]
                        desc='Fiber orientation [rad]',
                        modified=True)

    sigma_mu = Float(12.0, auto_set=False, enter_set=True,  # [N/mm^2]
                     desc='the matrix tensional strength as a scale parameter'
                     ' of the Weibull distribution [MPa]',
                     modified=True)

    sigma_fu = Float(720.0, auto_set=False, enter_set=True,  # [N/mm^2]
                     desc='the fiber tensional strength [MPa]',
                     modified=True)

    reinf_ratio = EitherType(names=['grid fiber layout', 'explicit value'],
                             klasses=[GridReinforcement, SimplyRatio],
                             modified=True)

    Pf = Range(low=0.0, high=1.0 - 1e-15, value=0.9,
               auto_set=False, enter_set=True,  # [-]
               desc='probability of crack spacing to be of final range',
               modified=True)

    rho = Property(Float, depends_on='reinf_ratio.rho,orientation')

    def _get_rho(self):
        if self.reinf_ratio.rho * cos(self.orientation) == 0:
            return 1e-15
        else:
            return self.reinf_ratio.rho * cos(self.orientation)

    V_f = Property(Float, depends_on='rho')

    @cached_property
    def _get_V_f(self):
        return self.rho

    V_m = Property(Float, depends_on='rho')

    @cached_property
    def _get_V_m(self):
        return 1 - self.rho

    alpha = Property(Float, depends_on='E_m,E_f,rho')

    @cached_property
    def _get_alpha(self):
        return (self.E_m * self.V_m) / (self.E_f * self.V_f)

    E_c = Property(Float, depends_on='E_m,E_f,rho')

    @cached_property
    def _get_E_c(self):
        return self.E_f * self.V_f + self.E_m * self.V_m

    delta_final = Property(Float, depends_on='E_m,E_f,rho,r,sigma_mu,tau,m')

    @cached_property
    def _get_delta_final(self):
        return self.sigma_mu * (self.V_m * self.r) / (self.V_f * 2 * self.tau)

    cs_final = Property(Float)

    def _get_cs_final(self):
        return 1.337 * self.delta_final

    def _get_delta(self, sigma_c):
        return sigma_c * (self.V_m * self.r * self.E_m) / (self.V_f * 2 * self.tau * self.E_c)

    # matrix strength scale parameter for the Weibull distribution with sigma_mu as
    # mean and m as shape parameter
    scale_sigma_m = Property(Float, depends_on='sigma_mu, m')

    @cached_property
    def _get_scale_sigma_m(self):
        return self.sigma_mu / gamma(1. + 1. / self.m)

    # composite scale parameter for the Weibull distribution with sigma_mu as
    # mean and m as shape parameter
    # TODO: the mean composite cracking stress increases with increasing
    # reinforcement ratio - this has yet to be implemented
    scale_sigma_c = Property(Float, depends_on='sigma_mu, m, E_m, E_f, rho')

    @cached_property
    def _get_scale_sigma_c(self):
        return self.scale_sigma_m / self.E_m * self.E_c

    def _get_cs(self, sigma_c):
        Pf = weibull_min.cdf(sigma_c, self.m, scale=self.scale_sigma_c)
        if Pf == 0:
            Pf = 1e-15
        return self.cs_final * 1.0 / Pf

    def eps_c(self, sigma_c):
        cs = self._get_cs(sigma_c)
        delta = self._get_delta(sigma_c)
        print('delta')
        print(delta)
        if cs > 2 * delta:
            print(sigma_c / self.E_c * (1 + self.alpha * delta / cs))
            return sigma_c / self.E_c * (1 + self.alpha * delta / cs)
        else:
            print(sigma_c * (1. / (self.E_f * self.V_f) -
                             (self.alpha * cs) / (4. * delta * self.E_c)))

            return sigma_c * (1. / (self.E_f * self.V_f) -
                              (self.alpha * cs) / (4. * delta * self.E_c))

    def _get_epsilon_c(self, sigma_c):
        get_epsilon_c = frompyfunc(self.eps_c, 1, 1)
        return get_epsilon_c(sigma_c)

    sigma_cu = Property(depends_on='rho, orientation, sigma_fu')

    @cached_property
    def _get_sigma_cu(self):
        '''Ultimate composite strength.

        The strength is given by the fiber strength related to the composite
        cross section by the reinforcement ratio rho and projected
        by the cosine of the fiber inclination into the loading direction.
        '''
        # 0.05 quantile strength of the matrix as a criterion for matrix failure
        # when this value is higher than the composite strength governed by the
        # reinforcement stress
        quantile = weibull_min.ppf(0.05, self.m, scale=self.scale_sigma_m)
        # composite failure due to matrix failure
        if self.sigma_fu * self.V_f * cos(self.orientation) < quantile / self.V_m:
            return quantile / self.V_m
        else:
            # composite failure due to fiber failure
            return self.sigma_fu * self.V_f * cos(self.orientation)

    csf = Property(
        depends_on='m, sigma_mu, Pf, sigma_fu, orientation, E_f, E_m')

    @cached_property
    def _get_csf(self):
        # composite stress at Pf probability for CS to be of final range
        # scale parameter for composite stress
        scale_sigma_c = self.scale_sigma_m / self.E_m * self.E_c

        sigma = weibull_min.ppf(self.Pf, self.m, scale=scale_sigma_c)

        # point of reaching final crack spacing
        epsilon_csf = hstack((0,
                              self._get_epsilon_c(sigma),
                              self._get_epsilon_c(sigma)))
        sigma_csf = hstack((sigma, sigma, 0))
        return epsilon_csf, sigma_csf

    sig_eps_fn = Property(depends_on='+modified, reinf_ratio.+modified')

    @cached_property
    def _get_sig_eps_fn(self):
        '''Get the stress and strain arrays'''
        print('update')
        n_points = 100
        sigma_c_arr = linspace(0, self.sigma_cu, n_points)
        if self.sigma_cu == self.sigma_fu * self.V_f * cos(self.orientation):
            epsilon_c_arr = self._get_epsilon_c(sigma_c_arr)
        else:
            epsilon_c_arr = sigma_c_arr / self.E_c

        # stress with respect to reinforcement
        sigma_f_arr = sigma_c_arr / self.rho

        # stress of reinforcement with no matrix interaction
        sigma_fiber = epsilon_c_arr[[0, -1]] * self.E_f * self.rho

        print(epsilon_c_arr)

        return epsilon_c_arr, sigma_c_arr, sigma_f_arr, sigma_fiber

    traits_view = View(
        Group(
            Tabbed(
                VGroup(
                    Group(
                        Item('E_f',      resizable=True, full_size=True,
                             label='E-modulus',
                             tooltip="Young's modulus of the fiber"),
                        Item('sigma_fu', resizable=True,
                             label='strength',
                             help="Strength of the fibers"
                             ),
                        label='fibers',
                    ),
                    Group(
                        Item('E_m',      resizable=True, full_size=True,
                             label='E-modulus',
                             help="Young's modulus of the matrix"),
                        Item('sigma_mu', resizable=True,
                             label='strength',
                             help="Scale parameter of the matrix strength"
                             'roughly corresponding to the mean strength'),
                        Item('m',        resizable=True,
                             label='Weibull-modulus',
                             help="Weibull modulus of the matrix strength distribution"
                             'defining the scatter of the strength'),
                        label='matrix',
                        scrollable=False,
                    ),
                    label='Components',
                    dock='tab',
                    id='scm.model.component_params',
                ),
                VGroup(
                    Item('tau', resizable=True, full_size=True, springy=False),
                    Item('r', resizable=False, springy=False),
                    springy=True,
                    label='Bond',
                    dock='tab',
                    id='scm.model.params',
                ),
                id='scm.model.allparams',
            ),
            VGroup(
                Item('reinf_ratio@', show_label=False, resizable=True),
                label='Cross section parameters',
                dock='tab',
                id='scm.model.reinf_ratio',
            ),
            VGroup(
                Item('orientation', label='fiber orientation'),
            ),
            id='scm.model.splitter',
            springy=False,
            layout='split',
        ),
        id='scm.model',
        dock='fixed',
        scrollable=True,
        resizable=True,
        buttons=[OKButton],
        height=0.8, width=0.8
    )


def run():
    '''Test w/o the user interface'''
    s = SCM()
    eps = s._get_epsilon_c(2.0)
    print('epsilon (should be ... 5.49622857811e-05 )', eps)
    s.E_f *= 0.9
    eps = s._get_epsilon_c(2.0)
    print('epsilon (should be ... 5.5626562923e-05 )', eps)
    s.reinf_ratio.h = 20
    eps = s._get_epsilon_c(2.0)
    print('epsilon (should be ... 5.3617517588e-05 )', eps)
    s.orientation = 1.56
    eps = s._get_epsilon_c(2.0)
    print('epsilon (should be ... 0.000112774785042 )', eps)
    # further dependency tests for changing the inputs and recalculation of results.
    # ....
    s = SCM()
    s.configure_traits()


if __name__ == '__main__':
    run()
