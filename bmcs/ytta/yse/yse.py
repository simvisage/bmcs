'''
Created on Sep 22, 2009

@author: rostislav
'''

from math import e, sqrt, log, pi

from numpy import array, frompyfunc, \
    log as ln, logspace, hstack
from scipy.optimize import brentq
from scipy.stats import norm
from traits.api import HasTraits, Float, Property, cached_property, \
    Int
from traitsui.api import \
    Spring, View, Item, VGroup, Group

from traitsui.menu import OKButton


class YSE(HasTraits):
    '''
    Size effect depending on the yarn length
    '''

    l_f = Float(0.5, auto_set=False, enter_set=True,  # [m]
                desc='yarn total length',
                modified=True)
    l_r = Float(0.05, auto_set=False, enter_set=True,  # [m]
                desc='reference specimen length - assumed to be a single bundle',
                modified=True)
    mu_r = Float(1955.8, auto_set=False, enter_set=True,  # [MPa]
                 desc='reference specimen mean strength',
                 modified=True)
    mu_x = Float(1586.0, auto_set=False, enter_set=True,  # [MPa]
                 desc='mean strength of evaluated specimen',
                 modified=True)
    m_f = Float(5., auto_set=False, enter_set=True,  # [-]
                desc='Weibull shape parameter for filaments',
                modified=True)
    Nf = Int(24000, auto_set=False, enter_set=True,  # [-]
             desc='number of filaments in yarn',
             modified=True)
    l_rho = Float(0.001, auto_set=False, enter_set=True,  # [m]
                  desc='autocorrelation length for fiber strength dispersion',
                  modified=True)

    # these parameters are called plot, but the model should not
    # interfere with the view (@todo resolve)
    l_plot = Float(80., auto_set=False, enter_set=True,  # [m]
                   desc='maximum yarn length',
                   modified=True)

    min_plot_length = Float(0.0001, auto_set=False, enter_set=True,  # [m]
                            desc='minimum yarn length',
                            modified=True)

    n_points = Int(100, auto_set=False, enter_set=True,
                   desc='points to plot',
                   modified=True)

    # autocorrelation length function
    def fl(self, l):
        return (self.l_rho / (self.l_rho + l))**(1. / self.m_f)
    '''second option'''
#        return (l/self.l_rho + self.l_rho/(self.l_rho + l))**(-1./self.m_f)

    # scale parameter for the yarn strength (23)
    s_r = Property(Float, depends_on='+modified')

    @cached_property
    def _get_s_r(self):
        c = e**(-1. / self.m_f)
        s_r = self.mu_r / (self.m_f**(-1. / self.m_f) * c
                           + self.Nf**(-2. / 3.) * self.m_f**(-1. / self.m_f -
                                                              1. / 3.) * e**(-1. / 3. / self.m_f) * 0.996
                           )
        # print 's_r =',s_r
        return s_r

    # strength standard deviation of the reference specimen (24)
    gamma_r = Property(Float, depends_on='+modified')

    @cached_property
    def _get_gamma_r(self):
        c = e**(-1. / self.m_f)
        gamma_r = self.s_r * \
            self.m_f**(-1. / self.m_f) * sqrt(c * (1 - c)) / sqrt(self.Nf)
#        print 'gamma_r =', gamma_r
        return gamma_r

    # Coefficient of variation for the first part with no filament interaction
    cov = Property(Float, depends_on='+modified')

    def _get_cov(self):
        #        print 'COV = ', self.gamma_r / self.mu_r
        return self.gamma_r / self.mu_r

# --------------------------------------
# GAUSSIAN NORMAL STRENGTH DISTRIBUTION
# --------------------------------------

    # method for getting the bundle length by iterating in get_l_b()
    def _get_gl_b(self, gl_b):
        l_r = self.l_r
        l_f = self.l_f
        mu_x = self.mu_x
        mu_r = self.mu_r

        # mean strength of one bundle at given length l_b (19)
        mu_b = self.fl(gl_b) / self.fl(l_r) * mu_r
#        print 'mu_b =', mu_b

        # probability of failure for 1 link of length l_b at load level mu_x
        # (21)
        Pf_x = 1. - 0.5**(gl_b / l_f)
#        print 'Pf_x =', Pf_x

        # strength standard deviation of mu_b for evaluating the chain of
        # bundles strength (3)
        gamma_b = mu_b * self.cov
#        print 'gamma_b =', gamma_b

        # for control - scale parameter for bundle of length l_b
        s_lb = self.fl(gl_b) / self.fl(l_r) * self.s_r
#        print 'slb =', s_lb

        # cumulative strength distribution of 1 bundle l_b at load level mu_x
        # (22)
        CDF_x = norm.cdf((mu_x - mu_b) / gamma_b)
#        print 'CDF_x =', CDF_x
        return Pf_x - CDF_x

    # bundle properties for Gaussian normal distribution of bundle strength
    gbundle_props = Property(depends_on='+modified')

    @cached_property
    def _get_gbundle_props(self):
        # rr: check the limits ( 2 * l_f for upper limit?)
        gl_b = brentq(self._get_gl_b, 1.e-10, 1. + self.l_f, xtol=0.00001)
        gmu_b = self.fl(gl_b) / self.fl(self.l_r) * self.mu_r
        ggamma_b = gmu_b * self.cov
        return [gl_b, gmu_b, ggamma_b]

    l_b = Property(depends_on='+modified')

    @cached_property
    def _get_l_b(self):
        return self._get_gbundle_props()[0]

    mu_b = Property(depends_on='+modified')

    @cached_property
    def _get_mu_b(self):
        return self._get_gbundle_props()[1]

    mu_sigma_0 = Float()

    # returns median strength of chob depending on length l
    def _get_gstrength(self, l):
        yarn = self._get_gbundle_props()
        test = norm.ppf(1. - 0.5**(yarn[0] / l)) * yarn[2] + yarn[1]
        return norm.ppf(1. - 0.5**(yarn[0] / l)) * yarn[2] + yarn[1]

    def mean_approx(self, l):
        yarn = self._get_gbundle_props()
        # bundle length
        l_b = yarn[0]
        # mean and stdev of Gaussian bundle
        mu_b = yarn[1]
        gamma_b = yarn[2]
        # No. of bundles in series
        nb = l / l_b
        if nb == 1:
            return mu_b, mu_b, mu_b
        w = ln(nb)
        # approximation of the mean for k = (1;300) (Mirek)
        mu = mu_b + gamma_b * (-0.007 * w**3 + 0.1025 * w**2 - 0.8684 * w)
        # approximation for the mean from extremes of Gaussian (tends to Gumbel
        # as mb grows large)
        a = gamma_b / sqrt(2 * w)
        b = mu_b + gamma_b * ((ln(w) + ln(4 * pi)) / sqrt(8 * w) - sqrt(2 * w))
        med = b + a * ln(ln(2))
        mean = b - 0.5772156649015328606 * a
        return mu, mean, med

    test_length_r = Property(depends_on='+modified')

    @cached_property
    def _get_test_length_r(self):
        xdata = array(
            [self.min_plot_length, self.l_r, self.l_r], dtype='float_')
        ydata = array(
            [self.mu_r, self.mu_r, self.mu_x * 0.0008], dtype='float_')
        return (xdata, ydata)

    test_length_f = Property(depends_on='+modified')

    @cached_property
    def _get_test_length_f(self):
        xdata = array(
            [self.min_plot_length, self.l_f, self.l_f], dtype='float_')
        ydata = array(
            [self.mu_x, self.mu_x, self.mu_x * 0.0008], dtype='float_')
        return (xdata, ydata)

    def _get_values(self):
        l_rho = self.l_rho
        n_points = self.n_points
        gl_b = self._get_gbundle_props()[0]
        gmu_b = self._get_gbundle_props()[1]
        m_f = self.m_f
        mu_r = self.mu_r
        l_r = self.l_r

        # for Gaussian bundle strength distribution
        if self.l_plot <= gl_b:
            gl_arr = logspace(
                log(self.min_plot_length, 10), log(gl_b, 10), n_points)
            gstrength_arr = self.fl(gl_arr) / self.fl(self.l_r) * self.mu_r
        elif self.l_plot > gl_b:
            gl_1 = logspace(
                log(self.min_plot_length, 10), log(gl_b, 10), n_points)
            gl_2 = logspace(log(gl_b, 10), log(self.l_plot, 10), n_points)
            gl_arr = hstack((gl_1, gl_2))
            gstrength_1 = self.fl(gl_1) / self.fl(self.l_r) * self.mu_r
            gstrength_22 = frompyfunc(self._get_gstrength, 1, 1)
            gstrength_2 = array(gstrength_22(gl_2), dtype='float64')
            gstrength_arr = hstack((gstrength_1, gstrength_2))
        # Mirek's mean approximation
            strength_22 = frompyfunc(self.mean_approx, 1, 3)
            strength_2 = array(strength_22(gl_2)[0], dtype='float64')
            mean_gumb = array(strength_22(gl_2)[1], dtype='float64')
            med_gumb = array(strength_22(gl_2)[2], dtype='float64')

        # asymptotes for the first two branches
        if self.l_plot <= l_rho:
            al_arr = array([self.min_plot_length, self.l_plot])
            astrength_arr = array([mu_r / self.fl(l_r), mu_r / self.fl(l_r)])
        elif l_rho < self.l_plot:
            al_arr = array([self.min_plot_length, l_rho, 10. * gl_b])
            astrength_1 = mu_r / self.fl(l_r)
            astrength_2 = (l_rho / al_arr[2])**(1 / m_f) * astrength_1
            astrength_arr = hstack((astrength_1, astrength_1, astrength_2))

        # left asymptote
        self.mu_sigma_0 = astrength_arr[0]

        # standard deviation for the first branch = before fragmentation
        if self.l_plot <= gl_b:
            stl_arr = logspace(log(self.min_plot_length, 10), log(self.l_plot, 10),
                               n_points / 2.)
            stdev_arr_plus = self.fl(
                stl_arr) / self.fl(l_r) * mu_r * (1 + self.cov)
            stdev_arr_minus = self.fl(
                stl_arr) / self.fl(l_r) * mu_r * (1 - self.cov)
        else:
            stl_arr = logspace(
                log(self.min_plot_length, 10), log(gl_b, 10), n_points)
            stdev_arr_plus = self.fl(
                stl_arr) / self.fl(l_r) * mu_r * (1 + self.cov)
            stdev_arr_minus = self.fl(
                stl_arr) / self.fl(l_r) * mu_r * (1 - self.cov)

        return gl_arr, al_arr, gstrength_arr, astrength_arr,\
            stl_arr, stdev_arr_plus, stdev_arr_minus, gl_2,\
            strength_2, mean_gumb, med_gumb

    traits_view = View(Group(
        VGroup(
            VGroup(
                Item('l_r',  label='test length [m]'),
                Item('mu_r', label='mean strength [MPa]'),
                label='I. test in length range of a single bundle',
            ),
            VGroup(
                Item('l_f', label='test length [m]'),
                Item('mu_x', label='mean strength [MPa]'),
                label='II. test in length range of chain-of-bundles',
            ),
            label='test data',
            dock='tab',
            id='yse.input_params'
        ),
        VGroup(
            Item('m_f', label='Weibull modulus [-]'),
            Item('l_rho', label='autocorrelation length [m]'),
            Item('Nf', label='number of filaments [-]'),
            Spring(resizable=True),
            label='material parameters',
            id='yse.material_parameters',
            dock='tab',
        ),
        VGroup(
            Item('l_b', label='bundle length (Gauss) [m]',
                 format_str='%.4f',
                 style='readonly',
                 emphasized=True),
            Item('mu_sigma_0', label='left strength asymptote [MPa]',
                 format_str='%.1f',
                 style='readonly',
                 emphasized=True),
            label='identified parameters',
            dock='tab',
            id='yse.identified_params'
        ),
        layout='split',
        dock='tab',
        id='yse.model',
    ),
        id='yse',
        dock='fixed',
        scrollable=True,
        resizable=True,
        height=0.8, width=0.8,
        buttons=[OKButton])


def run():
    yse = YSE()
    yse.configure_traits()

if __name__ == '__main__':
    run()
