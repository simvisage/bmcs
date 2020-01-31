'''
@brief Chain Of Bundles Strength Analysis Tool.

(C) 2007 Chudoba, R. and Vorechovsky, M.

Based on Pan 00

@todo Determine the fraction of broken filaments at peak
stress. Proceed as follows: Calculate the strain at peak stress
for the so called critical bundle. Recover the fractions of failed
filaments in other bundles. It should be possible to exploit the
kinematic assomption of a chain to recover the overall fraction of
broken filaments.

Implementation:


The classes WeibullDistrib, ScaledWeibullDistrib and
DanielsSmithDistrib remember the distribution parameters and return
the distribution values using the pdf, cdf, inv methods.

The classes RandomVariable and RandomChainOfBundlesVariable generate
the numerical data based on the associated distribution and
provide.

'''

from numpy import frompyfunc, arange, sqrt, exp, abs
from scipy.stats import weibull_min, norm
from traits.api import \
    HasTraits, Float, Int,  Property, cached_property, \
    Bool
from traitsui.api import Item, View, Group, \
    HGroup, VGroup


class WeibullDistrib(HasTraits):
    '''
    @brief Weibull distribution defined by scale and shape.

    The distribution holds for a given reference length. Length is not
    used for returning the characteristics of this distribution. It
    rather allows for scaling of derived distributions to other
    lengths.

    @param scale scale of the Weibull distribution
    @param shape shape of the Weibull distribution
    @param length length reference length for the current scale
    (required for scaling of the distribution - see class
    ScaledWeibullDistrib)
    '''

    scale = Float(0.0)
    shape = Float(1.0)
    length = Float(1.0)
    loc = Float(0.0)

    def cdf(self, x):
        '''
        @brief Return the value of a cummulative probability function for x.
        '''
        return weibull_min.cdf(x, self.shape,
                               loc=self.loc,
                               scale=self.scale)

    def pdf(self, x):
        '''
        @brief Return the value of a probability density function for x.
        '''
        return weibull_min.pdf(x, self.shape,
                               loc=self.loc,
                               scale=self.scale)

    def inv(self, p):
        '''
        @brief Return the inverse cummulative probability for p.

        @param p is the value between 0 and 1
        '''

        if p <= 0.0:
            p = 0.0000000000001
        if p >= 1.0:
            p = 0.9999999999999
        return weibull_min.ppf(p, self.shape,
                               loc=self.loc,
                               scale=self.scale)

    def get_limits(self):
        '''
        @brief Get limits of the range with relevant probabilities.

        Relevant range is between 1e-5 and 1-1e-5.
        '''
        x_min, x_max = self.inv(0.00001), self.inv(0.99999)
        return x_min, x_max


class ScaledWeibullDistrib(WeibullDistrib):
    '''
    @brief Scaled Weibull distribution derived from the Weibull distribution.

    As a subclass, this function reuses the methods cdf, pdf, inv and
    get_limits. The scale and shape parameters are derived from the
    reference distribution by scaling (see __init__).
    '''

    def __init__(self, wb, length, which=0):
        '''
        Constructor.

        Scaling is done by
        \f[
            s_1 = s_0 \left( \frac{l_0}{l_1} \right)^\frac{1}{m}
        \f]
        '''
        self.loc = 0.0
        self.wb = wb
        if which == 0:
            l_eff = (wb.length / length)
        else:
            l_eff = wb.length / (wb.length + length)
        self.scale = wb.scale * l_eff ** (1.0 / wb.shape)
        self.length = length
        self.shape = wb.shape


class DanielsSmithDistrib(HasTraits):
    '''
    @brief Strengh distribution of a bundle with nf number of fibers.

    The distribution follows the formula of Daniels and Smith for mean
    value and standard deviation of a normal distribution. The cdf and
    pdf values are thus computed using the normal distribution taken
    from scipy.stats.
    '''

    def __init__(self, wb, nf):
        '''
        @brief Constructor.

        @param wb source Weibull distribution
        @param nf number of filaments
        '''
        self.nf = nf
        self.wb = wb
        self.get_mean_std(nf)

    def get_mean_std(self, nf):
        '''
        @brief Daniels-Smith formula.

        \f[
            \mu_{\sigma,n}^*=\mu_{\sigma}^*+n^{\frac{-2}{3}}b^*\gamma
        \f]
        '''
        shape = self.wb.shape
        scale = self.wb.scale
        c = exp(-1.0 / shape)
        x = pow(shape, -1.0 / shape) * scale
        gamma = x * sqrt(c * (1.0 - c))
        b = scale * pow(shape, -(1.0 / shape + 1.0 / 3.0)) * \
            exp(-1.0 / (3.0 * shape))
        self.std = gamma / sqrt(nf)
        self.mean = x * c + pow(nf, (-2.0 / 3.0)) * b * 0.996

    def pdf(self, x):
        '''
        @brief Return the pdf of the normal distribution for the bundle.
        '''
        return norm.pdf(x, self.mean, self.std)

    def cdf(self, x):
        '''
        @brief Return the cdf of the normal distribution for the bundle.
        '''
        return norm.cdf(x, self.mean, self.std)

    def get_limits(self):
        '''
        @brief Get limits of the range with relevant probabilities.

        Take four standard deviations to the left and to the right.
        '''
        return self.mean - 4.0 * self.std, self.mean + 4.0 * self.std


class RandomVariable(HasTraits):
    '''
    @brief Random variable for with a given distribution.
    '''

    def __init__(self, distr, sims):
        '''
        @brief Constructor.

        @param sims number of sampling points of the distribution.
        '''
        self.distr = distr
        self.n = sims
        self.set_x()
        self.eval()

    def set_x(self):
        '''
        @brief Set the domain of the random variable.

        Use the method @see get_limits of the supplied distribution.
        '''
        x_min, x_max = self.distr.get_limits()
        dx = abs(x_max - x_min) / self.n
        self.x = arange(x_min, x_max, dx)

    def eval(self):
        '''
        @brief Get the pdf and cdf values of the distribution.

        Use the user functions of the numpy package. By applying the
        method frompyfunc, the distr.pdf and distr.cdf can be applied
        to an array of values representing the x-domain of the random
        variable.
        '''

        self.pdf = self.distr.pdf(self.x)
        self.cdf = self.distr.cdf(self.x)

    def get_mean(self):
        '''
        @brief Get the mean value of the random variable.
        '''
        return self.distr.stats('m')

    def plot(self, axes, style, label):
        '''
        @brief Plot the values of the function in the range of the x-domain.
        '''
        axes.plot(self.x, self.pdf, style, label=label,
                  linewidth=3)


class RandomChainOfBundlesVariable(RandomVariable):
    '''
    @brief Random variable for a chain of bundles.

    This variable is constructed using a BundleDistribution.
    '''

    def __init__(self, bundle_distr, nb, sims):
        '''
        @brief Constructor

        @param bundle_distr Instance of DanielsSmithDistrib to
        represent a single bundle with a specified length
        @param nb number of bundles chained
        @param sims number of sampling points to use for the bundle
        and chain-of-bundles variable.
        '''
        self.bundle_distr = bundle_distr
        self.bundle_var = RandomVariable(self.bundle_distr, sims)
        self.nb = nb
        self.eval()
        self.n = self.bundle_var.n
        self.x = self.bundle_var.x

    def eval(self):
        '''
        @brief Get the pdf and cdf values of the distribution.

        Use the user functions of the numpy package. By applying the
        method frompyfunc, the distr.pdf and distr.cdf can be applied
        to an array of values representing the pdf and cdf of the
        source random variable (of a single bundle). In this way the
        chaining of bundles is modeled.

        @todo write down the formula for pdf and cdf
        '''
        def pdf_min(cdf, pdf):
            return self.nb * pow(1.0 - cdf, self.nb - 1.0) * pdf
        pdf_min_func = frompyfunc(pdf_min, 2, 1)
        self.pdf = pdf_min_func(self.bundle_var.cdf, self.bundle_var.pdf)

        def cdf_min(cdf):
            return 1.0 - pow(1.0 - cdf, self.nb)
        cdf_min_func = frompyfunc(cdf_min, 1, 1)
        self.cdf = cdf_min_func(self.bundle_var.cdf)


class ChainOfBundlesAnalysis(HasTraits):
    '''
    Manage the distributions representing the chaoined bundle.
    '''
    l_rho = Float(1.0, auto_set=False, enter_set=True, input=True)
    scale_rho = Float(1.095208512, auto_set=False, enter_set=True, input=True)
    m = Float(4.5422149741521, auto_set=False, enter_set=True, input=True)
    l_tot = Float(1.0, auto_set=False, enter_set=True, input=True)
    nb = Float(5, auto_set=False, enter_set=True, input=True)
    nf = Int(10, auto_set=False, enter_set=True, input=True)
    fiber_total = Bool(True, input=True)
    fiber_bundle = Bool(True, input=True)
    bundle_total = Bool(True, input=True)
    bundle_bundle = Bool(True, input=True)
    yarn_total = Bool(True, input=True)

    distribs = Property(depends_on='+input')

    @cached_property
    def _get_distribs(self):
        l_rho = self.l_rho
        scale_rho = self.scale_rho
        m = self.m
        l_tot = self.l_tot
        nb = self.nb
        nf = self.nf

        #
        # Source distribution for the reference length (l_rho)
        #
        wd_r = WeibullDistrib(scale=scale_rho,
                              shape=m,
                              length=l_rho)
        #
        # Scaled distribution for a single fiber with the length l_tot
        #
        sfd_t = ScaledWeibullDistrib(wd_r, length=l_tot)
        #
        # Bundle distribution with the length l_tot
        #
        dsd_t = DanielsSmithDistrib(sfd_t, nf)
        #
        # Scaled distribution for a single fiber with the bundle length (l_tot/nb)
        #
        sfd_b = ScaledWeibullDistrib(wd_r, length=l_tot / nb)
        #
        # Bundle distribution with the length l_tot/nb
        #
        dsd_b = DanielsSmithDistrib(sfd_b, nf)
        #
        # Construct the random variables - by specifying the distribution and
        # the number of sampling points
        #
        # The Chain of Bundle random variable is special in that it is
        # constructed by making an instance of bundle distribution and
        # constructing the cdf and pdf by its integration.
        #
        return  RandomVariable( sfd_t, 500 ), \
            RandomVariable( dsd_t, 500 ), \
            RandomVariable( sfd_b, 500 ), \
            RandomVariable( dsd_b, 500 ), \
            RandomChainOfBundlesVariable(dsd_b, nb, sims=500)

    # Distribution of the single filament strength on the length l_tot
    sfd_t = Property(depends_on='+input')

    def _get_sfd_t(self):
        return self.distribs[0]

    view_chob = View(VGroup(HGroup(Group(Item('l_rho', label='reference length'),
                                         Item(
                                             'scale_rho', label='scale parameter'),
                                         Item('m', label='shape parameter'),
                                         dock='tab'
                                         ),
                                   Group(Item('nf', label='number of filaments'),
                                         Item('nb', label='number of bundles'),
                                         Item('l_tot', label='total length'),
                                         dock='tab'
                                         ),
                                   Group(Item('fiber_total', label='fiber (t)'),
                                         Item(
                                             'fiber_bundle', label='fiber (b)'),
                                         Item(
                                             'bundle_total', label='bundle (t)'),
                                         dock='tab'
                                         ),
                                   Group(Item('bundle_bundle', label='bundle (b)'),
                                         Item('yarn_total', label='yarn (t)'),
                                         dock='tab'
                                         ),

                                   ),
                            ),
                     resizable=True,
                     scrollable=True,
                     height=0.8, width=0.8
                     )

if __name__ == '__main__':
    chob = ChainOfBundlesAnalysis()
    chob
    chob.configure_traits()
