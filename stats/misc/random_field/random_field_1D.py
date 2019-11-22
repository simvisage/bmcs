'''
Created on 24.06.2011

@author: rrypl
'''

from traits.api import HasStrictTraits, Float, Array, Int, Property, \
    cached_property, Bool, Event, Enum
from traitsui.api import View, Item
from math import e
from numpy import dot, transpose, ones, array, eye, linspace, reshape
from numpy.linalg import eig
from numpy.random import shuffle
from scipy.linalg import toeplitz
from scipy.stats import norm, weibull_min
import numpy as np


class RandomField(HasStrictTraits):

    '''Class for generating a 1D random field by scaling a standardized
    normally distributed random field. The random field array is stored
    in the property random_field. Gaussian or Weibull local distributions
    are available.
    The parameters of the Weibull random field are related to the minimum
    extreme along the whole length of the field.
    '''

    # Parameters to be set
    lacor = Float(1., auto_set=False, enter_set=True,
                  desc='autocorrelation length', modified=True)
    nsim = Int(1, auto_set=False, enter_set=True,
               desc='No of fields to be simulated', modified=True)
    mean = Float(0, auto_set=False, enter_set=True,
                 desc='mean value', modified=True)
    stdev = Float(1., auto_set=False, enter_set=True,
                  desc='standard deviation', modified=True)
    shape = Float(10., auto_set=False, enter_set=True,
                  desc='shape for Weibull distr', modified=True)
    scale = Float(5., auto_set=False, enter_set=True,
                  desc='scale for Weibull distr. corresp. to a length < lacor', modified=True)
    loc = Float(auto_set=False, enter_set=True,
                desc='location for 3 params weibull', modified=True)
    length = Float(1000., auto_set=False, enter_set=True,
                   desc='length of the random field', modified=True)
    nx = Int(200, auto_set=False, enter_set=True,
             desc='number of discretization points', modified=True)
    non_negative_check = False
    reevaluate = Event
    seed = Bool(False)
    distr_type = Enum('Weibull', 'Gauss', modified=True)

    xgrid = Property(Array, depends_on='length,nx')

    @cached_property
    def _get_xgrid(self):
        '''get the discretized grid for the random field'''
        return np.linspace(0, self.length, self.nx)

    gridpoint_scale = Property(depends_on='scale,shape,length,nx,lacor')

    @cached_property
    def _get_gridpoint_scale(self):
        '''Scaling of the defined distribution to the distribution of a single
        grid point. This option is only available for Weibull random field'''
        delta_x = self.xgrid[1] - self.xgrid[0]
        return self.scale * (self.lacor / (delta_x + self.lacor)) ** (-1. / self.shape)

    def acor(self, dx, lcorr):
        '''autocorrelation function'''
        return e ** (-(dx / lcorr) ** 2)

    eigenvalues = Property(depends_on='lacor,length,nx')

    @cached_property
    def _get_eigenvalues(self):
        '''evaluates the eigenvalues and eigenvectors of the autocorrelation matrix'''
        # creating a symm. toeplitz matrix with (xgrid, xgrid) data points
        Rdist = toeplitz(self.xgrid, self.xgrid)
        # apply the autocorrelation func. to get the correlation matrix
        R = self.acor(Rdist, self.lacor)
        # evaluate the eigenvalues and eigenvectors of the autocorrelation
        # matrix
        print('evaluating eigenvalues for random field...')
        eigenvalues = eig(R)
        print('complete')
        return eigenvalues

    random_field = Property(Array, depends_on='+modified, reevaluate')

    @cached_property
    def _get_random_field(self):
        if self.seed == True:
            np.random.seed(101)
        '''simulates the Gaussian random field'''
        # evaluate the eigenvalues and eigenvectors of the autocorrelation
        # matrix
        _lambda, phi = self.eigenvalues
        # simulation points from 0 to 1 with an equidistant step for the LHS
        randsim = linspace(0, 1, len(self.xgrid) + 1) - 0.5 / (len(self.xgrid))
        randsim = randsim[1:]
        # shuffling points for the simulation
        shuffle(randsim)
        # matrix containing standard Gauss distributed random numbers
        xi = transpose(
            ones((self.nsim, len(self.xgrid))) * array([norm().ppf(randsim)]))
        # eigenvalue matrix
        LAMBDA = eye(len(self.xgrid)) * _lambda
        # cutting out the real part
        ydata = dot(dot(phi, (LAMBDA) ** 0.5), xi).real
        if self.distr_type == 'Gauss':
            # scaling the std. distribution
            scaled_ydata = ydata * self.stdev + self.mean
        elif self.distr_type == 'Weibull':
            # setting Weibull params
            Pf = norm().cdf(ydata)
            scaled_ydata = weibull_min(
                self.shape, scale=self.scale, loc=self.loc).ppf(Pf)
        self.reevaluate = False
        rf = reshape(scaled_ydata, len(self.xgrid))
        if self.non_negative_check == True:
            if (rf < 0).any():
                raise ValueError('negative value(s) in random field')
        return rf

    view_traits = View(Item('lacor'),
                       Item('nsim'),
                       Item('shape'),
                       Item('scale'),
                       Item('length'),
                       Item('nx'),
                       Item('distr_type'),
                       )

if __name__ == '__main__':
    from matplotlib import pyplot as p

#     rf = RandomField(seed=False,
#                      lacor=10.,
#                      length=500,
#                      nx=500,
#                      nsim=1,
#                      loc=.0,
#                      shape=10.,
#                      scale=2.4,
#                      distr_type='Weibull'
#                      )

    rf = RandomField(seed=False,
                     lacor=10.,
                     length=500,
                     nx=500,
                     nsim=1,
                     mean=2.8,
                     stdev=0.15,
                     distr_type='Gauss'
                     )

    rf2 = RandomField(seed=False,
                      lacor=10.,
                      length=500,
                      nx=500,
                      nsim=1,
                      mean=2.8,
                      stdev=0.000005,
                      distr_type='Gauss'
                      )

    p.plot(rf.xgrid, rf.random_field, lw=2, color='black', label='Weibull')
    p.plot(rf2.xgrid, rf2.random_field, lw=2, color='blue', label='Gauss')
    p.legend(loc='best')
    p.ylim(0)
    p.show()
