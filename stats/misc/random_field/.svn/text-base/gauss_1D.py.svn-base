'''
Created on 24.06.2011

@author: rrypl
'''

from enthought.traits.api import HasTraits, Float, Array, Int, Property, \
    cached_property, Bool
from math import e
from numpy import dot, transpose, ones, array, eye, real, linspace, reshape
from numpy.linalg import eig
from numpy.random import shuffle
from scipy.linalg import toeplitz
from scipy.stats import norm


class GaussRandomField(HasTraits):
    '''Generating Gaussian distributed random field by scaling a standardized normal distribution random
     field. The random field array is stored in the property random_field'''

    #Parameters to be set
    lacor = Float(1. , auto_set = False, enter_set = True,
                   desc = 'autocorrelation  of the field', modified = True)
    nsim = Int(1 , auto_set = False, enter_set = True,
                desc = 'No of Fields to be simulated', modified = True)
    mean = Float(0, auto_set = False, enter_set = True,
                  desc = 'mean value', modified = True)
    stdev = Float(1., auto_set = False, enter_set = True,
                   desc = 'standard deviation', modified = True)

    xgrid = Array

    non_negative_check = False
    reevaluate = Bool(False)

    def acor(self, dx, lcorr):
        '''autocorrelation function'''
        return e ** (-(dx / lcorr) ** 2)

    eigenvalues = Property(depends_on = '+modified')
    @cached_property
    def _get_eigenvalues(self):
        '''evaluates the eigenvalues and eigenvectors of the autocorrelation matrix'''
        #creating a symm. toeplitz matrix with (xgrid, xgrid) data points
        Rdist = toeplitz(self.xgrid, self.xgrid)
        #apply the autocorrelation func. to get the correlation matrix
        R = self.acor(Rdist , self.lacor)
        #evaluate the eigenvalues and eigenvectors of the autocorrelation matrix
        print 'evaluating eigenvalues for random field...'
        eigenvalues = eig(R)
        print 'complete'
        return eigenvalues

    random_field = Property(Array , depends_on = '+modified, reevaluate')
    @cached_property
    def _get_random_field(self):
        '''simulates the Gaussian random field'''
        #evaluate the eigenvalues and eigenvectors of the autocorrelation matrix
        _lambda, phi = self.eigenvalues
        #simulation points from 0 to 1 with an equidistant step for the LHS
        randsim = linspace(0, 1, len(self.xgrid) + 1) - 0.5 / (len(self.xgrid))
        randsim = randsim[1:]
        #shuffling points for the simulation
        shuffle(randsim)
        #matrix containing standard Gauss distributed random numbers
        xi = transpose(ones((self.nsim, len(self.xgrid))) * array([ norm().ppf(randsim) ]))
        #eigenvalue matrix 
        LAMBDA = eye(len(self.xgrid)) * _lambda
        #cutting out the real part
        ydata = dot(dot(phi, (LAMBDA) ** 0.5), xi).real
        # scaling the std. distribution
        scaled_ydata = ydata * self.stdev + self.mean
        self.reevaluate = False
        rf = reshape(scaled_ydata, len(self.xgrid))
        if self.non_negative_check == True:
            if (rf < 0).any():
                raise ValueError, 'negative value(s) in random field'
        return rf



if __name__ == '__main__':

    from matplotlib import pyplot as p
    rf = GaussRandomField(lacor = 6. , xgrid = linspace(0, 100., 100), mean = 0., stdev = 1.)
    x = rf.xgrid
    for sim in range(3):
        p.plot(x, rf.random_field, lw = 2)
        rf.reevaluate = True
    p.show()
