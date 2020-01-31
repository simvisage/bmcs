'''
Created on Jan 25, 2015
implements the 3D random field
@author: rostislavrypl
'''

from etsproxy.traits.api import HasTraits, Float, Property, \
                                cached_property, Array, Enum, \
                                Event, Bool, List
from scipy.linalg import eigh, toeplitz
import numpy as np
from math import e
from scipy.stats import norm, weibull_min


class RandomField(HasTraits):
    '''
    This class implements a 3D random field on a regular grid
    and allows for interpolation using the EOLE method
    '''
    lacor_arr = Array(Float, modified=True) #(nD,1) array of autocorrelation lengths
    nDgrid = List(Array, modified=True) # list of nD entries: each entry is an array of points in the part. dimension
    reevaluate = Event
    seed = Bool(False)
    distr_type = Enum('Gauss', 'Weibull', modified=True)
    stdev = Float(1.0, modified=True)
    mean = Float(0.0, modified=True)
    shape = Float(5.0, modified=True)
    scale = Float(1.0, modified=True)
    loc = Float(0.0, modified=True)
    
    def acor(self, dx, lacor):
        '''autocorrelation function'''
        C = e ** (-(dx / lacor) ** 2)
        return C

    eigenvalues = Property(depends_on='+modified')
    @cached_property
    def _get_eigenvalues(self):
        '''evaluates the eigenvalues and eigenvectors of the covariance matrix'''
        # creating distances from the first coordinate
        for i, grid_i in enumerate(self.nDgrid):
            self.nDgrid[i] -= grid_i[0]
        # creating a symm. toeplitz matrix with (xgrid, xgrid) data points
        coords_lst = [toeplitz(grid_i) for grid_i in self.nDgrid]
        # apply the autocorrelation func. on the coord matrices to obtain the covariance matrices
        C_matrices = [self.acor(coords_i, self.lacor_arr[i]) for i, coords_i in enumerate(coords_lst)]
        # evaluate the eigenvalues and eigenvectors of the autocorrelation matrices
        eigen_lst = []
        for i, C_i in enumerate(C_matrices):
            print(('evaluating eigenvalues for dimension ' + str(i+1)))
            lambda_i, Phi_i = eigh(C_i)
            # truncate the eigenvalues at 99% of tr(C)
            truncation_limit = 0.99 * np.trace(C_i)
            argsort = np.argsort(lambda_i)
            cum_sum_lambda = np.cumsum(np.sort(lambda_i)[::-1])
            idx_trunc = int(np.sum(cum_sum_lambda < truncation_limit))
            eigen_lst.append([lambda_i[argsort[::-1]][:idx_trunc], Phi_i[:, argsort[::-1]][:,:idx_trunc]])
        print('complete')
        Lambda_C = 1.0
        Phi_C = 1.0
        for lambda_i, Phi_i in eigen_lst:
            Lambda_i = np.diag(lambda_i)
            Lambda_C = np.kron(Lambda_C, Lambda_i)
            Phi_C = np.kron(Phi_C, Phi_i)
        
        return Lambda_C, Phi_C
    
    generated_random_vector = Property(Array, depends_on='reevaluate')
    @cached_property
    def _get_generated_random_vector(self):
        if self.seed == True:
            np.random.seed(141)
        # points between 0 to 1 with an equidistant step for the LHS
        # No. of points = No. of truncated eigenvalues
        npts = self.eigenvalues[0].shape[0]
        randsim = np.linspace(0.5/npts, 1 - 0.5/npts, npts)
        # shuffling points for the simulation
        np.random.shuffle(randsim)
        # matrix containing standard Gauss distributed random numbers
        xi = norm().ppf(randsim)
        return xi
    
    random_field = Property(Array, depends_on='+modified')
    @cached_property
    def _get_random_field(self):
        '''simulates the Gaussian random field'''
        # evaluate the eigenvalues and eigenvectors of the autocorrelation matrix
        Lambda_C_sorted, Phi_C_sorted = self.eigenvalues
        # generate the RF with standardized Gaussian distribution
        ydata = np.dot(np.dot(Phi_C_sorted, (Lambda_C_sorted) ** 0.5), self.generated_random_vector)
        # transform the standardized Gaussian distribution
        if self.distr_type == 'Gauss':
            # scaling the std. distribution
            scaled_ydata = ydata * self.stdev + self.mean
        elif self.distr_type == 'Weibull':
            # setting Weibull params
            Pf = norm().cdf(ydata)
            scaled_ydata = weibull_min(self.shape, scale=self.scale, loc=self.loc).ppf(Pf)
        shape = tuple([len(grid_i) for grid_i in self.nDgrid])
        rf = np.reshape(scaled_ydata, shape)
        return rf
    
    def interpolate_rf(self, coords):
        '''interpolate RF values using the EOLE method
        coords = list of 1d arrays of coordinates'''
        # check consistency of dimensions
        if len(coords) != len(self.nDgrid):
            raise ValueError('point dimension differs from random field dimension')
        # create the covariance matrix
        C_matrices = [self.acor(coords_i.reshape(1, len(coords_i)) - self.nDgrid[i].reshape(len(self.nDgrid[i]),1), self.lacor_arr[i]) for i, coords_i in enumerate(coords)]
        
        C_u = 1.0
        for i, C_ui in enumerate(C_matrices):
            if i == 0:
                C_u *= C_ui
            else:
                C_u = C_u.reshape(C_u.shape[0], 1, C_u.shape[1]) * C_ui
            grid_size = 1.0
            for j in np.arange(i+1):
                grid_size *= len(self.nDgrid[j])
            C_u = C_u.reshape(grid_size,len(coords[0]))

        Lambda_Cx, Phi_Cx = self.eigenvalues
        # values interpolated in the standardized Gaussian rf 
        u = np.sum(self.generated_random_vector / np.diag(Lambda_Cx) ** 0.5 * np.dot(C_u.T, Phi_Cx), axis=1)
        if self.distr_type == 'Gauss':
            scaled_u = u * self.stdev + self.mean
        elif self.distr_type == 'Weibull':
            Pf = norm().cdf(u)
            scaled_u = weibull_min(self.shape, scale=self.scale, loc=self.loc).ppf(Pf)
        return scaled_u

if __name__ == '__main__':
    example1D = False
    example2D = True
    example3D = False
    
    if example1D is True:
        ''' 1D plot '''
        import matplotlib
        matplotlib.use('WxAgg')
        import matplotlib.pyplot as plt
        # random field instance
        rf = RandomField(seed=False,
                         distr_type='Gauss',
                         lacor_arr=np.array([1.0]),
                         nDgrid=[np.linspace(0.0, 10., 100)]
                         )
        plt.plot(rf.nDgrid[0], rf.random_field)
        # interpolation example
        interp_coords = [np.array([1., 2., 3., 5., 8.])]
        u = rf.interpolate_rf(interp_coords)
        plt.plot(interp_coords[0], u, 'ro')
        plt.show()
    
    if example2D is True:
        ''' 2D plot '''
        import os
        os.environ['ETS_TOOLKIT'] = 'qt4'
        os.environ['QT_API'] = 'pyqt'
        from mayavi import mlab
        # random field instance
        rf = RandomField(distr_type='Gauss',
                         seed=True,
                         lacor_arr=np.array([10.0, 1.]),
                    nDgrid=[np.linspace(0.0, 50., 500),
                            np.linspace(0.0, 30., 300)]
                    )
        rand_field_2D = rf.random_field
        x, y = rf.nDgrid
        mlab.surf(x, y, rand_field_2D)
        # interpolation example
        interp_coords = [np.array([10., 20., 30., 35.]),np.array([1.0, 5., 13., 19.])]
        u = rf.interpolate_rf(interp_coords)
        mlab.points3d(interp_coords[0], interp_coords[1], u, scale_factor=.8, color=(1,0,0))
        mlab.show()
        mlab.close()
        
    if example3D is True:
        ''' 3D plot '''
        import os
        os.environ['ETS_TOOLKIT'] = 'qt4'
        os.environ['QT_API'] = 'pyqt'
        from mayavi import mlab
        # random field instance
        rf = RandomField(distr_type='Gauss',
                         lacor_arr=np.array([5.0, 4.0, 2.0]),
                    nDgrid=[np.linspace(0.0, 20., 20),
                            np.linspace(0.0, 30., 30),
                            np.linspace(0.0, 40., 40)]
                    )
        rand_field_3D = rf.random_field
        x,y,z = rf.nDgrid
        cx,cy,cz = np.mgrid[x[0]:x[-1]:complex(0,len(x)), y[0]:y[-1]:complex(0,len(y)), z[0]:z[-1]:complex(0,len(z))]
        mlab.contour3d(cx,cy,cz, rand_field_3D, contours=7, transparent=True)
        mlab.show()
        mlab.close()
 