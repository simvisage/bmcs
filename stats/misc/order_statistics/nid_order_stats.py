'''
Created on 31.8.2011

class for evaluating the kth statistics of a subset of n realizations
each of which has a different distribution.
Example: k = 1, n = 10 evaluates the PDF of the weakest of 10 elements
taken from 10 different distributions

@author: Q
'''

from etsproxy.traits.api import HasTraits, Instance, Int, Array, List, \
    cached_property, Property
import numpy as np
import scipy as sp
import itertools as it
import time



class NIDOrderStats( HasTraits ):

    distr_list = List( desc = 'NID distributions' )

    n = Property( Int, depends_on = 'distr_list',
            desc = 'number of realizations' )
    def _get_n( self ):
        return len( self.distr_list )

    k = Int( 1, auto_set = False, enter_set = True,
            desc = 'kth order statistics to evaluate' )

    x_arr = np.linspace( 0, 10, 200 )

    cdf_arr = Property( Array, depends_on = 'distr_list' )
    @cached_property
    def _get_cdf_arr( self ):
        '''creates a 2D array of shape (m,x_arr) containing m CDFs'''
        cdf_arr = np.ones( ( self.n + 2, len( self.x_arr ) ) )
        for i, distr in enumerate( self.distr_list ):
            cdf_arr[i] = distr.cdf( self.x_arr )
        return cdf_arr

    sf_arr = Property( Array, depends_on = 'distr_list' )
    @cached_property
    def _get_sf_arr( self ):
        '''creates a 2D array of shape (m,x_arr) containing m SFs'''
        sf_arr = np.ones( ( self.n + 2, len( self.x_arr ) ) )
        for i, distr in enumerate( self.distr_list ):
            sf_arr[i] = distr.sf( self.x_arr )
        return sf_arr

    pdf_arr = Property( Array, depends_on = 'distr_list' )
    @cached_property
    def _get_pdf_arr( self ):
        '''creates a 2D array of shape (m,x_arr) containing m PDFs'''
        pdf_arr = np.ones( ( self.n + 2, len( self.x_arr ) ) )
        for i, distr in enumerate( self.distr_list ):
            pdf_arr[i] = distr.pdf( self.x_arr )
        return pdf_arr

    def kth_pdf( self ):
        '''evaluates the PDF of the kth entry; cases k = 1 and k > 1 are distinguished
            The most general formula is used here. For higher performance use CIDOrderStats'''
        if len( self.distr_list ) == self.n:
            n = self.n
            k = self.k
            if n >= k:
                fct = sp.misc.factorial
                # the constant actually tells how many redundances are present in the summation
                constant = 1. / ( fct( k - 1 ) * fct( n - k ) )
                summation = np.zeros( len( self.x_arr ) )
                permutations = it.permutations( list(range( n)), n )
                loop_run = True
                t = time.clock()
                while loop_run == True:
                    try:
                        idx = list( next(permutations) )
                        if k == 1:
                            id_pdf = idx[k - 1]
                            pdf = self.pdf_arr[id_pdf]
                            if n == 1:
                                PDF = pdf
                                summation += PDF.flatten()
                            elif n == 2:
                                id_sf = idx[1]
                                sf = self.sf_arr[id_sf]
                                PDF = pdf * sf
                                summation += PDF.flatten()
                            else:
                                id_sf = idx[k:]
                                sf = self.sf_arr[id_sf]
                                PDF = pdf * sf.prod( axis = 0 )
                                summation += PDF.flatten()
                        else:
                            id_cdf = idx[:k - 1]
                            cdf = self.cdf_arr[id_cdf]
                            id_pdf = idx[k - 1]
                            pdf = self.pdf_arr[id_pdf]
                            id_sf = idx[k:]
                            sf = self.sf_arr[id_sf]
                            PDF = cdf.prod( axis = 0 ) * pdf * sf.prod( axis = 0 )
                            summation += PDF.flatten()
                    except StopIteration:
                        loop_run = False

                print(('NID', time.clock() - t, 's'))
                return constant * summation

            else:
                raise ValueError('n < k')
        else:
            raise ValueError('%i distributions required, %i given' % ( self.n, len( self.distr_list ) ))

if __name__ == '__main__':

    from matplotlib import pyplot as p
    from scipy.stats import norm, weibull_min, uniform
    import numpy as np


    distr1 = uniform( 4, 4 )
    distr2 = norm( 5, 0.7 )
    distr3 = weibull_min( 1, 2, loc = 3 )
    distr4 = norm( 3, 2 )
    distr5 = norm( 6, 1 )
    distr6 = norm( 6, 2 )
    nid = NIDOrderStats( distr_list = [distr1,
                                       distr2,
                                       distr3,
                                       distr4,
                                       distr5,
                                       distr6,
                                       ], k = 1, x_arr = np.linspace( -3, 15, 500 ) )

    for i, distr in enumerate( [distr1, distr2, distr3, distr4, distr5, distr6] ):
        p.plot( nid.x_arr, distr.pdf( nid.x_arr ), lw = 1, label = 'distr%s' % ( i + 1 ) )

    nid.k = 1
    p.plot( nid.x_arr, nid.kth_pdf(), lw = 2, color = 'black', label = 'k = 1' )
    nid.k = 6
    p.plot( nid.x_arr, nid.kth_pdf(), lw = 2, color = 'black', label = 'k = 6' )
    p.legend()
    p.title( 'order stats for 6 non-identically distributed RVs' )
    p.show()
