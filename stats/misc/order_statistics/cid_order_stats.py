'''
Created on 5.9.2011

class for evaluating the kth statistics of a subset of n realizations
each of which can have different distribution but a number of identically
distributed realization can occur. The algorithm efficiently reduces the
number of permutation to the necessary ones needed for the non identically
distributed realizations.
Example: k = 1, distr_list = [norm, poisson, weibull], iid_list = [3,3,2]
evaluates the PDF of the weakest of 8 elements taken from three different
distributions

@author: Q
'''


from etsproxy.traits.api import HasTraits, Int, Array, List, \
    cached_property, Property
import numpy as np
import scipy as sp
from itertools import combinations_with_replacement as cbs
import time

class CIDOrderStats( HasTraits ):

    distr_list = List( desc = 'NID distributions' )
    iid_list = List( desc = '''list of length len(distr_list),
                            entries = how many times is the
                            particular distribution repeated''' )

    m = Property( Int, depends_on = 'distr_list' )
    @cached_property
    def _get_m( self ):
        return len( self.distr_list )

    nid_indexes = Property( List, depends_on = 'distr_list' )
    @cached_property
    def _get_nid_indexes( self ):
        return list(range( self.m))

    k = Int( 1, auto_set = False, enter_set = True,
            desc = 'kth order statistics to evaluate' )

    n = Property( Int, depends_on = 'iid_list' )
    @cached_property
    def _get_n( self ):
        return sum( self.iid_list )

    x_arr = np.linspace( 0, 10, 200 )

    cdf_arr = Property( Array, depends_on = 'distr_list' )
    @cached_property
    def _get_cdf_arr( self ):
        '''creates a 2D array of shape (m,x_arr) containing m CDFs'''
        cdf_arr = np.ones( ( len( self.distr_list ) + 2, len( self.x_arr ) ) )
        for i, distr in enumerate( self.distr_list ):
            cdf_arr[i] = distr.cdf( self.x_arr )
        return cdf_arr

    sf_arr = Property( Array, depends_on = 'distr_list' )
    @cached_property
    def _get_sf_arr( self ):
        '''creates a 2D array of shape (m,x_arr) containing m SFs'''
        sf_arr = np.ones( ( len( self.distr_list ) + 2, len( self.x_arr ) ) )
        for i, distr in enumerate( self.distr_list ):
            sf_arr[i] = distr.sf( self.x_arr )
        return sf_arr

    pdf_arr = Property( Array, depends_on = 'distr_list' )
    @cached_property
    def _get_pdf_arr( self ):
        '''creates a 2D array of shape (m,x_arr) containing m PDFs'''
        pdf_arr = np.ones( ( len( self.distr_list ) + 2, len( self.x_arr ) ) )
        for i, distr in enumerate( self.distr_list ):
            pdf_arr[i] = distr.pdf( self.x_arr )
        return pdf_arr

    def get_com_list( self, nid_idx, iid_list ):
        '''creates tuples of indexes for combining the CDFs'''
        k = self.k
        m = len( iid_list )
        comb = cbs( nid_idx, k - 1 )
        unique_combs = []
        count_arr = np.zeros( m )

        try:
            while True:
                c = next(comb)
                for i, idx in enumerate( nid_idx ):
                    count_arr[i] = c.count( idx )
                if np.all( count_arr <= iid_list ):
                    unique_combs.append( c )
        except StopIteration:
            pass
        return unique_combs

    def kth_pdf( self ):
        '''evaluates the PDF of the kth entry; cases k = 1 and k > 1 are distinguished'''
        t = time.clock()
        n = self.n
        m = self.m
        k = self.k
        fct = sp.misc.factorial
        cbns = sp.misc.comb
        if len( self.iid_list ) == len( self.distr_list ):
            if n >= k:
                summation = np.zeros( len( self.x_arr ) )
                if k == 1:
                    for i, exp in enumerate( self.iid_list ):
                        pdf = self.pdf_arr[i] * exp
                        sf_exp = self.iid_list[:]
                        sf_exp[i] -= 1
                        sf_exp = np.array( sf_exp )[:, np.newaxis]
                        sf = self.sf_arr[list(range( m))] ** sf_exp
                        summation += pdf * sf.prod( axis = 0 )
                else:
                    # PDF taken from every NID
                    for pdfi, exp in enumerate( self.iid_list ):
                        pdf = self.pdf_arr[pdfi] * exp
                        iid_copy = self.iid_list[:]
                        nid_copy = self.nid_indexes[:]
                        # substract the index which is taken for the PDF
                        iid_copy[pdfi] -= 1
                        try:
                            # if the PDF is taken from a unique distribution
                            # the distribution is popped out of the list
                            zero_id = iid_copy.index( 0 )
                            iid_copy.pop( zero_id )
                            nid_copy.pop( zero_id )
                        except ValueError:
                            pass
                        # combs is a list of k-1 unique combinations for the CDFs
                        combs = self.get_com_list( nid_copy, iid_copy )
                        for i, cdf_comb in enumerate( combs ):
                            # the SFs are built from the remaining indexes
                            sf_exp = self.iid_list[:]
                            sf_exp[pdfi] -= 1
                            multipl = 1
                            # every combination can occur comb(IID[i], IID in CDF) times 
                            for m_idx in self.nid_indexes:
                                index_occurs = cdf_comb.count( m_idx )
                                index_total = sf_exp[m_idx]
                                if index_occurs != 0:
                                    times = cbns( index_total, index_occurs )
                                    multipl *= times
                            for index in cdf_comb:
                                sf_exp[index] -= 1
                            sf_exp = np.array( sf_exp )[:, np.newaxis]
                            sf = self.sf_arr[list(range( m))] ** sf_exp
                            sf = sf.prod( axis = 0 )
                            cdf = self.cdf_arr[np.array( cdf_comb )].prod( axis = 0 )
                            # a summand is the product of the combination of CDFs, SFs, one PDF
                            # and the particular multiplier
                            summation += cdf * pdf * sf * multipl
                print(('CID', time.clock() - t, 's', 'k = ', self.k))
                self.performance.append( ( time.clock() - t, self.k ) )
                return summation
            else:
                raise ValueError('n < k')
        else:
            raise ValueError('len(distr_list) != len(iid_list)')

    performance = List

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from scipy.stats import norm, weibull_min
    from .nid_order_stats import NIDOrderStats

    distr1 = norm( 4, 1 )
    distr2 = norm( 5, 0.3 )
    distr3 = weibull_min( 4, 1 )
    distr4 = norm( 3, 2 )
    distr5 = norm( 4.6, 0.4 )
    distr6 = norm( 6, 2 )
    cid = CIDOrderStats( distr_list = [distr1,
                                       distr2,
                                       distr3,
                                       distr4,
                                       distr5,
                                       distr6,
                                       distr6,
                                       distr6,
                                       distr6,
                                       distr6,
                                       ],
                        iid_list = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ],
                        k = 4, x_arr = np.linspace( -4, 6, 200 ) )

    nid = NIDOrderStats( distr_list = [distr1,
                                       distr1,
                                       distr2,
                                       distr2,
                                       distr2,
                                       distr3,
                                       distr4,
                                       distr4,
                                       ], k = 4, x_arr = np.linspace( -2, 3, 200 ) )

    times = []
    for k in range( 5 ):
        nid.k = k + 1
        cid.k = k + 1
        kth_pdf = cid.kth_pdf()
        plt.plot( cid.x_arr, kth_pdf, label = 'k = %i' % cid.k )
        times.append( cid.performance[-1][0] )
        #plt.plot( nid.x_arr, nid.kth_pdf(), color = 'red', lw = 2, ls = 'dashed', label = 'NID' )
    #plt.plot( np.array( range( cid.k ) ) + 1, times )
    plt.legend( loc = 'best' )
    plt.show()
