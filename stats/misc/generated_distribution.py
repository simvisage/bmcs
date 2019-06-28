'''
Created on 1.9.2011

GenDistr takes the attributes x_values (np.array) and corresponding pdf_values
or cdf_values. The class provides the same statistical methods as
scipy.stats.distribution classes but evaluates them numerically.
If a PDF or CDF is evaluated numerically as an array, the statistical moments
and functions can be obtained using this class.

@author: Q
'''

from etsproxy.traits.api import HasTraits, Int, Array, Property, cached_property

import scipy as sp
import numpy as np

from mathkit.mfn import MFnLineArray

class GenDistr( HasTraits ):

    x_values = Array
    pdf_values = Array
    cdf_values = Array

    def check( self ):
        if len( self.x_values ) == 0:
            raise ValueError('x_values not defined')
        if len( self.pdf_values ) == 0 and len( self.cdf_values ) == 0:
            raise ValueError('either pdf_values or cdf_values have to be given')
        else:
            pass

    cached_pdf = Property( depends_on = 'x_values, pdf_values, cdf_values' )
    @cached_property
    def _get_cached_pdf( self ):
        if len( self.pdf_values ) == 0:
            cdf_line = MFnLineArray( xdata = self.x_values, ydata = self.cdf_values )
            pdf = []
            for x in self.x_values:
                pdf.append( cdf_line.get_diff( x ) )
            return MFnLineArray( xdata = self.x_values, ydata = pdf )
        else:
            return MFnLineArray( xdata = self.x_values, ydata = self.pdf_values )

    def pdf( self, x ):
        self.check()
        return self.cached_pdf.get_values( x )

    cached_cdf = Property( depends_on = 'x_values, pdf_values, cdf_values' )
    @cached_property
    def _get_cached_cdf( self ):
        if len( self.cdf_values ) == 0:
            cdf = []
            for i in range( len( self.x_values ) ):
                cdf.append( np.trapz( self.pdf_values[:i], self.x_values[:i] ) )
            return MFnLineArray( xdata = self.x_values, ydata = cdf )
        else:
            return MFnLineArray( xdata = self.x_values, ydata = self.cdf_values )

    def cdf( self, x ):
        self.check()
        return self.cached_cdf.get_values( x )

    def sf( self, x ):
        self.check()
        return 1 - self.cdf( x )

    def ppf( self, x ):
        self.check()
        if len( self.cdf_values ) != 0:
            xx, cdf = self.x_values, self.cdf_values
        else:
            xx, cdf = self.x_values, self.cdf( self.x_values )
        ppf_mfn_line = MFnLineArray( xdata = cdf, ydata = xx )
        return ppf_mfn_line.get_values( x )

    def stats( self, str ):
        self.check()
        if str == 'm':
            return self.mean()
        elif str == 'v':
            return self.var()
        elif str == 'mv':
            return ( self.mean(), self.var() )

    def median( self ):
        self.check()
        return self.ppf( 0.5 )

    def mean( self ):
        self.check()
        if len( self.pdf_values ) != 0:
            x, pdf = self.x_values, self.pdf_values
        else:
            x, pdf = self.x_values, self.pdf( self.x_values )
        return np.trapz( x * pdf , x )

    def var( self ):
        self.check()
        if len( self.pdf_values ) != 0:
            x, pdf = self.x_values, self.pdf_values
        else:
            x, pdf = self.x_values, self.pdf( self.x_values )
        mu = self.mean()
        return np.trapz( ( x ** 2 - mu ) * pdf , x )

    def std( self ):
        self.check()
        return np.sqrt( self.var() )

if __name__ == '__main__':

    from scipy.stats import norm
    from matplotlib import pyplot as plt
    gd = GenDistr()
    x = np.linspace( -4, 4, 300 )
    gd.x_values = x
    gd.cdf_values = norm( 0, 1 ).cdf( x )
    plt.plot( gd.x_values, gd.cdf_values, lw = 2, label = 'input CDF' )
    plt.plot( x, gd.pdf( x ), lw = 2, label = 'evaluated PDF' )
    plt.plot( x, gd.sf( x ), lw = 2, label = 'evaluated SF' )
    plt.legend( loc = 'best' )
    plt.show()
