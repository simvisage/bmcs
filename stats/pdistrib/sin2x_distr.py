'''
Created on 20.06.2011

@author: axel
'''

from scipy.stats.distributions import rv_continuous
from numpy import pi, cos, sin, arccos

class sin2x_gen( rv_continuous ):
    def _pdf( self, x ):
        return sin( 2 * x )
    def _cdf( self, x ):
        return - 0.5 * cos ( 2 * x ) + 0.5
    def _stats( self ):
        return pi / 4, 0.0625 * pi ** 2 - 0.5 , 0.0, 0.0
    def _ppf( self, x, *args, **kw ):
        return arccos( -2 * x + 1 ) / 2.
sin2x = sin2x_gen( a = 0, b = pi / 2., name = 'sine(2x)', extradoc = """

sine(2x) distribution.
""" )
