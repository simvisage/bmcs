
from scipy.stats.distributions import rv_continuous
from numpy import pi, cos, sin, arccos

class sin_gen( rv_continuous ):
    def _pdf( self, x ):
        return sin( x )
    def _cdf( self, x ):
        return 1 - cos( x )
    def _stats( self ):
        return 1.0, pi - 3.0, 0.0, 0.0
    def ppf( self, x, *args, **kw ):
        return arccos( 1 - x )
sin_distr = sin_gen( a = 0, b = pi / 2., name = 'sine', extradoc = """

sine distribution.
""" )
