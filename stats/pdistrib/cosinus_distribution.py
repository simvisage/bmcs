
from scipy.stats.distributions import rv_continuous
from numpy import pi, cos, sin, arcsin

class cos_gen( rv_continuous ):
    def _pdf( self, x ):
        return sin( x )
    def _cdf( self, x ):
        return 1 - cos( x )
    def _stats( self ):
        return 1.0, pi - 2.0, 0.0, 0.0
    def ppf( self, x, *args, **kw ):
        return arcsin( x )
cos_distr = cos_gen( a=0, b=pi / 2., name='cosine', extradoc="""

Cosine distribution.
""" )
