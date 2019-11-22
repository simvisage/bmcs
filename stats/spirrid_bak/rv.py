#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 8, 2011 by: rch

from etsproxy.traits.api import HasTraits, Property, Float, cached_property, \
    Str, Int
from stats.pdistrib import PDistrib as PD


#===============================================================================
# Probability distribution specification
#===============================================================================
class RV(HasTraits):

    def __init__(self, type, loc = 0.0, scale = 0.0, shape = 1.0,
                  *args, **kw):
        '''Convenience initialization'''
        super(RV, self).__init__(*args, **kw)
        self.type = type
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.args = args
        self.kw = kw

    def __str__(self):
        return '%s( loc = %g, scale = %g, shape = %g)[n_int = %s]' % \
            (self.type, self.loc, self.scale, self.shape, str(self.n_int))

    # number of integration points
    n_int = Int(None)

    # location parameter
    loc = Float

    # scale parameter
    scale = Float

    # shape parameter
    shape = Float

    # type specifier
    type = Str

    # hidden property instance of the scipy stats distribution
    _distr = Property(depends_on = 'mu,std,loc,type')
    @cached_property
    def _get__distr(self):
        '''Construct a distribution.
        '''
        if self.n_int == None:
            n_segments = 10
        else:
            n_segments = self.n_int
        pd = PD(distr_choice = self.type, n_segments = n_segments)
        pd.distr_type.set(scale = self.scale, shape = self.shape, loc = self.loc)
        return pd

    # access methods to pdf, ppf, rvs
    def pdf(self, x):
        return self._distr.pdf(x)

    def ppf(self, x):
        return self._distr.ppf(x)

    def rvs(self, x):
        return self._distr.rvs(x)

