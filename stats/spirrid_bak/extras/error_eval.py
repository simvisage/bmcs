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

from etsproxy.traits.api import HasTraits, Array, Property, Float, \
    cached_property
import math
import numpy as np # import numpy package


#===============================================================================
# Error evaluation
#===============================================================================

class ErrorEval(HasTraits):
    '''Evaluate the error for a given exact solution'''

    #===========================================================================
    # Configuration of the error evaluation 
    #===========================================================================

    # array containing the exact solution
    exact_arr = Array(value = [], dtype = float)

    # distance of the data points (equidistant resolution assumed)
    d_e = Float(1.0)

    # get the peak value of the exact solution
    exact_max = Property(Float, depends_on = 'exact_arr')
    @cached_property
    def _get_exact_max(self):
        return np.max(self.exact_arr)

    # get the index of the peak value of the exact solution
    exact_argmax = Property(Float, depends_on = 'exact_arr')
    @cached_property
    def _get_exact_argmax(self):
        return np.argmax(self.exact_arr)

    # get the integral of the exact solution
    exact_E = Property(Float, depends_on = 'exact_arr')
    @cached_property
    def _get_exact_E(self):
        return np.sum(self.exact_arr * self.d_e)

    #===========================================================================
    # Error evaluation methods
    #===========================================================================

    def eval_error_max(self, approx_arr):
        '''
        Evaluate the relative error as a maximum difference to the exact solution
        normalized by the peak value of the exact solution
        '''
        squared_err = (approx_arr - self.exact_arr) ** 2
        max_err = math.sqrt(np.max(squared_err))
        return max_err / self.exact_max

    def eval_error_peak(self, approx_arr):
        '''
        Evaluate the relative error as a maximum peak difference to the exact solution
        normalized by the peak value of the exact solution
        '''
        max_exact = self.exact_arr[ self.exact_argmax ]
        peak_err = math.fabs(self.exact_max - max_exact)
        return peak_err / max_exact

    def eval_error_energy(self, approx_arr):
        '''
        evaluate the relative error as a difference of energy to the exact solution
        normalized by the energy of the exact solution
        '''
        delta_E = np.sqrt(np.sum((approx_arr - self.exact_arr) ** 2 * self.d_e))
        return delta_E / self.exact_E

    def eval_error_rms(self, approx_arr):
        '''
        Evaluate the relative error as a rms
        '''
        n_e = self.exact_arr.size
        squared_err = (approx_arr - self.exact_arr) ** 2
        return np.sqrt(np.sum(squared_err) / n_e) / self.exact_max

    def eval_error_all(self, approx_arr):
        '''
        Evaluate all errors and store them in an array
        '''
        return np.array([self.eval_error_max(approx_arr),
                          self.eval_error_energy(approx_arr),
                          self.eval_error_rms(approx_arr)],
                        dtype = float
                        )
