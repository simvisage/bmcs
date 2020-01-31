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
# Created on Nov 8, 2011 by: rch

from stats.spirrid_bak import CodeGen
import numpy as np

#===============================================================================
# Generator of the numpy code
#===============================================================================
class CodeGenNumpy(CodeGen):
    '''
        Numpy code is identical for all types of sampling, 
        no special treatment needed. 
    '''
    def get_code(self):
        '''
            Return the code for the given sampling of the random domain.
        '''
        s = self.spirrid
        n = len(s.evar_lst)
        targs = dict(list(zip(s.tvar_names, s.sampling.theta)))

        if self.implicit_var_eval:

            def mu_q_method(*e):
                '''Template for the evaluation of the mean response.
                '''
                eargs = dict(list(zip(s.evar_names, e)))
                args = dict(eargs, **targs)

                Q_dG = s.q(**args)
                Q2_dG = Q_dG ** 2

                Q_dG *= s.sampling.dG # in-place multiplication
                Q2_dG *= s.sampling.dG

                # sum all squared values to get the variance
                mu_q = np.sum(Q_dG)
                var_q = np.sum(Q2_dG) - mu_q ** 2

                return mu_q, var_q
        else:

            def mu_q_method(*e):
                '''Template for the evaluation of the mean response.
                '''
                eargs = dict(list(zip(s.evar_names, e)))
                args = dict(eargs, **targs)

                Q_dG = s.q(**args)

                Q_dG *= s.sampling.dG # in-place multiplication

                # sum all squared values to get the variance
                mu_q = np.sum(Q_dG)

                return mu_q, None

        otypes = [ float for i in range(n * 2)]
        return np.vectorize(mu_q_method, otypes = otypes)

    def __str__(self):
        return 'var_eval: %s\n' % repr(self.implicit_var_eval)

