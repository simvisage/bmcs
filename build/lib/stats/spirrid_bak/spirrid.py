#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This softwe_arare is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 8, 2011 by: rch

from stats.spirrid_bak import CodeGenNumpyFactory, CodeGenCFactory, \
    CodeGenCythonFactory
from etsproxy.traits.api import Property, cached_property, Callable, Trait, \
    on_trait_change, Event
from stats.spirrid_bak import FunctionRandomization, TGrid, PGrid, MonteCarlo, \
    LatinHypercubeSampling, make_ogrid
import platform
import string
import types
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock


#===============================================================================
# Generic implementation of the integral
#===============================================================================
class SPIRRID(FunctionRandomization):
    '''Set of parallel independent responses with random identical distributions.
    '''
    #===========================================================================
    # type of the sampling of the random domain
    #===========================================================================
    sampling_type = Trait('TGrid', {'TGrid' : TGrid,
                                      'PGrid' : PGrid,
                                      'MCS' : MonteCarlo,
                                      'LHS': LatinHypercubeSampling },
                          input_change = True)

    sampling = Property(depends_on = 'input_change')
    @cached_property
    def _get_sampling(self):
        return self.sampling_type_(randomization = self)

    #===========================================================================
    # Code generator
    #===========================================================================

    codegen_type = Trait('numpy', {'numpy' : CodeGenNumpyFactory(),
                                   'c' : CodeGenCFactory(),
                                   'cython' : CodeGenCythonFactory()},
                         input_change = True)

    # object representing the code generator
    codegen = Property(depends_on = 'codegen_type,sampling_type')
    @cached_property
    def _get_codegen(self):
        return self.codegen_type_(spirrid = self)

    #===========================================================================
    # Inspection methods
    #===========================================================================
    def get_samples(self, n = 20):
        '''Return the first n randomly selected samples.
        '''
        return self.sampling.get_samples(n)

    #===========================================================================
    # Template for the integration of a response function in the time loop
    #===========================================================================
    mu_q_method = Property(Callable, depends_on = 'input_change,alg_option,recalc')
    @cached_property
    def _get_mu_q_method(self):
        '''Generate an integrator method for the particular data type
        of dG and variables. 
        '''
        return self.codegen.get_code()

    #===========================================================================
    # Run the estimation of the mean response
    #===========================================================================
    results = Property(depends_on = 'input_change,codegen_option, recalc')
    @cached_property
    def _get_results(self):
        '''Estimate the mean value function given the randomization pattern.
        '''
        self.tvar_names
        self.evar_names

        time__start = sysclock()
        e_orth = make_ogrid(self.evar_lst)
        self.sampling.theta
        self.sampling.dG
        time__data_setup = sysclock()
        self.mu_q_method
        time__method_setup = sysclock()
        mu_q_arr, var_q_arr = self.mu_q_method(*e_orth)
        time__exec = sysclock()
        return mu_q_arr, var_q_arr, [time__data_setup - time__start,
                                     time__method_setup - time__data_setup,
                                     time__exec - time__method_setup]

    #===========================================================================
    # Access results
    #===========================================================================
    mu_q_arr = Property()
    def _get_mu_q_arr(self):
        '''Mean value of q'''
        return self.results[0]

    var_q_arr = Property()
    def _get_var_q_arr(self):
        '''Variance of q'''
        # switch on the implicit evaluation of variance 
        # if it has not been the case so far
        if not self.codegen.implicit_var_eval:
            self.codegen.implicit_var_eval = True
        return self.results[1]

    exec_time = Property()
    def _get_exec_time(self):
        return self.results[2]

    #===========================================================================
    # state monitors
    #===========================================================================
    # enable recalculation of results property
    # (for time efficiency analysis)
    recalc = Event
    @on_trait_change('recalc')
    def set_recalc(self):
        self.sampling.recalc = True
        self.codegen.recalc = True

    # Change propagation (traits having input_change metadata)
    # are monitored using the input_change event
    input_change = Event
    @on_trait_change('+input_change')
    def set_input_change(self):
        self.input_change = True

    # The subcomponents codegen and sampling can use
    # this method to trigger a change inducing
    # a recalculation
    codegen_option = Event

    # Change in the sampling configuration
    # (this might be thresholds for covering
    # the random domain). 
    sampling_option = Event

    #===========================================================================
    # Introspection
    #===========================================================================
    # report the current configuration of the integrator
    def __str__(self):

        # get the name either of the method or of the class
        try:
            qname = self.q.__name__
        except AttributeError:
            qname = self.q.__class__.__name__

        s = '# function:\n'
        s += 'q = %s(%s)\n' % (qname, string.join(self.var_names, ','))

        s += '# evars:\n'
        s += self.evar_str

        s += '\n'
        s += '# tvars[n_int = %d]: \n' % self.n_int
        s += self.tvar_str

        s += '\n'
        s += '# sampling: %s\n' % self.sampling_type
        s += '# codegen: %s\n' % self.codegen_type
        s += str(self.codegen)
        return s
