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
# Created on Sep 22, 2011 by: rch

from etsproxy.traits.api import HasStrictTraits, Array, Property, Float, \
    cached_property, Callable, Str, Int, WeakRef, Dict, Event
from stats.spirrid_bak import RV
import inspect
import numpy as np
import string
import types
from functools import reduce


#===============================================================================
# Helper methoods to produce an-dimensional array from a list of arrays 
#===============================================================================
def make_ogrid(args):
    '''Orthogonalize a list of one-dimensional arrays.
    scalar values are left untouched.
    '''
    # count the number of arrays in the list
    dt = list(map(type, args))
    n_arr = dt.count(np.ndarray)

    oargs = []
    i = 0
    for arg in args:
        if isinstance(arg, np.ndarray):
            shape = np.ones((n_arr,), dtype = 'int')
            shape[i] = len(arg)
            i += 1
            oarg = np.copy(arg).reshape(tuple(shape))
            oargs.append(oarg)
        elif isinstance(arg, float):
            oargs.append(arg)
    return oargs

def make_ogrid_full(args):
    '''Orthogonalize a list of one-dimensional arrays.\
    including scalar values in args.
    '''
    oargs = []
    n_args = len(args)
    for i, arg in enumerate(args):
        if isinstance(arg, float):
            arg = np.array([arg], dtype = 'd')

        shape = np.ones((n_args,), dtype = 'int')
        shape[i] = len(arg)
        i += 1
        oarg = np.copy(arg).reshape(tuple(shape))
        oargs.append(oarg)
    return oargs

#===============================================================================
# Function randomization
#===============================================================================
class FunctionRandomization(HasStrictTraits):

    # response function
    q = Callable(input = True)

    #===========================================================================
    # Inspection of the response function parameters
    #===========================================================================
    var_spec = Property(depends_on = 'q')
    @cached_property
    def _get_var_spec(self):
        '''Get the names of the q_parameters'''
        if type(self.q) is types.FunctionType:
            arg_offset = 0
            q = self.q
        else:
            arg_offset = 1
            q = self.q.__call__
        argspec = inspect.getargspec(q)
        args = np.array(argspec.args[ arg_offset:])
        dflt = np.array(argspec.defaults)
        return args, dflt

    var_names = Property(depends_on = 'q')
    @cached_property
    def _get_var_names(self):
        '''Get the array of default values.
        None - means no default has been specified
        '''
        return self.var_spec[0]

    var_defaults = Property(depends_on = 'q')
    @cached_property
    def _get_var_defaults(self):
        '''Get the array of default values.
        None - means no default has been specified
        '''
        dflt = self.var_spec[1]
        defaults = np.repeat(None, len(self.var_names))
        start_idx = min(len(dflt), len(defaults))
        defaults[ -start_idx: ] = dflt[ -start_idx:]
        return defaults

    #===========================================================================
    # Control variable specification
    #===========================================================================
    evars = Dict(Str, Array, input_change = True)
    def __evars_default(self):
        return { 'e': [0, 1] }

    evar_lst = Property()
    def _get_evar_lst(self):
        ''' sort entries according to var_names.'''
        return [ self.evars[ nm ] for nm in self.evar_names ]

    evar_names = Property(depends_on = 'evars')
    @cached_property
    def _get_evar_names(self):
        evar_keys = list(self.evars.keys())
        return [nm for nm in self.var_names if nm in evar_keys ]

    evar_str = Property()
    def _get_evar_str(self):
        s_list = ['%s = [%g, ..., %g] (%d)' % (name, value[0], value[-1], len(value))
                  for name, value in zip(self.evar_names, self.evar_lst)]
        return string.join(s_list, '\n')

    # convenience property to specify a single control variable without
    # the need to send a dictionary
    e_arr = Property
    def _set_e_arr(self, e_arr):
        '''Get the first free argument of var_names and set it to e vars
        '''
        self.evars[self.var_names[0]] = e_arr

    #===========================================================================
    # Specification of parameter value / distribution
    #===========================================================================

    tvars = Dict(input_change = True)

    tvar_lst = Property(depends_on = 'tvars')
    @cached_property
    def _get_tvar_lst(self):
        '''sort entries according to var_names
        '''
        return [ self.tvars[ nm ] for nm in self.tvar_names ]

    tvar_names = Property
    def _get_tvar_names(self):
        '''get the tvar names in the order given by the callable'''
        tvar_keys = list(self.tvars.keys())
        return np.array([nm for nm in self.var_names if nm in tvar_keys ], dtype = str)

    tvar_str = Property()
    def _get_tvar_str(self):
        s_list = ['%s = %s' % (name, str(value))
                  for name, value in zip(self.tvar_names, self.tvar_lst)]
        return string.join(s_list, '\n')

    # number of integration points
    n_int = Int(10, input_change = True)

#===============================================================================
# Randomization classes
#===============================================================================
class RandomSampling(HasStrictTraits):
    '''Deliver the discretized theta and dG
    '''
    randomization = WeakRef(FunctionRandomization)

    recalc = Event

    # count the random variables
    n_rand_vars = Property
    def _get_n_rand_vars(self):
        dt = list(map(type, self.randomization.tvar_lst))
        return dt.count(RV)

    n_sim = Property
    def _get_n_sim(self):
        '''Get the total number of sampling points.
        '''
        n_int = self.randomization.n_int
        return n_int ** self.n_rand_vars

    def get_theta_range(self, tvar):
        '''Return minimu maximum and delta of the variable.
        '''
        if tvar.n_int != None:
            n_int = tvar.n_int
        else:
            n_int = self.randomization.n_int
        min_theta = tvar.ppf(1e-5)
        max_theta = tvar.ppf(1 - 1e-5)
        len_theta = max_theta - min_theta
        d_theta = len_theta / n_int
        return min_theta, max_theta, d_theta

    theta = Property()

    def get_samples(self, n):
        '''Get the fully expanded samples (for plotting)
        '''
        raise NotImplemented

class RegularGrid(RandomSampling):
    '''Grid shape randomization
    '''
    theta_list = Property(Array(float), depends_on = 'recalc')
    @cached_property
    def _get_theta_list(self):
        '''Get the orthogonally oriented arrays of random variables. 
        '''
        theta_list = []
        for tvar in self.randomization.tvar_lst:
            if isinstance(tvar, float):
                theta_list.append(tvar)
                continue
            theta_list.append(self.get_theta_for_distrib(tvar))
        return theta_list

    theta = Property(Array(float), depends_on = 'recalc')
    @cached_property
    def _get_theta(self):
        return make_ogrid(self.theta_list)

    dG = Property(Array(float), depends_on = 'recalc')
    @cached_property
    def _get_dG(self):
        if len(self.dG_ogrid) == 0:
            # deterministic case
            return 1.0
        else:
            # cross product of dG marginal values
            return reduce(lambda x, y: x * y, self.dG_ogrid)

    def get_samples(self, n):
        '''Get the fully expanded samples.
        '''
        # make the random permutation of the simulations and take n of them
        idx = np.random.permutation(np.arange(self.n_sim))[:n]
        # full orthogonalization (including scalars)
        otheta = make_ogrid_full(self.theta_list)
        # array of ones used for expansion 
        oarray = np.ones(np.broadcast(*otheta).shape, dtype = float)
        # expand (broadcast), flatten and stack the arrays
        return np.vstack([ (t * oarray).flatten()[idx] for t in otheta ])

class TGrid(RegularGrid):
    '''
        Regular grid of random variables theta.
    '''
    theta_11 = Array(float)
    def _theta_11_default(self):
        ''' 'discretize the range (-1,1) symmetrically with n_int points '''
        n_int = self.randomization.n_int
        return np.linspace(-(1.0 - 1.0 / n_int),
                             (1.0 - 1.0 / n_int) , n_int)

    def get_theta_for_distrib(self, tvar):
        if tvar.n_int != None:
            n_int = tvar.n_int
        else:
            n_int = self.randomization.n_int
        min_theta, max_theta, d_theta = self.get_theta_range(tvar)
        return np.linspace(min_theta + 0.5 * d_theta,
                            max_theta - 0.5 * d_theta, n_int)

    dG_ogrid = Property(Array(float), depends_on = 'recalc')
    @cached_property
    def _get_dG_ogrid(self):
        dG_ogrid = [ 1.0 for i in range(len(self.theta)) ]
        for i, (tvar, theta) in \
            enumerate(zip(self.randomization.tvar_lst, self.theta)):
            if not isinstance(tvar, float):
                # get the size of the integration cell
                min_theta, max_theta, d_theta = self.get_theta_range(tvar)
                dG_ogrid[i] = tvar.pdf(theta) * d_theta
        return dG_ogrid

class PGrid(RegularGrid):
    '''
        Regular grid of probabilities
    '''
    pi = Array(float)
    def _pi_default(self):
        n_int = self.randomization.n_int
        return np.linspace(0.5 / n_int,
                            1. - 0.5 / n_int, n_int)

    def get_theta_for_distrib(self, distrib):
        return distrib.ppf(self.pi)

    dG_ogrid = Property(Array(float), depends_on = 'recalc')
    @cached_property
    def _get_dG_ogrid(self):
        return np.repeat(1. / self.randomization.n_int, self.n_rand_vars)

    def _get_dG(self):
        return 1.0 / self.n_sim

class IrregularSampling(RandomSampling):
    '''Irregular sampling based on Monte Carlo concept
    '''
    dG = Property(Array(float))
    @cached_property
    def _get_dG(self):
        return 1. / self.n_sim

    def get_samples(self, n):
        n = min(self.n_sim, n)
        idx = np.random.permutation(np.arange(self.n_sim))[:n]
        s_list = []
        for t in self.theta:
            if isinstance(t, np.ndarray):
                s_list.append(t[idx])
            else:
                s_list.append(np.repeat(t, n))
        return np.vstack(s_list)

class MonteCarlo(IrregularSampling):
    '''
        Standard Monte Carlo randomization:
        For each variable generate n_sim = n_int ** n_rv 
        number of sampling points.
    '''

    theta = Property(Array(float), depends_on = 'recalc')
    @cached_property
    def _get_theta(self):

        theta_list = []
        for tvar in self.randomization.tvar_lst:
            if isinstance(tvar, float):
                theta_list.append(tvar)
            else:
                theta_arr = tvar.rvs(self.n_sim)
                theta_list.append(theta_arr)
        return theta_list

class LatinHypercubeSampling(IrregularSampling):
    '''
        Latin hypercube sampling generated from the 
        samples of the individual random variables 
        with random perturbation.
    '''

    pi = Array(float)
    def _pi_default(self):
        return np.linspace(0.5 / self.n_sim,
                            1. - 0.5 / self.n_sim, self.n_sim)

    theta = Property(Array(float), depends_on = 'recalc')
    @cached_property
    def _get_theta(self):

        theta_list = []
        for tvar in self.randomization.tvar_lst:
            if isinstance(tvar, float):
                theta_list.append(tvar)
            else:
                # point probability function
                theta_arr = tvar.ppf(self.pi)
                theta_list.append(np.random.permutation(theta_arr))
        return theta_list

if __name__ == '__main__':

    from stats.spirrid_bak import SPIRRID, Heaviside

    class fiber_tt_2p:

        la = Float(0.1)

        def __call__(self, e, la, xi = 0.017):
            ''' Response function of a single fiber '''
            return la * e * Heaviside(xi - e)

    s = SPIRRID(q = fiber_tt_2p(),
                sampling_type = 'LHS',
                evars = {'e':[0, 1]},
                tvars = {'la':1.0, # RV( 'norm', 10.0, 1.0 ),
                          'xi': RV('norm', 1.0, 0.1)}
                )

    print(('tvar_names', s.tvar_names))
    print(('tvars', s.tvar_lst))
    print(('evar_names', s.evar_names))
    print(('evars', s.evar_lst))
    print(('var_defaults', s.var_defaults))

    #print 'mu_q', s.mu_q_arr

    s = SPIRRID(q = fiber_tt_2p(),
                 sampling_type = 'LHS',
                 evars = {'e' : [0, 1], 'la' : [0, 1]},
                 tvars = {'xi': RV('norm', 1.0, 0.1),
#                          'la': RV('norm', 10.0, 1.0)
                          })

    print(('tvars', s.tvar_lst))
    print(('evars', s.evar_lst))

    print(('la:', s.tvars['xi']))
    s.tvars['xi'] = 1.0

    print(('mu_q', s.mu_q_arr))

