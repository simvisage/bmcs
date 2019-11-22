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
# Created on May 25, 2010 by: rch, kelidas, rr, mv

# @todo: the order of parameters must be defined - the only possibility is
# to define a list in the response function as it was the case orginally. 
# The current implementation # does not allow for the call to array-
# based RF evaluation.
# SOLVED - the order of the 
#
# @todo: Compiled call to rf_grid for the calculation of the standard deviation.
#

import platform
from functools import reduce
if platform.system() == 'Linux':
    from time import time as sysclock
elif platform.system() == 'Windows':
    from time import clock as sysclock

from etsproxy.traits.api import \
    HasTraits, Instance, List, Property, Array, Int, Any, cached_property, Dict, \
    Event, on_trait_change, Bool, Float, WeakRef, Str, Enum, CTrait

from etsproxy.traits.ui.api import \
    View, Item

from numpy import \
    ogrid, trapz, linspace, array, sum, arange, zeros_like, \
    ones_like, max, argmax, sqrt, ones, copy as ncopy, repeat, where

import copy

from i_rf import \
    IRF

from rf_filament import \
    Filament

from scipy.weave import \
    inline, converters

from stats.pdistrib.pdistrib import \
    PDistrib, IPDistrib

from types import \
    ListType

import os

import time

from mathkit.mfn.mfn_line.mfn_line import \
    MFnLineArray

def orthogonalize( arr_list ):
    '''Orthogonalize a list of one-dimensional arrays.
    '''
    n_arr = len( arr_list )
    ogrid = []
    for i, arr in enumerate( arr_list ):
        shape = ones( ( n_arr, ), dtype = 'int' )
        shape[i] = len( arr )
        arr_i = ncopy( arr ).reshape( tuple( shape ) )
        ogrid.append( arr_i )
    return ogrid

class RandomVariable( HasTraits ):
    '''Class representing the definition and discretization of a random variable.
    '''
    trait_value = Float

    source_trait = CTrait

    # name of the parameter
    name = Str

    pd = Instance( IPDistrib )
    def _pd_changed( self ):
        self.pd.n_segments = self._n_int

    changed = Event
    @on_trait_change( 'pd.changed,+changed' )
    def _set_changed( self ):
        self.changed = True

    # Switch the parameter random 
    random = Bool( False, changed = True )

    def set_random( self, distribution = 'uniform', discr_type = 'T grid',
                    loc = 0., scale = 1., shape = 1., n_int = 30 ):

        possible_distr = self.source_trait.distr
        if distribution and distribution not in possible_distr:
            raise AssertionError('distribution type %s not allowed for parameter %s' \
                % ( distribution, self.name ))

        self.pd = PDistrib( distr_choice = distribution, n_segments = n_int )
        self.pd.distr_type.set( scale = scale, shape = shape, loc = loc )

        self.n_int = n_int
        self.discr_type = discr_type
        self.random = True

    def unset_random( self ):
        self.random = False

    # Number of integration points (relevant only for grid based methods)
    _n_int = Int
    n_int = Property
    def _set_n_int( self, value ):
        if self.pd:
            self.pd.n_segments = value
        self._n_int = value
    def _get_n_int( self ):
        return self.pd.n_segments

    # type of the RandomVariable discretization 
    discr_type = Enum( 'T grid', 'P grid', 'MC',
                        changed = True )
    def _discr_type_default( self ):
        return 'T grid'

    theta_arr = Property( Array( 'float_' ), depends_on = 'changed' )
    @cached_property
    def _get_theta_arr( self ):
        '''Get the discr_type of the pdistrib
        '''
        if not self.random:
            return array( [ self.trait_value ], dtype = float )

        if self.discr_type == 'T grid':

            # Get the discr_type from pdistrib and shift it
            # such that the midpoints of the segments are used for the
            # integration.

            x_array = self.pd.x_array
            # Note assumption of T grid discr_type
            theta_array = x_array[:-1] + self.pd.dx / 2.0

        elif self.discr_type == 'P grid':

            # P grid disretization generated from the inverse cummulative
            # probability
            #
            distr = self.pd.distr_type.distr
            # Grid of constant probabilities
            pi_arr = linspace( 0.5 / self.n_int, 1. - 0.5 / self.n_int, self.n_int )
            theta_array = distr.ppf( pi_arr )

        return theta_array

    dG_arr = Property( Array( 'float_' ), depends_on = 'changed' )
    @cached_property
    def _get_dG_arr( self ):

        if not self.random:
            return array( [ 1.0 ], dtype = float )

        if self.discr_type == 'T grid':

            d_theta = self.theta_arr[1] - self.theta_arr[0]
            return self.pd.get_pdf_array( self.theta_arr ) * d_theta

        elif self.discr_type == 'P grid':

            # P grid disretization generated from the inverse cummulative
            # probability
            #
            return array( [ 1.0 / float( self.n_int ) ], dtype = 'float_' )

    def get_rvs_theta_arr( self, n_samples ):
        if self.random:
            return self.pd.get_rvs_array( n_samples )
        else:
            return array( [self.trait_value], float )

class Randomization( HasTraits ):
    '''Multidimensional statistical integration.
    
    Its name SPIRRID is an acronym for 
    Set of Parallel Independent Random Responses with Identical Distributions
    
    The package implements the evaluation of an integral over a set of 
    random variables affecting a response function RF and distributed 
    according to a probabilistic distribution PDistrib.
    
    The input parameters are devided in four categories in order
    to define state consistency of the evaluation. The outputs 
    are define as cached properties that are reevaluated in response
    to changes in the inputs.
    
    The following events accummulate changes in the input parameters of spirrid:
    rf_change - change in the response function
    rand_change - change in the randomization
    conf_change - change in the configuration of the algorithm
    eps_change - change in the studied range of the process control variable       
    '''
    #--------------------------------------------------------------------
    # Response function 
    #--------------------------------------------------------------------
    #
    rf = Instance( IRF )
    def _rf_changed( self ):
        self.on_trait_change( self._set_rf_change, 'rf.changed' )

    #--------------------------------------------------------------------
    # Specification of random parameters 
    #--------------------------------------------------------------------
    # 
    rv_list = Property( List( RandomVariable ), depends_on = 'rf_change' )
    @cached_property
    def _get_rv_list( self ):
        rf = self.rf
        param_tuple = list(zip( rf.param_keys, rf.param_values, rf.param_traits ))
        return [ RandomVariable( spirrid = self, rf = self.rf,
                     name = nm, trait_value = tv, source_trait = st )
                 for nm, tv, st in param_tuple ]

    # key-based access to the random variables
    rv_dict = Property( Dict, depends_on = 'rf_change' )
    @cached_property
    def _get_rv_dict( self ):
        d = {}
        for rv in self.rv_list:
            d[rv.name] = rv
        return d

    def set_random( self, variable, distribution = 'uniform', discr_type = 'T grid',
                    loc = 0., scale = 1., shape = 1., n_int = 30 ):
        '''Declare a variable as random 
        '''
        self.rv_dict[ variable ].set_random( distribution, discr_type,
                                             loc, scale, shape, n_int )

    def unset_random( self, variable ):
        '''Delete declaration of random variable
        '''
        self.rv_dict[ variable ].unset_random()

    def unset_all_random( self ):
        '''Set all variables to detereministic'''
        list([rv.unset_random for rv in self.rv_list])

    n_rv = Property( depends_on = 'rnd_change' )
    @cached_property
    def _get_n_rv( self ):
        return self.rv_random_keys.size

    # subsidiary methods for sorted access to the random variables.
    # (note dictionary has not defined order of its items)
    rv_random_keys = Property( List, depends_on = 'rf_change' )
    @cached_property
    def _get_rv_random_keys( self ):
        random_pattern = array( [ rv.random for rv in self.rv_list ], dtype = bool )
        ridx = where( random_pattern == True )[0]
        return self.rv_keys[ ridx ]

    # subsidiary methods for sorted access to the random variables.
    # (note dictionary has not defined order of its items)
    rv_keys = Property( List, depends_on = 'rf_change' )
    @cached_property
    def _get_rv_keys( self ):
        return array( self.rf.param_keys )

    #--------------------------------------------------------------------
    # Define which changes in the response function and in the 
    # statistical parameters are relevant for reevaluation of the response
    #--------------------------------------------------------------------
    rf_change = Event
    @on_trait_change( 'rf.changed' )
    def _set_rf_change( self ):
        self.rf_change = True

    rand_change = Event
    @on_trait_change( 'rv_list, rv_list.changed' )
    def _set_rand_change( self ):
        self.rand_change = True

    conf_change = Event
    @on_trait_change( '+alg_option' )
    def _set_conf_change( self ):
        self.conf_change = True

    eps_change = Event
    @on_trait_change( '+eps_range' )
    def _set_eps_change( self ):
        self.eps_change = True

    # List of discretized statistical domains
    # 
    theta_arr_list = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_theta_arr_list( self ):
        '''Get list of arrays with discretized RandomVariables.
        '''
        return [ rv.theta_arr for rv in self.rv_list ]

    # Discretized statistical domain
    # 
    theta_ogrid = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_theta_ogrid( self ):
        '''Get orthogonal list of arrays with discretized RandomVariables.
        '''
        return orthogonalize( self.theta_arr_list )

    #---------------------------------------------------------------------------------
    # PDF * Theta arrays oriented in enumerated dimensions - broadcasting possible
    #---------------------------------------------------------------------------------
    dG_ogrid = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_dG_ogrid( self ):
        '''Get orthogonal list of arrays with PDF * Theta product of.
        '''
        dG_arr_list = [ rv.dG_arr for rv in self.rv_list ]
        return orthogonalize( dG_arr_list )

    #---------------------------------------------------------------------------------
    # PDF grid - mutually multiplied arrays of PDF
    #---------------------------------------------------------------------------------
    dG_grid = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_dG_grid( self ):
        if len( self.dG_ogrid ):
            return reduce( lambda x, y: x * y, self.dG_ogrid )
        else:
            return 1.0

class SPIRRID( Randomization ):

    #---------------------------------------------------------------------------------------------
    # Range of the control process variable epsilon
    # Define particular control variable points with
    # the cv array or an equidistant range with (min, max, n)
    #---------------------------------------------------------------------------------------------

    cv = Array( eps_range = True )
    min_eps = Float( 0.0, eps_range = True )
    max_eps = Float( 0.0, eps_range = True )
    n_eps = Float( 80, eps_range = True )

    eps_arr = Property( depends_on = 'eps_change' )
    @cached_property
    def _get_eps_arr( self ):

        # @todo: !!! 
        # This is a side-effect in a property - CRIME !! [rch]
        # No clear access interface points - naming inconsistent.
        # define a clean property with a getter and setter
        # 
        # if the array of control variable points is not given
        if len( self.cv ) == 0:

            n_eps = self.n_eps
            min_eps = self.min_eps
            max_eps = self.max_eps

            return linspace( min_eps, max_eps, n_eps )
        else:
            return self.cv

    #------------------------------------------------------------------------------------
    # Configuration of the algorithm
    #------------------------------------------------------------------------------------
    # 
    # cached_dG_grid:
    # If set to True, the cross product between the pdf values of all random variables
    # will be precalculated and stored in an n-dimensional grid
    # otherwise the product is performed for every epsilon in the inner loop anew
    # 
    cached_dG = Bool( True, alg_option = True )

    # compiled_eps_loop:
    # If set True, the loop over the control variable epsilon is compiled
    # otherwise, python loop is used.
    compiled_eps_loop = Bool( False, alg_option = True )

    # compiled_QdG_loop:
    # If set True, the integration loop over the product between the response function
    # and the pdf . theta product is performed in c
    # otherwise the numpy arrays are used.
    compiled_QdG_loop = Bool( False, alg_option = True )
    def _compiled_QdG_loop_changed( self ):
        '''If the inner loop is not compiled, the outer loop 
        must not be compiled as well.
        '''
        if self.compiled_QdG_loop == False:
            self.compiled_eps = False

    arg_list = Property( depends_on = 'rf_change, rand_change, conf_change' )
    @cached_property
    def _get_arg_list( self ):

        arg_list = []
        # create argument string for inline function
        if self.compiled_eps_loop:
            arg_list += [ 'mu_q_arr', 'e_arr' ]
        else:
            arg_list.append( 'e' )

        arg_list += ['%s_flat' % name for name in self.rv_keys ]

        if self.cached_dG:
            arg_list += [ 'dG_grid' ]
        else:
            arg_list += [ '%s_pdf' % name for name in self.rv_keys ]

        return arg_list

    dG_C_code = Property( depends_on = 'rf_change, rand_change, conf_change' )
    @cached_property
    def _get_dG_C_code( self ):
        if self.cached_dG: # q_g - blitz matrix used to store the grid
            code_str = '\tdouble pdf = dG_grid(' + \
                       ','.join( [ 'i_%s' % name
                                  for name in self.rv_random_keys ] ) + \
                       ');\n'
        else: # qg
            code_str = '\tdouble pdf = ' + \
                       '*'.join( [ ' *( %s_pdf + i_%s)' % ( name, name )
                                  for name in self.rv_random_keys ] ) + \
                       ';\n'
        return code_str

    #------------------------------------------------------------------------------------
    # Configurable generation of C-code for mean curve evaluation
    #------------------------------------------------------------------------------------
    C_code = Property( depends_on = 'rf_change, rand_change, conf_change, eps_change' )
    @cached_property
    def _get_C_code( self ):

        code_str = ''
        if self.compiled_eps_loop:

            # create code string for inline function
            #
            code_str += 'for( int i_eps = 0; i_eps < %g; i_eps++){\n' % self.n_eps

            if self.cached_dG:

                # multidimensional index needed for dG_grid 
                # blitz arrays must be used also for other arrays
                #
                code_str += 'double eps = e_arr( i_eps );\n'

            else:
                # pointer access possible for single dimensional arrays 
                # use the pointer arithmetics for accessing the pdfs
                code_str += '\tdouble eps = *( e_arr + i_eps );\n'

        else:

            # create code string for inline function
            #
            code_str += 'double eps = e;\n'

        code_str += 'double mu_q(0);\n'
        code_str += 'double q(0);\n'

        code_str += '#line 100\n'
        # create code for constant params
        for name, rv in list(self.rv_dict.items()):
            if not rv.random:
                code_str += 'double %s = %g;\n' % ( name, rv.trait_value )

        # generate loops over random params

        for rv in self.rv_list:

            # skip the terministic variable
            if not rv.random:
                continue

            name = rv.name
            n_int = rv.n_int

            # create the loop over the random variable
            #
            code_str += 'for( int i_%s = 0; i_%s < %g; i_%s++){\n' % ( name, name, n_int, name )
            if self.cached_dG:

                # multidimensional index needed for pdf_grid - use blitz arrays
                #
                code_str += '\tdouble %s = %s_flat( i_%s );\n' % ( name, name, name )
            else:

                # pointer access possible for single dimensional arrays 
                # use the pointer arithmetics for accessing the pdfs
                code_str += '\tdouble %s = *( %s_flat + i_%s );\n' % ( name, name, name )

        if self.n_rv > 0:
            code_str += self.dG_C_code
            code_str += self.rf.C_code + \
                       '// Store the values in the grid\n' + \
                       '\tmu_q +=  q * pdf;\n'
        else:
            code_str += self.rf.C_code + \
                       '\tmu_q += q;\n'

        # close the random loops
        #
        for name in self.rv_random_keys:
            code_str += '};\n'

        if self.compiled_eps_loop:
            if self.cached_dG: # blitz matrix
                code_str += 'mu_q_arr(i_eps) = mu_q;\n'
            else:
                code_str += '*(mu_q_arr + i_eps) = mu_q;\n'
            code_str += '};\n'
        else:
            code_str += 'return_val = mu_q;'
        return code_str

    compiler_verbose = Int( 0 )
    compiler = Str( 'gcc' )

    # Option of eval that induces the calculation of variation
    # in parallel with the mean value so that interim values of Q_grid 
    # are directly used for both.
    #
    implicit_var_eval = Bool( False, alg_option = True )

    def _eval( self ):
        '''Evaluate the integral based on the configuration of algorithm.
        '''
        if self.cached_dG == False and self.compiled_QdG_loop == False:
            raise NotImplementedError('Configuration for pure Python integration is too slow and is not implemented')

        self._set_compiler()
        # prepare the array of the control variable discretization
        #
        eps_arr = self.eps_arr
        mu_q_arr = zeros_like( eps_arr )

        # prepare the variable for the variance
        var_q_arr = None
        if self.implicit_var_eval:
            var_q_arr = zeros_like( eps_arr )

        # prepare the parameters for the compiled function in 
        # a separate dictionary
        c_params = {}

        if self.compiled_eps_loop:

            # for compiled eps_loop the whole input and output array must be passed to c
            #
            c_params['e_arr'] = eps_arr
            c_params['mu_q_arr'] = mu_q_arr
            #c_params['n_eps' ] = n_eps

        if self.compiled_QdG_loop:

            # prepare the lengths of the arrays to set the iteration bounds
            #
            for rv in self.rv_list:
                c_params[ '%s_flat' % rv.name ] = rv.theta_arr

        if len( self.rv_list ) > 0:
            if self.cached_dG:
                c_params[ 'dG_grid' ] = self.dG_grid
            else:
                for rv in self.rv_list:
                    c_params['%s_pdf' % rv.name] = rv.dG_arr
        else:
                c_params[ 'dG_grid' ] = self.dG_grid

        if self.cached_dG:
            conv = converters.blitz
        else:
            conv = converters.default

        t = sysclock()

        if self.compiled_eps_loop:

            # C loop over eps, all inner loops must be compiled as well
            #
            if self.implicit_var_eval:
                raise NotImplementedError('calculation of variance not available in the compiled version')

            inline( self.C_code, self.arg_list, local_dict = c_params,
                    type_converters = conv, compiler = self.compiler,
                    verbose = self.compiler_verbose )

        else:

            # Python loop over eps
            #
            for idx, e in enumerate( eps_arr ):

                if self.compiled_QdG_loop:

                    if self.implicit_var_eval:
                        raise NotImplementedError('calculation of variance not available in the compiled version')

                    # C loop over random dimensions
                    #
                    c_params['e'] = e # prepare the parameter
                    mu_q = inline( self.C_code, self.arg_list, local_dict = c_params,
                                   type_converters = conv, compiler = self.compiler,
                                   verbose = self.compiler_verbose )
                else:

                    # Numpy loops over random dimensions
                    #
                    # get the rf grid for all combinations of
                    # parameter values
                    #          
                    Q_grid = self.rf( e, *self.theta_ogrid )

                    # multiply the response grid with the contributions
                    # of pdf distributions (weighted by the delta of the
                    # random variable disretization)
                    #

                    if not self.implicit_var_eval:
                        # only mean value needed, the multiplication can be done
                        # in-place
                        Q_grid *= self.dG_grid
                        # sum all the values to get the integral 
                        mu_q = sum( Q_grid )
                    else:
                        # get the square values of the grid
                        Q_grid2 = Q_grid ** 2
                        # make an inplace product of Q_grid with the weights
                        Q_grid *= self.dG_grid
                        # make an inplace product of the squared Q_grid with the weights
                        Q_grid2 *= self.dG_grid
                        # sum all values to get the mean
                        mu_q = sum( Q_grid )
                        # sum all squared values to get the variance
                        var_q = sum( Q_grid2 ) - mu_q ** 2

                # add the value to the return array
                mu_q_arr[idx] = mu_q

                if self.implicit_var_eval:
                    var_q_arr[idx] = var_q

        duration = sysclock() - t
        return  mu_q_arr, var_q_arr, duration

    def eval_i_dG_grid( self ):
        '''Get the integral of the pdf * theta grid.
        '''
        return sum( self.dG_grid )

    #---------------------------------------------------------------------------------------------
    # Output properties
    #---------------------------------------------------------------------------------------------

    # container for the data obtained in the integration
    #
    # This is not only the mean curve but also variance and 
    # execution statistics. Such an implementation 
    # concentrates the critical part of the algorithmic 
    # evaluation and avoids duplication of code and 
    # repeated calls. The results are cached in the tuple.
    # They are accessed by the convenience properties defined
    # below. 
    #  
    results = Property( depends_on = 'rand_change, conf_change, eps_change' )
    @cached_property
    def _get_results( self ):
        return self._eval()

    #---------------------------------------------------------------------------------------------
    # Output accessors
    #---------------------------------------------------------------------------------------------
    # the properties that access the cached results and give them a name

    mu_q_arr = Property()
    def _get_mu_q_arr( self ):
        '''mean of q at eps'''
        return self.results[0]

    var_q_arr = Property()
    def _get_var_q_arr( self ):
        '''variance of q at eps'''
        # switch on the implicit evaluation of variance 
        # if it has not been the case so far
        if not self.implicit_var_eval:
            self.implicit_var_eval = True

        return self.results[1]

    exec_time = Property()
    def _get_exec_time( self ):
        '''Execution time of the last evaluation.
        '''
        return self.results[2]

    mean_curve = Property()
    def _get_mean_curve( self ):
        '''Mean response curve.
        '''
        return MFnLineArray( xdata = self.eps_arr, ydata = self.mu_q_arr )

    var_curve = Property()
    def _get_var_curve( self ):
        '''variance of q at eps'''
        return MFnLineArray( xdata = self.eps_arr, ydata = self.var_q_arr )

    #---------------------------------------------------------------------------------------------
    # Auxiliary methods
    #---------------------------------------------------------------------------------------------
    def _set_compiler( self ):
        '''Catch eventual mismatch between scipy.weave and compiler 
        '''
        try:
            uname = os.uname()[3]
        except:
            # it is not Linux - just let it go and suffer
            return

        #if self.compiler == 'gcc':
            #os.environ['CC'] = 'gcc-4.1'
            #os.environ['CXX'] = 'g++-4.1'
            #os.environ['OPT'] = '-DNDEBUG -g -fwrapv -O3'

    traits_view = View( Item( 'rf@', show_label = False ),
                        width = 0.3, height = 0.3,
                        resizable = True,
                        scrollable = True,
                        )

if __name__ == '__main__':

    from matplotlib import pyplot as plt

    rf = Filament()

    s = SPIRRID( rf = rf, max_eps = 0.05, n_eps = 80,
                 compiled_QdG_loop = False,
                 cached_dG = True,
                 compiler_verbose = 0,
                 implicit_var_eval = False
                 )

    s.set_random( 'xi', distribution = 'weibull_min', discr_type = 'T grid',
              scale = 0.02, shape = 10., n_int = 5 )
    s.set_random( 'theta', distribution = 'uniform', discr_type = 'T grid',
              loc = 0.0, scale = 0.01, n_int = 5 )

    print([x.random for x in s.rv_list])
    for i in s.theta_ogrid:
        print((s.theta_ogrid))

    s.mean_curve.plot( plt, color = 'black' , linewidth = 2,
                       label = 'T grid grid' )
    s.var_curve.plot( plt, color = 'green', label = 'T grid variance' )
    # change the discr type to P grid
    #
    s.rv_dict['xi'].discr_type = 'P grid'
    s.rv_dict['theta'].discr_type = 'P grid'

    s.mean_curve.plot( plt, color = 'red' , linewidth = 2, label = 'P grid ' )
    s.var_curve.plot( plt, color = 'red', label = 'P grid variance' )

    s.rv_dict['xi'].discr_type = 'P grid'
    s.rv_dict['theta'].discr_type = 'T grid'

    s.mean_curve.plot( plt, color = 'blue' , linewidth = 2, label = 'T/P grid' )
    s.var_curve.plot( plt, color = 'blue', label = 'T/P grid variance' )

    plt.legend( loc = 'best' )
    plt.show()
