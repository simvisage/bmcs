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
# Created on May 25, 2010 by: rch

# @todo: the order of parameters must be defined - the only possibility is
# to define a list in the response function as it was the case orginally. 
# The current implementation # does not allow for the call to array-
# based RF evaluation.
#
# @todo: Compiled call to rf_grid for the calculation of the standard deviation.
#

from etsproxy.traits.api import \
    HasTraits, Instance, List, Property, Array, Int, Any, cached_property, Dict, \
    Event, on_trait_change, Bool, Float, WeakRef, Str

from etsproxy.traits.ui.api import \
    View, Item

from numpy import \
    ogrid, linspace, array, sum, arange, zeros, \
    max, argmax, sqrt, ones, copy as ncopy, repeat, mgrid, c_

import copy

from i_rf import \
    IRF

from rf_filament import \
    Filament

from scipy.weave import \
    inline, converters

from stats.pdistrib.pdistrib import \
    PDistrib, IPDistrib

from string import \
    split

from types import \
    ListType

import os

import time

from mathkit.mfn.mfn_line.mfn_line import \
    MFnLineArray
from functools import reduce

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

class RV( HasTraits ):
    '''Class representing the definition and discretization of a random variable.
    '''
    name = Str

    pd = Instance( IPDistrib )
    def _pd_changed( self ):
        self.pd.n_segments = self._n_int

    changed = Event
    @on_trait_change( 'pd.changed' )
    def _set_changed( self ):
        self.changed = True

    _n_int = Int
    n_int = Property
    def _set_n_int( self, value ):
        if self.pd:
            self.pd.n_segments = value
        self._n_int = value
    def _get_n_int( self ):
        return self.pd.n_segments

    # index within the randomization
    idx = Int( 0 )

    theta_arr = Property( Array( 'float_' ), depends_on = 'changed' )
    @cached_property
    def _get_theta_arr( self ):
        '''Get the discretization from pdistrib and shift it
        such that the midpoints of the segments are used for the
        integration.
        '''
        x_array = self.pd.x_array
        # Note assumption of equidistant discretization
        theta_array = x_array[:-1] + self.pd.dx / 2.0
        return theta_array

    pdf_arr = Property( Array( 'float_' ), depends_on = 'changed' )
    @cached_property
    def _get_pdf_arr( self ):
        return self.pd.get_pdf_array( self.theta_arr )

    dG_arr = Property( Array( 'float_' ), depends_on = 'changed' )
    @cached_property
    def _get_dG_arr( self ):
        d_theta = self.theta_arr[1] - self.theta_arr[0]
        return self.pdf_arr * d_theta

    def get_rvs_theta_arr( self, n_samples ):
        return self.pd.get_rvs_array( n_samples )

class SPIRRID( HasTraits ):
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
        self.rv_dict = {}
    #--------------------------------------------------------------------
    # Specification of random parameters 
    #--------------------------------------------------------------------
    # 
    rv_dict = Dict
    def add_rv( self, variable, distribution = 'uniform',
                loc = 0., scale = 1., shape = 1., n_int = 30 ):
        '''Declare a variable as random 
        '''
        if variable not in self.rf.param_keys:
            raise AssertionError('parameter %s not defined by the response function' \
                % variable)

        params_with_distr = self.rf.traits( distr = lambda x: type( x ) == ListType
                                            and distribution in x )
        if variable not in params_with_distr:
            raise AssertionError('distribution type %s not allowed for parameter %s' \
                % ( distribution, variable ))

        # @todo - let the RV take care of PDistrib specification.
        # isolate the dirty two-step definition of the distrib from spirrid 
        #
        pd = PDistrib( distr_choice = distribution, n_segments = n_int )
        pd.distr_type.set( scale = scale, shape = shape, loc = loc )
        self.rv_dict[variable] = RV( name = variable, pd = pd, n_int = n_int )

    def del_rv( self, variable ):
        '''Delete declaration of random variable
        '''
        del self.rv_dict[ variable ]

    def clear_rv( self ):
        self.rv_dict = {}

    # subsidiary methods for sorted access to the random variables.
    # (note dictionary has not defined order of its items)
    rv_keys = Property( List, depends_on = 'rv_dict' )
    @cached_property
    def _get_rv_keys( self ):
        rv_keys = sorted( self.rv_dict.keys() )
        # the random variable gets an index based on the 
        # sorted keys
        for idx, key in enumerate( rv_keys ):
            self.rv_dict[ key ].idx = idx
        return rv_keys

    rv_list = Property( List, depends_on = 'rv_dict' )
    @cached_property
    def _get_rv_list( self ):
        return list(map( self.rv_dict.get, self.rv_keys ))

    #--------------------------------------------------------------------
    # Define which changes in the response function and in the 
    # statistical parameters are relevant for reevaluation of the response
    #--------------------------------------------------------------------
    rf_change = Event
    @on_trait_change( 'rf.changed' )
    def _set_rf_change( self ):
        self.rf_change = True

    rand_change = Event
    @on_trait_change( 'rv_dict, rv_dict.changed' )
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

    # Dictionary with key = rf parameters
    # and values = default param values for the resp func 
    #
    param_dict = Property( Dict, depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_param_dict( self ):
        '''Gather all the traits with the metadata distr specified.
        '''
        dict = {}
        for name, value in zip( self.rf.param_keys, self.rf.param_values ):
            rv = self.rv_dict.get( name, None )
            if rv == None:
                dict[ name ] = value
            else:
                dict[ name ] = self.theta_ogrid[ rv.idx ]
        return dict

    ##### - experimental #####
    # @deprecated: ful coverage of the sampling domain - for orientation
    full_theta_arr_list = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_full_theta_arr_list( self ):
        '''Get list of arrays with both deterministic and statistic arrays.
        '''
        param_arr_list = [ array( [value], dtype = 'float_' ) for value in self.rf.param_values ]
        for idx, name in enumerate( self.rf.param_keys ):
            rv = self.rv_dict.get( name, None )
            if rv:
                param_arr_list[ idx ] = rv.theta_arr
        return param_arr_list

    def get_rvs_theta_arr( self, n_samples ):
        rvs_theta_arr = array( [ repeat( value, n_samples ) for value in self.rf.param_values ] )
        for idx, name in enumerate( self.rf.param_keys ):
            rv = self.rv_dict.get( name, None )
            if rv:
                rvs_theta_arr[ idx, :] = rv.get_rvs_theta_arr( n_samples )
        return rvs_theta_arr

    # Constant parameters
    #
    const_param_dict = Property( Dict, depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_const_param_dict( self ):
        const_param_dict = {}
        for name, v in zip( self.rf.param_keys, self.rf.param_values ):
            if name not in self.rv_keys:
                const_param_dict[ name ] = v
        return const_param_dict

    # List of discretized statistical domains
    # 
    theta_arr_list = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_theta_arr_list( self ):
        '''Get list of arrays with discretized RVs.
        '''
        return [ rv.theta_arr for rv in self.rv_list ]

    # Discretized statistical domain
    # 
    theta_ogrid = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_theta_ogrid( self ):
        '''Get orthogonal list of arrays with discretized RVs.
        '''
        return orthogonalize( self.theta_arr_list )

    #---------------------------------------------------------------------------------
    # PDF arrays oriented in enumerated dimensions - broadcasting possible
    #---------------------------------------------------------------------------------
    pdf_ogrid = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_pdf_ogrid( self ):
        '''Get orthogonal list of arrays with PDF values of RVs.
        '''
        pdf_arr_list = [ rv.pdf_arr for rv in self.rv_list ]
        return orthogonalize( pdf_arr_list )

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
    compiled_eps_loop = Bool( True, alg_option = True )

    # compiled_QdG_loop:
    # If set True, the integration loop over the product between the response function
    # and the pdf . theta product is performed in c
    # otherwise the numpy arrays are used.
    compiled_QdG_loop = Bool( True, alg_option = True )
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

    C_code_qg = Property( depends_on = 'rf_change, rand_change, conf_change' )
    @cached_property
    def _get_C_code_qg( self ):
        if self.cached_dG: # q_g - blitz matrix used to store the grid
            code_str = '\tdouble pdf = dG_grid(' + \
                       ','.join( [ 'i_%s' % name
                                  for name in self.rv_keys ] ) + \
                       ');\n'
        else: # qg
            code_str = '\tdouble pdf = ' + \
                       '*'.join( [ ' *( %s_pdf + i_%s)' % ( name, name )
                                  for name in self.rv_keys ] ) + \
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
                # use blitz arrays must be used also for other arrays
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
        for name, value in list(self.const_param_dict.items()):
            code_str += 'double %s = %g;\n' % ( name, value )

        # generate loops over random params

        for rv in self.rv_list:

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

        if len( self.rv_keys ) > 0:
            code_str += self.C_code_qg
            code_str += self.rf.C_code + \
                       '// Store the values in the grid\n' + \
                       '\tmu_q +=  q * pdf;\n'
        else:
            code_str += self.rf.C_code + \
                       '\tmu_q += q;\n'

        # close the random loops
        #
        for name in self.rv_keys:
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

    eps_grid_shape = Property( depends_on = 'eps_change' )
    @cached_property
    def _get_eps_grid_shape( self ):
        return tuple( [ len( eps ) for eps in self.eps_list ] )

    eps_list = Property( depends_on = 'eps_change' )
    @cached_property
    def _get_eps_list( self ):
        ctrl_list = self.rf.ctrl_traits
        # generate the slices to produce the grid of the control values
        eps_list = [ linspace( *cv.ctrl_range )
                    for cv in ctrl_list ]
        # produce the tuple of expanded arrays with n-dimensions - values broadcasted
        return eps_list

    eps_grid = Property( depends_on = 'eps_change' )
    @cached_property
    def _get_eps_grid( self ):
        '''Generate the grid of control variables.
        The array can be multidimensional depending on the dimension
        of the input variable of the current response function.
        '''
        ctrl_list = self.rf.ctrl_traits
        # generate the slices to produce the grid of the control values
        slices = [ slice( cv.ctrl_range[0], cv.ctrl_range[1], complex( 0, cv.ctrl_range[2] ) )
                  for cv in ctrl_list ]
        # produce the tuple of expanded arrays with n-dimensions - values broadcasted
        return mgrid[ tuple( slices ) ]

    eps_arr = Property( depends_on = 'eps_change' )
    @cached_property
    def _get_eps_arr( self ):
        '''
        flatten the arrays and order them as columns of an array containing all combinations
        of the control variable values.
        '''
        return c_[ tuple( [ eps_arr.flatten() for eps_arr in self.eps_grid ] ) ]

    compiler_verbose = Int( 0 )
    compiler = Str( 'gcc' )

    def _eval( self ):
        '''Evaluate the integral based on the configuration of algorithm.
        '''

        if self.cached_dG == False and self.compiled_QdG_loop == False:
            raise NotImplementedError('Configuration for pure Python integration is too slow and is not implemented')

        self._set_compiler()
        # prepare the array of the control variable discretization
        #
        eps_arr = self.eps_arr
        mu_q_arr = zeros( ( eps_arr.shape[0], ), dtype = 'float_' )

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

        t = time.clock()

        if self.compiled_eps_loop:

            # C loop over eps, all inner loops must be compiled as well
            #
            inline( self.C_code, self.arg_list, local_dict = c_params,
                    type_converters = conv, compiler = self.compiler,
                    verbose = self.compiler_verbose )

        else:

            # Python loop over eps
            #
            for idx, e in enumerate( eps_arr ):

                if self.compiled_QdG_loop:

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
                    Q_grid = self.rf( *e, **self.param_dict )

                    # multiply the response grid with the contributions
                    # of pdf distributions (weighted by the delta of the
                    # random variable disretization)
                    #
                    Q_grid *= self.dG_grid

                    # sum all the values to get the integral 
                    mu_q = sum( Q_grid )

                # add the value to the return array
                mu_q_arr[idx] = mu_q

        duration = time.clock() - t

        return  mu_q_arr, duration

    def eval_i_dG_grid( self ):
        '''Get the integral of the pdf * theta grid.
        '''
        return sum( self.dG_grid )

    def _eval_mu_q( self ):
        # configure eval and call it
        pass

    def _eval_stdev_q( self ):
        # configure eval and call it
        pass

    #--------------------------------------------------------------------------------------------
    # Numpy implementation
    #--------------------------------------------------------------------------------------------
    def get_rf( self, eps ):
        '''
        Numpy based evaluation of the response function.
        '''
        return self.rf( eps, **self.param_dict )

    #---------------------------------------------------------------------------------------------
    # Output properties
    #---------------------------------------------------------------------------------------------

    # container for the data obtained in the integration
    #
    # This is not only the mean curve but also stdev and 
    # execution statistics. Such an implementation 
    # concentrates the critical part of the algorithmic 
    # evaluation and avoids duplication of code and 
    # repeated calls. The results are cached in the tuple.
    # They are accessed by the convenience properties defined
    # below. 
    #  
    results = Property( depends_on = 'rf_change, rand_change, conf_change, eps_change' )
    @cached_property
    def _get_results( self ):
        return self._eval()

    #---------------------------------------------------------------------------------------------
    # Output accessors
    #---------------------------------------------------------------------------------------------
    # the properties that access the cached results and give them a name

    mu_q_arr = Property()
    def _get_mu_q_arr( self ):
        return self.results[0]

    mu_q_grid = Property()
    def _get_mu_q_grid( self ):
        return self.mu_q_arr.reshape( self.eps_grid_shape )

    exec_time = Property()
    def _get_exec_time( self ):
        '''Execution time of the last evaluation.
        '''
        return self.results[1]

    mean_curve = Property()
    def _get_mean_curve( self ):
        '''Mean response curve.
        '''
        return MFnLineArray( xdata = self.eps_arr, ydata = self.mu_q_arr )

    mu_q_peak_idx = Property()
    def _get_mu_q_peak_idx( self ):
        '''Get mean peak response value'''
        return argmax( self.mu_q_arr )

    mu_q_peak = Property()
    def _get_mu_q_peak( self ):
        '''Get mean peak response value'''
        return self.mu_q_arr[ self.mu_q_peak_idx ]

    eps_at_peak = Property()
    def _get_eps_at_peak( self ):
        '''Get strain at maximum middle response mu_q
        '''
        return self.eps_arr[ self.mu_q_peak_idx ]

    stdev_mu_q_peak = Property()
    def _get_stdev_mu_q_peak( self ):
        '''
        Numpy based evaluation of the time integral.
        '''
        mu_q_peak = self.mu_q_peak
        eps_at_peak = self.eps_at_peak

        q_quad_grid = self.get_rf( eps_at_peak ) ** 2
        q_quad_grid *= self.dG_grid
        q_quad_peak = sum( q_quad_grid )
        stdev_mu_q_peak = sqrt( q_quad_peak - mu_q_peak ** 2 )

        return stdev_mu_q_peak

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

    s = SPIRRID( rf = rf, max_eps = 0.05, n_eps = 80 )

    from etsproxy.traits.traits_listener \
            import TraitsListener, ListenerParser, ListenerHandler, \
                   ListenerNotifyWrapper

    dict = s.__dict__.get( TraitsListener )
    print('----------')
    for n, l in list(dict.items()):
        print(n, ':', end=' ')
        for li in l:
            print(li.listener, end=' ')
        print()
    print('----------')

    rf.xi = 0.175

    print('xi changed')

    s.add_rv( 'xi', distribution = 'weibull_min', scale = 0.02, shape = 10., n_int = 30 )
    s.add_rv( 'theta', distribution = 'uniform', loc = 0.0, scale = 0.01, n_int = 30 )

    s.compiled_eps_loop = False
    s.cached_dG = False

    s.mean_curve.plot( plt, color = 'b' , linewidth = 3 )

    print(s.eps_at_peak)
    print(s.mu_q_peak)
    print(s.stdev_mu_q_peak)

    plt.errorbar( s.eps_at_peak, s.mu_q_peak, s.stdev_mu_q_peak, color = 'r', linewidth = 2 )

    plt.show()
