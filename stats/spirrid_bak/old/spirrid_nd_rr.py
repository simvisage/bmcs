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
    max, argmax, sqrt, ones, copy as ncopy, repeat, mgrid, indices, interp

import copy

from i_rf import \
    IRF

from quaducom.resp_func.cb_clamped_fiber import CBClampedFiber

from stats.pdistrib.pdistrib import \
    PDistrib, IPDistrib

from types import \
    ListType

import time

from scipy.optimize import newton

from mathkit.mfn.mfn_line.mfn_line import \
    MFnLineArray

from scipy import ndimage
from functools import reduce

def orthogonalize( arr_list, ctrl_var = 0, rand_var = 0 ):
    '''Orthogonalize a list of one-dimensional arrays.
    '''
    n_arr = len( arr_list ) + ctrl_var + rand_var
    ogrid = []
    for i, arr in enumerate( arr_list ):
        shape = ones( ( n_arr, ), dtype = 'int' )
        shape[i + rand_var] = len( arr )
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
                if not name in self.ctrl_keys:
                    dict[ name ] = value
                else:
                    dict[ name ] = self.ctrl_ogrid[self.ctrl_keys.index( name )]
            else:
                dict[ name ] = self.theta_ogrid[ rv.idx ]
        return dict

    ##### - experimental #####
    # @deprecated: full coverage of the sampling domain - for orientation
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

    ctrl_keys = List
    ctrl_list = List
    def add_ctrl_var( self, variable, min = 0, max = 1, np = 20 ):
        if variable in self.rv_keys:
            raise AssertionError('variable %s is defined as a random variable' \
                % ( variable ))
        else:
            self.ctrl_keys.append( variable )
            self.ctrl_list.append( linspace( min, max, np ) )


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
        return orthogonalize( self.theta_arr_list, ctrl_var = len( self.ctrl_list ) )


    # Discretized control domain
    # 
    ctrl_ogrid = Property( depends_on = 'rf_change' )
    @cached_property
    def _get_ctrl_ogrid( self ):
        '''Get orthogonal list of arrays with discretized ctrl vars.
        '''
        return orthogonalize( self.ctrl_list, rand_var = len( self.rv_list ) )

    #---------------------------------------------------------------------------------
    # PDF arrays oriented in enumerated dimensions - broadcasting possible
    #---------------------------------------------------------------------------------
    pdf_ogrid = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_pdf_ogrid( self ):
        '''Get orthogonal list of arrays with PDF values of RVs.
        '''
        pdf_arr_list = [ rv.pdf_arr for rv in self.rv_list ]
        return orthogonalize( pdf_arr_list, ctrl_var = len( self.ctrl_list ) )

    #---------------------------------------------------------------------------------
    # PDF * Theta arrays oriented in enumerated dimensions - broadcasting possible
    #---------------------------------------------------------------------------------
    dG_ogrid = Property( depends_on = 'rf_change, rand_change' )
    @cached_property
    def _get_dG_ogrid( self ):
        '''Get orthogonal list of arrays with PDF * Theta product of.
        '''
        dG_arr_list = [ rv.dG_arr for rv in self.rv_list ]
        return orthogonalize( dG_arr_list, ctrl_var = len( self.ctrl_list ) )

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

    # main control variable (epsilon, displacement, crack width..)

    ctrl_max = Float( 1.0, eps_range = True )
    ctrl_min = Float( 0.0, eps_range = True )
    np = Float( 1.0, eps_range = True )
    eps_arr = Property( depends_on = 'eps_change' )
    @cached_property
    def _get_eps_arr( self ):

        return linspace( self.ctrl_min, self.ctrl_max, self.np )

    def _eval( self ):
        '''Evaluate the integral based on the configuration of algorithm.
        '''
        t = time.clock()

        Q_grid = self.rf( **self.param_dict )
        # multiply the response grid with the contributions
        # of pdf distributions (weighted by the delta of the
        # random variable disretization)

        Q_grid *= self.dG_grid

        # sum (integrate) only over the axes of random variables
        # arrays of ctrl variables stay as grid
        rand_axes = arange( len( self.rv_keys ) )[::-1]
        for ax in rand_axes:
            Q_grid = sum( Q_grid, axis = ax )
        mu_q_grid = Q_grid
        duration = time.clock() - t

        return  mu_q_grid, duration

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

    mu_q_grid = Property()
    def _get_mu_q_grid( self ):
        return self.results[0]

    exec_time = Property()
    def _get_exec_time( self ):
        '''Execution time of the last evaluation.
        '''
        return self.results[1]

    # ---------------------------------------------------------------------
    #                        RESULT INTERPOLATION
    # ---------------------------------------------------------------------

    ctrl_indices = Property( depends_on = 'ctrl_list' )
    @cached_property
    def _get_ctrl_indices( self ):
        interp_indices_list = []
        for i, value in enumerate( self.ctrl_list ):
            interp_array = MFnLineArray( xdata = value, ydata = arange( len( value ) ) )
            interp_indices_list.append( interp_array )
        return interp_indices_list

    def get_interp_indices( self, dict ):
        indices = []
        values = list(map( dict.get, self.ctrl_keys ))
        for i, interp_obj in enumerate( self.ctrl_indices ):
            idx = interp_obj.get_value( values[i] )
            indices.append( idx )
        return indices

    def coordinates( self, dict ):
        # create the coords array shape
        coord_shape = ones( len( dict ) )
        coord_shape[-1] = len( dict )
        # create the coords values = indices and reshape them
        coords = array( self.get_interp_indices( dict ) ).reshape( coord_shape )
        # the ndimage method takes the transpose array as values
        return coords.T


    def mean_e_P( self, order = 1, **kw ):
        '''Mean response curve.
        '''
        # check if the number of key words given equals the ndims of the result grid
        # minus the dimension of the control variable eps
        if len( self.ctrl_keys ) == len( kw ):
            # create a zeros array to be filled with interpolated forces for every eps 
            P = zeros( len( self.eps_arr ) )
            # the value to be interpolated is defined by coordinates assigning
            # the data grid; e.g. if coords = array([[1.5, 2.5]]) then coords.T
            # point to the half distance between the 1. and 2. entry in the first
            # dimension and the half distance between the 2. and 3. entry of the 2. dim
            coords = self.coordinates( kw )
            # specify the data for the interpolation - one eps slice of the mu_q_grid
            data = self.mu_q_grid
            # interpolate the value (linear)
            P = ndimage.map_coordinates( data, coords, order = order, mode = 'nearest' )
            return P
        else:
            raise TypeError('mean_curve() takes {req} arguments ({given} given)'.format( req = \
                len( self.ctrl_keys ), given = len( kw ) ))

    def force_residuum( self, w, order, P, kw ):
        kw['w'] = w
        p = self.mean_e_P( order = order, **kw ).flatten()
        return P - p

    def p_x( self, x_arr, P = 0.2, order = 1, **kw ):
        '''
        returns p for a range of x
        '''
        kw['x'] = 0.0
        kw['w'] = 0.0

        w = newton( self.force_residuum, 1e-3, args = ( order, P, kw ) )
        kw['w'] = w
        #x_arr = linspace( -kw['Ll'], kw['Lr'], np )
        p_arr = []
        for i, x in enumerate( x_arr ):
            kw['x'] = x
            p_arr.append( self.mean_e_P( order = order, **kw ).flatten() )
        return MFnLineArray( xdata = x_arr, ydata = array( p_arr ).flatten() )


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from quaducom.resp_func.stress_profiles.cb_clamped_fiber_sp import CBClampedFiberSP

    rf = CBClampedFiberSP( w = 0.9, x = 2, tau = 0.12, l = 10.5, D_f = 26e-3, E_f = 72e3, theta = 0.0,
                          xi = 0.0179, phi = 1., Ll = 50, Lr = 10 )
    s = SPIRRID( rf = rf )
    s.add_rv( 'l', distribution = 'uniform', loc = 5.0, scale = 20.0, n_int = 20 )
    s.add_rv( 'tau', distribution = 'uniform', loc = 0.0, scale = 1., n_int = 20 )
    s.add_rv( 'xi', distribution = 'weibull_min', scale = 0.017, shape = 5, n_int = 20 )
    #s.add_ctrl_var( 'Ll', min = 49.1, max = 70.1, np = 50 )
    #s.add_ctrl_var( 'Lr', min = 10.1, max = 70.1, np = 50 )
    s.add_ctrl_var( 'x', min = -50.1, max = 10.5, np = 100 )
    s.add_ctrl_var( 'w', min = 0.0, max = 0.8, np = 100 )

    p = array( [0.1, 0.2, 0.25] )
    x = linspace( -50, 10, 100 )
    profile = s.p_x( x, P = p[2] )
    plt.plot( profile.xdata, profile.ydata, lw = 2,
              color = 'black', ls = '-.', label = 'P = %.2f N' % p[2] )

    profile = s.p_x( x, P = p[1] )
    plt.plot( profile.xdata, profile.ydata, lw = 2,
              color = 'black', ls = '--', label = 'P = %.2f N' % p[1] )
    profile = s.p_x( x, P = p[0] )
    plt.plot( profile.xdata, profile.ydata, lw = 2,
              color = 'black', label = 'P = %.2f N' % p[0] )
    plt.ylim( 0, 0.28 )
    plt.legend( loc = 'best' )
    plt.show()


    e = linspace( 0, 0.7, 300 )
    p = []
    for i, wi in enumerate( e ):
        p_i = s.mean_e_P( order = 1, w = wi, x = 0 )
        p.append( p_i.flatten() )
    plt.plot( e, p, color = 'black', lw = 2, label = 'P_w at x = 0' )
    plt.legend( loc = 'best' )
    plt.show()







