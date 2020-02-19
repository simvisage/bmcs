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
# Created on Jun 2, 2010 by: rch

# Saved method usable for data analysis of the measured arrays
#
# - evaluation of the step histogram, plotting, ironing
#

class ExTypeSave( object ):
    
    continuity_profile_dict = Property( Dict )
    def _get_continuity_profile_dict(self):
        ''' Get the histogram and the cummulative distribution of
        the step size in all measured data arrays.
        '''
        hist_dict = {}
        for i, factor in enumerate( self.factor_list ):
            data_arr = self.data_array[:,i]
            delta_arr = fabs( data_arr[1:] - data_arr[:-1] )
            ydata, xdata = histogram( delta_arr, bins = 1000, normed = True )
            cs_hg = cumsum( ydata )
            norm_cs_hg = hstack( [zeros( (1,), dtype = 'float_' ), cs_hg / fabs( cs_hg[-1] )] )
            max_delta = fabs( xdata[-1] )
            norm_xdata = xdata / max_delta
            #print 'hy',norm_cs_hg
            hist_dict[ factor ] = ( norm_xdata, norm_cs_hg, max_delta )
        return hist_dict
       
    def _plot_continuity_profiles( self, axes ):
    
        axes.set_xlabel( 'normalized delta' )
        axes.set_ylabel( 'normalized cummulative frequency' )

        factors = list(self.continuity_profile_dict.keys())

        for factor in factors:
            hist = self.continuity_profile_dict[ factor ]
            axes.plot( hist[0], hist[1]
                       # color = c, linewidth = w, linestyle = s 
                       )
                    
        axes.legend( factors )
        
    # Factor used to cut off the delta in the measured field
    # If there is a too big step in the array it should be 
    # removed since it most probably corresponds to 
    # disturbance in the measuring process 
    #
    cutoff_delta_factor = Float( 100, 
                      auto_set = False, enter_set = True,
                      ironing_param = True )
    
    
    data_array_ironed = Property( Array( float ),
                                   depends_on = 'data_array, +ironing_param, +axis_selection' )
    @cached_property
    def _get_data_array_ironed(self):
        '''remove the jumps in the displacement curves 
        due to resetting the displacement gauges. 
        '''
        print('*** curve ironing 2 activated ***')
        
        # each column from the data array corresponds to a measured parameter 
        # e.g. displacement at a given point as function of time u = f(t))
        #
        data_array_ironed = copy( self.data_array )
        
        for idx, factor in enumerate( self.factor_list ):

            if self.names_and_units[0][ idx ] == 'Kraft' or \
                self.names_and_units[0][ idx ] == 'Bezugskanal' or \
                self.names_and_units[0][ idx ] == 'Weg':
                continue
            
            rel_delta, cum_hg, max_delta = self.continuity_profile_dict[ factor ]

            rel_delta_median = interp( 0.5, cum_hg, rel_delta )

            print('factor', factor)
            print('rel_delta_median', rel_delta_median)
            print('cutoff fraction', ( 1.0 / self.cutoff_delta_factor ))

#            if rel_delta_median <= ( 1.0 / self.cutoff_delta_factor ):
#                
            print('ironing')

            data_arr = copy( data_array_ironed[:,idx] )
            delta_median = rel_delta_median * max_delta
            print('delta_median', delta_median)
            delta_threshold = delta_median * self.cutoff_delta_factor
            print('delta_threshold', delta_threshold)

            # get the difference between each point and its successor
            delta_arr =  fabs( data_arr[1:] - data_arr[:-1] )
            print('delta_arr.shape', delta_arr.shape)
            # get the indexes in 'data_column' after which a 
            # jump exceeds the defined tolerance criteria
            delta_iron_idx = where( delta_arr > delta_threshold )[0]
            print('delta_iron_idx.shape', delta_iron_idx) 
            # glue the curve at each jump together
            for jidx in delta_iron_idx:
                # get the offsets at each jump of the curve
                shift = data_arr[jidx+1] - data_arr[jidx]
                #print 'shifting index', jidx, 'by', shift
                # shift all succeeding values by the calculated offset
                data_arr[jidx+1:] -= shift

            data_array_ironed[:,idx] = data_arr

        return data_array_ironed
