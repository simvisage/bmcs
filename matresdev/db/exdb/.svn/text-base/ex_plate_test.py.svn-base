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
# Created on Feb 15, 2010 by: rch

from enthought.traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements, DelegatesTo

from enthought.util.home_directory import \
    get_home_directory

from enthought.traits.ui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group, \
    HGroup, Spring

## overload the 'get_label' method from 'Item' to display units in the label
from util.traits.ui.item import \
    Item

from enthought.traits.ui.table_column import \
    ObjectColumn

from enthought.traits.ui.menu import \
    OKButton, CancelButton

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter

from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure

import os

import csv

from numpy import array, fabs, where, copy, ones, argsort

from numpy import \
    loadtxt, argmax, polyfit, poly1d, frompyfunc, dot

from enthought.traits.ui.table_filter \
    import EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate, \
           EvalTableFilter

#-- Tabular Adapter Definition -------------------------------------------------

from string import replace
from os.path import exists

#-----------------------------------------------------------------------------------
# ExDesignReader
#-----------------------------------------------------------------------------------
from enthought.traits.ui.file_dialog  \
    import open_file, FileInfo, TextInfo, ImageInfo

from enthought.traits.ui.api \
    import View, Item, TabularEditor, VGroup, HGroup

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter

from ex_type import ExType

data_file_editor = FileEditor( filter = ['*.DAT'] )

from ex_type import ExType
from i_ex_type import IExType

from mathkit.array.smoothing import smooth

from promod.matdb.trc.fabric_layup \
    import FabricLayUp

from promod.matdb.trc.fabric_layout \
    import FabricLayOut

from promod.matdb.trc.concrete_mixture \
    import ConcreteMixture

from promod.matdb.trc.composite_cross_section import \
    CompositeCrossSection, plain_concrete



class ExPlateTest( ExType ):
    '''Read the data from the directory
    '''

    implements( IExType )

    #--------------------------------------------------------------------
    # register a change of the traits with metadata 'input'
    #--------------------------------------------------------------------

    input_change = Event
    @on_trait_change( '+input, ccs.input_change, +ironing_param' )
    def _set_input_change( self ):
        self.input_change = True

    #--------------------------------------------------------------------------------
    # specify inputs:
    #--------------------------------------------------------------------------------

    edge_length = Float( 1.25, unit = 'm', input = True, table_field = True,
                           auto_set = False, enter_set = True )
    thickness = Float( 0.03, unit = 'm', input = True, table_field = True,
                           auto_set = False, enter_set = True )

    # age of the concrete at the time of testing
    age = Int( 28, unit = 'd', input = True, table_field = True,
                             auto_set = False, enter_set = True )
    loading_rate = Float( 0.50, unit = 'mm/min', input = True, table_field = True,
                            auto_set = False, enter_set = True )

    #--------------------------------------------------------------------------
    # composite cross section
    #--------------------------------------------------------------------------

    ccs = Instance( CompositeCrossSection )
    def _ccs_default( self ):
        '''default settings correspond to 
        setup '7u_MAG-07-03_PZ-0708-1'
        '''
        fabric_layout_key = 'MAG-07-03'
#        fabric_layout_key = '2D-02-06a'
        concrete_mixture_key = 'PZ-0708-1'
#        concrete_mixture_key = 'FIL-10-09'
#        orientation_fn_key = 'all0'                                           
        orientation_fn_key = '90_0'
        n_layers = 10
        s_tex_z = 0.030 / ( n_layers + 1 )
        ccs = CompositeCrossSection ( 
                    fabric_layup_list = [
                            plain_concrete( s_tex_z * 0.5 ),
                            FabricLayUp ( 
                                   n_layers = n_layers,
                                   orientation_fn_key = orientation_fn_key,
                                   s_tex_z = s_tex_z,
                                   fabric_layout_key = fabric_layout_key
                                   ),
                            plain_concrete( s_tex_z * 0.5 )
                                        ],
                    concrete_mixture_key = concrete_mixture_key
                    )
        return ccs

    #--------------------------------------------------------------------------
    # Get properties of the composite 
    #--------------------------------------------------------------------------

    # E-modulus of the composite at the time of testing 
    E_c = Property( Float, unit = 'MPa', depends_on = 'input_change', table_field = True )
    def _get_E_c( self ):
        return self.ccs.get_E_c_time( self.age )

    # E-modulus of the composite after 28 days
    E_c28 = DelegatesTo( 'ccs', listenable = False )

    # reinforcement ration of the composite 
    rho_c = DelegatesTo( 'ccs', listenable = False )

    #--------------------------------------------------------------------------------
    # define processing
    #--------------------------------------------------------------------------------

    # put this into the ironing procedure processor
    #
    jump_rtol = Float( 0.03,
                      auto_set = False, enter_set = True,
                      ironing_param = True )


    data_array_ironed = Property( Array( float ),
                                  depends_on = 'data_array, +ironing_param, +axis_selection' )
    @cached_property
    def _get_data_array_ironed( self ):
        '''remove the jumps in the displacement curves 
        due to resetting the displacement gauges. 
        '''
        print '*** curve ironing activated ***'

        # each column from the data array corresponds to a measured parameter 
        # e.g. displacement at a given point as function of time u = f(t))
        #
        data_array_ironed = copy( self.data_array )

        for idx in range( self.data_array.shape[1] ):

            # use ironing method only for columns of the displacement gauges.
            #
            if self.names_and_units[0][ idx ] != 'Kraft' and \
                self.names_and_units[0][ idx ] != 'Bezugskanal' and \
                self.names_and_units[0][ idx ] != 'Weg':

                # 1d-array corresponding to column in data_array
                data_arr = copy( data_array_ironed[:, idx] )

                # get the difference between each point and its successor
                jump_arr = data_arr[1:] - data_arr[0:-1]

                # get the range of the measured data 
                data_arr_range = max( data_arr ) - min( data_arr )

                # determine the relevant criteria for a jump
                # based on the data range and the specified tolerances:
                jump_crit = self.jump_rtol * data_arr_range

                # get the indexes in 'data_column' after which a 
                # jump exceeds the defined tolerance criteria
                jump_idx = where( fabs( jump_arr ) > jump_crit )[0]

                print 'number of jumps removed in data_arr_ironed for', self.names_and_units[0][ idx ], ': ', jump_idx.shape[0]
                # glue the curve at each jump together
                for jidx in jump_idx:
                    # get the offsets at each jump of the curve
                    shift = data_arr[jidx + 1] - data_arr[jidx]
                    # shift all succeeding values by the calculated offset
                    data_arr[jidx + 1:] -= shift

                data_array_ironed[:, idx] = data_arr[:]

        return data_array_ironed


    @on_trait_change( '+ironing_param' )
    def process_source_data( self ):
        '''read in the measured data from file and assign
        attributes after array processing.        
        
        NOTE: if center displacement gauge ('WA_M') is missing the measured 
        displacement of the cylinder ('Weg') is used instead.
        A minor mistake is made depending on how much time passes
        before the cylinder has contact with the plate.
        '''
        print '*** process source data ***'

        self._read_data_array()
        # curve ironing:
        #
        self.processed_data_array = self.data_array_ironed

        # set attributes:
        #
        self._set_array_attribs()

        if 'WA_M' not in self.factor_list:
            print '*** NOTE: Displacement gauge at center ("WA_M") missing. Cylinder displacement ("Weg") is used instead! ***'
            self.WA_M = self.Weg


    #--------------------------------------------------------------------------------
    # plot templates
    #--------------------------------------------------------------------------------

    plot_templates = {'force / deflection (center)'          : '_plot_force_deflection_center',
                      'smoothed force / deflection (center)' : '_plot_smoothed_force_deflection_center',
                      'force / deflection (edges)' : '_plot_force_edge_deflection',
                      'continuity profiles'                : '_plot_continuity_profiles'
                     }
    default_plot_template = 'force / deflection (center)'


    def _plot_force_deflection_center( self, axes ):

        xkey = 'deflection [mm]'
        ykey = 'force [kN]'
        xdata = -self.WA_M
        ydata = -self.Kraft

        axes.set_xlabel( '%s' % ( xkey, ) )
        axes.set_ylabel( '%s' % ( ykey, ) )
        axes.plot( xdata, ydata
                       # color = c, linewidth = w, linestyle = s 
                       )

    n_fit_window_fraction = Float( 0.1 )

    def _plot_smoothed_force_deflection_center( self, axes ):

        # get the index of the maximum stress
        max_force_idx = argmax( -self.Kraft )
        # get only the ascending branch of the response curve
        f_asc = -self.Kraft[:max_force_idx + 1]
        w_asc = -self.WA_M[:max_force_idx + 1]

        f_max = f_asc[-1]
        w_max = w_asc[-1]

        n_points = int( self.n_fit_window_fraction * len( w_asc ) )
        f_smooth = smooth( f_asc, n_points, 'flat' )
        w_smooth = smooth( w_asc, n_points, 'flat' )

        axes.plot( w_smooth, f_smooth, color = 'blue', linewidth = 2 )

        secant_stiffness_w10 = ( f_smooth[10] - f_smooth[0] ) / ( w_smooth[10] - w_smooth[0] )
        w0_lin = array( [0.0, w_smooth[10] ], dtype = 'float_' )
        f0_lin = array( [0.0, w_smooth[10] * secant_stiffness_w10 ], dtype = 'float_' )

        #axes.plot( w0_lin, f0_lin, color = 'black' )


    def _plot_force_edge_deflection( self, axes ):

        # get the index of the maximum stress
        max_force_idx = argmax( -self.Kraft )
        # get only the ascending branch of the response curve
        f_asc = -self.Kraft[:max_force_idx + 1]
        w_v_asc = -self.WA_V[:max_force_idx + 1]
        w_h_asc = -self.WA_H[:max_force_idx + 1]
        w_l_asc = -self.WA_L[:max_force_idx + 1]
        w_r_asc = -self.WA_R[:max_force_idx + 1]

        w_vh_asc = ( w_v_asc + w_h_asc ) / 2
        w_lr_asc = ( w_l_asc + w_r_asc ) / 2

#        axes.plot( w_vh_asc, f_asc, color = 'blue', linewidth = 3 )
        axes.plot( w_v_asc, f_asc, color = 'blue', linewidth = 1 )
        axes.plot( w_h_asc, f_asc, color = 'blue', linewidth = 1 )
#        axes.plot( w_lr_asc, f_asc, color = 'green', linewidth = 3 )
        axes.plot( w_l_asc, f_asc, color = 'green', linewidth = 1 )
        axes.plot( w_r_asc, f_asc, color = 'green', linewidth = 1 )


    def _plot_force_center_edge_deflection( self, axes ):

        # get the index of the maximum stress
        max_force_idx = argmax( -self.Kraft )
        # get only the ascending branch of the response curve
        f_asc = -self.Kraft[:max_force_idx + 1]
        w_ml_asc = -self.WA_ML[:max_force_idx + 1]
        w_mr_asc = -self.WA_MR[:max_force_idx + 1]

        w_mlmr_asc = ( w_ml_asc + w_mr_asc ) / 2

        axes.plot( w_mlmr_asc, f_asc, color = 'blue', linewidth = 3 )
#        axes.plot( w_ml_asc, f_asc, color = 'blue', linewidth = 1 )
#        axes.plot( w_mr_asc, f_asc, color = 'blue', linewidth = 1 )

    #--------------------------------------------------------------------------------
    # view
    #--------------------------------------------------------------------------------

    traits_view = View( VGroup( 
                         Group( 
                              Item( 'jump_rtol', format_str = "%.4f" ),
                              label = 'curve_ironing'
                              ),
                         Group( 
                              Item( 'thickness', format_str = "%.3f" ),
                              Item( 'edge_length', format_str = "%.3f" ),
                              label = 'geometry'
                              ),
                         Group( 
                              Item( 'loading_rate' ),
                              Item( 'age' ),
                              label = 'loading rate and age'
                              ),
                         Group( 
                              Item( 'E_c', show_label = True, style = 'readonly', format_str = "%.0f" ),
                              Item( 'ccs@', show_label = False ),
                              label = 'composite cross section'
                              )
                         ),
                        scrollable = True,
                        resizable = True,
                        height = 0.8,
                        width = 0.6
                        )

if __name__ == '__main__':
    et = ExPlateTest()
    print 'loading_rate', et.loading_rate
    et.configure_traits()
