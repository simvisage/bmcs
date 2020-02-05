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
    Dict, Button, Bool, Enum, Event, implements, DelegatesTo, \
    Callable

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

from mathkit.mfn import MFnLineArray
from mathkit.mfn.mfn_line.mfn_matplotlib_editor import \
    MFnMatplotlibEditor

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

from loadtxt_bending import loadtxt_bending

from promod.simdb import \
    SimDB

# Access to the toplevel directory of the database
#
simdb = SimDB()


class ExBendingTest( ExType ):
    '''Read the data from the directory
    '''

    implements( IExType )

    file_ext = 'raw'

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

    length = Float( 1.15, unit = 'm', input = True, table_field = True,
                           auto_set = False, enter_set = True )
    width = Float( 0.2, unit = 'm', input = True, table_field = True,
                           auto_set = False, enter_set = True )
    thickness = Float( 0.06, unit = 'm', input = True, table_field = True,
                           auto_set = False, enter_set = True )

    # age of the concrete at the time of testing
    age = Int( 28, unit = 'd', input = True, table_field = True,
                             auto_set = False, enter_set = True )
    loading_rate = Float( 3.0, unit = 'mm/min', input = True, table_field = True,
                            auto_set = False, enter_set = True )

    #--------------------------------------------------------------------------
    # composite cross section
    #--------------------------------------------------------------------------

    ccs = Instance( CompositeCrossSection )
    def _ccs_default( self ):
        '''default settings correspond to 
        setup '7u_MAG-07-03_PZ-0708-1'
        '''
#        fabric_layout_key = 'MAG-07-03'
#        fabric_layout_key = '2D-02-06a'
        fabric_layout_key = '2D-05-11'
#        concrete_mixture_key = 'PZ-0708-1'
        concrete_mixture_key = 'FIL-10-09'
        orientation_fn_key = 'all0'
#        orientation_fn_key = 'all90'                                           
#        orientation_fn_key = '90_0'
        n_layers = 12
        s_tex_z = 0.060 / ( n_layers + 1 )
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

    def _read_data_array( self ):
        ''' Read the experiment data. 
        '''
        if exists( self.data_file ):

            print 'READ FILE'
            # change the file name dat with asc  
            file_split = self.data_file.split( '.' )

            file_name = file_split[0] + '.csv'
            if not os.path.exists( file_name ):

                file_name = file_split[0] + '.raw'
                if not os.path.exists( file_name ):
                    raise IOException, 'file %s does not exist' % file_name

            print 'file_name', file_name

            _data_array = loadtxt_bending( file_name )

            self.data_array = _data_array

    names_and_units = Property
    @cached_property
    def _get_names_and_units( self ):
        ''' Set the names and units of the measured data.
        '''
        names = ['w', 'eps_c', 'F']
        units = ['mm', '%', 'kN']
        return names, units

#    mfn_elastomer = Instance( MFnLineArray )
#    def _mfn_elastomer_default( self ):
#        elastomer_path = os.path.join( simdb.exdata_dir, 'bending_tests', 'ZiE_2011-06-08_BT-12c-6cm-0-TU', 'elastomer_f-w.raw' )
#        # loadtxt_bending returns an array with three columns: 
#        # 0: deformation; 1: eps_c; 2: force
#        _data_array_elastomer = loadtxt_bending( elastomer_path )
#        return MFnLineArray( xdata = _data_array_elastomer[:, 0], ydata = _data_array_elastomer[:, 2] )
#
#    elastomer_force = Callable
#    def elastomer_force_default( self ):
#        return frompyfunc( self.mfn_elastomer.get_value, 2, 1 )

    def process_source_data( self ):
        '''read in the measured data from file and assign
        attributes after array processing.        
        '''
        super( ExBendingTest, self ).process_source_data()

        elastomer_path = os.path.join( simdb.exdata_dir, 'bending_tests', 'ZiE_2011-06-08_BT-12c-6cm-0-TU', 'elastomer_f-w.raw' )
        _data_array_elastomer = loadtxt_bending( elastomer_path )

        # force [kN]:
        #
        xdata = -0.001 * _data_array_elastomer[:, 2].flatten()

        # displacement [mm]:
        #
        ydata = -1.0 * _data_array_elastomer[:, 0].flatten()

        mfn_displacement_elastomer = MFnLineArray( xdata = xdata, ydata = ydata )
        displacement_elastomer_vectorized = frompyfunc( mfn_displacement_elastomer.get_value, 1, 1 )

        # convert data from 'N' to 'kN' and change sign
        #
        self.F = -0.001 * self.F

        # change sign in positive values for vertical displacement [mm]
        #
        self.w = -1.0 * self.w

        # substract the deformation of the elastomer cushion between the cylinder
        # 
        self.w = self.w - displacement_elastomer_vectorized( self.F )



    #--------------------------------------------------------------------------------
    # plot templates
    #--------------------------------------------------------------------------------

    plot_templates = {'force / deflection (center)'          : '_plot_force_deflection_center',
                      'smoothed force / deflection (center)' : '_plot_smoothed_force_deflection_center',
                     }

    default_plot_template = 'force / deflection (center)'

    def _plot_force_deflection_center( self, axes ):
        xkey = 'deflection [mm]'
        ykey = 'force [kN]'
        # NOTE: processed data returns positive values for force and displacement
        #
        xdata = self.w
        ydata = self.F

        axes.set_xlabel( '%s' % ( xkey, ) )
        axes.set_ylabel( '%s' % ( ykey, ) )
        axes.plot( xdata, ydata
                       # color = c, linewidth = w, linestyle = s 
                       )

    n_fit_window_fraction = Float( 0.1 )

    def _plot_smoothed_force_deflection_center( self, axes ):

        # get the index of the maximum stress
        max_force_idx = argmax( self.F )
        # get only the ascending branch of the response curve
        f_asc = self.F[:max_force_idx + 1]
        w_asc = self.w[:max_force_idx + 1]

        f_max = f_asc[-1]
        w_max = w_asc[-1]

        n_points = int( self.n_fit_window_fraction * len( w_asc ) )
        f_smooth = smooth( f_asc, n_points, 'flat' )
        w_smooth = smooth( w_asc, n_points, 'flat' )

        axes.plot( w_smooth, f_smooth, color = 'blue', linewidth = 2 )

#        secant_stiffness_w10 = ( f_smooth[10] - f_smooth[0] ) / ( w_smooth[10] - w_smooth[0] )
#        w0_lin = array( [0.0, w_smooth[10] ], dtype = 'float_' )
#        f0_lin = array( [0.0, w_smooth[10] * secant_stiffness_w10 ], dtype = 'float_' )

        #axes.plot( w0_lin, f0_lin, color = 'black' )


    #--------------------------------------------------------------------------------
    # view
    #--------------------------------------------------------------------------------

    traits_view = View( VGroup( 
                         Group( 
                              Item( 'length', format_str = "%.3f" ),
                              Item( 'width', format_str = "%.3f" ),
                              Item( 'thickness', format_str = "%.3f" ),
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
    et = ExBendingTest()
    print 'loading_rate', et.loading_rate
    et.configure_traits()
