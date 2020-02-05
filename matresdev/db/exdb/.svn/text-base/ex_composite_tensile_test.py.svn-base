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
# Created on Feb 15, 2010 by: rch, ascholzen

# @todo - construct the class for fabric layout returning calculating the 
#         cs-area of the reinforcement.
#       - instead of processed array - construct the array traits accessible
#         with the name of the measured channels
#       - reread the pickle file without processing the data (take care to reestablish
#         the link from the ex_type to the ex_run
#       - define the exdb_browser showing the inputs and outputs in a survey
#       - define the ExTreatment class with cummulative evaluation of the response values.
#       
#

from enthought.traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements, \
    DelegatesTo

from enthought.util.home_directory import \
    get_home_directory

from enthought.traits.ui.api import \
    View, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group, \
    InstanceEditor, HGroup, Spring

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

from numpy import \
    array, fabs, where, copy, ones, linspace, ones_like, hstack

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
    import View, Item, TabularEditor

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter

from promod.simdb import \
    SimDB

import os
import pickle
import string

# Access to the toplevel directory of the database
#
simdb = SimDB()

from ex_type import ExType
from i_ex_type import IExType

from mathkit.array.smoothing import smooth

from promod.matdb.trc.fabric_layup \
    import FabricLayUp

from promod.matdb.trc.fabric_layout \
    import FabricLayOut

from promod.matdb.trc.concrete_mixture \
    import ConcreteMixture

from promod.matdb.trc.composite_cross_section \
    import CompositeCrossSection, plain_concrete


class ExCompositeTensileTest( ExType ):
    '''Read the data from the directory
    '''

    implements( IExType )

    #--------------------------------------------------------------------
    # register a change of the traits with metadata 'input'
    #--------------------------------------------------------------------

    input_change = Event
    @on_trait_change( '+input, ccs.input_change' )
    def _set_input_change( self ):
        print '*** raising input change in CTT'
        self.input_change = True

    #--------------------------------------------------------------------------------
    # specify inputs:
    #--------------------------------------------------------------------------------

    width = Float( 0.140, unit = 'm', input = True, table_field = True,
                           auto_set = False, enter_set = True )
    gauge_length = Float( 0.550, unit = 'm', input = True, table_field = True,
                           auto_set = False, enter_set = True )

    # age of the concrete at the time of testing
    age = Int( 9, unit = 'd', input = True, table_field = True,
                           auto_set = False, enter_set = True )
    loading_rate = Float( 2.0, unit = 'mm/min', input = True, table_field = True,
                           auto_set = False, enter_set = True )

    #--------------------------------------------------------------------------
    # composite cross section
    #--------------------------------------------------------------------------

    ccs = Instance( CompositeCrossSection )
    def _ccs_default( self ):
        '''default settings correspond to 
        setup '9u_MAG-07-03_PZ-0708-1'
        '''
        print 'ccs default used'
#        fabric_layout_key = 'MAG-07-03'
#        fabric_layout_key = '2D-02-06a'
#        fabric_layout_key = '2D-14-10'
#        fabric_layout_key = '2D-14-10'
#        fabric_layout_key = '2D-18-10'
#        fabric_layout_key = '2D-04-11'
        fabric_layout_key = '2D-05-11'
#        concrete_mixture_key = 'PZ-0708-1'
        concrete_mixture_key = 'FIL-10-09'
        orientation_fn_key = 'all90'
#        orientation_fn_key = '90_0'
        n_layers = 12
        s_tex_z = 0.06 / ( n_layers + 1 )
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
    # Indicate whether the test is suitable and prepared for
    # calibration.
    #--------------------------------------------------------------------------
    ready_for_calibration = Property( Bool )
    def _get_ready_for_calibration( self ):
        # return False by default
        # the subclasses shall overload this 
        # and define the rules
        return self.ccs.is_regular

    #--------------------------------------------------------------------------
    # Get properties of the composite 
    #--------------------------------------------------------------------------

    # E-modulus of the composite at the time of testing 
    E_c = Property( Float, unit = 'MPa', depends_on = 'input_change', table_field = True )
    @cached_property
    def _get_E_c( self ):
        return self.ccs.get_E_c_time( self.age )

    # cross-sectional-area of the composite 
    A_c = Property( Float, unit = 'm^2', depends_on = 'input_change' )
    @cached_property
    def _get_A_c( self ):
        return self.width * self.ccs.thickness

    # total cross-sectional-area of the textile reinforcement 
    A_tex = Property( Float, unit = 'mm^2', depends_on = 'input_change' )
    @cached_property
    def _get_A_tex( self ):
        return self.ccs.a_tex * self.width

    # E-modulus of the composite after 28 days
    E_c28 = DelegatesTo( 'ccs', listenable = False )

    # reinforcement ration of the composite 
    rho_c = DelegatesTo( 'ccs', listenable = False )

    #--------------------------------------------------------------------------------
    # define processing
    #--------------------------------------------------------------------------------

    def process_source_data( self ):
        '''read in the measured data from file and assign
        attributes after array processing.        
        If necessary modify the assigned data, e.i. change
        the sign or specify an offset for the specific test setup.
        '''
        super( ExCompositeTensileTest, self ).process_source_data()

        # NOTE: the small tensile tests (INSTRON) with width = 0.10 m have only 3 displacement gauges
        #
        if hasattr( self, "W10_re" ) and hasattr( self, "W10_li" ) and hasattr( self, "W10_vo" ):
            self.W10_re -= self.W10_re[0]
            self.W10_re *= -1
            self.W10_li -= self.W10_li[0]
            self.W10_li *= -1
            self.W10_vo -= self.W10_vo[0]
            self.W10_vo *= -1

        if hasattr( self, "W10_vli" ):
            print 'change_varname'
            self.WA_VL = self.W10_vli
        if hasattr( self, "W10_vre" ):
            self.WA_VR = self.W10_vre
        if hasattr( self, "W10_hli" ):
            self.WA_HL = self.W10_hli
        if hasattr( self, "W20_hre" ):
            self.WA_HR = self.W20_hre

        # NOTE: the large tensile tests (PSB1000) with width = 0.14 m have 4 displacement gauges
        #
        if hasattr( self, "WA_VL" ) and hasattr( self, "WA_VR" ) and hasattr( self, "WA_HL" ) and hasattr( self, "WA_HR" ):
            self.WA_VL -= self.WA_VL[0]
            self.WA_VL *= -1
            self.WA_VR -= self.WA_VR[0]
            self.WA_VR *= -1
            self.WA_HL -= self.WA_HL[0]
            self.WA_HL *= -1
            self.WA_HR -= self.WA_HR[0]
            self.WA_HR *= -1


    #-------------------------------------------------------------------------------
    # Get the strain and state arrays
    #-------------------------------------------------------------------------------
    eps = Property( Array( 'float_' ), output = True,
                    depends_on = 'input_change' )
    @cached_property
    def _get_eps( self ):
        print 'CALCULATING STRAINS'

        if hasattr( self, "W10_re" ) and hasattr( self, "W10_li" ) and hasattr( self, "W10_vo" ):
            # measured strains 
            eps_li = self.W10_li / ( self.gauge_length * 1000. )  #[mm/mm]
            eps_re = self.W10_re / ( self.gauge_length * 1000. )
            eps_vo = self.W10_vo / ( self.gauge_length * 1000. )
            # average strains 
            eps_m = ( ( eps_li + eps_re ) / 2. + eps_vo ) / 2.

        if hasattr( self, "WA_VL" ) and hasattr( self, "WA_VR" ) and hasattr( self, "WA_HL" ) and hasattr( self, "WA_HR" ):
            # measured strains 
            eps_V = ( self.WA_VL + self.WA_VR ) / 2. / ( self.gauge_length * 1000. )  #[mm/mm]
            eps_H = ( self.WA_HL + self.WA_HR ) / 2. / ( self.gauge_length * 1000. )
            # average strains 
            eps_m = ( eps_V + eps_H ) / 2.

        min_eps = min( eps_m[:10] )
        return eps_m - min_eps

    sig_c = Property( Array( 'float_' ), output = True,
                      depends_on = 'input_change' )
    @cached_property
    def _get_sig_c( self ):
        print 'CALCULATING COMPOSITE STRESS'
        # measured force: 
        force = self.Kraft # [kN]
        # cross sectional area of the concrete [m^2]: 
        A_c = self.A_c
        # calculated stress: 
        sig_c = ( force / 1000. ) / A_c  # [MPa]
        return sig_c

    sig_tex = Property( Array( 'float_' ),
                        output = True, depends_on = 'input_change' )
    @cached_property
    def _get_sig_tex( self ):
        # measured force: 
        force = self.Kraft # [kN]
        # cross sectional area of the reinforcement: 
        A_tex = self.A_tex
        # calculated stresses:
        sig_tex = ( force * 1000. ) / self.A_tex  # [MPa]
        return sig_tex

    #-------------------------------------------------------------------------------
    # Get the maximum stress index to cut off the descending part of the curves
    #-------------------------------------------------------------------------------
    max_stress_idx = Property( Int, depends_on = 'input_change' )
    @cached_property
    def _get_max_stress_idx( self ):
        return argmax( self.sig_c )

    #-------------------------------------------------------------------------------
    # Get only the ascending branch of the response curve
    #-------------------------------------------------------------------------------
    eps_asc = Property( Array( 'float_' ), depends_on = 'input_change' )
    @cached_property
    def _get_eps_asc( self ):
        return self.eps[:self.max_stress_idx + 1]

    sig_c_asc = Property( Array( 'float_' ), depends_on = 'input_change' )
    @cached_property
    def _get_sig_c_asc( self ):
        return self.sig_c[:self.max_stress_idx + 1]

    sig_tex_asc = Property( Array( 'float_' ), depends_on = 'input_change' )
    @cached_property
    def _get_sig_tex_asc( self ):
        return self.sig_tex[:self.max_stress_idx + 1]

    #-------------------------------------------------------------------------------
    # Get maximum values of the variables
    #-------------------------------------------------------------------------------
    eps_max = Property( Float, depends_on = 'input_change',
                            output = True, table_field = True, unit = 'MPa' )
    @cached_property
    def _get_eps_max( self ):
        return self.eps_asc[-1]

    sig_c_max = Property( Float, depends_on = 'input_change',
                            output = True, table_field = True, unit = 'MPa' )
    @cached_property
    def _get_sig_c_max( self ):
        return self.sig_c_asc[-1]

    sig_tex_max = Property( Float, depends_on = 'input_change',
                            output = True, table_field = True, unit = '-' )
    @cached_property
    def _get_sig_tex_max( self ):
        return self.sig_tex_asc[-1]

    #-------------------------------------------------------------------------------
    # Smoothing parameters
    #-------------------------------------------------------------------------------
    n_smooth_window_fraction = Float( 0.1 )

    n_points = Property( Int )
    def _get_n_points( self ):
        # get the fit with n-th-order polynomial
        return int( self.n_smooth_window_fraction * len( self.eps ) )

    #-------------------------------------------------------------------------------
    # Smoothed variables
    #-------------------------------------------------------------------------------
    eps_smooth = Property( Array( 'float_' ), output = True,
                            depends_on = 'input_change' )
    @cached_property
    def _get_eps_smooth( self ):
        return smooth( self.eps_asc, self.n_points, 'flat' )

    sig_c_smooth = Property( Array( 'float_' ), output = True,
                            depends_on = 'input_change' )
    @cached_property
    def _get_sig_c_smooth( self ):
        print 'COMPUTING SMOOTHED COMPOSITE STRESS'
        sig_c_smooth = smooth( self.sig_c_asc, self.n_points, 'flat' )
        sig_lin = self.E_c * self.eps_smooth
        cut_sig = where( sig_c_smooth > sig_lin )
        sig_c_smooth[ cut_sig ] = sig_lin[ cut_sig ]
        return sig_c_smooth

    sig_tex_smooth = Property( Array( 'float_' ), output = True,
                            depends_on = 'input_change' )
    @cached_property
    def _get_sig_tex_smooth( self ):
        return smooth( self.sig_tex_asc, self.n_points, 'flat' )

    #--------------------------------------------------------------------------------
    # plot templates
    #--------------------------------------------------------------------------------

    plot_templates = {'force / gauge displacement'         : '_plot_force_displacement',
                      'concrete stress / strain'           : '_plot_c_stress_strain',
                      'smoothed concrete stress / strain'  : '_plot_c_smooth_stress_strain',
                      'textile stress / strain'            : '_plot_tex_stress_strain',
                      'smoothed textile stress / strain'   : '_plot_tex_smooth_stress_strain',
                      'continuity profiles'                : '_plot_continuity_profiles' }

    default_plot_template = 'force / gauge displacement'

    def _plot_c_stress_strain( self, axes ):
        xkey = 'eps [-]'
        ykey = 'sig_c [MPa]'
        xdata = self.eps
        ydata = self.sig_c

        axes.set_xlabel( '%s' % ( xkey, ) )
        axes.set_ylabel( '%s' % ( ykey, ) )

        axes.plot( xdata, ydata
                       # color = c, linewidth = w, linestyle = s 
                       )

    def _plot_c_smooth_stress_strain( self, axes ):

        #self._plot_c_stress_strain(axes)

        axes.set_xlabel( 'eps_asc [-]' )
        axes.set_ylabel( 'sig_c_smooth [MPa]' )

        axes.plot( self.eps_asc, self.sig_c_asc, color = 'green'
                       # color = c, linewidth = w, linestyle = s 
                       )
        sig_lin = array( [0, self.sig_c_max], dtype = 'float_' )
        eps_lin = array( [0, self.sig_c_max / self.E_c ], dtype = 'float_' )
        axes.plot( eps_lin, sig_lin, color = 'red' )

        axes.plot( self.eps_smooth, self.sig_c_smooth, color = 'blue', linewidth = 2
                       # color = c, linewidth = w, linestyle = s 
                       )

    def _plot_tex_stress_strain( self, axes ):
        xkey = 'eps [-]'
        ykey = 'sig_tex [MPa]'
        xdata = self.eps
        ydata = self.sig_tex

        axes.set_xlabel( '%s' % ( xkey, ) )
        axes.set_ylabel( '%s' % ( ykey, ) )

        axes.plot( xdata, ydata
                       # color = c, linewidth = w, linestyle = s 
                       )

    def _plot_tex_smooth_stress_strain( self, axes ):

        axes.set_xlabel( 'eps_asc [-]' )
        axes.set_ylabel( 'sig_tex_smooth [MPa]' )

        axes.plot( self.eps_asc, self.sig_tex_asc
                       # color = c, linewidth = w, linestyle = s 
                       )

        axes.plot( self.eps_smooth, self.sig_tex_smooth
                       # color = c, linewidth = w, linestyle = s 
                       )

        eps_lin = array( [0, self.eps_smooth[-1]], dtype = 'float_' )
        sig_lin = self.ccs.E_tex * eps_lin
        # plot the textile secant stiffness at fracture state 
        axes.plot( eps_lin, sig_lin
                       # color = c, linewidth = w, linestyle = s 
                       )

    def _plot_force_displacement( self, axes ):

        if hasattr( self, "W10_re" ) and hasattr( self, "W10_li" ) and hasattr( self, "W10_vo" ):
            # 
            axes.plot( self.W10_re, self.Kraft )
            axes.plot( self.W10_li, self.Kraft )
            axes.plot( self.W10_vo, self.Kraft )
            axes.set_xlabel( '%s' % ( 'Weg [mm]', ) )
            axes.set_ylabel( '%s' % ( 'Kraft [kN]', ) )

        if hasattr( self, "WA_VL" ) and hasattr( self, "WA_VR" ) and hasattr( self, "WA_HL" ) and hasattr( self, "WA_HR" ):
            # 
            axes.plot( self.WA_VL, self.Kraft )
            axes.plot( self.WA_VR, self.Kraft )
            axes.plot( self.WA_HL, self.Kraft )
            axes.plot( self.WA_HR, self.Kraft )
            axes.set_xlabel( '%s' % ( 'Weg [mm]', ) )
            axes.set_ylabel( '%s' % ( 'Kraft [kN]', ) )


    traits_view = View( VSplit( 
                         HSplit( Group( 
                                  Item( 'width'       , format_str = "%.3f" ),
                                  Item( 'gauge_length', format_str = "%.3f" ),
                                  springy = True,
                                  label = 'geometry',
                                  id = 'promod.exdb.ex_composite_tensile_test.geometry',
                                  dock = 'tab',
                                  ),
                               Group( 
                                  Item( 'loading_rate' ),
                                  Item( 'age' ),
                                  springy = True,
                                  label = 'loading rate and age',
                                  id = 'promod.exdb.ex_composite_tensile_test.loading',
                                  dock = 'tab', ),
                                  id = 'promod.exdb.ex_composite_tensile_test.xxx',
                                  dock = 'tab',
                             ),
                            Group( 
                                  Item( 'ccs@', resizable = True, show_label = False ),
                                  label = 'composite cross section'
                                  ),
#                               label = 'input variables',
#                               id = 'promod.exdb.ex_composite_tensile_test.vgroup.inputs',
#                               dock = 'tab',
#                               scrollable = True,
#                               ),
                         Group( 
                               Item( 'E_c', visible_when = 'derived_data_available',
                                                style = 'readonly', show_label = True , format_str = "%.0f" ),
                               Item( 'sig_c_max', visible_when = 'derived_data_available',
                                                style = 'readonly', emphasized = True , format_str = "%.2f" ),
                               Item( 'sig_tex_max', visible_when = 'derived_data_available',
                                                style = 'readonly', emphasized = True , format_str = "%.2f" ),
                               Item( 'eps_max', visible_when = 'derived_data_available',
                                                style = 'readonly', emphasized = True , format_str = "%.4f" ),
                               label = 'output characteristics',
                               id = 'promod.exdb.ex_composite_tensile_test.vgroup.outputs',
                               dock = 'tab',
                               scrollable = True,
                               ),
                         scrollable = True,
                         id = 'promod.exdb.ex_composite_tensile_test.vgroup',
                         dock = 'tab',
                         ),
                         id = 'promod.exdb.ex_composite_tensile_test',
                         dock = 'tab',
                         scrollable = True,
                         resizable = True,
                         height = 0.8,
                         width = 0.5,
                         )


#--------------------------------------------------------------    

if __name__ == '__main__':

    et = ExCompositeTensileTest()
    et.configure_traits()
