'''
Created on 06.05.2011

@author: rrypl
'''
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
# Created on Jan 11, 2011 by: rch

from etsproxy.traits.api import HasTraits, Float, Instance, on_trait_change, Int, \
                                Event, Button, Str

from etsproxy.traits.ui.api import \
    View, Item, Tabbed, VGroup, HGroup, ModelView, HSplit, OKButton

from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure

from numpy import \
    linspace, frompyfunc

from stats.spirrid_bak.old.spirrid import SPIRRID
from util.traits.either_type import EitherType

# IMPORT RESPONSE FUNCTIONS
from stats.spirrid_bak.i_rf import IRF
from quaducom.resp_func.brittle_filament import Filament
from quaducom.resp_func.po_short_fiber import POShortFiber
from quaducom.resp_func.cb_short_fiber import CBShortFiber
from quaducom.resp_func.po_clamped_fiber import POClampedFiber
from quaducom.resp_func.cb_clamped_fiber import CBClampedFiber
from quaducom.resp_func.po_infinite_fiber import POInfiniteFiber
from quaducom.resp_func.cb_infinite_fiber import CBInfiniteFiber


class RFModelView( ModelView ):

    title = Str( 'RF browser' )

    model = Instance( IRF )

    child = Instance( SPIRRID )
    def _child_default( self ):
        return SPIRRID()

    rf = EitherType( names = ['brittle filament',
                              'pullout short fiber',
                              'crack bridge short fiber',
                              'pullout clamped fiber',
                              'crack bridge clamped fiber',
                              'pullout infinite fiber',
                              'crack bridge infinite fiber',
                              ],

                            klasses = [Filament,
                                       POShortFiber,
                                       CBShortFiber,
                                       POClampedFiber,
                                       CBClampedFiber,
                                       POInfiniteFiber,
                                       CBInfiniteFiber,
                                       ],

                            config_change = True )

    def _model_default( self ):
        return self.rf

    def _rf_changed( self ):
        self.model = self.rf
        self.child.rf = self.model
        for name in self.model.param_keys:
            self.on_trait_change( self._redraw, 'rf.' + name )

    def init( self, info ):
        for name in self.model.param_keys:
            self.on_trait_change( self._redraw, 'rf.' + name )

    def close( self, info, is_ok ):
        for name in self.model.param_keys:
            self.on_trait_change( self._redraw, 'rf.' + name, remove = True )
        return is_ok

    figure = Instance( Figure )
    def _figure_default( self ):
        figure = Figure( facecolor = 'white' )
        figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        return figure

    data_changed = Event( True )

    max_x = Float( 0.04, enter_set = True, auto_set = False, config_change = True )
    x_points = Int( 80, enter_set = True, auto_set = False, config_change = True )

    @on_trait_change( '+config_change' )
    def _redraw( self ):
        in_arr = linspace( 0.0, self.max_x, self.x_points )
        self.model = self.rf
        args = [ in_arr ] + self.model.param_values

        # get the number of parameters of the response function
        n_args = len( args )
        fn = frompyfunc( self.model.__call__, n_args, 1 )

        out_arr = fn( *args )

        axes = self.figure.axes[0]
        axes.plot( in_arr, out_arr,
                   linewidth = 2, label = self.model.title )

        axes.set_xlabel( self.model.x_label )
        axes.set_ylabel( self.model.y_label )
        #axes.legend( loc = 'best' )

        self.data_changed = True
    show = Button
    def _show_fired( self ):
        self._redraw()

    clear = Button
    def _clear_fired( self ):
        axes = self.figure.axes[0]
        axes.clear()
        self.data_changed = True

    def default_traits_view( self ):
        '''
        Generates the view from the param items.
        '''
        #rf_param_items = [ Item( 'model.' + name, format_str = '%g' ) for name in self.model.param_keys ]
        D2_plot_param_items = [ VGroup( Item( 'max_x' , label = 'max x value' ),
                                       Item( 'x_points', label = 'No of plot points' ) ) ]

        if hasattr( self.rf, 'get_q_x' ):
            D3_plot_param_items = [ VGroup( Item( 'min_x', label = 'min x value' ),
                                       Item( 'max_x', label = 'max x value' ),
                                       Item( 'min_y', label = 'min y value' ),
                                       Item( 'max_y', label = 'max y value' ) )
                                       ]
        else:
            D3_plot_param_items = []

        control_items = [
                        Item( 'show', show_label = False ),
                        Item( 'clear', show_label = False ),
                        ]
        view = View( HSplit( VGroup( Item( '@rf', show_label = False ),
                                     label = 'Function parameters',
                                     id = 'stats.spirrid_bak.rf_model_view.rf_params',
                                     scrollable = True
                                     ),
                             VGroup( HGroup( *D2_plot_param_items ),
                                     label = 'plot parameters',
                                     id = 'stats.spirrid_bak.rf_model_view.2Dplot_params'
                                     ),
#                            VGroup( HGroup( *D3_plot_param_items ),
#                                     label = '3D plot parameters',
#                                     id = 'stats.spirrid_bak.rf_model_view.3Dplot_params' ),
                             VGroup( Item( 'model.comment', show_label = False,
                                           style = 'readonly' ),
                                     label = 'Comment',
                                     id = 'stats.spirrid_bak.rf_model_view.comment',
                                     scrollable = True,
                                     ),
                             VGroup( 
                                    HGroup( *control_items ),
                                    Item( 'figure', editor = MPLFigureEditor(),
                                     resizable = True, show_label = False ),
                                     label = 'Plot',
                                     id = 'stats.spirrid_bak.rf_model_view.plot'
                                     ),
                             dock = 'tab',
                             id = 'stats.spirrid_bak.rf_model_view.split'
                             ),
                    kind = 'modal',
                    resizable = True,
                    dock = 'tab',
                    buttons = [OKButton],
                    id = 'stats.spirrid_bak.rf_model_view'
                    )
        return view

def run():

    rf = RFModelView( model = Filament() )
    rf._redraw()
    rf.configure_traits( kind = 'live' )

if __name__ == '__main__':
    run()
