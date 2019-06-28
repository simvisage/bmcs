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

from enthought.traits.api import HasTraits, Float, Property, cached_property, \
                                Instance, List, on_trait_change, Int, Tuple, Bool, \
                                Event, Button, Str

from enthought.traits.ui.api import \
    View, Item, Tabbed, VGroup, HGroup, Group, ModelView, HSplit, VSplit, Spring, OKButton

from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure

from numpy import \
    linspace, frompyfunc

from stats.spirrid_bak.i_rf import IRF
from util.traits.either_type import EitherType
from .brittle_filament import Filament
from .free_stress_double_pullout import ConstantFrictionFiniteFiber

class RFModelView( ModelView ):

    model = Instance( IRF )
    def _model_default( self ):
        return Filament()

    resp_func = EitherType( names = ['brittle filament',
                                     'dbl pullout const friction short fiber'],

                            klasses = [Filament,
                                       ConstantFrictionFiniteFiber],

                            config_change = True )

    def init( self, info ):
        for name in self.model.param_keys:
            self.on_trait_change( self._redraw, 'model.' + name )

    def close( self, info, is_ok ):
        for name in self.model.param_keys:
            self.on_trait_change( self._redraw, 'model.' + name, remove = True )
        return is_ok

    figure = Instance( Figure )
    def _figure_default( self ):
        figure = Figure( facecolor = 'white' )
        figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        return figure

    data_changed = Event( True )

    max_x = Float( 0.01, enter_set = True, auto_set = False, config_change = True )
    n_points = Int( 20, enter_set = True, auto_set = False, config_change = True )

    @on_trait_change( '+config_change' )
    def _redraw( self ):

        self.model = self.resp_func
        figure = self.figure
        axes = self.figure.axes[0]

        in_arr = linspace( 0.0, self.max_x, self.n_points )

        args = [ in_arr ] + self.model.param_values

        # get the number of parameters of the response function

        n_args = len( args )
        fn = frompyfunc( self.model.__call__, n_args, 1 )

        out_arr = fn( *args )

        axes = self.figure.axes[0]
        axes.plot( in_arr, out_arr,
                   linewidth = 2 )

        axes.set_xlabel( self.model.x_label )
        axes.set_ylabel( self.model.y_label )
        axes.legend( loc = 'best' )

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
        plot_param_items = [ Item( 'max_x' , label = 'max x value' ),
                            Item( 'n_points', label = 'No of plot points' ) ]
        control_items = [
                        Item( 'show', show_label = False ),
                        Item( 'clear', show_label = False ),
                        ]
        view = View( HSplit( VGroup( Item( '@resp_func', show_label = False ), #*rf_param_items,
                                     label = 'Function Parameters',
                                     id = 'stats.spirrid_bak.rf_model_view.rf_params',
                                     scrollable = True
                                     ),
                             VGroup( *plot_param_items,
                                     label = 'Plot Parameters',
                                     id = 'stats.spirrid_bak.rf_model_view.plot_params'
                                     ),
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

    rf = RFModelView( model = ConstantFrictionFiniteFiber() )

    rf.configure_traits( kind = 'live' )

if __name__ == '__main__':
    run()
