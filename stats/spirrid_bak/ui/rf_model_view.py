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

from etsproxy.traits.api import HasTraits, Float, Property, cached_property, \
                                Instance, List, on_trait_change, Int, Tuple, Bool, \
                                Event, Button, Str

from etsproxy.traits.ui.api import \
    View, Item, Tabbed, VGroup, HGroup, Group, ModelView, HSplit, VSplit, Spring

from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure

from numpy import \
    linspace, frompyfunc

from stats.spirrid_bak import IRF

class RFModelView(ModelView):
    '''
    Size effect depending on the yarn length
    '''
    model = Instance(IRF)

    title = Str('RF browser')

    def init(self, info):
        for name in self.model.param_keys:
            self.on_trait_change(self._redraw, 'model.' + name)

    def close(self, info, is_ok):
        for name in self.model.param_keys:
            self.on_trait_change(self._redraw, 'model.' + name, remove = True)
        return is_ok

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure(facecolor = 'white')
        figure.add_axes([0.08, 0.13, 0.85, 0.74])
        return figure

    data_changed = Event(True)

    eps_max = Float(0.1, enter_set = True, auto_set = False, config_change = True)

    n_eps = Int(20, enter_set = True, auto_set = False, config_change = True)

    x_name = Str('epsilon', enter_set = True, auto_set = False)
    y_name = Str('sigma', enter_set = True, auto_set = False)

    @on_trait_change('+config_change')
    def _redraw(self):

        figure = self.figure
        axes = self.figure.axes[0]

        in_arr = linspace(0.0, self.eps_max, self.n_eps)

        args = [ in_arr ] + self.model.param_values

        # get the number of parameters of the response function

        n_args = len(args)
        fn = frompyfunc(self.model.__call__, n_args, 1)

        out_arr = fn(*args)

        axes = self.figure.axes[0]
        axes.plot(in_arr, out_arr,
                   linewidth = 2)

        axes.set_xlabel(self.x_name)
        axes.set_ylabel(self.y_name)
        axes.legend(loc = 'best')

        self.data_changed = True

    show = Button
    def _show_fired(self):
        self._redraw()

    clear = Button
    def _clear_fired(self):
        axes = self.figure.axes[0]
        axes.clear()
        self.data_changed = True

    def default_traits_view(self):
        '''
        Generates the view from the param items.
        '''
        rf_param_items = [ Item('model.' + name, format_str = '%g') for name in self.model.param_keys ]
        plot_param_items = [ Item('eps_max'), Item('n_eps'),
                            Item('x_name', label = 'x-axis'),
                            Item('y_name', label = 'y-axis') ]
        control_items = [
                        Item('show', show_label = False),
                        Item('clear', show_label = False),
                        ]
        view = View(HSplit(VGroup(*rf_param_items,
                                     label = 'Function Parameters',
                                     id = 'stats.spirrid_bak.rf_model_view.rf_params',
                                     scrollable = True
                                     ),
                             VGroup(*plot_param_items,
                                     label = 'Plot Parameters',
                                     id = 'stats.spirrid_bak.rf_model_view.plot_params'
                                     ),
                             VGroup(Item('model.comment', show_label = False,
                                           style = 'readonly'),
                                     label = 'Comment',
                                     id = 'stats.spirrid_bak.rf_model_view.comment',
                                     scrollable = True,
                                     ),
                             VGroup(
                                    HGroup(*control_items),
                                    Item('figure', editor = MPLFigureEditor(),
                                     resizable = True, show_label = False),
                                     label = 'Plot',
                                     id = 'stats.spirrid_bak.rf_model_view.plot'
                                     ),
                             dock = 'tab',
                             id = 'stats.spirrid_bak.rf_model_view.split'
                             ),
                    kind = 'modal',
                    resizable = True,
                    dock = 'tab',
                    buttons = ['Ok', 'Cancel' ],
                    id = 'stats.spirrid_bak.rf_model_view'
                    )
        return view

def run():
    from rf_filament import Filament
    from quaducom.pullout.constant_friction_finite_fiber import ConstantFrictionFiniteFiber

    rf = RFModelView(model = Filament())

    rf.configure_traits(kind = 'live')

    rf.model = ConstantFrictionFiniteFiber()

    rf.configure_traits(kind = 'live')

if __name__ == '__main__':
    run()
