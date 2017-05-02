
from mathkit.mfn import MFnLineArray
from traits.api import \
    Str, Enum, \
    Range, on_trait_change, Float, Int
from traitsui.api import \
    View, Item, Group, VGroup, VSplit, UItem
from util.traits.editors import MPLFigureEditor
from view.plot2d import Viz2D
from view.ui import BMCSLeafNode

import numpy as np


class LoadingScenario(MFnLineArray, BMCSLeafNode):

    def reset(self):
        return
    node_name = Str('loading scenario')
    loading_type = Enum("monotonic", "cyclic",
                        enter_set=True, auto_set=False,
                        BC=True,
                        input=True)
    number_of_cycles = Int(1,
                           enter_set=True, auto_set=False,
                           BC=True,
                           input=True)
    maximum_loading = Float(1.0,
                            enter_set=True, auto_set=False,
                            BC=True,
                            input=True)
    unloading_ratio = Range(0., 1., value=0.5,
                            enter_set=True, auto_set=False,
                            BC=True,
                            input=True)
    number_of_increments = Int(20,
                               enter_set=True, auto_set=False,
                               BC=True,
                               input=True)
    amplitude_type = Enum("increasing", "constant",
                          enter_set=True, auto_set=False,
                          BC=True,
                          input=True)
    loading_range = Enum("non-symmetric", "symmetric",
                         enter_set=True, auto_set=False,
                         BC=True,
                         input=True)

    t_max = Float(1.)

    def __init__(self, *arg, **kw):
        super(LoadingScenario, self).__init__(*arg, **kw)
        self._update_xy_arrays()

    @on_trait_change('+BC')
    def _update_xy_arrays(self):
        if(self.loading_type == "monotonic"):
            self.number_of_cycles = 1
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if(self.amplitude_type == "increasing" and
                self.loading_range == "symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= -1
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if(self.amplitude_type == "increasing" and
                self.loading_range == "non-symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if(self.amplitude_type == "constant" and
                self.loading_range == "symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] = -self.maximum_loading
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if(self.amplitude_type == "constant" and
                self.loading_range == "non-symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:,
                                    0] = self.maximum_loading * self.unloading_ratio
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        t_arr = np.linspace(0, self.t_max, len(d_arr))
        self.xdata = t_arr
        self.ydata = d_arr
        self.replot()

    traits_view = View(
        VGroup(
            VSplit(
                VGroup(
                    Group(
                        Item('loading_type',
                             full_size=True, resizable=True
                             )
                    ),
                    Group(
                        Item('maximum_loading',
                             full_size=True, resizable=True)
                    ),
                    Group(
                        Item('number_of_cycles',
                             full_size=True, resizable=True),
                        Item('amplitude_type'),
                        Item('loading_range'),
                        Item('unloading_ratio'),
                        show_border=True, label='Cyclic load inputs'),
                    scrollable=True
                ),
                UItem('figure', editor=MPLFigureEditor(),
                      height=300,
                      resizable=True,
                      springy=True),
            )
        )
    )

    tree_view = traits_view


class Viz2DLoadControlFunction(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'Load control'

    def plot(self, ax, vot, *args, **kw):
        bc = self.vis2d.control_bc
        val = bc.value
        tloop = self.vis2d.tloop
        t_arr = np.array(tloop.t_record, np.float_)
        if len(t_arr) == 0:
            return
        f_arr = val * bc.time_function(t_arr)
        ax.plot(t_arr, f_arr, color='black')
        ax.set_ylabel('load factor')
        ax.set_xlabel('time')
        vot_idx = tloop.get_time_idx(vot)
        ax.plot([t_arr[vot_idx]], [f_arr[vot_idx]], 'o',
                color='black', markersize=10)
