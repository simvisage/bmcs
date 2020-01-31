'''
Created on Mar 4, 2017

@author: rch
'''

from mathkit.mfn import MFnLineArray
from traits.api import \
    List, Float, Int, Range, Property,\
    cached_property, Bool, Callable, on_trait_change
from traitsui.api import \
    View, UItem, Item, Include, \
    VGroup, VSplit, spring, Tabbed
from util.traits.editors import MPLFigureEditor
from view.plot2d import Vis2D, Viz2D
from view.ui.bmcs_tree_node import BMCSLeafNode

import numpy as np


class TFViz2D(Viz2D):

    def plot(self, ax, vot):
        t, y = self.vis2d.get_ty_data(vot)
        ax.plot(t, y)
        y_min, y_max = np.min(y), np.max(y)
        ax.plot([vot, vot], [y_min, y_max])


class TFunPWLInteractive(MFnLineArray, BMCSLeafNode, Vis2D):
    '''Interactive time function.
    '''
    node_name = 'time function'
    t_values = List(Float, [0])
    f_values = List(Float, [0])

    def reset(self):
        self.f_values = [0]
        self.t_values = [0]
        self.f_value = self.f_min

    n_f_values = Int(10,
                     input=True,
                     auto_set=False, enter_set=True)

    f_min = Float(0.0, input=True,
                  auto_set=False, enter_set=True,
                  label='F minimum')

    f_max = Float(1.0,
                  input=True,
                  auto_set=False, enter_set=True,
                  label='F maximum')

    t_ref = Float(1.0, auto_set=False, enter_set=True,
                  label='Initial time range')

    f_value = Range(low='f_min', high='f_max', value=0,
                    input=True,
                    auto_set=False, enter_set=True)

    enable_slider = Bool(True, disable_on_run=True)

    run_eagerly = Bool(True, label='Run eagerly')

    t_snap = Float(0.1, label='Time step to snap to',
                   auto_set=False, enter_set=True)

    def __init__(self, *arg, **kw):
        super(TFunPWLInteractive, self).__init__(*arg, **kw)
        self.xdata = np.array(self.t_values)
        self.ydata = np.array(self.f_values)

    d_t = Property(depends_on='t_ref,n_f_values')

    @cached_property
    def _get_d_t(self):
        return self.t_ref / self.n_f_values

    def _update_xy_arrays(self):
        delta_f = self.f_value - self.f_values[-1]
        self.f_values.append(self.f_value)
        rel_step = delta_f / (self.f_max - self.f_min)
        delta_t = rel_step * self.t_ref
        t_value = np.fabs(delta_t) + self.t_values[-1]
        n_steps = int(t_value / self.t_snap) + 1
        t_value = n_steps * self.t_snap
        self.t_values.append(t_value)
        self.xdata = np.array(self.t_values)
        self.ydata = np.array(self.f_values)
        self.replot()

    def _f_value_changed(self):
        self._update_xy_arrays()
        t_value = self.t_values[-1]
        f_value = self.f_values[-1]
        if self.ui:
            self.ui.model.set_tmax(t_value)
            if self.run_eagerly:
                print('LS-run', t_value, f_value)
                self.ui.run()

    def get_ty_data(self, vot):
        return self.t_values, self.f_values

    viz2d_classes = {
        'time_function': TFViz2D,
    }

    tree_view = View(
        VGroup(
            VSplit(
                VGroup(
                    VGroup(
                        Include('actions'),
                    ),
                    Tabbed(
                        VGroup(
                            VGroup(
                                UItem('f_value',
                                      full_size=True, resizable=True,
                                      enabled_when='enable_slider'
                                      ),
                            ),
                            VGroup(
                                Item('f_max',
                                     full_size=True, resizable=True),
                                Item('f_min',
                                     full_size=True),
                                Item('n_f_values',
                                     full_size=True),
                                Item('t_snap', tooltip='Snap value to round off'
                                     'the value to',
                                     full_size=True),
                            ),
                            spring,
                            label='Steering',
                        ),
                        VGroup(
                            Item('run_eagerly',
                                 full_size=True, resizable=True,
                                 tooltip='True - run calculation immediately'
                                 'after moving the value slider; \nFalse - user must'
                                 'start calculation by clicking Run button'),
                            spring,
                            label='Mode',
                        ),
                    ),
                ),
                UItem('figure', editor=MPLFigureEditor(),
                      height=300,
                      resizable=True,
                      springy=True),
            ),
        )
    )

    traits_view = tree_view

if __name__ == '__main__':
    bc = TFunPWLInteractive()
    bc.set_traits_with_metadata(True, disable_on_run=True)
    bc.replot()
    bc.configure_traits()
