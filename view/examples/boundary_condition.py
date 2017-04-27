'''
Created on Mar 4, 2017

@author: rch
'''

from traits.api import \
    List, Float, Int, Range, Property,\
    cached_property
from traitsui.api import \
    View, UItem, Item, Include, \
    VGroup, spring

import numpy as np
from view.plot2d import Vis2D, Viz2D
from view.ui.bmcs_tree_node import BMCSLeafNode


class TFViz2D(Viz2D):

    def plot(self, ax, vot):
        t, y = self.vis2d.get_ty_data(vot)
        ax.plot(t, y)
        y_min, y_max = np.min(y), np.max(y)
        ax.plot([vot, vot], [y_min, y_max])


class BoundaryCondition(BMCSLeafNode, Vis2D):

    node_name = 'boundary condition'
    t_values = List(Float, [0])
    f_values = List(Float, [0])
    n_f_values = Int(10, auto_set=False, enter_set=True)
    f_min = Float(0.0, auto_set=False, enter_set=True)
    f_max = Float(1.0, auto_set=False, enter_set=True)
    t_ref = Float(1.0, auto_set=False, enter_set=True)

    f_value = Range(low='f_min', high='f_max', value=0,
                    auto_set=False, enter_set=True)

    d_t = Property(depends_on='t_ref,n_f_values')

    @cached_property
    def _get_d_t(self):
        return self.t_ref / self.n_f_values

    def _f_value_changed(self):
        delta_f = self.f_value - self.f_values[-1]
        self.f_values.append(self.f_value)
        rel_step = delta_f / self.f_max
        delta_t = rel_step * self.t_ref
        t_value = np.fabs(delta_t) + self.t_values[-1]
        self.t_values.append(t_value)
        if self.ui:
            self.ui.set_vot(t_value)

    def get_ty_data(self, vot):
        return self.t_values, self.f_values

    viz2d_classes = {
        'time_function': TFViz2D,
    }

    tree_view = View(
        VGroup(
            VGroup(
                Include('actions'),
            ),
            VGroup(
                UItem('f_value', full_size=True, resizable=True),
                label='Steering slider'
            ),
            VGroup(
                Item('f_min', full_size=True, resizable=True),
                Item('f_max', full_size=True),
                Item('n_f_values', full_size=True),
                spring,
                label='Steering range',
            ),
        ),
        resizable=True
    )

if __name__ == '__main__':
    bc = BoundaryCondition()
    bc.configure_traits(view='tree_view')
