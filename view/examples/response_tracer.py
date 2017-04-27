'''
Created on Mar 4, 2017

@author: rch
'''

from traits.api import \
    List, Float, Int, Range, Property,\
    cached_property
from traitsui.api import View, UItem, Item, Include, HGroup

import numpy as np
from view.plot2d import Vis2D, Viz2D
from view.ui.bmcs_tree_node import BMCSLeafNode


class TP(Viz2D):

    def plot(self, ax, vot):
        t, x, y = self.vis2d.get_sim_results(vot)
        ax.plot(x, y)


class ResponseTracer(BMCSLeafNode, Vis2D):

    node_name = 'response tracer'

    def get_sim_results(self, vot):
        t = np.linspace(0, 1, 100)
        x = t
        y = x**2 - vot
        return t, x, y

    viz2d_classes = {
        'time_profile': TP
    }

    tree_view = View(
        HGroup(
            Include('actions')
        )
    )

if __name__ == '__main__':
    bc = ResponseTracer()
    bc.configure_traits(view='tree_view')
