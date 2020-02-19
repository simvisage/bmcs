'''
Created on Nov 21, 2018

@author: rch
'''

from traitsui.api import View, Include, HGroup

from view.plot2d import Vis2D, Viz2D
from view.ui import BMCSLeafNode

import numpy as np
import traits.api as tr


class ShearCrackViz2D(Viz2D):

    def plot(self, ax, vot):
        sc = self.vis2d
        sz = sc.parent
        sz._plot_shear_zone(ax, vot)
        ax.plot(self.vis2d.x, self.vis2d.y, linewidth=2, color='black')


class ShearCrack(BMCSLeafNode, Vis2D):
    '''Class representing the geometry of a crack using
    a piecewise linear representation
    '''

    node_name = 'shear crack geometry'

    x = tr.Array(np.float_,
                 value=[0.2, 0.15, 0.1], GEO=True)
    y = tr.Array(np.float_,
                 value=[0.0, 0.15, 0.2], GEO=True)

    viz2d_classes = {
        'shear_crack': ShearCrackViz2D
    }

    tree_view = View(
        HGroup(
            Include('actions')
        )
    )
