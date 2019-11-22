'''
Created on Mar 16, 2017

@author: rch
'''

import numpy as np
from .viz2d import Viz2D


class Viz2DTimeFunction(Viz2D):

    def plot(self, ax, vot):
        x, y = self.vis2d.get_viz2d_data()
        ax.plot(x, y)
        ymin, ymax = np.min(y), np.max(y)
        ax.plot([vot, vot], [ymin, ymax])
