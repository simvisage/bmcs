'''
Created on Dec 5, 2018

@author: rch
'''

import numpy as np
from view.plot2d import Viz2D


class Viz2DSigEps(Viz2D):

    def plot(self, ax, vot=1.0):
        prim = self.vis2d
        U = np.array(prim.U, np.float_)
        F = np.array(prim.F, np.float_)
        ax.plot(U[:, 0], F[:, 0])
        ax.plot(U[:, 1], F[:, 0])
