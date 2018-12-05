'''
Created on Dec 5, 2018

@author: rch
'''

from view.plot2d import Viz2D
import numpy as np


class Viz2DSigEps(Viz2D):

    def plot(self, ax, vot=1.0):
        U = np.array(self.vis2d.tloop.U_record, np.float_)
        F = np.array(self.vis2d.tloop.F_record, np.float_)
        ax.plot(U[:, 0], F[:, 0])
        ax.plot(U[:, 1], F[:, 0])
