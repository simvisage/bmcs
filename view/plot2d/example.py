'''
Created on Mar 4, 2017

@author: rch
'''

import numpy as np
from vis2d import Vis2D
from viz2d import Viz2D


class TimeProfile(Vis2D):

    def plot(self, ax, vot=0):
        x = np.linspace(0, 1, 100)
        y = x**2 + vot * x
        ax.plot(x, y)

    viz2d_classes = {'default': Viz2D}


class TimeFunction(Vis2D):

    def plot(self, ax, vot=0):
        x = np.linspace(0, 1, 100)
        y = x**2
        ax.plot(x, y)
        y_max = np.max(y)
        ax.plot([vot, vot], [0, y_max])

    viz2d_classes = {'default': Viz2D}

mpl1 = TimeFunction()
mpl2 = TimeProfile()
mpl3 = TimeProfile()


class TF(Viz2D):

    def plot(self, ax, vot):
        t, x, y = self.vis2d.get_sim_results(vot)
        ax.plot(t, y)
        y_min, y_max = np.min(y), np.max(y)
        ax.plot([vot, vot], [0, y_max])


class TP(Viz2D):

    def plot(self, ax, vot):
        t, x, y = self.vis2d.get_sim_results(vot)
        ax.plot(x, y)


class ResponseTracer(Vis2D):

    def get_sim_results(self, vot):
        t = np.linspace(0, 1, 100)
        x = t
        y = x**2 - vot
        return t, x, y

    viz2d_classes = {
        'default': TF,
        'time_profile': TP
    }

rt = ResponseTracer()
