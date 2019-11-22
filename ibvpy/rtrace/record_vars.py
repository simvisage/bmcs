'''
Created on Feb 11, 2018

@author: rch
'''

from copy import deepcopy
import numpy as np
import traits.api as tr
from view.plot2d.vis2d import Vis2D


class RecordVars(Vis2D):

    U = tr.List
    F = tr.List
    t = tr.List
    states = tr.List

    def update(self):
        ts = self.sim.tstep
        self.U.append(np.copy(ts.U_k))
        self.F.append(np.copy(ts.F_k))
        self.t.append(np.copy(ts.t_n1))
        self.states.append(deepcopy(ts.state_k))
