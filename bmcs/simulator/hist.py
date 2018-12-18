'''
Created on Dec 17, 2018

@author: rch
'''

from traits.api import \
    HasStrictTraits, provides, \
    List, WeakRef

import numpy as np

from .i_hist import IHist
from .i_model import IModel


@provides(IHist)
class Hist(HasStrictTraits):
    '''Object storing and managing the history of calculation.
    '''

    model = WeakRef(IModel)

    def add_timestep(self, t):
        '''Add the time step and record the 
        corresponding state of the model.
        '''
        self.timesteps.append(t)
        self.model.record_state(t)
        self.tline.val = min(t, self.tline.max)

    timesteps = List()

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        x = np.array(self.timesteps, dtype=np.float_)
        idx = np.array(np.arange(len(x)), dtype=np.float_)
        t_idx = np.interp(vot, x, idx)
        return np.array(t_idx, np.int_)

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))
