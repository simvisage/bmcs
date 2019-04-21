'''
Created on Dec 17, 2018

@author: rch
'''

from traits.api import \
    HasStrictTraits, provides, \
    List, Dict, WeakRef, Property, cached_property

import numpy as np

from .i_hist import IHist


@provides(IHist)
class Hist(HasStrictTraits):
    '''Object storing and managing the history of calculation.
    '''

    sim = WeakRef

    timesteps = List()

    U_list = List()
    F_list = List()

    record_dict = Property(
        Dict, depends_on='sim.record, sim.record_items'
    )

    @cached_property
    def _get_record_dict(self):
        for vis in self.sim.record.values():
            vis.sim = self.sim
            vis.setup()
        return {key: vis for key, vis in self.sim.record.items()}

    def __getitem__(self, key):
        return self.record_dict[key]

    state_vars = List()

    def init_state(self):
        self.timesteps = []
        self.U_list = []
        self.F_list = []
        self.state_vars = []
        for vis in self.record_dict.values():
            vis.setup()

    def record_timestep(self, t, U, F,
                        state_vars=None):
        '''Add the time step and record the 
        corresponding state of the model.
        '''
        self.timesteps.append(t)
        self.U_list.append(np.copy(U))
        self.F_list.append(np.copy(F))
        self.state_vars.append(state_vars)
        for vis in self.record_dict.values():
            vis.update()

    U_t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_U_t(self):
        return np.array(self.U_list, dtype=np.float_)

    F_t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_F_t(self):
        return np.array(self.F_list, dtype=np.float_)

    t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_t(self):
        return np.array(self.timesteps, dtype=np.float_)

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        t = self.t
        if len(t) == 0:
            return 0
        idx = np.array(np.arange(len(t)), dtype=np.float_)
        t_idx = np.interp(vot, t, idx)
        return np.array(t_idx, np.int_)

    def get_time(self, vot):
        return self.t[self.get_time_idx(vot)]

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))
