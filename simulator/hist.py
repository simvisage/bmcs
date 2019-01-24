'''
Created on Dec 17, 2018

@author: rch
'''

from traits.api import \
    HasStrictTraits, provides, \
    List, Dict, WeakRef, Property, cached_property

import numpy as np

from .i_hist import IHist
from .tstep import TStep


@provides(IHist)
class Hist(HasStrictTraits):
    '''Object storing and managing the history of calculation.
    '''

    sim = WeakRef

    timesteps = List()

    U_list = List()
    F_list = List()

    state_vars = List()

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
        print('U_t recalculated', self.timesteps)
        return np.array(self.U_list, dtype=np.int)

    F_t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_F_t(self):
        return np.array(self.U_list, dtype=np.int)

    t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_t(self):
        return np.array(self.timesteps, dtype=np.float_)

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        t = self.t
        idx = np.array(np.arange(len(t)), dtype=np.float_)
        t_idx = np.interp(vot, t, idx)
        return np.array(t_idx, np.int_)

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))

    record_dict = Property(
        Dict, depends_on='sim.record, sim.record_items'
    )

    @cached_property
    def _get_record_dict(self):
        for viz in self.sim.record.values():
            viz.sim = self.sim
            viz.setup()
        return {key: viz for key, viz in self.sim.record.items()}

    def __getitem__(self, key):
        return self.record_dict[key]
