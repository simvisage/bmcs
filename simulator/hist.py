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

    prim_vars = List()
    conj_vars = List()

    state_vars = List()

    def record_timestep(self, t,
                        prim_var, conj_var,
                        state_vars=None):
        '''Add the time step and record the 
        corresponding state of the model.
        '''
        self.timesteps.append(t)
        self.prim_vars.append(prim_var)
        self.conj_vars.append(conj_var)
        self.state_vars.append(state_vars)
        for vis in self.record_dict.values():
            vis.update(prim_var, t)

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        x = np.array(self.timesteps, dtype=np.float_)
        idx = np.array(np.arange(len(x)), dtype=np.float_)
        t_idx = np.interp(vot, x, idx)
        return np.array(t_idx, np.int_)

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))

    def get_t_arr(self):
        return np.array(self.timesteps, dtype=np.float_)

    record_dict = Property(
        Dict, depends_on='sim.record, sim.record_items'
    )

    @cached_property
    def _get_record_dict(self):
        for viz in self.sim.record.values():
            viz.tstep = self.sim.tstep
            viz.setup()
        return {key: viz for key, viz in self.sim.record.items()}

    def __getitem__(self, key):
        return self.record_dict[key]
