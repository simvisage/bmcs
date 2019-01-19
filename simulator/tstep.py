'''
'''

from traits.api import \
    HasStrictTraits, WeakRef, \
    Property, Float, Instance, DelegatesTo, \
    cached_property, provides

import numpy as np

from .i_hist import IHist
from .i_model import IModel
from .i_tstep import ITStep


@provides(ITStep)
class TStep(HasStrictTraits):
    '''Manage the data and metadata of a time step within an interation loop.
    '''
    model = Instance(IModel)

    hist = WeakRef(IHist)

    t_n1 = Float(0.0, auto_set=False, enter_set=True)
    '''Target value of the control variable.
    '''
    U_n = Float(0.0, auto_set=False, enter_set=True)
    '''Current fundamental value of the primary variable.
    '''
    U_k = Float(0.0, auto_set=False, enter_set=True)
    '''Current trial value of the primary variable.
    '''

    def init_state(self):
        '''Initialize state.
        '''
        self.U_n = 0.0
        self.t_n1 = 0.0
        self.U_k = 0.0

    def record_state(self):
        '''Provide the current state for history recording.
        '''
        pass

    _corr_pred = Property(depends_on='U_k,t_n1')

    @cached_property
    def _get__corr_pred(self):
        return self.model.get_corr_pred(self.U_k, self.t_n1)

    R = Property

    def _get_R(self):
        R, dR = self._corr_pred
        return R

    dR = Property

    def _get_dR(self):
        R, dR = self._corr_pred
        return dR

    R_norm = Property

    def _get_R_norm(self):
        R = self.R
        return np.sqrt(R * R)

    def make_iter(self):
        d_U = self.R / self.dR
        self.U_k += d_U

    def make_incr(self):
        '''Update the control, primary and state variables..
        '''
        self.U_n = self.U_k
        # self.hist.record_timestep()
