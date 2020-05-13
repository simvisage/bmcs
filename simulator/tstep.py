'''
'''

from simulator.sim_base import Simulator

import numpy as np
import traits.api as tr

from .hist import Hist
from .i_hist import IHist
from .i_model import IModel
from .i_tloop import ITLoop
from .i_tstep import ITStep


@tr.provides(ITStep, IModel)
class TStep(tr.HasStrictTraits):
    '''Manage the data and metadata of a time step within an interation loop.
    '''
    title = tr.Str('<unnamed>')

    tloop_type = tr.Type(ITLoop)
    '''Type of time loop to be used with the model
    '''

    #=========================================================================
    # HISTORY
    #=========================================================================
    hist_type = tr.Type(Hist)

    hist = tr.Property(tr.Instance(IHist))
    r'''History representation of the model response.
    '''
    @tr.cached_property
    def _get_hist(self):
        return self.hist_type(tstep_source=self)

    debug = tr.Bool(False)

    t_n1 = tr.Float(0.0, auto_set=False, enter_set=True)
    '''Target value of the control variable.
    '''
    U_n = tr.Float(0.0, auto_set=False, enter_set=True)
    '''Current fundamental value of the primary variable.
    '''
    U_k = tr.Float(0.0, auto_set=False, enter_set=True)
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

    _corr_pred = tr.Property(depends_on='U_k,t_n1')

    @tr.cached_property
    def _get__corr_pred(self):
        return self.get_corr_pred(self.U_k, self.t_n1)

    R = tr.Property

    def _get_R(self):
        R, _ = self._corr_pred
        return R

    dR = tr.Property

    def _get_dR(self):
        _, dR = self._corr_pred
        return dR

    R_norm = tr.Property

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

    sim = tr.Property()
    '''Launch a simulator - currently only one simulator is allowed
    for a model. Mutiple might also make sense when different solvers
    are to be compared. The simulator pulls the time loop type
    from the model.
    '''
    @tr.cached_property
    def _get_sim(self):
        return Simulator(tstep=self)
