
import time

from simulator.api import \
    TStep, Simulator
from simulator.tloop_implicit import TLoopImplicit
from traits.api import \
    Float
import traits.has_traits

import numpy as np

from .interaction_scripts import run_rerun_test


traits.has_traits.CHECK_INTERFACES = 2


class DemoQuadFNModel(TStep):
    '''Model implementing both the value of the residual of the governing
    equations and of their derivatives to be plugged into the implicit
    time stepping algorithm.
    '''
    tloop_type = TLoopImplicit

    R_0 = Float(1.0, auto_set=False, enter_set=True)
    '''Target value of a function (load).
    '''

    def get_corr_pred(self, U_k, t_n1):
        '''Return the value and the derivative of a function
        '''
        R = U_k * U_k - (self.R_0 * t_n1)
        dR = 2 * U_k
        # handle the case of zero derivative - return small number
        dR = max(np.fabs(dR), 1e-3)
        # sleep a bit to make the calculation slower - sleep for 100 msecs
        # to demonstrate the pause-resume, stop-restart functionality
        time.sleep(0.01)
        return R, dR


# Construct a Simulator
m = DemoQuadFNModel()
s = m.sim
run_rerun_test(s)