
import time

from traits.api import \
    Float, Bool
import traits.has_traits
from bmcs.simulator import \
    Model, Simulator
import numpy as np

from .tloop_implicit import TLoopImplicit
traits.has_traits.CHECK_INTERFACES = 2


class DemoQuadFNModel(Model):
    '''Model implementing both the value of the residual of the governing
    equations and of their derivatives to be plugged into the implicit
    time stepping algorithm.
    '''
    tloop_type = TLoopImplicit

    R_0 = Float(1.0, auto_set=False, enter_set=True)
    '''Target value of a function (load).
    '''

    def get_corr_pred(self, U_k, t_n):
        '''Return the value and the derivative of a function
        '''
        R = U_k * U_k - (self.R_0 * t_n)
        dR = 2 * U_k
        # handle the case of zero derivative - return small number
        dR = max(np.fabs(dR), 1e-3)
        # sleep a bit to make the calculation slower - sleep for 100 msecs
        # to demonstrate the pause-resume, stop-restart functionality
        time.sleep(0.1)
        return R, dR


# Construct a Simulator
m = DemoQuadFNModel()
s = Simulator(model=m)
print(s.tloop)
# Start calculation in a thread
print('\nRUN the calculation thread from t = 0.0')
s.run()
print('\nWAIT in main thread for 3 secs')
time.sleep(3)
print('\nPAUSE the calculation thread')
s.pause()
print('\nPAUSED wait 1 sec')
time.sleep(1)
print('\nRESUME the calculation thread')
s.run()
print('\nWAIT in the main thread for 3 secs again')
time.sleep(3)
print('\nSTOP the calculation thread')
s.stop()
print('\nRUN a new calculation thread from t = 0.0')
s.run()
print('\nJOIN the calculation thread into main thread to end simultaneously')
s.join()
print('END all threads')
