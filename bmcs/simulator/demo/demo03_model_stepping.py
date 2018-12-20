'''
The model interface is changed such that it makes the stepping. 
on its own. The time loop could thus be simplified and generalized.
As a result applications can only inherit from model.

@author: rch
'''

import time

from traits.api import \
    Int, Float, Property, cached_property
import traits.has_traits
traits.has_traits.CHECK_INTERFACES = 2
from bmcs.simulator import \
    Simulator, TLoop, Model
import numpy as np


class DemoNRTLoop(TLoop):
    '''Demonstration loop running from min to max with a defined step.
    '''

    k_max = Int(30, enter_set=True, auto_set=False)

    acc = Float(1e-4, enter_set=True, auto_set=False)

    def eval(self):
        t_n = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step
        while t_n < t_max:
            print('\ttime: %g' % t_n, end='')
            k = 0
            # run the iteration loop
            while (k < self.k_max) and not self.user_wants_abort:
                if self.model.R_norm < self.acc:
                    print('\titer: %g' % k)
                    break
                self.model.trial_step()
                k += 1
            else:  # handle unfinished iteration loop
                if k >= self.k_max:  # add step size reduction
                    # no success abort the simulation
                    self.restart = True
                    print('Warning: '
                          'convergence not reached in %g iterations', k)
                return
            # accept the time step
            self.model.update_state(t_n)
            self.hist.record_timestep(t_n)
            # time line launches notifiers to announce a new step to
            # subscribers
            t_n += dt
            self.tline.val = t_n
        return


class DemoQuadFNModel(Model):
    tloop_type = DemoNRTLoop

    R_0 = Float(1.0, auto_set=False, enter_set=True)
    '''Target value of a function (load).
    '''
    U_n = Float(0.0, auto_set=False, enter_set=True)
    '''Current fundamental value of the primary variable.
    '''
    t_n = Float(0.0, auto_set=False, enter_set=True)
    '''Current value of the control variable.
    '''
    U_k = Float(0.0, auto_set=False, enter_set=True)
    '''Current trial value of the primary variable.
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

    def init_state(self):
        '''Initialize state.
        '''
        self.U_n = 0.0
        self.t_n = 0.0
        self.U_k = 0.0

    def update_state(self, t_n):
        '''Update the control, primary and state variables..
        '''
        self.t_n = t_n
        self.U_n = self.U_k

    def record_state(self):
        '''Provide the current state for history recording.
        '''
        pass

    _corr_pred = Property(depends_on='U_k,t_n')

    @cached_property
    def _get__corr_pred(self):
        return self.get_corr_pred(self.U_k, self.t_n)

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

    def trial_step(self):
        d_U = self.R / self.dR
        self.U_k += d_U


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
