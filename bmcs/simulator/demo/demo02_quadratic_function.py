'''
This script is used to demonstrate the machinery
of the simulation framework on the example a simple
quadratic function. 

@author: rch
'''

import time

from traits.api import \
    Int, Float
import traits.has_traits

from bmcs.simulator import \
    Simulator, TLoop, Model
import numpy as np

from .interaction_scripts import run_rerun_test
traits.has_traits.CHECK_INTERFACES = 2


class DemoNRTLoop(TLoop):
    '''Demonstration loop running from min to max with a defined step.
    '''

    k_max = Int(30, enter_set=True, auto_set=False)

    def eval(self):
        t_n1 = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step
        U_k = self.model.get_state()
        while t_n1 < t_max:
            print('\ttime: %g' % t_n1, end='')
            k = 0
            # run the iteration loop
            while (k < self.k_max) and not self.user_wants_abort:
                R, dR = self.model.get_corr_pred(U_k, t_n1)
                if np.sqrt(R * R) < 1e-5:
                    print('\titer: %g' % k)
                    break
                dU = R / dR
                U_k += dU
                k += 1
            else:  # handle unfinished iteration loop
                if k >= self.k_max:  # add step size reduction
                    # no success abort the simulation
                    self.restart = True
                    print('Warning: '
                          'convergence not reached in %g iterations', k)
                return
            # accept the time step
            self.model.update_state(U_k, t_n1)
            self.hist.record_timestep(t_n1, U_k)
            # tline launches notifiers to announce a new step to subscribers
            self.tline.val = t_n1
            # set a new target time
            t_n1 += dt
        return


class DemoQuadFNModel(Model):
    tloop_type = DemoNRTLoop

    R0 = Float(1.0, auto_set=False, enter_set=True)
    '''Target value of a function (load).
    '''
    U_n = Float(0.0, auto_set=False, enter_set=True)
    '''Current value of the primary variable.
    '''

    t_n = Float(0.0, auto_set=False, enter_set=True)
    '''Current value of the control variable.
    '''

    def get_corr_pred(self, U_k, t_n1):
        '''Return the value and the derivative of a function
        '''
        R = U_k * U_k - (self.R0 * t_n1)
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

    def get_state(self):
        '''Get the current control and primary variables.
        '''
        return self.U_n

    def update_state(self, U_k, t_n):
        '''Update the control, primary and state variables..
        '''
        self.t_n = t_n
        self.U_n = U_k

    def record_state(self):
        '''Provide the current state for history recording.
        '''
        pass


# Construct a Simulator
m = DemoQuadFNModel()
s = Simulator(model=m)
print(s.tloop)
run_rerun_test(s)
