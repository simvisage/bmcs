'''
This script is used to demonstrate the machinery
of the simulation framework on the example a simple
quadratic function. 

@author: rch
'''

import time
from traits.api import \
    Int, Float
from bmcs.simulator import \
    Simulator, TLoop, Model
import numpy as np


class DemoNRTLoop(TLoop):
    '''Demonstration loop running with equidistant steps
    from min to max with a defined step.
    '''

    k_max = Int(30)

    def eval(self):
        self.init()
        interrupt = False
        t_n = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step
        U_k = self.model.init_state()
        while t_n < t_max:
            print('\ttime: %g' % t_n, end='')
            k = 0
            while (k < self.k_max):
                if self.restart or self.paused:
                    interrupt = True
                R, dR = self.model.get_corr_pred(U_k, t_n)
                if np.sqrt(R * R) < 1e-5:
                    print('\titer: %g' % k)
                    break
                dU = R / dR
                U_k += dU
                k += 1
            if interrupt:
                break
            t_n += dt
            self.model.update_state(U_k, t_n)
            self.hist.record_timestep(t_n)
            self.tline.val = t_n
        return


class DemoQuadFNModel(Model):
    tloop_type = DemoNRTLoop

    R0 = Float(1.0, auto_set=False, enter_set=True)

    U_n = Float(0.0, auto_set=False, enter_set=True)

    t_n = Float(0.0, auto_set=False, enter_set=True)

    def get_corr_pred(self, U_k, t_n):
        '''Return the value and the derivative of a function
        '''
        R = U_k * U_k - (self.R0 * t_n)
        dR = 2 * U_k
        time.sleep(0.1)
        return R, max(np.fabs(dR), 1.e-3)

    def init_state(self):
        '''Initialize state.
        '''
        return self.U_n

    def update_state(self, U_k, t_n):
        '''Record state in history.
        '''
        t_n = t_n
        self.U_n = U_k

    def record_state(self):
        pass


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
