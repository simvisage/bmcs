'''

This script is used to demonstrate the states of a model.

@author: rch
'''

import time

import traits.has_traits
traits.has_traits.CHECK_INTERFACES = 2

from bmcs.simulator import \
    Simulator, TLoop, Model


class DemoExplicitTLoop(TLoop):
    '''Demonstration loop running explicitly evaluating
    a function and saving the results in the history.
    This kind of time loop is used e.g. for the explicit
    calculation of stress strain curve with all components
    prescribed.
    '''

    def eval(self):
        t_n = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step
#        U_k, t_nn = self.model.get_state()
#        assert(np.fabs(t_n - t_nn) < 1e-5)  # implementation verification
        while t_n < t_max:
            print('\ttime %g' % t_n)
            if self.user_wants_abort:
                break
            time.sleep(1)
            t_n += dt
#            self.model.update_state(U_k, t_n)
            self.hist.record_timestep(t_n)
            self.tline.val = t_n
        return


class DemoStatesModel(Model):
    tloop_type = DemoExplicitTLoop


# Construct a Simulator
m = DemoStatesModel()
s = Simulator(model=m)
print(s.tloop)
# Start calculation in a thread
print('RUN the calculation thread from t = 0.0')
s.run()
print('WAIT in main thread for 3 secs')
time.sleep(3)
print('PAUSE the calculation thread')
s.pause()
print('PAUSED wait 1 sec')
time.sleep(1)
print('RESUME the calculation thread')
s.run()
print('WAIT in the main thread for 3 secs again')
time.sleep(3)
print('STOP the calculation thread')
s.stop()
print('RUN a new calculation thread from t = 0.0')
s.run()
print('JOIN the calculation thread into main thread to end simultaneously')
s.join()
print('END all threads')
