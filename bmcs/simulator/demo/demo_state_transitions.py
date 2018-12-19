'''

This script is used to demonstrate the states of a model.

...todo:: Handle the END event of the simulator in case 
that the calculation thread is still running. In such a case,
the request should be confirmed by the user action.

@author: rch
'''

import time

from bmcs.simulator import \
    Simulator, TLoop, Model
import numpy as np


class DemoStatesTLoop(TLoop):
    '''Demonstration loop running with equidistant steps
    from min to max with a defined step.
    '''

    def eval(self):
        self.init()
        t_min = self.tline.val
        t_max = self.tline.max
        t_step = self.tline.step
        n_steps = (t_max - t_min) / t_step
        tarray = np.linspace(t_min, t_max, n_steps + 1)
        for t in tarray:
            print('\ttime %g' % t)
            if self.restart or self.paused:
                break
            time.sleep(1)
            self.hist.record_timestep(t)
            self.tline.val = t
        return


class DemoStatesModel(Model):
    tloop_type = DemoStatesTLoop


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
