'''

This script is used to demonstrate the states of a model.

@author: rch
'''

import time

from simulator.api import \
    Simulator, TLoop, TStep
import traits.has_traits

from .interaction_scripts import run_rerun_test


traits.has_traits.CHECK_INTERFACES = 2


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
        while t_n < t_max:
            print('\ttime %g' % t_n)
            if self.user_wants_abort:
                break
            time.sleep(1)
            t_n += dt
            self.hist.record_timestep(t_n, None, None)
            self.tline.val = t_n
        return


class DemoStatesModel(TStep):
    tloop_type = DemoExplicitTLoop


# Construct a Simulator
m = DemoStatesModel()
s = m.sim
run_rerun_test(s)
