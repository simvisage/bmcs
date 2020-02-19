
import time

from traits.api import Instance,  HasStrictTraits,\
    Bool
import numpy as np
from view.examples.demo_model.tline import TLine


class TimeLoop(HasStrictTraits):

    tline = Instance(TLine)

    paused = Bool(False)
    restart = Bool(True)

    def init_loop(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = 0
            self.restart = False

    def eval(self):
        '''this method is just called by the tloop_thread'''

        self.init_loop()
        t_min = self.tline.val
        t_max = self.tline.max
        n_steps = 5
        tarray = np.linspace(t_min, t_max, n_steps)
        for idx, t in enumerate(tarray):
            if self.restart or self.paused:
                break
            time.sleep(1)
            self.tline.val = t
