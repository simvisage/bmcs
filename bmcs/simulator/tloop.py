'''
'''

import time
from traits.api import \
    HasStrictTraits,\
    Bool, provides, WeakRef

import numpy as np

from .i_hist import IHist
from .i_model import IModel
from .i_tloop import ITLoop
from .tline import TLine


@provides(ITLoop)
class TLoop(HasStrictTraits):
    '''Handle the time loop with interactive state management.

    The time loop serves as the base class for application time loops.
    That can be interrupted paused, resumed or restarted.

    The implementation of the loop must contain the break criterion

    while True:
        if self.restart or self.paused:
            break    
        #calculation

    '''
    tline = WeakRef(TLine)

    model = WeakRef(IModel)

    hist = WeakRef(IHist)

    paused = Bool(False)

    restart = Bool(True)

    def init(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = 0
            self.restart = False

    def eval(self):
        '''This method is called by the tloop_thread.
        '''
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
            self.tline.val = t
        return

    def __call__(self):
        return self.eval()