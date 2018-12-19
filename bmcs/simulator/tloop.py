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
        raise NotImplementedError

    def __call__(self):
        return self.eval()