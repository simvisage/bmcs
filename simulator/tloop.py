'''
'''

import traits.api as tr

from .i_tloop import ITLoop
from .i_tstep import ITStep
from .tline_mixin import TLineMixIn


@tr.provides(ITLoop)
class TLoop(TLineMixIn):
    '''The time loop serves as the base class for application time loops.
    That can be interrupted paused, resumed or restarted.

    The implementation of the loop must contain the break criterion

    while True:
        if self.restart or self.paused:
            break    
        #calculation

    '''

    tstep = tr.WeakRef(ITStep)

    sim = tr.Property

    def _get_sim(self):
        return self.tstep.sim

    hist = tr.Property

    def _get_hist(self):
        return self.tstep.hist

    paused = tr.Bool(False)

    restart = tr.Bool(True)

    user_wants_abort = tr.Property

    def _get_user_wants_abort(self):
        return self.restart or self.paused

    def init(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = self.tline.min
            self.tstep.init_state()
            self.hist.init_state()
            self.restart = False

    def eval(self):
        '''This method is called by the tloop_thread.
        '''
        raise NotImplementedError

    def __call__(self):
        self.init()
        return self.eval()
