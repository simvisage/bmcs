'''
'''

from traits.api import \
    HasStrictTraits,\
    Bool, WeakRef, cached_property,\
    Property, DelegatesTo, Instance

from .hist import Hist
from .i_tstep import ITStep
from .tline import TLine


class TLoop(HasStrictTraits):
    '''The time loop serves as the base class for application time loops.
    That can be interrupted paused, resumed or restarted.

    The implementation of the loop must contain the break criterion

    while True:
        if self.restart or self.paused:
            break    
        #calculation

    '''
    sim = WeakRef

    tline = WeakRef(TLine)

    tstep = WeakRef(ITStep)

    model = DelegatesTo('tstep')

    hist = WeakRef(Hist)

    paused = Bool(False)

    restart = Bool(True)

    user_wants_abort = Property()

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
