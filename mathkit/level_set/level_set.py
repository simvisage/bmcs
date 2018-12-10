
from math import sin

from traits.api import \
    HasTraits, Float, provides, Interface


class ILevelSetFn(Interface):
    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        raise NotImplementedError


@provides(ILevelSetFn)
class SinLSF(HasTraits):
    a = Float(1.5, enter_set=True, auto_set=False)
    b = Float(2.0, enter_set=True, auto_set=False)

    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return y - (sin(self.b * x) + self.a)


@provides(ILevelSetFn)
class PlaneLSF(HasTraits):
    a = Float(.5, enter_set=True, auto_set=False)
    b = Float(2.0, enter_set=True, auto_set=False)
    c = Float(-2.5, enter_set=True, auto_set=False)

    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return self.a * x + self.b * y + self.c


@provides(ILevelSetFn)
class ElipseLSF(HasTraits):
    a = Float(.5, enter_set=True, auto_set=False)
    b = Float(2.0, enter_set=True, auto_set=False)
    c = Float(-2.5, enter_set=True, auto_set=False)

    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return self.a * x * x + self.b * y * y - self.c
