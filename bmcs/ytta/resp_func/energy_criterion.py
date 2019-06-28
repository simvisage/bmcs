""" defining the pullout model as a traited class """

from math import e

from numpy import sqrt, hstack, linspace, infty, array, linspace, tanh, argmax, \
    where, sign
from traits.api import Property, HasTraits, Instance, Bool, \
    on_trait_change, Tuple, Array
from traitsui.api import View, Item, VGroup
from traitsui.menu import OKButton
from util.traits.either_type import EitherType

from .pull_out import PullOut


class EnergyCriterion(PullOut):

    param_names = ['Ef', 'rf', 'k', 'qf', 'G', 'L', 'l', 'phi', 'f', 'fu']

    def get_P(self, a):
        ''' Pu assures displacement continuity but violates the constitutive relation q = kU '''
        Pu_deb = self.qf * a + (self.qf / 2. / self.w +
                                sqrt((self.qf / 2. / self.G) ** 2 + 2. * self.Ef * self.Af * self.p * self.G)) * \
            self.get_clamp(a)
        ''' Pq assures the constitutive relation q = kU but violates the displacement continuity '''
        Pq_deb = self.qf * a + (self.qf / self.w + sqrt(2. * self.Ef * self.Af * self.p * self.G)) * \
            self.get_clamp(a)
        return Pq_deb

    bool_infinite = Bool(False)
    bool_finite = Bool(False)
    bool_clamp = Bool(False)

    @on_trait_change('event_infinite, event_finite, event_clamp')
    def get_value(self):
        if self.bool_infinite == True:
            return self.prepare_infinite()
        elif self.bool_finite == True:
            return self.prepare_finite()
        elif self.bool_clamp == True:
            return self.prepare_clamp()

    traits_view = View(VGroup(VGroup(
        Item('material_choice', label='material'),
        Item('Ef'),
        Item('f'),
        Item('beta'),
        Item('fu'),
        Item('Pu', style='readonly'),
        Item('include_fu', label='fu on/off'),
        Item('yvalues'),
        label='physical parameters',
        id='energy_criterion.physics',
    ),
        VGroup(
        Item('G'),
        Item('k'),
        Item('qf'),
        Item('tau', style='readonly'),
        label='bond parameters',
        id='energy_criterion.bond_law',
    ),
        id='energy_criterion.vgroup',
        dock='tab'
    ),
        resizable=True,
        kind='live',
        height=0.8,
        width=0.8,
        dock='tab',
        id='energy_criterion.view'
    )


if __name__ == "__main__":
    pullout_fn = EnergyCriterion()
    pullout_fn.get_value()
