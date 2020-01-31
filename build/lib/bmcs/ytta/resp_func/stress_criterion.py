""" defining the pullout model as a traited class """

from math import e

from numpy import sqrt, hstack, linspace, infty, array, linspace, tanh, argmax, \
    where
from traits.api import HasTraits, Instance, on_trait_change, \
    Property, DelegatesTo, Tuple, Array, Bool, Any
from traitsui.api import View, Item, VGroup
from traitsui.menu import OKButton
from util.traits.either_type import EitherType

from .parameters import Geometry
from .parameters import Material
from .pull_out import PullOut


# from stress_criterion_bond import StressCriterionBond
class StressCriterion(PullOut):

    param_names = ['Ef', 'k', 'qf', 'qy', 'f', 'fu']

    material = Instance(Material)

    def _material_default(self):
        return Material()

    def get_P(self, a):
        return (self.qf * a + self.qy / self.w * self.get_clamp(a))

#    bond = Instance( StressCriterionBond )
#    def _bond_default( self ):
#        return StressCriterionBond( material=self.material )

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
        id='stress_criterion.physics',
    ),
        VGroup(
        Item('qf'),
        Item('k'),
        Item('qy'),
        #                                            Item( '@bond', show_label = False ),
        Item('tau', style='readonly'),
        label='bond law',
        id='stress_criterion.bond',
    ),
        id='stress_criterion.vgroup',
        dock='tab'
    ),
        resizable=True,
        kind='live',
        height=0.8,
        width=0.8,
        buttons=[OKButton],
        dock='tab',
        id='stress_criterion.view'
    )


if __name__ == "__main__":
    pullout_fn = StressCriterion()
    pullout_fn.get_value()
    pullout_fn.configure_traits()
