'''
Created on Feb 9, 2010

@author: rostislav
'''

from numpy import infty, sign, pi, cos
from traits.api import HasTraits, Float, Property, cached_property, \
    Instance, List, on_trait_change, Int, Tuple, Bool, DelegatesTo, Event, Enum, \
    provides, Str, Interface, Button
from traitsui.api import View, Item, Tabbed, VGroup, HGroup, \
    ModelView, HSplit, VSplit, Group, Include, InstanceEditor
from traitsui.menu import Action, CloseAction, HelpAction, Menu, \
    MenuBar, NoButtons, Separator, ToolBar, OKButton

from util.traits.either_type import EitherType

from .parameters import Geometry
from .parameters import Plot


def Heaviside(x):
    return sign(sign(x) + 1)


class IBoundary(Interface):
    """
    Abstract class representing the single boundary condition.
    """
    pass


class BaseBC(HasTraits):

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    plot = Instance(Plot)

    def _plot_default(self):
        return Plot()

    type = Instance(HasTraits)
    BC = Str
    L = DelegatesTo('geometry')
    l = DelegatesTo('geometry')
    z = DelegatesTo('geometry')
    phi = DelegatesTo('geometry')
    Lf = DelegatesTo('geometry')
    rf = DelegatesTo('geometry')
    Af = DelegatesTo('geometry')
    lambd = DelegatesTo('geometry')
    theta = DelegatesTo('geometry')

    u_plot = DelegatesTo('plot')
    w_plot = DelegatesTo('plot')

    traits_view = View(Item('@type', show_label=False),
                       scrollable=False,
                       resizable=True)


class InfFreeOne(BaseBC):

    BC = 'one-sided pull-out with infinite embedded length'
    traits_view = View(Item('u_plot'),
                       Item('l'),
                       Item('rf'),
                       Item('Af', style='readonly'),
                       Item('phi'), )


class InfFreeDbl(BaseBC):

    BC = 'double-sided pull-out with infinite embedded length'
    traits_view = View(Item('w_plot'),
                       Item('rf'),
                       Item('Af', style='readonly'),
                       Item('phi'), )


@provides(IBoundary)
class InfiniteEmbeddedLength(BaseBC):

    def __init__(self, **kw):
        super(InfiniteEmbeddedLength, self).__init__(**kw)
        self.L = infty

    type = EitherType(names=['double sided',
                             'one sided'],
                      klasses=[InfFreeDbl,
                               InfFreeOne])


class FinFreeOne(BaseBC):

    BC = 'one-sided pull-out with finite embedded length'
    traits_view = View(Item('l'),
                       Item('L'),
                       Item('rf'),
                       Item('Af', style='readonly'),
                       Item('phi'), )


class FinFreeDbl(BaseBC):

    BC = 'double-sided pull-out with finite embedded length'
    Le = Property(depends_on='Lf, z, phi', label='Le', desc='embedded length')

    def _get_Le(self):
        return (self.Lf / 2. - self.z / cos(self.phi)) * \
            Heaviside(self.Lf / 2. - self.z / cos(self.phi))

    traits_view = View(Item('Lf'),
                       Item('z'),
                       Item('Le', style='readonly'),
                       Item('rf'),
                       Item('Af', style='readonly'),
                       Item('phi'), )


@provides(IBoundary)
class FiniteEmbeddedLength(BaseBC):

    type = EitherType(names=['double sided',
                             'one sided'],
                      klasses=[FinFreeDbl,
                               FinFreeOne])


class FinClampOne(BaseBC):

    BC = 'one-sided pull-out with clamped fibre end'
    Le = Property(depends_on='Lf, z, phi', label='Le', desc='embedded length')

    def _get_Le(self):
        return (self.Lf / 2. - self.z / cos(self.phi)) * \
            Heaviside(self.Lf / 2. - self.z / cos(self.phi))
    l = Property(depends_on='Lf, z, phi', label='l', desc='free length')

    traits_view = View(Item('u_plot'),
                       Item('Lf'),
                       Item('z'),
                       Item('Le', style='readonly'),
                       Item('rf'),
                       Item('Af', style='readonly'),
                       Item('phi'), )


class FinClampDbl(BaseBC):

    BC = 'double-sided pull-out with clamped fibre end'
    Le = Property(depends_on='Lf, z, phi', label='Le', desc='embedded length')

    def _get_Le(self):
        return (self.Lf / 2. - self.z / cos(self.phi)) * \
            Heaviside(self.Lf / 2. - self.z / cos(self.phi))
    traits_view = View(Item('w_plot'),
                       Item('Lf'),
                       Item('z'),
                       Item('Le', style='readonly'),
                       Item('rf'),
                       Item('Af', style='readonly'),
                       Item('phi'), )


@provides(IBoundary)
class ClampedFibre(BaseBC):

    type = EitherType(names=['one sided',
                             'double sided'],
                      klasses=[FinClampOne,
                               FinClampDbl])


if __name__ == '__main__':
    boundary1 = FiniteEmbeddedLength()
    boundary2 = InfiniteEmbeddedLength()
    boundary3 = ClampedFibre()
    boundary1.configure_traits()
    boundary2.configure_traits()
    boundary3.configure_traits()
