'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, Dict, Property, Float, \
    WeakRef, DelegatesTo, cached_property, \
    Str, List, Button

from traitsui.api import \
    View, Group, UItem, Include, EnumEditor, HGroup, \
    HSplit, Item, VGroup


class Viz2D(HasStrictTraits):
    '''Base class of the visualization adaptors
    '''
    name = Str('<unnamed>')
    label = Property(depends_on='label')

    @cached_property
    def _get_label(self):
        return self.name
    vis2d = WeakRef

    def plot(self, ax, vot=0):
        self.vis2d.plot(ax, vot)

    def reset(self, ax):
        pass

    view = View(
        HSplit(
            VGroup(
                UItem('label'),
                label='Vizualization inteerface',
                springy=True
            )),
        resizable=True
    )
