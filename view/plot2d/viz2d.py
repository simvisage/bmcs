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
    '''Visualization interface
    '''
    label = Str('<unnamed>')
    vis2d = WeakRef

    def plot(self, ax, vot=0):
        self.vis2d.plot(ax, vot)

    view = View(
        HSplit(
            VGroup(
                UItem('label'),
                label='Vizualization inteerface',
                springy=True
            ),
            Group(
                Item('vis2d@'),
                label='Visualized object',
                springy=True
            ),
        ),
        resizable=True
    )
