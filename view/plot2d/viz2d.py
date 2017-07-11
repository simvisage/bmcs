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

import matplotlib.pyplot as plt
from reporter import ROutputRecord


class Viz2D(ROutputRecord):
    '''Base class of the visualization adaptors
    '''

    viz_sheet = WeakRef

    name = Str('<unnamed>')
    label = Property(depends_on='label')

    @cached_property
    def _get_label(self):
        return self.name
    vis2d = WeakRef

    def plot(self, ax, vot=0):
        self.vis2d.plot(ax, vot)

    def write_figure(self, f, fname):
        f.write(r'''
\includegraphics[width=7.5cm]{%s}
''' % fname)
        self.savefig(fname)

    def savefig(self, fname):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot(ax, 0.25)
        fig.savefig(fname)

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
