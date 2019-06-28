'''
Created on Dec 3, 2015

@author: rch
'''

from os.path import join

from traits.api import \
    HasStrictTraits, Dict, Property, Float, \
    Bool, WeakRef, DelegatesTo, cached_property, \
    Str, List, Button
from traitsui.api import \
    View, Group, UItem, Include, EnumEditor, HGroup, \
    HSplit, Item, VGroup

import matplotlib.pyplot as plt
from reporter import ROutputItem


class Viz2D(ROutputItem):
    '''Base class of the visualization adaptors
    '''

    viz_sheet = WeakRef

    name = Str('<unnamed>')
    label = Property(depends_on='label')
    visible = Bool(True)

    @cached_property
    def _get_label(self):
        return self.name
    vis2d = WeakRef

    def reset(self, ax):
        pass

    def clear(self):
        pass

    def plot(self, ax, vot=0):
        raise NotImplemented

    def plot_tex(self, ax, vot, *args, **kw):
        raise NotImplemented

    def write_figure(self, f, rdir, rel_study_path):
        fname = 'fig_' + self.name.replace(' ', '_') + '.pdf'
        f.write(r'''
\includegraphics[width=7.5cm]{%s}
''' % join(rel_study_path, fname))
        self.savefig(join(rdir, fname))

    def savefig_animate(self, vot, fname, figsize=(9, 6)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.plot(ax, vot)
        fig.tight_layout()
        fig.savefig(fname)

    def savefig(self, fname):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_tex(ax, 0.25)
        fig.savefig(fname)

    trait_view = View(
        HSplit(
            VGroup(
                UItem('label'),
                Item('visible'),
                label='Vizualization interface',
                springy=True
            )),
        resizable=True
    )
