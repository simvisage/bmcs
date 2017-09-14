'''
Created on Dec 3, 2015

@author: rch
'''

from os.path import join

from reporter import ROutputItem
from traits.api import \
    HasStrictTraits, Dict, Property, Float, \
    WeakRef, DelegatesTo, cached_property, \
    Str, List, Button
from traitsui.api import \
    View, Group, UItem, Include, EnumEditor, HGroup, \
    HSplit, Item, VGroup

import matplotlib.pyplot as plt


class Viz2D(ROutputItem):
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

    def plot_tex(self, ax, vot, *args, **kw):
        self.plot(ax, vot, *args, **kw)

    def write_figure(self, f, rdir, rel_study_path):
        fname = 'fig_' + self.name.replace(' ', '_') + '.pdf'
        f.write(r'''
\includegraphics[width=7.5cm]{%s}
''' % join(rel_study_path, fname))
        self.savefig(join(rdir, fname))

    def savefig_animate(self, vot, fname, *args, **kw):
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        self.plot_tex(ax, vot)
        fig.tight_layout()
        fig.savefig(fname, *args, **kw)

    def savefig(self, fname):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_tex(ax, 0.25)
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
