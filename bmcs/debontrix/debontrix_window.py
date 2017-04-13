'''
Created on Dec 20, 2016


@author: rch
'''

from matplotlib.figure import \
    Figure
from traits.api import \
    HasStrictTraits, Instance, Button, Event, Range
from traitsui.api import \
    View, Item, Group, \
    HSplit, HGroup
from util.traits.editors.mpl_figure_editor import \
    MPLFigureEditor

from debontrix import Bondix as DebonTrixModel, FETS1D52L4ULRH


class DebonTrixWindow(HasStrictTraits):

    '''View object for a cross section state.
    '''
    model = Instance(DebonTrixModel)

    w = Range(low=0.0, high=0.1, value=0.0,
              enter_set=True, auto_set=False)

    def _w_changed(self):
        self.model.w = self.w
        self.replot = True

    G = Range(low=0.1, high=100, value=0.1,
              enter_set=True, auto_set=False)

    def _G_changed(self):
        self.model.G = self.G
        self.replot = True

    eta = Range(low=0.01, high=100, value=0.1,
                enter_set=True, auto_set=False)

    def _eta_changed(self):
        self.model.eta = self.eta
        self.replot = True

    n_E = Range(low=3, high=100, value=10,
                enter_set=True, auto_set=False)

    def _n_E_changed(self):
        self.model.n_E = self.n_E
        self.replot = True

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure()
        return figure

    data_changed = Event

    replot = Button

    def _replot_fired(self):
        self.figure.clear()
        self.model.plot(self.figure)
        self.data_changed = True

    clear = Button()

    def _clear_fired(self):
        self.figure.clear()
        self.data_changed = True

    view = View(HSplit(Group(Item('model',
                                  resizable=True,
                                  show_label=False,
                                  width=400,
                                  height=200),
                             ),
                       Group(HGroup(Item('replot', show_label=False),
                                    Item('clear', show_label=False)
                                    ),
                             Item('figure', editor=MPLFigureEditor(),
                                  resizable=True, show_label=False,
                                  springy=True),
                             Item('n_E'),
                             Item('G'),
                             Item('eta'),
                             Item('w'),
                             # Item('self.root.time', label='t/T_max'),
                             label='plot sheet',
                             dock='tab',
                             )
                       ),
                id='bmcstreeview_id',
                width=0.9,
                height=0.8,
                title='Debontrix BMCS App',
                resizable=True,
                )

if __name__ == '__main__':

    tv = DebonTrixWindow(model=DebonTrixModel())
    tv.configure_traits()
