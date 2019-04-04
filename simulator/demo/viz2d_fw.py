'''
Created on Apr 3, 2019

@author: rch
'''
from ibvpy.api import IBCond
from traits.api import \
    Bool, \
    Button
from traits.api import \
    Tuple, HasStrictTraits, Array, WeakRef
from traitsui.api import \
    View, Item, Group
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from view.plot2d import Viz2D, Vis2D

import numpy as np


class DataSheet(HasStrictTraits):

    data = Array(np.float_)

    view = View(
        Item('data',
             show_label=False,
             resizable=True,
             editor=ArrayViewEditor(titles=['x', 'y', 'z'],
                                    format='%.4f',
                                    # Font fails with wx in OSX;
                                    #   see traitsui issue #13:
                                    # font   = 'Arial 8'
                                    )
             ),
        width=0.5,
        height=0.6
    )


class Vis2DFW(Vis2D):

    Pw = Tuple()
    bc = WeakRef(IBCond)

    def _Pw_default(self):
        return ([0], [0])

    def update(self):
        sim = self.sim
        dofs = self.bc.dofs
        U_ti = sim.hist.U_t
        F_ti = sim.hist.F_t
        P = np.sum(F_ti[:, dofs], axis=1)
        w = np.average(U_ti[:, dofs], axis=1)
        self.Pw = P, w


class Viz2DFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    show_legend = Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        sim = self.vis2d.sim
        P_t, u_t = sim.hist['Pw'].Pw
        ymin, ymax = np.min(P_t), np.max(P_t)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(u_t), np.max(u_t)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        ax.plot(u_t, P_t, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('pull-out force P [N]')
        ax.set_xlabel('pull-out displacement [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        sim = self.vis2d.sim
        P_t, u_t = sim.hist['Pw'].Pw
        idx = sim.hist.get_time_idx(vot)
        P, w = P_t[idx], u_t[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)

    show_data = Button()

    def _show_data_fired(self):
        sim = self.vis2d.sim
        P_t, u_t = sim.hist['Pw'].Pw
        data = np.vstack([u_t, P_t]).T
        show_data = DataSheet(data=data)
        show_data.edit_traits()

    def plot_tex(self, ax, vot, *args, **kw):
        self.plot(ax, vot, *args, **kw)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
        Item('show_data')
    )
