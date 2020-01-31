'''
Created on Apr 3, 2019

@author: rch
'''
from ibvpy.api import IBCond
from traits.api import \
    Bool, \
    Button, Str
from traits.api import \
    Tuple, HasStrictTraits, Array, WeakRef
from traitsui.api import \
    View, Item
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
    bc_right = Str
    bc_left = Str

    def _Pw_default(self):
        return ([0], [0], [0], [0])

    def update(self):
        sim = self.sim
        bc_right = sim.trait_get(self.bc_right)[self.bc_right]
        bc_left = sim.trait_get(self.bc_left)[self.bc_left]
        dofs_right = np.unique(bc_right.dofs)
        dofs_left = np.unique(bc_left.dofs)
        U_ti = sim.hist.U_t
        F_ti = sim.hist.F_t
        P = np.sum(F_ti[:, dofs_right], axis=1)
        P0 = np.sum(F_ti[:, dofs_left], axis=1)
        w = np.average(U_ti[:, dofs_right], axis=1)
        w0 = np.average(U_ti[:, dofs_left], axis=1)
        self.Pw = P, P0, w, w0


class Viz2DFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    show_legend = Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        sim = self.vis2d.sim
        P_t, P0_t, w_t, w0_t = sim.hist['Pw'].Pw
        ymin, ymax = np.min(P_t), np.max(P_t)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(w_t), np.max(w_t)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        ax.plot(w_t, P_t, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
        ax.plot(w0_t, P_t, linewidth=2, color='orange', alpha=0.4,
                label='P(w;x=0)')
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('pull-out force P [N]')
        ax.set_xlabel('pull-out displacement [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        sim = self.vis2d.sim
        P_t, P0_t, w_t, w0_t = sim.hist['Pw'].Pw
        idx = sim.hist.get_time_idx(vot)
        P, P0, w, w0 = P_t[idx], P0_t[idx], w_t[idx], w0_t[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)
        ax.plot([w0], [P], 'o', color='orange', markersize=10)

    show_data = Button()

    def _show_data_fired(self):
        sim = self.vis2d.sim
        P_t, u_t = sim.hist['Pw'].Pw_right
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
