#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 21, 2009 by: rrypl

from matplotlib.figure import Figure
from traits.api import \
    Instance, Enum, Bool, on_trait_change, Int, Event
from traitsui.api import \
    View, Item, VGroup, HGroup, ModelView, HSplit, VSplit
from traitsui.menu import OKButton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from .scm import SCM


#--------------------------------------------------------------------------
# MODEL_VIEW
#--------------------------------------------------------------------------
class SCMModelView (ModelView):

    model = Instance(SCM)

    def _model_default(self):
        return SCM()

    # Number of all curves which can be plotted by this module
    n_curves = Int(6)

    # switches between stresses with respect to area of composite and
    # reinforcement, respectively
    stresses = Enum('composite', 'reinforcement')
    csf = Bool(True)

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.08, 0.13, 0.85, 0.74])
        return figure

    data_changed = Event

    @on_trait_change('model.+modified, model.reinf_ratio.+modified, stresses, csf')
    def refresh(self):

        figure = self.figure
        figure.clear()
        axes = figure.gca()

        scm = self.model
        epsilon_c_arr = scm.sig_eps_fn[0]
        sigma_c_arr = scm.sig_eps_fn[1]
        sigma_f_arr = scm.sig_eps_fn[2]
        sigma_fiber = scm.sig_eps_fn[3]
        epsilon_csf = scm.csf[0]
        sigma_csf = scm.csf[1]

        axes.set_xlabel('strain', weight='semibold')
        axes.set_ylabel('stress [MPa]', weight='semibold')
        axes.set_axis_bgcolor(color='white')
        axes.ticklabel_format(scilimits=(-3., 4.))
        axes.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        if self.stresses == 'composite':

            axes.plot(epsilon_c_arr, sigma_c_arr, lw=2, color='blue')
            axes.plot(epsilon_c_arr[[0, -1]], sigma_fiber, lw=2, color='grey')
            legend = ['composite', 'reference stiffness']
            if self.csf:
                axes.plot(
                    epsilon_csf, sigma_csf, lw=1, color='green', linestyle='dotted')
                legend.append('strain at $CS_f$')
            axes.legend(legend, 'upper left')
        else:

            axes.plot(epsilon_c_arr, sigma_f_arr, lw=2, color='red')
            axes.plot(
                epsilon_c_arr[[0, -1]], sigma_fiber / scm.rho, lw=2, color='grey')
            legend = ['reinforcement', 'reference stiffness']
            if self.csf:
                axes.plot(epsilon_csf, sigma_csf / scm.rho, lw=1,
                          color='green', linestyle='dotted')
                legend.append('strain at $CS_f$')
            axes.legend(legend, 'upper left')

        self.data_changed = True

    traits_view = View(
        HSplit(
            VGroup(
                Item('model@', show_label=False, springy=True, width=400),
                label='Material parameters',
                id='scm.viewmodel.model',
                dock='tab',
            ),
            VSplit(
                VGroup(
                    Item('figure',
                         editor=MPLFigureEditor(), springy=True,
                         resizable=True, show_label=False),
                    label='Strain hardening response',
                    id='scm.viewmodel.plot_window',
                    dock='tab',
                ),
                HGroup(
                    VGroup(Item('stresses',
                                label='stress refered to cross-sectional area of',
                                resizable=False, height=30), scrollable=False),
                    VGroup(Item('csf', label='show csf'),
                           scrollable=False),
                    VGroup(Item('model.Pf', label='probability for csf'),
                           scrollable=False),
                    label='plot parameters',
                    id='scm.viewmodel.plot_params',
                    group_theme='blue',
                ),
                id='scm.viewmodel.right',
            ),
            id='scm.viewmodel.splitter',
        ),
        title='Stochastic Cracking Model',
        id='scm.viewmodel',
        dock='tab',
        resizable=True,
        buttons=[OKButton],
        height=0.8, width=0.8
    )


if __name__ == '__main__':
    s = SCMModelView(model=SCM())
    s.refresh()
    s.configure_traits()
#    run()
