'''
Created on 12.01.2016
@author: Yingxiong, ABaktheer, RChudoba

@todo: enable recalculation after the initial offline run
@todo: reset viz adapters upon recalculation to forget their axes lims
@todo: introduce a switch for left and right supports
'''

from traits.api import \
    Property, Instance, cached_property, \
    HasStrictTraits, Bool, List, Float, Trait, Int, Enum, \
    Array, Button
from traitsui.api import \
    View, Item, Group
from traitsui.ui_editors.array_view_editor import ArrayViewEditor

from bmcs.time_functions import \
    LoadingScenario
import matplotlib.pyplot as plt
import numpy as np
from reporter import RInputRecord
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, TLine


class CrossSection(BMCSLeafNode, RInputRecord):
    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    A_m = Float(15240,
                CS=True,
                input=True,
                unit=r'$\mathrm{mm}^2$',
                symbol=r'A_\mathrm{m}',
                auto_set=False, enter_set=True,
                desc='matrix area')
    A_f = Float(153.9,
                CS=True,
                input=True,
                unit='$\\mathrm{mm}^2$',
                symbol='A_\mathrm{f}',
                auto_set=False, enter_set=True,
                desc='reinforcement area')
    P_b = Float(44,
                CS=True,
                input=True,
                unit='$\\mathrm{mm}$',
                symbol='p_\mathrm{b}',
                auto_set=False, enter_set=True,
                desc='perimeter of the bond interface')

    view = View(
        Item('A_m'),
        Item('A_f'),
        Item('P_b')
    )

    tree_view = view


class Geometry(BMCSLeafNode, RInputRecord):

    node_name = 'geometry'
    L_x = Float(45,
                GEO=True,
                input=True,
                unit='$\mathrm{mm}$',
                symbol='L',
                auto_set=False, enter_set=True,
                desc='embedded length')

    view = View(
        Item('L_x'),
    )

    tree_view = view


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


class Viz2DPullOutFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    show_legend = Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        P_t = self.vis2d.get_P_t()
        ymin, ymax = np.min(P_t), np.max(P_t)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        w_0, w_L = self.vis2d.get_w_t()
        xmin, xmax = np.min(w_L), np.max(w_L)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        ax.plot(w_L, P_t, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
        ax.plot(w_0, P_t, linewidth=1, color='magenta', alpha=0.4,
                label='P(w;x=0)')
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('pull-out force P [N]')
        ax.set_xlabel('pull-out slip w [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        P_t = self.vis2d.get_P_t()
        w_0, w_L = self.vis2d.get_w_t()
        idx = self.vis2d.tloop.get_time_idx(vot)
        P, w = P_t[idx], w_L[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)
        P, w = P_t[idx], w_0[idx]
        ax.plot([w], [P], 'o', color='magenta', markersize=10)

    show_data = Button()

    def _show_data_fired(self):
        P_t = self.vis2d.get_P_t()
        w_0, w_L = self.vis2d.get_w_t()
        data = np.vstack([w_0, w_L, P_t]).T
        show_data = DataSheet(data=data)
        show_data.edit_traits()

    def plot_tex(self, ax, vot, *args, **kw):
        self.plot(ax, vot, *args, **kw)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
        Item('show_data')
    )


class Viz2DPullOutField(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = Property(depends_on='plot_fn')

    @cached_property
    def _get_label(self):
        return 'field: %s' % self.plot_fn

    plot_fn = Trait('u_C',
                    {'eps_C': 'plot_eps_C',
                     'sig_C': 'plot_sig_C',
                     'u_C': 'plot_u_C',
                     's': 'plot_s',
                     'sf': 'plot_sf',
                     'omega': 'plot_omega',
                     'Fint_C': 'plot_Fint_C',
                     'eps_f(s)': 'plot_eps_s',
                     },
                    label='Field',
                    tooltip='Select the field to plot'
                    )

    def plot(self, ax, vot, *args, **kw):
        ymin, ymax = getattr(self.vis2d, self.plot_fn_)(ax, vot, *args, **kw)
        if self.adaptive_y_range:
            if self.initial_plot:
                self.y_max = ymax
                self.y_min = ymin
                self.initial_plot = False
                return
        self.y_max = max(ymax, self.y_max)
        self.y_min = min(ymin, self.y_min)
        ax.set_ylim(ymin=self.y_min, ymax=self.y_max)

    def savefig(self, fname):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for vot in [0.25, 0.5, 0.75, 1.0]:
            self.plot(ax, vot)
        fig.savefig(fname)

    y_max = Float(1.0, label='Y-max value',
                  auto_set=False, enter_set=True)
    y_min = Float(0.0, label='Y-min value',
                  auto_set=False, enter_set=True)

    adaptive_y_range = Bool(True)
    initial_plot = Bool(True)

    traits_view = View(
        Item('plot_fn', resizable=True, full_size=True),
        Item('y_min', ),
        Item('y_max', ),
        Item('adaptive_y_range')
    )


class Viz2DEnergyPlot(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'line plot'

    def plot(self, ax, vot,
             label_U='U(t)', label_W='W(t)',
             color_U='blue', color_W='red'):
        t = self.vis2d.get_t()
        U_bar_t = self.vis2d.get_U_bar_t()
        W_t = self.vis2d.get_W_t()
        ax.plot(t, W_t, color=color_W, label=label_W)
        ax.plot(t, U_bar_t, color=color_U, label=label_U)
        ax.fill_between(t, W_t, U_bar_t, facecolor='gray', alpha=0.5,
                        label='G(t)')
        ax.set_ylabel('energy [Nmm]')
        ax.set_xlabel('time [-]')
        ax.legend()


class Viz2DEnergyReleasePlot(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'line plot'

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d.get_t()

        dG = self.vis2d.get_dG_t()

        ax.plot(t, dG, color='black', label='dG/dt')
        ax.fill_between(t, 0, dG, facecolor='blue', alpha=0.2)
        ax.legend()


class PullOutModelBase(BMCSModel, Vis2D):

    node_name = 'pull out simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.loading_scenario,
            self.mats_eval,
            self.cross_section,
            self.geometry
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.loading_scenario,
            self.mats_eval,
            self.cross_section,
            self.geometry,
        ]

    tree_view = View(
        Group(
            Item('control_variable', resizable=True, full_size=True),
            Item('w_max', resizable=True, full_size=True),
            Item('n_e_x', resizable=True, full_size=True),
            Item('fixed_boundary'),
            Group(
                Item('loading_scenario@', show_label=False),
            )
        )
    )

    #=========================================================================
    # Test setup parameters
    #=========================================================================
    loading_scenario = Instance(
        LoadingScenario,
        report=True,
        desc='object defining the loading scenario'
    )

    def _loading_scenario_default(self):
        return LoadingScenario()

    cross_section = Instance(
        CrossSection,
        report=True,
        desc='cross section parameters'
    )

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(
        Geometry,
        report=True,
        desc='geometry parameters of the boundary value problem'
    )

    def _geometry_default(self):
        return Geometry()

#
#     loading_scenario = Instance(LoadingScenario, report=False)
#
#     def _loading_scenario_default(self):
#         return LoadingScenario()
#
#     cross_section = Instance(CrossSection, report=True)
#
#     def _cross_section_default(self):
#         return CrossSection()
#
#     geometry = Instance(Geometry, report=True)
#
#     def _geometry_default(self):
#         return Geometry()

    control_variable = Enum('u', 'f',
                            auto_set=False, enter_set=True,
                            BC=True)

    #=========================================================================
    # Discretization
    #=========================================================================
    n_e_x = Int(20,
                auto_set=False,
                enter_set=True,
                MESH=True,
                symbol='n_\mathrm{e}',
                unit='-',
                desc='number of finite elements')

    w_max = Float(1, BC=True,
                  symbol='w_{\max}',
                  unit='mm',
                  desc='maximum pullout slip',
                  auto_set=False, enter_set=True)

    u_f0_max = Property(depends_on='u_f0_max')

    @cached_property
    def _get_u_f0_max(self):
        return self.w_max

    def _set_u_f0_max(self, value):
        self.w_max = value

    n_e_x = Int(20, MESH=True, auto_set=False, enter_set=True,
                symbol='n_\mathrm{E}',
                unit='-',
                desc='number of finite elements along the embedded length'
                )

    fixed_boundary = Enum('non-loaded end (matrix)',
                          'loaded end (matrix)',
                          'non-loaded end (reinf)',
                          'clamped left',
                          BC=True,
                          desc='which side of the specimen is fixed [non-loaded end [matrix], loaded end [matrix], non-loaded end [reinf]]')

    fixed_dofs = Property

    def _get_fixed_dofs(self):
        if self.fixed_boundary == 'non-loaded end (matrix)':
            return [0]
        elif self.fixed_boundary == 'non-loaded end (reinf)':
            return [1]
        elif self.fixed_boundary == 'loaded end (matrix)':
            return [self.controlled_dof - 1]
        elif self.fixed_boundary == 'clamped left':
            return [0, 1]

    controlled_dof = Property

    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    free_end_dof = Property

    def _get_free_end_dof(self):
        return self.n_e_x + 1

    k_max = Int(400,
                unit='\mathrm{mm}',
                symbol='k_{\max}',
                desc='maximum number of iterations',
                ALG=True)

    tolerance = Float(1e-4,
                      unit='-',
                      symbol='\epsilon',
                      desc='required accuracy',
                      ALG=True)

    def plot_u_C(self, ax, vot,
                 label_m='matrix', label_f='reinf'):
        X_M = self.tstepper.X_M
        L = self.geometry.L_x
        u_C = self.get_u_C(vot).T
        ax.plot(X_M, u_C[0], linewidth=2, color='blue', label=label_m)
        ax.fill_between(X_M, 0, u_C[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, u_C[1], linewidth=2, color='orange', label=label_f)
        ax.fill_between(X_M, 0, u_C[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('displacement')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)
        return np.min(u_C), np.max(u_C)

    def plot_eps_C(self, ax, vot,
                   label_m='matrix', label_f='reinf'):
        X_M = self.tstepper.X_M
        L = self.geometry.L_x
        eps_C = self.get_eps_C(vot).T
        ax.plot(X_M, eps_C[0], linewidth=2, color='blue', label=label_m)
        ax.fill_between(X_M, 0, eps_C[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, eps_C[1], linewidth=2, color='orange', label=label_f)
        ax.fill_between(X_M, 0, eps_C[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('strain')
        ax.set_xlabel('bond length')
        return np.min(eps_C), np.max(eps_C)

    def plot_sig_C(self, ax, vot):
        X_M = self.tstepper.X_M
        sig_C = self.get_sig_C(vot).T

        A_m = self.cross_section.A_m
        A_f = self.cross_section.A_f
        L = self.geometry.L_x
        F_m = sig_C[0]
        F_f = sig_C[1]
        ax.plot(X_M, F_m, linewidth=2, color='blue', )
        ax.fill_between(X_M, 0, F_m, facecolor='blue', alpha=0.2)
        ax.plot(X_M, F_f, linewidth=2, color='orange')
        ax.fill_between(X_M, 0, F_f, facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('stress [MPa]')
        ax.set_xlabel('bond length')
        F_min = min(np.min(F_m), np.min(F_f))
        F_max = max(np.max(F_m), np.max(F_f))
        return F_min, F_max

    def plot_s(self, ax, vot):
        X_J = self.tstepper.X_J
        s = self.get_s(vot)
        ax.fill_between(X_J, 0, s, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, s, linewidth=2, color='lightcoral')
        ax.set_ylabel('slip')
        ax.set_xlabel('bond length')
        return np.min(s), np.max(s)

    def plot_sf(self, ax, vot):
        X_J = self.tstepper.X_J
        sf = self.get_sf(vot)
        ax.fill_between(X_J, 0, sf, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, sf, linewidth=2, color='lightcoral')
        ax.set_ylabel('shear flow')
        ax.set_xlabel('bond length')
        return np.min(sf), np.max(sf)

    def plot_omega(self, ax, vot):
        X_J = self.tstepper.X_J
        omega = self.get_omega(vot)
        ax.fill_between(X_J, 0, omega, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, omega, linewidth=2, color='lightcoral', label='bond')
        ax.set_ylabel('damage')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)
        return 0.0, 1.05

    def plot_eps_s(self, ax, vot):
        eps_C = self.get_eps_C(vot).T
        s = self.get_s(vot)
        ax.plot(eps_C[1], s, linewidth=2, color='lightcoral')
        ax.set_ylabel('reinforcement strain')
        ax.set_xlabel('slip')
