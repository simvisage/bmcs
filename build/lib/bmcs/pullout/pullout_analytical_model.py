'''
Created on 12.01.2016
@author: Yingxiong, ABaktheer, RChudoba

@todo: enable recalculation after the initial offline run
@todo: reset viz adapters upon recalculation to forget their axes lims
@todo: introduce a switch for left and right supports
'''

from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from traits.api import \
    Property, Instance, cached_property, \
    Bool, List, Float, Trait, Int, on_trait_change, \
    Dict, Str
from traitsui.api import \
    View, Item, VGroup, Group
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow, TLine

import numpy as np


class MaterialParams(BMCSLeafNode):
    '''Record of material parameters of an interface layer
    '''
    node_name = Str('material parameters')

    E_f = Float(240000,
                MAT=True,
                input=True,
                symbol="$E_\mathrm{f}$",
                unit='$\mathrm{MPa}$',
                desc="reinforcement stiffness",
                enter_set=True,
                auto_set=False)

    tau_pi_bar = Float(4.5,
                       MAT=True,
                       input=True,
                       symbol=r'$\bar{\tau}$',
                       unit='$\mathrm{MPa}$',
                       desc="frictional bond strength",
                       enter_set=True,
                       auto_set=False)

    view = View(VGroup(Group(Item('E_f', full_size=True, resizable=True),
                             Item('tau_pi_bar'), show_border=True,
                             label='Material parameters'),
                       ))

    tree_view = view


class CrossSection(BMCSLeafNode):
    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    A_f = Float(153.9,
                CS=True,
                input=True,
                auto_set=False, enter_set=True,
                symbol='$A_\mathrm{f}$', unit='$\mathrm{mm}^2$',
                desc='reinforcement area')
    P_b = Float(44,
                input=True,
                CS=True,
                auto_set=False, enter_set=True,
                symbol='$P_\mathrm{b}$', unit='$\mathrm{mm}$',
                desc='perimeter of the bond interface')

    view = View(
        Item('A_m', full_size=True, resizable=True),
        Item('P_b')
    )

    tree_view = view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    L_x = Float(20,
                GEO=True,
                input=True,
                symbol='$L$', unit='mm',
                auto_set=False, enter_set=True,
                desc='embedded length')

    view = View(
        Item('L_x', full_size=True, resizable=True),
    )

    tree_view = view


class Viz2DPullOutFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    def plot(self, ax, vot, *args, **kw):
        P_t = self.vis2d.get_P_t()
        ymin, ymax = np.min(P_t), np.max(P_t)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.0 * L_y
        w_L = self.vis2d.get_w_t()
        xmin, xmax = np.min(w_L), np.max(w_L)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.0 * L_x
        ax.plot(w_L, P_t, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=0)')
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('pull-out force P [N]')
        ax.set_xlabel('pull-out slip w [mm]')
        self.plot_marker(ax, vot)
        ax.legend(loc=4)

    def plot_marker(self, ax, vot):
        idx = self.vis2d.get_time_idx(vot)
        P_t = self.vis2d.get_P_t()
        w_L = self.vis2d.get_w_t()
        P, w = P_t[idx], w_L[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)

    def plot_tex(self, ax, vot):
        self.plot(ax, vot)


class Viz2DPullOutField(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = Property(depends_on='plot_fn')

    @cached_property
    def _get_label(self):
        return 'field: %s' % self.plot_fn

    plot_fn = Trait('u',
                    {'eps': 'plot_eps',
                     'sig': 'plot_sig',
                     'u': 'plot_u',
                     'sf': 'plot_sf',
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

    def plot_tex(self, ax, vot, *args, **kw):
        self.plot(ax, vot, *args, **kw)

    y_max = Float(1.0, label='Y-max value',
                  auto_set=False, enter_set=True)
    y_min = Float(0.0, label='Y-min value',
                  auto_set=False, enter_set=True)

    adaptive_y_range = Bool(True)
    initial_plot = Bool(True)

    traits_view = View(
        Item('plot_fn'),
        Item('y_min', ),
        Item('y_max', )
    )


class PullOutModel(BMCSModel, Vis2D):

    node_name = 'pull out simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.material,
            self.cross_section,
            self.geometry,
        ]

    sv_names = Property(List(Str), depends_on='material_model')
    '''Names of state variables of the current material model.
    '''
    @cached_property
    def _get_sv_names(self):
        return ['t', 'w', 'u_x', 'eps_x', 'sf_x']

    sv_hist = Dict((Str, List))
    '''Record of state variables. The number of names of the variables
    depend on the active material model. See s_names.
    '''

    def _sv_hist_default(self):
        sv_hist = {}
        for sv_name in self.sv_names:
            sv_hist[sv_name] = []
        return sv_hist

    @on_trait_change('MAT,BC')
    def _sv_hist_reset(self):
        print('sv_hist_reset')
        for sv_name in self.sv_names:
            self.sv_hist[sv_name] = []

    def paused(self):
        self._paused = True

    def stop(self):
        self._sv_hist_reset()
        self._restart = True
        self.loading_scenario.reset()

    _paused = Bool(False)
    _restart = Bool(True)

    n_steps = Int(100, ALG=True,
                  symbol='$n_\mathrm{E}$', unit='-',
                  desc='number of time steps',
                  enter_set=True, auto_set=False)

    def init(self):
        if self._paused:
            self._paused = False
        if self._restart:
            self.tline.val = self.tline.min
            self._restart = False

    x = Property(depends_on='GEO')

    @cached_property
    def _get_x(self):
        return np.linspace(0, self.geometry.L_x, self.n_x)

    def analytical_po_model(self, w):
        x = self.x
        Ef = self.material.E_f
        Pb = self.cross_section.P_b
        Af = self.cross_section.A_f
        L = self.geometry.L_x
        tau_bar = self.material.tau_pi_bar
        T = Pb * tau_bar
        a = np.sqrt(2 * w * Ef * Af * T) / T
        idx = np.where(x < a)[0]
        u_x = np.zeros_like(x)
        u_x[idx] = (1.0 / (2.0 * Ef * Af) * (
            T * x[idx] * x[idx] + 2 * w * Ef * Af - 2 *
            np.sqrt(2 * w * Ef * Af * T) * x[idx]
        ))
        eps_x = np.zeros_like(x)
        eps_x[idx] = (1.0 / (Ef * Af) *
                      (T * x[idx] - np.sqrt(2.0 * w * Ef * Af * T)))

        sig_x = np.zeros_like(x)
        sig_x[idx] = Ef * Af * eps_x[idx]

        sf_x = np.zeros_like(x)
        sf_x[idx] = T
        return u_x, eps_x, sf_x

    def eval(self):
        t_max = self.loading_scenario.xdata[-1]
        self.set_tmax(t_max)
        t_min = self.tline.val
        n_steps = (self.tline.max - self.tline.min) / self.tline.step
        tarray = np.linspace(t_min, t_max, n_steps)
        sv_names = self.sv_names
        sv_records = [[] for s_n in sv_names]
        s_last = 0
        for idx, t in enumerate(tarray):
            if self._restart or self._paused:
                break
            w = self.w_max * self.loading_scenario(t)
            state_vars = \
                self.analytical_po_model(w)
            t_ = np.array([t], dtype=np.float_)
            for sv_record, state in zip(sv_records,
                                        (t_, w) + state_vars):
                sv_record.append(np.copy(state))

        # append the data to the previous simulation steps
        for sv_name, sv_record in zip(sv_names, sv_records):
            self.sv_hist[sv_name].append(np.array(sv_record))

        self.tline.val = min(t, self.tline.max)

    def get_sv_hist(self, sv_name):
        return np.vstack(self.sv_hist[sv_name])

    material = Instance(MaterialParams, report=True)

    def _material_default(self):
        return MaterialParams()

    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    cross_section = Instance(CrossSection, report=True)

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(Geometry, report=True)

    def _geometry_default(self):
        return Geometry()

    n_x = Int(100, auto_set=False, enter_set=True)

    w_max = Float(1,
                  BC=True,
                  symbol='$u_{\mathrm{f},0}$',
                  unit='mm',
                  desc='control displacement',
                  auto_set=False, enter_set=True)

    tline = Instance(TLine)

    def _tline_default(self):
        # assign the parameters for solver and loading_scenario
        t_max = 1.0  # self.loading_scenario.t_max
        d_t = 0.1  # self.loading_scenario.d_t
        return TLine(min=0.0, step=d_t, max=t_max,
                     time_change_notifier=self.time_changed,
                     )

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        x = self.get_sv_hist('t').T[0]
        idx = np.array(np.arange(len(x)), dtype=np.float_)
        t_idx = np.interp(vot, x, idx)
        return np.array(t_idx + 0.5, np.int_)

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))

    def get_P_t(self):
        Ef = self.material.E_f
        Af = self.cross_section.A_f
        return - Ef * Af * self.get_sv_hist('eps_x')[:, 0]

    def get_w_t(self):
        return self.get_sv_hist('u_x')[:, 0]

    def plot_u(self, ax, vot,
               color='orange', label='reinf',
               facecolor='orange', alpha=0.2):
        L = self.geometry.L_x
        x = -self.x

        idx = self.get_time_idx(vot)

        u_x = self.get_sv_hist('u_x')[idx]
        ax.plot(x, u_x, linewidth=2, color=color, label=label)
        ax.fill_between(x, 0, u_x, facecolor=facecolor, alpha=alpha)
        ax.plot([0, -L], [0, 0], color='black')
        ax.set_ylabel('displacement: u [mm]')
        ax.set_xlabel('bond length: x [mm]')
        ax.legend(loc=2)
        return np.min(u_x), np.max(u_x)

    def plot_eps(self, ax, vot,
                 color='orange', label='reinf',
                 facecolor='orange', alpha=0.2):
        L = self.geometry.L_x
        x = -self.x
        idx = self.get_time_idx(vot)
        eps_x = -self.get_sv_hist('eps_x')[idx]
        ax.plot(x, eps_x, linewidth=2, color=color, label=label)
        ax.fill_between(x, 0, eps_x, facecolor=facecolor, alpha=alpha)
        ax.plot([0, -L], [0, 0], color='black')
        ax.legend(loc=2)
        ax.set_ylabel('strain: eps [-]')
        ax.set_xlabel('bond length: x [mm]')
        return np.min(eps_x), np.max(eps_x)

    def plot_sig(self, ax, vot,
                 color='orange', label='bond',
                 facecolor='orange', alpha=0.2):
        L = self.geometry.L_x
        x = -self.x
        idx = self.get_time_idx(vot)
        eps_x = -self.get_sv_hist('eps_x')
        sig_x = self.material.E_f * eps_x
        ax.plot(x, sig_x, linewidth=2, color=color, label='reinf')
        ax.fill_between(x, 0, sig_x, facecolor=color, alpha=0.2)
        ax.plot([0, -L], [0, 0], color='black')
        ax.legend(loc=2)
        ax.set_ylabel('stress: sig [MPa]')
        ax.set_xlabel('bond length: x [mm]')
        return np.min(sig_x), np.max(sig_x)

    def plot_sf(self, ax, vot,
                color='blue', label='bond',
                facecolor='blue', alpha=0.2):
        L = self.geometry.L_x
        x = -self.x
        idx = self.get_time_idx(vot)
        sf_x = self.get_sv_hist('sf_x')[idx]
        ax.plot(x, sf_x, linewidth=2, color=color, label=label)
        ax.fill_between(x, 0, sf_x, facecolor=facecolor, alpha=alpha)
        ax.plot([0, -L], [0, 0], color='black')
        ax.legend(loc=2)
        ax.set_ylabel('shear flow: p * tau [N/mm]')
        ax.set_xlabel('bond length: x [mm]')
        return np.min(sf_x), np.max(sf_x)

#     tree_view = View(
#         VGroup(
#             Group(
#                 Item('w_max', full_size=True, resizable=True),
#             ),
#         )
#     )
    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW,
                     'load function': Viz2DLoadControlFunction,
                     }


def run_pullout_const_shear(*args, **kw):
    po = PullOutModel(name='t32_analytical_pullout', n_x=200, w_max=1.5)
    po.geometry.trait_set(L_x=800)
#     po.cross_section.trait_set(A_f=4.5, P_b=1.0)
#     po.material.trait_set(E_f=9 * 180000.0, tau_pi_bar=2.77 * 9)
    po.tline.step = 0.01

    w = BMCSWindow(model=po)
    # po.add_viz2d('load function')
    po.add_viz2d('F-w', 'load-displacement')
    po.add_viz2d('field', 'shear flow', plot_fn='sf')
    po.add_viz2d('field', 'displacement', plot_fn='u')
    po.add_viz2d('field', 'strain', plot_fn='eps')
#    po.add_viz2d('field', 'sig', plot_fn='sig')
    w.viz_sheet.monitor_chunk_size = 2

    w.run()
    w.offline = False
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_const_shear()
