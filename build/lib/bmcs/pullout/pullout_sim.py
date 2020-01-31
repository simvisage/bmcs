'''
Created on 12.01.2016
@author: ABaktheer, RChudoba

@todo: enable recalculation after the initial offline run
@todo: reset viz adapters upon recalculation to forget their axes lims
@todo: introduce a switch for left and right supports
'''
import copy

from bmcs.time_functions import \
    LoadingScenario
from ibvpy.api import BCDof, IMATSEval
from ibvpy.fets.fets1D5 import FETS1D52ULRH
from ibvpy.mats.mats1D5.vmats1D5_bondslip1D import \
    MATSBondSlipMultiLinear, MATSBondSlipDP, \
    MATSBondSlipD, MATSBondSlipEP, MATSBondSlipFatigue
from reporter import RInputRecord
from scipy import interpolate as ip
from simulator.api import \
    Simulator, XDomainFEInterface1D
from traits.api import \
    Property, Instance, cached_property, \
    HasStrictTraits, Bool, List, Float, Trait, Int, Enum, \
    Array, Button
from traits.api import \
    on_trait_change, Tuple
from traitsui.api import \
    View, Item, Group
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.ui.bmcs_tree_node import itags_str
from view.window import BMCSWindow
import matplotlib.pyplot as plt
import numpy as np


class PulloutRecord(Vis2D):

    Pw = Tuple()

    def _Pw_default(self):
        return ([0], [0], [0])

    sig_t = List([])
    eps_t = List([])

    def setup(self):
        self.Pw = ([0], [0], [0])
        self.eps_t = []
        self.sig_t = []

    def update(self):
        sim = self.sim
        c_dof = sim.controlled_dof
        f_dof = sim.free_end_dof
        U_ti = self.sim.hist.U_t
        F_ti = self.sim.hist.F_t
        P = F_ti[:, c_dof]
        w_L = U_ti[:, c_dof]
        w_0 = U_ti[:, f_dof]
        self.Pw = P, w_0, w_L

        t = self.sim.tstep.t_n1
        self.eps_t.append(
            self.sim.get_eps_Ems(t)
        )
        self.sig_t.append(
            self.sim.get_sig_Ems(t)
        )

    def get_t(self):
        return self.sim.hist.t

    def get_U_bar_t(self):
        sim = self.sim
        xdomain = sim.tstep.fe_domain[0].xdomain
        fets = xdomain.fets
        A = xdomain.A
        sig_t = np.array(self.sig_t)
        eps_t = np.array(self.eps_t)
        w_ip = fets.ip_weights
        J_det = xdomain.det_J_Em
        U_bar_t = 0.5 * np.einsum('m,Em,s,tEms,tEms->t',
                                  w_ip, J_det, A, sig_t, eps_t)
        return U_bar_t

    def get_W_t(self):
        P_t, _, w_L = self.sim.get_Pw_t()
        W_t = []
        for i, _ in enumerate(w_L):
            W_t.append(np.trapz(P_t[:i + 1], w_L[:i + 1]))
        return W_t

    def get_dG_t(self):
        sim = self.sim
        t = sim.hist.t
        U_bar_t = self.get_U_bar_t()
        W_t = self.get_W_t()
        G = W_t - U_bar_t
        tck = ip.splrep(t, G, s=0, k=1)
        return ip.splev(t, tck, der=1)


class Viz2DPullOutFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    show_legend = Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        sim = self.vis2d.sim
        P_t, w_0_t, w_L_t = sim.hist['Pw'].Pw
        ymin, ymax = np.min(P_t), np.max(P_t)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(w_L_t), np.max(w_L_t)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        ax.plot(w_L_t, P_t, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
        ax.plot(w_0_t, P_t, linewidth=1, color='magenta', alpha=0.4,
                label='P(w;x=0)')
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('pull-out force P [N]')
        ax.set_xlabel('pull-out slip w [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        sim = self.vis2d.sim
        P_t, w_0_t, w_L_t = sim.hist['Pw'].Pw
        idx = sim.hist.get_time_idx(vot)
        P, w = P_t[idx], w_L_t[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)
        P, w = P_t[idx], w_0_t[idx]
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

    plot_fn = Trait('eps_p',
                    {'eps_p': 'plot_eps_p',
                     'sig_p': 'plot_sig_p',
                     'u_p': 'plot_u_p',
                     's': 'plot_s',
                     'sf': 'plot_sf',
                     'omega': 'plot_omega',
                     'Fint_p': 'plot_Fint_p',
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
        if len(W_t) == 0:
            return
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


class PullOutModel(Simulator):

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
            Item('mats_eval_type', resizable=True, full_size=True),
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

    control_variable = Enum('u', 'f',
                            auto_set=False, enter_set=True,
                            BC=True)

    #=========================================================================
    # Discretization
    #=========================================================================
    n_e_x = Int(20,
                MESH=True,
                auto_set=False,
                enter_set=True,
                symbol='n_\mathrm{E}',
                unit='-',
                desc='number of finite elements along the embedded length'
                )

    #=========================================================================
    # Algorithimc parameters
    #=========================================================================
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

    mats_eval_type = Trait('multilinear',
                           {'multilinear': MATSBondSlipMultiLinear,
                            'damage': MATSBondSlipD,
                            'elasto-plasticity': MATSBondSlipEP,
                            'damage-plasticity': MATSBondSlipDP,
                            'cumulative fatigue': MATSBondSlipFatigue},
                           MAT=True,
                           desc='material model type')

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()
        self._update_node_list()

    mats_eval = Instance(IMATSEval, report=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    mm = Property

    def _get_mm(self):
        return self.mats_eval

    material = Property

    def _get_material(self):
        return self.mats_eval

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS1D52ULRH),
                         depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRH(A_m=self.cross_section.A_m,
                            P_b=self.cross_section.P_b,
                            A_f=self.cross_section.A_f)

    dots_grid = Property(Instance(XDomainFEInterface1D),
                         depends_on=itags_str)
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        geo = self.geometry
        return XDomainFEInterface1D(
            dim_u=2,
            coord_max=[geo.L_x],
            shape=[self.n_e_x],
            fets=self.fets_eval
        )

    fe_grid = Property

    def _get_fe_grid(self):
        return self.dots_grid.mesh

    domains = Property(depends_on=itags_str)

    @cached_property
    def _get_domains(self):
        return [(self.dots_grid, self.mats_eval)]

    #=========================================================================
    # Boundary conditions
    #=========================================================================
    w_max = Float(1, BC=True,
                  symbol='w_{\max}',
                  unit='mm',
                  desc='maximum pullout slip',
                  auto_set=False, enter_set=True)

    u_f0_max = Property(depends_on='BC')

    @cached_property
    def _get_u_f0_max(self):
        return self.w_max

    def _set_u_f0_max(self, value):
        self.w_max = value

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
        return 1

    fixed_bc = Property(depends_on='BC,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_bc(self):
        return BCDof(node_name='fixed left end', var='u',
                     dof=0, value=0.0)

    control_bc = Property(depends_on='BC,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCDof(node_name='pull-out displacement',
                     var=self.control_variable,
                     dof=self.controlled_dof, value=self.w_max,
                     time_function=self.loading_scenario)

    bc = Property(depends_on='BC,MESH')

    @cached_property
    def _get_bc(self):
        return [self.fixed_bc, self.control_bc]

    X_M = Property()

    def _get_X_M(self):
        state = self.tstep.fe_domain[0]
        return state.xdomain.x_Ema[..., 0].flatten()

    #=========================================================================
    # Getter functions @todo move to the PulloutStateRecord
    #=========================================================================

    def get_u_p(self, vot):
        '''Displacement field
        '''
        idx = self.hist.get_time_idx(vot)
        U = self.hist.U_t[idx]
        state = self.tstep.fe_domain[0]
        dof_Epia = state.xdomain.o_Epia
        fets = state.xdomain.fets
        u_Epia = U[dof_Epia]
        N_mi = fets.N_mi
        u_Emap = np.einsum('mi,Epia->Emap', N_mi, u_Epia)
        return u_Emap.reshape(-1, 2)

    def get_eps_Ems(self, vot):
        '''Epsilon in the components'''
        state = self.tstep.fe_domain[0]
        idx = self.hist.get_time_idx(vot)
        U = self.hist.U_t[idx]
        return state.xdomain.map_U_to_field(U)

    def get_eps_p(self, vot):
        '''Epsilon in the components'''
        eps_Ems = self.get_eps_Ems(vot)
        return eps_Ems[..., (0, 2)].reshape(-1, 2)

    def get_s(self, vot):
        '''Slip between the two material phases'''
        eps_Ems = self.get_eps_Ems(vot)
        return eps_Ems[..., 1].flatten()

    def get_sig_Ems(self, vot):
        '''Get streses in the components 
        '''
        txdomain = self.tstep.fe_domain[0]
        idx = self.hist.get_time_idx(vot)
        U = self.hist.U_t[idx]
        t_n1 = self.hist.t[idx]
        eps_Ems = txdomain.xdomain.map_U_to_field(U)
        state_vars_t = self.tstep.hist.state_vars[idx]
        state_k = copy.deepcopy(state_vars_t)
        sig_Ems, _ = txdomain.tmodel.get_corr_pred(
            eps_Ems, t_n1, **state_k[0]
        )
        return sig_Ems

    def get_sig_p(self, vot):
        '''Epsilon in the components'''
        sig_Ems = self.get_sig_Ems(vot)
        return sig_Ems[..., (0, 2)].reshape(-1, 2)

    def get_sf(self, vot):
        '''Get the shear flow in the interface
        '''
        sig_Ems = self.get_sig_Ems(vot)
        return sig_Ems[..., 1].flatten()

    def get_shear_integ(self):
        #         d_ECid = self.get_d_ECid(vot)
        #         s_Emd = np.einsum('Cim,ECid->Emd', self.tstepper.sN_Cim, d_ECid)
        #         idx = self.tloop.get_time_idx(vot)
        #         sf = self.tloop.sf_Em_record[idx]

        sf_t_Em = np.array(self.tloop.sf_Em_record)
        w_ip = self.fets_eval.ip_weights
        J_det = self.tstepper.J_det
        P_b = self.cross_section.P_b
        shear_integ = np.einsum('tEm,m,em->t', sf_t_Em, w_ip, J_det) * P_b
        return shear_integ

    def get_Pw_t(self):
        sim = self
        c_dof = sim.controlled_dof
        f_dof = sim.free_end_dof
        U_ti = sim.hist.U_t
        F_ti = sim.hist.F_t
        P = F_ti[:, c_dof]
        w_L = U_ti[:, c_dof]
        w_0 = U_ti[:, f_dof]
        return P, w_0, w_L

    #=========================================================================
    # Plot functions
    #=========================================================================
    def plot_u_p(self, ax, vot,
                 label_m='matrix', label_f='reinf'):
        X_M = self.X_M
        L = self.geometry.L_x
        u_p = self.get_u_p(vot).T
        ax.plot(X_M, u_p[0], linewidth=2, color='blue', label=label_m)
        ax.fill_between(X_M, 0, u_p[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, u_p[1], linewidth=2, color='orange', label=label_f)
        ax.fill_between(X_M, 0, u_p[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('displacement')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)
        return np.min(u_p), np.max(u_p)

    def plot_eps_p(self, ax, vot,
                   label_m='matrix', label_f='reinf'):
        X_M = self.X_M
        L = self.geometry.L_x
        eps_p = self.get_eps_p(vot).T
        ax.plot(X_M, eps_p[0], linewidth=2, color='blue', label=label_m)
        ax.fill_between(X_M, 0, eps_p[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, eps_p[1], linewidth=2, color='orange', label=label_f)
        ax.fill_between(X_M, 0, eps_p[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('strain')
        ax.set_xlabel('bond length')
        return np.min(eps_p), np.max(eps_p)

    def plot_sig_p(self, ax, vot):
        X_M = self.X_M
        sig_p = self.get_sig_p(vot).T

#        A_m = self.cross_section.A_m
#        A_f = self.cross_section.A_f
        L = self.geometry.L_x
        F_m = sig_p[0]
        F_f = sig_p[1]
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
        X_J = self.X_M
        s = self.get_s(vot)
        ax.fill_between(X_J, 0, s, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, s, linewidth=2, color='lightcoral')
        ax.set_ylabel('slip')
        ax.set_xlabel('bond length')
        return np.min(s), np.max(s)

    def plot_sf(self, ax, vot):
        X_J = self.X_M
        sf = self.get_sf(vot)
        ax.fill_between(X_J, 0, sf, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, sf, linewidth=2, color='lightcoral')
        ax.set_ylabel('shear flow')
        ax.set_xlabel('bond length')
        return np.min(sf), np.max(sf)

    def plot_omega(self, ax, vot):
        X_J = self.X_J
        omega = self.get_omega(vot)
        ax.fill_between(X_J, 0, omega, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, omega, linewidth=2, color='lightcoral', label='bond')
        ax.set_ylabel('damage')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)
        return 0.0, 1.05

    def plot_eps_s(self, ax, vot):
        eps_p = self.get_eps_p(vot).T
        s = self.get_s(vot)
        ax.plot(eps_p[1], s, linewidth=2, color='lightcoral')
        ax.set_ylabel('reinforcement strain')
        ax.set_xlabel('slip')

    def get_window(self):
        self.record['Pw'] = PulloutRecord()
        fw = Viz2DPullOutFW(name='pullout-curve', vis2d=self.hist['Pw'])
        u_p = Viz2DPullOutField(name='displacement along the bond',
                                plot_fn='u_p', vis2d=self)
        eps_p = Viz2DPullOutField(name='strain along the bond',
                                  plot_fn='eps_p', vis2d=self)
        sig_p = Viz2DPullOutField(name='stress along the bond',
                                  plot_fn='sig_p', vis2d=self)
        s = Viz2DPullOutField(name='slip along the bond',
                              plot_fn='s', vis2d=self)
        sf = Viz2DPullOutField(name='shear flow along the bond',
                               plot_fn='sf', vis2d=self)
        energy = Viz2DEnergyPlot(name='energy',
                                 vis2d=self.hist['Pw'])
        dissipation = Viz2DEnergyReleasePlot(name='energy release',
                                             vis2d=self.hist['Pw'])
        w = BMCSWindow(sim=self)
        w.viz_sheet.viz2d_list.append(fw)
        w.viz_sheet.viz2d_list.append(u_p)
        w.viz_sheet.viz2d_list.append(eps_p)
        w.viz_sheet.viz2d_list.append(sig_p)
        w.viz_sheet.viz2d_list.append(s)
        w.viz_sheet.viz2d_list.append(sf)
        w.viz_sheet.viz2d_list.append(energy)
        w.viz_sheet.viz2d_list.append(dissipation)
        w.viz_sheet.monitor_chunk_size = 10
        return w
