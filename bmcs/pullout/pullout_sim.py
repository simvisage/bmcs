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
from scipy.integrate import cumtrapz
from simulator.api import \
    TStepBC, Hist, XDomainFEInterface1D
from traits.api import \
    Property, Instance, cached_property, \
    HasStrictTraits, Bool, List, Float, Trait, Int, Enum, \
    Array, Button
from traits.api import \
    on_trait_change, Tuple
from traitsui.api import \
    View, Item, Group
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from view.plot2d import Vis2D
from view.ui import BMCSLeafNode, itags_str, BMCSRootNode
from view.window import BMCSWindow, PlotPerspective
from view.window.i_bmcs_model import IBMCSModel

import numpy as np
import traits.api as tr


class PulloutHist(Hist, Vis2D):

    record_traits = tr.List(
        ['P', 'w_0', 'w_L', ]
    )

    record_t = tr.Dict

    Pw = Tuple()

    def _Pw_default(self):
        return ([0], [0], [0])

    sig_t = List([])
    eps_t = List([])

    def init_state(self):
        super(PulloutHist, self).init_state()
        self.Pw = ([0], [0], [0])
        self.eps_t = []
        self.sig_t = []
        for rt in self.record_traits:
            self.record_t[rt] = [0]

    def record_timestep(self, t, U, F,
                        state_vars=None):
        super(PulloutHist, self).record_timestep(t, U, F, state_vars)
        c_dof = self.tstep_source.controlled_dof
        f_dof = self.tstep_source.free_end_dof
        U_ti = self.U_t
        F_ti = self.F_t
        P = F_ti[:, c_dof]
        w_L = U_ti[:, c_dof]
        w_0 = U_ti[:, f_dof]
        self.Pw = P, w_0, w_L

        t = self.tstep_source.t_n1
        self.eps_t.append(
            self.tstep_source.get_eps_Ems(t)
        )
        self.sig_t.append(
            self.tstep_source.get_sig_Ems(t)
        )
        for rt in self.record_traits:
            self.record_t[rt].append(getattr(self.tstep_source, rt))

    def get_Pw_t(self):
        c_dof = self.tstep_source.controlled_dof
        f_dof = self.tstep_source.free_end_dof
        U_ti = self.U_t
        F_ti = self.F_t
        P = F_ti[:, c_dof]
        w_L = U_ti[:, c_dof]
        w_0 = U_ti[:, f_dof]
        return P, w_0, w_L

    def get_U_bar_t(self):
        xdomain = self.tstep_source.fe_domain[0].xdomain
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
        P_t, _, w_L = self.get_Pw_t()
        W_t = cumtrapz(P_t, w_L, initial=0)
        return W_t

    def get_dG_t(self):
        t = self.t
        U_bar_t = self.get_U_bar_t()
        W_t = self.get_W_t()
        G = W_t - U_bar_t
#         G0 = G[:-1]
#         G1 = G[1:]
#         dG = np.hstack([0, G1 - G0])
#         return dG
        tck = ip.splrep(t, G, s=0, k=1)
        return ip.splev(t, tck, der=1)

    show_legend = Bool(True, auto_set=False, enter_set=True)

    def plot_Pw(self, ax, vot, *args, **kw):
        P_t, w_0_t, w_L_t = self.get_Pw_t()
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
        P_t, w_0_t, w_L_t = self.Pw
        idx = self.get_time_idx(vot)
        P, w = P_t[idx], w_L_t[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)
        P, w = P_t[idx], w_0_t[idx]
        ax.plot([w], [P], 'o', color='magenta', markersize=10)

    def plot_G_t(self, ax, vot,
                 label_U='U(t)', label_W='W(t)',
                 color_U='blue', color_W='red'):

        t = self.t
        U_bar_t = self.get_U_bar_t()
        W_t = self.get_W_t()
        if len(W_t) == 0:
            return
        ax.plot(t, W_t, color=color_W, label=label_W)
        ax.plot(t, U_bar_t, color=color_U, label=label_U)
        ax.fill_between(t, W_t, U_bar_t, facecolor='gray', alpha=0.5,
                        label='G(t)')
        ax.set_ylabel('energy [Nmm]')
        ax.set_xlabel('time [-]')
        ax.legend()

    def plot_dG_t(self, ax, vot, *args, **kw):
        t = self.t
        dG = self.get_dG_t()
        ax.plot(t, dG, color='black', label='dG/dt')
        ax.fill_between(t, 0, dG, facecolor='blue', alpha=0.05)
        ax.legend()

    show_data = Button()

    def _show_data_fired(self):
        show_data = DataSheet(data=self.U_t)
        show_data.edit_traits()


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
                                    )
             ),
        width=0.5,
        height=0.6
    )


class PullOutModel(TStepBC, BMCSRootNode, Vis2D):

    hist_type = PulloutHist

    node_name = 'pull out simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.loading_scenario,
            self.mats_eval,
            self.cross_section,
            self.geometry,
            self.sim
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.loading_scenario,
            self.mats_eval,
            self.cross_section,
            self.geometry,
            self.sim
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

    @tr.on_trait_change(itags_str)
    def report_change(self):
        self.model_structure_changed = True

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
                            desc=r'displacement or force control: [u|f]',
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
                unit='-',
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

    mats_eval = Instance(IMATSEval, report=True,
                         desc='material model of the interface')
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

    domains = Property(depends_on=itags_str + 'model_structure_changed')

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

    fixed_dofs = Property(depends_on=itags_str)

    @cached_property
    def _get_fixed_dofs(self):
        if self.fixed_boundary == 'non-loaded end (matrix)':
            return [0]
        elif self.fixed_boundary == 'non-loaded end (reinf)':
            return [1]
        elif self.fixed_boundary == 'loaded end (matrix)':
            return [self.controlled_dof - 1]
        elif self.fixed_boundary == 'clamped left':
            return [0, 1]

    controlled_dof = Property(depends_on=itags_str)

    @cached_property
    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    free_end_dof = Property(depends_on=itags_str)

    @cached_property
    def _get_free_end_dof(self):
        return 1

    fixed_bc_list = Property(depends_on=itags_str)
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_bc_list(self):
        return [
            BCDof(node_name='fixed left end', var='u',
                  dof=dof, value=0.0) for dof in self.fixed_dofs
        ]

    control_bc = Property(depends_on=itags_str)
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCDof(node_name='pull-out displacement',
                     var=self.control_variable,
                     dof=self.controlled_dof, value=self.w_max,
                     time_function=self.loading_scenario)

    bc = Property(depends_on=itags_str)

    @cached_property
    def _get_bc(self):
        return [self.control_bc] + self.fixed_bc_list

    X_M = Property()

    def _get_X_M(self):
        state = self.fe_domain[0]
        return state.xdomain.x_Ema[..., 0].flatten()

    #=========================================================================
    # Getter functions @todo move to the PulloutStateRecord
    #=========================================================================

    P = tr.Property

    def _get_P(self):
        c_dof = self.controlled_dof
        return self.F_k[c_dof]

    w_L = tr.Property

    def _get_w_L(self):
        c_dof = self.controlled_dof
        return self.U_n[c_dof]

    w_0 = tr.Property

    def _get_w_0(self):
        f_dof = self.free_end_dof
        return self.U_n[f_dof]

    def get_u_p(self, vot):
        '''Displacement field
        '''
        idx = self.hist.get_time_idx(vot)
        U = self.hist.U_t[idx]
        state = self.fe_domain[0]
        dof_Epia = state.xdomain.o_Epia
        fets = state.xdomain.fets
        u_Epia = U[dof_Epia]
        N_mi = fets.N_mi
        u_Emap = np.einsum('mi,Epia->Emap', N_mi, u_Epia)
        return u_Emap.reshape(-1, 2)

    def get_eps_Ems(self, vot):
        '''Epsilon in the components'''
        state = self.fe_domain[0]
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
        txdomain = self.fe_domain[0]
        idx = self.hist.get_time_idx(vot)
        U = self.hist.U_t[idx]
        t_n1 = self.hist.t[idx]
        eps_Ems = txdomain.xdomain.map_U_to_field(U)
        state_vars_t = self.hist.state_vars[idx]
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
        sf_t_Em = np.array(self.tloop.sf_Em_record)
        w_ip = self.fets_eval.ip_weights
        J_det = self.tstepper.J_det
        P_b = self.cross_section.P_b
        shear_integ = np.einsum('tEm,m,em->t', sf_t_Em, w_ip, J_det) * P_b
        return shear_integ

    #=========================================================================
    # Plot functions
    #=========================================================================
    def plot_geo(self, ax, vot):
        u_p = self.get_u_p(vot).T

        f_dof = self.free_end_dof
        w_L_b = u_p.flatten()[f_dof]
        c_dof = self.controlled_dof
        w = u_p.flatten()[c_dof]

        A_m = self.cross_section.A_m
        A_f = self.cross_section.A_f
        h = A_m
        d = h * 0.1  # A_f / A_m

        L_b = self.geometry.L_x
        x_C = np.array([[-L_b, 0], [0, 0], [0, h], [-L_b, h]], dtype=np.float_)
        ax.fill(*x_C.T, color='gray', alpha=0.3)

        f_top = h / 2 + d / 2
        f_bot = h / 2 - d / 2
        ax.set_xlim(xmin=-1.05 * L_b,
                    xmax=max(0.05 * L_b, 1.1 * self.w_max))

        if False:
            a_val = self.get_aw_pull(vot)
            width = d * 0.5
            x_a = np.array([[a_val, f_bot - width], [0, f_bot - width],
                            [0, f_top + width], [a_val, f_top + width]],
                           dtype=np.float_)
            line_aw = ax.fill([], [], color='white', alpha=1)
            line_aw.set_xy(x_a)

        line_F, = ax.fill([], [], color='black', alpha=0.8)
        x_F = np.array([[-L_b + w_L_b, f_bot], [w, f_bot],
                        [w, f_top], [-L_b + w_L_b, f_top]], dtype=np.float_)
        line_F.set_xy(x_F)
        x_F0 = np.array([[-L_b, f_bot], [-L_b + w_L_b, f_bot],
                         [-L_b + w_L_b, f_top], [-L_b, f_top]], dtype=np.float_)
        line_F0, = ax.fill([], [], color='white', alpha=1)
        line_F0.set_xy(x_F0)

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
        color = 'green'
        ax.fill_between(X_J, 0, s, facecolor=color, alpha=0.3)
        ax.plot(X_J, s, linewidth=2, color=color)
        ax.set_ylabel('slip')
        ax.set_xlabel('bond length')
        return np.min(s), np.max(s)

    def plot_sf(self, ax, vot):
        X_J = self.X_M
        sf = self.get_sf(vot)
        color = 'red'
        ax.fill_between(X_J, 0, sf, facecolor=color, alpha=0.2)
        ax.plot(X_J, sf, linewidth=2, color=color)
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
        Pw = self.hist.plt('plot_Pw', label='pullout curve')
        geo = self.plt('plot_geo', label='geometry')
        u_p = self.plt('plot_u_p', label='displacement along the bond')
        eps_p = self.plt('plot_eps_p', label='strain along the bond')
        sig_p = self.plt('plot_sig_p', label='stress along the bond')
        s = self.plt('plot_s', label='slip along the bond')
        sf = self.plt('plot_sf', label='shear flow along the bond')
        energy = self.hist.plt('plot_G_t', label='energy')
        dissipation = self.hist.plt('plot_dG_t', label='energy release')
        pp0 = PlotPerspective(
            name='geo',
            viz2d_list=[geo],
            positions=[111],
        )
        pp1 = PlotPerspective(
            name='history',
            viz2d_list=[Pw, geo, energy, dissipation],
            positions=[221, 222, 223, 224],
        )
        pp2 = PlotPerspective(
            name='fields',
            viz2d_list=[s, u_p, eps_p, sig_p],
            twinx=[(s, sf, False)],
            positions=[221, 222, 223, 224],
        )
        win = BMCSWindow(model=self)
        win.viz_sheet.pp_list = [pp0, pp1, pp2]
        win.viz_sheet.selected_pp = pp0
        win.viz_sheet.monitor_chunk_size = 10
        return win
