'''
Created on 12.01.2016
@author: Yingxiong, ABaktheer, RChudoba

@todo: enable recalculation after the initial offline run
@todo: reset viz adapters upon recalculation to forget their axes lims
@todo: introduce a switch for left and right supports
'''

from bmcs.bond_slip.bond_material_params import MaterialParams
from bmcs.bond_slip.mats_bondslip import MATSBondSlipDP
from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.mats.mats_bondslip import MATSEvalFatigue
from bmcs.mats.tloop import TLoop
from bmcs.mats.tstepper import TStepper
from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import BCDof, FEGrid, BCSlice
from ibvpy.core.bcond_mngr import BCondMngr
from traits.api import \
    Property, Instance, cached_property, \
    Bool, List, Float, Trait, Int
from traitsui.api import \
    View, Item
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow, TLine

import numpy as np


class CrossSection(BMCSLeafNode):
    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    A_m = Float(15240,
                CS=True,
                input=True,
                auto_set=False, enter_set=True,
                desc='matrix area [mm2]')
    A_f = Float(153.9,
                CS=True,
                input=True,
                auto_set=False, enter_set=True,
                desc='reinforcement area [mm2]')
    P_b = Float(44,
                CS=True,
                input=True,
                auto_set=False, enter_set=True,
                desc='perimeter of the bond interface [mm]')

    view = View(
        Item('A_m'),
        Item('A_f'),
        Item('P_b')
    )

    tree_view = view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    L_x = Float(45,
                GEO=True,
                input=True,
                auto_set=False, enter_set=True,
                desc='embedded length')

    view = View(
        Item('L_x'),
    )

    tree_view = view


class Viz2DPullOutFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    def plot(self, ax, vot, *args, **kw):
        idx = self.vis2d.tloop.get_time_idx(vot)
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
        P, w = P_t[idx], w_L[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)
        P, w = P_t[idx], w_0[idx]
        ax.plot([w], [P], 'o', color='magenta', markersize=10)
        ax.legend(loc=4)


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
                     'w': 'plot_w',
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
            self.bcond_mngr,
        ]

    def init(self):
        self.tloop.init()

    def eval(self):
        return self.tloop.eval()

    def pause(self):
        self.tloop.paused = True

    def stop(self):
        self.tloop.restart = True

    material = Instance(MaterialParams)

    def _material_default(self):
        return MaterialParams()

    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    cross_section = Instance(CrossSection)

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    n_e_x = Int(20, auto_set=False, enter_set=True)

    w_max = Float(1, auto_set=False, enter_set=True)

    free_end_dof = Property

    def _get_free_end_dof(self):
        return self.n_e_x + 1

    controlled_dof = Property

    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    fixed_dof = Property

    def _get_fixed_dof(self):
        #fe_grid = self.tstepper.sdomain
        #fe_grid[-1, -1].dofs
        return 2 + 2 * self.n_e_x - 1

    mats_eval = Property(Instance(MATSEvalFatigue),
                         depends_on='MAT')
    '''Material model'''
    @cached_property
    def _get_mats_eval(self):
        # assign the material parameters
        print 'new material model'
        return MATSEvalFatigue(E_b=self.material.E_b,
                               gamma=self.material.gamma,
                               S=self.material.S,
                               tau_pi_bar=self.material.tau_pi_bar,
                               r=self.material.r,
                               K=self.material.K,
                               c=self.material.c,
                               a=self.material.a,
                               pressure=self.material.pressure)


#     mats_eval = Property(Instance(MATSBondSlipDP),
#                          depends_on='MAT')
#     '''Material model'''
#     @cached_property
#     def _get_mats_eval(self):
#         # assign the material parameters
#         print 'new material model'
#         return MATSBondSlipDP(E_b=self.material.E_b,
#                                gamma=self.material.gamma,
#                                tau_bar=self.material.tau_bar,
#                                K=self.material.K,
#                                )

    fets_eval = Property(Instance(FETS1D52ULRHFatigue),
                         depends_on='CS')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRHFatigue(A_m=self.cross_section.A_m,
                                   P_b=self.cross_section.P_b,
                                   A_f=self.cross_section.A_f)

    bcond_mngr = Instance(BCondMngr)
    '''Boundary condition manager
    '''

    def _bcond_mngr_default(self):
        bc_list = [BCDof(node_name='fixed left end', var='u',
                         dof=0, value=0.0),
                   BCDof(node_name='pull-out displacement', var='u',
                         dof=self.controlled_dof, value=self.w_max,
                         time_function=self.loading_scenario)]
        return BCondMngr(bcond_list=bc_list)

    control_bc = Property(depends_on='BC')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return self.bcond_mngr.bcond_list[1]

    fe_grid = Property(Instance(FEGrid), depends_on='GEO,MESH,FE')
    '''Diescretization object.
    '''
    @cached_property
    def _get_fe_grid(self):
        # Element definition
        return FEGrid(coord_max=(self.geometry.L_x,),
                      shape=(self.n_e_x,),
                      fets_eval=self.fets_eval)

    tstepper = Property(Instance(TStepper),
                        depends_on='MAT,GEO,MESH,CS,TIME,ALG,BC')
    '''Objects representing the state of the model providing
    the predictor and corrector functionality needed for time-stepping
    algorithm.
    '''
    @cached_property
    def _get_tstepper(self):
        return TStepper(mats_eval=self.mats_eval,
                        fets_eval=self.fets_eval,
                        sdomain=self.fe_grid,
                        bcond_mngr=self.bcond_mngr
                        )

    tline = Instance(TLine)

    def _tline_default(self):
        # assign the parameters for solver and loading_scenario
        t_max = 1.0  # self.loading_scenario.t_max
        d_t = 0.02  # self.loading_scenario.d_t
        return TLine(min=0.0, step=d_t, max=t_max,
                     time_change_notifier=self.time_changed,
                     )

    k_max = Int(200,
                ALG=True)
    tolerance = Float(1e-4,
                      ALG=True)

    tloop = Property(Instance(TLoop),
                     depends_on='MAT,GEO,MESH,CS,TIME,ALG,BC')
    '''Algorithm controlling the time stepping.
    '''
    @cached_property
    def _get_tloop(self):

        k_max = self.k_max
        tolerance = self.tolerance
        return TLoop(ts=self.tstepper, k_max=k_max,
                     tolerance=tolerance,
                     tline=self.tline)

    def get_d_ECid(self, vot):
        '''Get the displacements as a four-dimensional array 
        corresponding to element, material component, node, spatial dimension
        '''
        idx = self.tloop.get_time_idx(vot)
        d = self.tloop.U_record[idx]
        dof_ECid = self.tstepper.dof_ECid
        return d[dof_ECid]

    def get_u_C(self, vot):
        '''Displacement field
        '''
        d_ECid = self.get_d_ECid(vot)
        N_mi = self.fets_eval.N_mi
        u_EmdC = np.einsum('mi,ECid->EmdC', N_mi, d_ECid)
        return u_EmdC.reshape(-1, 2)

    def get_eps_C(self, vot):
        '''Epsilon in the components'''
        d_ECid = self.get_d_ECid(vot)
        eps_EmdC = np.einsum('Eimd,ECid->EmdC', self.tstepper.dN_Eimd, d_ECid)
        return eps_EmdC.reshape(-1, 2)

    def get_sig_C(self, vot):
        '''Get streses in the components 
        @todo: unify the index protocol
        for eps and sig. Currently eps uses multi-layer indexing, sig
        is extracted from the material model format.
        '''
        idx = self.tloop.get_time_idx(vot)
        return self.tloop.sig_EmC_record[idx].reshape(-1, 2)

    def get_s(self, vot):
        '''Slip between the two material phases'''
        d_ECid = self.get_d_ECid(vot)
        s_Emd = np.einsum('Cim,ECid->Emd', self.tstepper.sN_Cim, d_ECid)
        return s_Emd.flatten()

    def get_sf(self, vot):
        '''Get the shear flow in the interface
        @todo: unify the index protocol
        for eps and sig. Currently eps uses multi-layer indexing, sig
        is extracted from the material model format.
        '''
        idx = self.tloop.get_time_idx(vot)
        sf = self.tloop.sf_Em_record[idx].flatten()
        return sf

    def get_P_t(self):
        F_array = np.array(self.tloop.F_record, dtype=np.float_)
        return F_array[:, self.controlled_dof]

    def get_w_t(self):
        d_t = self.tloop.U_record
        dof_ECid = self.tstepper.dof_ECid
        d_t_ECid = d_t[:, dof_ECid]
        w_0 = d_t_ECid[:, 0, 1, 0, 0]
        w_L = d_t_ECid[:, -1, 1, -1, -1]
        return w_0, w_L

    def get_w(self, vot):
        '''Damage variables
        '''
        idx = self.tloop.get_time_idx(vot)
        w_Emd = self.tloop.w_record[idx]
        return w_Emd.flatten()

    def plot_u_C(self, ax, vot):
        X_M = self.tstepper.X_M
        L = self.geometry.L_x
        u_C = self.get_u_C(vot).T
        ax.plot(X_M, u_C[0], linewidth=2, color='blue', label='matrix')
        ax.fill_between(X_M, 0, u_C[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, u_C[1], linewidth=2, color='orange', label='reinf')
        ax.fill_between(X_M, 0, u_C[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('displacement: u [mm]')
        ax.set_xlabel('bond length: x [mm]')
        ax.legend(loc=2)

    def plot_eps_C(self, ax, vot):
        X_M = self.tstepper.X_M
        L = self.geometry.L_x
        eps_C = self.get_eps_C(vot).T
        ax.plot(X_M, eps_C[0], linewidth=2, color='blue',)
        ax.fill_between(X_M, 0, eps_C[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, eps_C[1], linewidth=2, color='orange',)
        ax.fill_between(X_M, 0, eps_C[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('strain: eps [-]')
        ax.set_xlabel('bond length: x [mm]')

    def plot_sig_C(self, ax, vot):
        X_M = self.tstepper.X_M
        sig_C = self.get_sig_C(vot).T
        A_m = self.cross_section.A_m
        A_f = self.cross_section.A_f
        L = self.geometry.L_x
        F_m = A_m * sig_C[0]
        F_f = A_f * sig_C[1]
        ax.plot(X_M, F_m, linewidth=2, color='blue', )
        ax.fill_between(X_M, 0, F_m, facecolor='blue', alpha=0.2)
        ax.plot(X_M, F_f, linewidth=2, color='orange')
        ax.fill_between(X_M, 0, F_f, facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('stress flow: A * sig [N]')
        ax.set_xlabel('bond length: x [mm]')

    def plot_s(self, ax, vot):
        X_J = self.tstepper.X_J
        s = self.get_s(vot)
        ax.fill_between(X_J, 0, s, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, s, linewidth=2, color='lightcoral')
        ax.set_ylabel('slip: s [mm]')
        ax.set_xlabel('bond length: x [mm]')

    def plot_sf(self, ax, vot):
        X_J = self.tstepper.X_J
        sf = self.get_sf(vot)
        ax.fill_between(X_J, 0, sf, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, sf, linewidth=2, color='lightcoral')
        ax.set_ylabel('shear flow: shear flow [MPa]')
        ax.set_xlabel('bond length: x [mm]')

    def plot_w(self, ax, vot):
        X_J = self.tstepper.X_J
        w = self.get_w(vot)
        ax.fill_between(X_J, 0, w, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, w, linewidth=2, color='lightcoral', label='bond')
        ax.set_ylabel('damage')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)

    def plot_eps_s(self, ax, vot):
        eps_C = self.get_eps_C(vot).T
        s = self.get_s(vot)
        ax.plot(eps_C[1], s, linewidth=2, color='lightcoral')
        ax.set_ylabel('reinforcement strain')
        ax.set_xlabel('slip')

    trait_view = View(Item('fets_eval'),
                      )

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW,
                     'load function': Viz2DLoadControlFunction,
                     }


def run_pullout():
    po = PullOutModel(n_e_x=100, k_max=500)
    po.tline.step = 0.005
    po.bcond_mngr.bcond_list[1].value = 0.01

    po.loading_scenario.set(loading_type='cyclic',
                            amplitude_type='constant'
                            )
    po.loading_scenario.set(number_of_cycles=6,
                            amplitude_type='increasing',
                            maximum_loading=1)
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function')
    po.add_viz2d('F-w')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'w', plot_fn='w')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')

    w.offline = False
    w.finish_event = True
    w.configure_traits()


if __name__ == '__main__':
    run_pullout()
