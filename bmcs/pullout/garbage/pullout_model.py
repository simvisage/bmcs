'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.

'''

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.mats.mats_bondslip import \
    MATSBondSlipDP, MATSBondSlipMultiLinear
from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import \
    BCDof, FEGrid, BCSlice, TStepper, IMATSEval
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.core.tline import TLine
from ibvpy.core.vtloop import TimeLoop as TLoop
from ibvpy.dots.vdots_grid3d import DOTSGrid
from ibvpy.rtrace.rt_dof import RTDofGraph
from traits.api import \
    Property, Instance, cached_property, \
    Bool, List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item
from view.plot2d import Viz2D, Vis2D
from view.window import BMCSModel, BMCSWindow

import numpy as np
from .pullout import Viz2DPullOutFW, Viz2DPullOutField, \
    CrossSection, Geometry, PullOutModelBase


class PullOutModel(PullOutModelBase):

    #=========================================================================
    # Tree node attributes
    #=========================================================================
    node_name = 'pull out simulation'

    #=========================================================================
    # Interactive control of the time loop
    #=========================================================================
    def init(self):
        self.tloop.init()

    def eval(self):
        self.rt_Pu.clear()
        return self.tloop.eval()

    def pause(self):
        self.tloop.paused = True

    def stop(self):
        self.tloop.restart = True

    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type = Trait('multilinear',
                           {'damage-plasticity': MATSBondSlipDP,
                            'multilinear': MATSBondSlipMultiLinear},
                           MAT=True,
                           desc='material model type [damage-plasticity, multilinear]')

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()
        self._update_node_list()

    mats_eval = Instance(IMATSEval, report=True,
                         desc='object representing the material model')
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    material = Property

    def _get_material(self):
        return self.mats_eval

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS1D52ULRHFatigue),
                         depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRHFatigue(A_m=self.cross_section.A_m,
                                   P_b=self.cross_section.P_b,
                                   A_f=self.cross_section.A_f,
                                   mats_eval=self.mats_eval)

    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='BC,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bcond_mngr(self):
        bc_list = [self.fixed_bc,
                   self.control_bc]
        return BCondMngr(bcond_list=bc_list)

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
        return BCDof(node_name='pull-out displacement', var='u',
                     dof=self.controlled_dof, value=self.w_max,
                     time_function=self.loading_scenario)

    dots_grid = Property(Instance(DOTSGrid),
                         depends_on='CS,MAT,GEO,MESH,FE')
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        geo = self.geometry
        return DOTSGrid(
            L_x=geo.L_x,
            n_x=self.n_e_x,
            fets=self.fets_eval, mats=self.mats_eval
        )

    fe_grid = Property

    def _get_fe_grid(self):
        return self.dots_grid.mesh

#
#     fe_grid = Property(Instance(FEGrid), depends_on='MAT,GEO,MESH,FE')
#     '''Diescretization object.
#     '''
#     @cached_property
#     def _get_fe_grid(self):
#         # Element definition
#         return FEGrid(coord_max=(self.geometry.L_x,),
#                       shape=(self.n_e_x,),
#                       fets_eval=self.fets_eval)

    rt_Pu = Property(depends_on='BC,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_rt_Pu(self):
        return RTDofGraph(name='P(u)',
                          var_y='F_int', idx_y=self.controlled_dof,
                          var_x='U_k', idx_x=self.controlled_dof)

    tstepper = Property(Instance(TStepper),
                        depends_on='MAT,GEO,MESH,CS,ALG,BC')
    '''Objects representing the state of the model providing
    the predictor and corrector functionality needed for time-stepping
    algorithm.
    '''
    @cached_property
    def _get_tstepper(self):
        #self.fe_grid.dots = self.dots
        ts = TStepper(
            sdomain=self.fe_grid,
            bcond_mngr=self.bcond_mngr,
            rtrace_list=[self.rt_Pu,
                         #                         RTraceDomainField(name = 'Stress' ,
                         #                         var = 'sig_app', idx = 0,
                         #                         record_on = 'update'),
                         # RTraceDomainListField(name='Displacement',
                         #                      var='u', idx=0),
                         #                             RTraceDomainField(name = 'N0' ,
                         #                                          var = 'N_mtx', idx = 0,
                         # record_on = 'update')

                         ]

        )
        return ts

    tline = Instance(TLine)

    def _tline_default(self):
        # assign the parameters for solver and loading_scenario
        t_max = 1.0  # self.loading_scenario.t_max
        d_t = 0.02  # self.loading_scenario.d_t
        return TLine(min=0.0, step=d_t, max=t_max,
                     time_change_notifier=self.time_changed,
                     )

    fixed_dof = Property

    def _get_fixed_dof(self):
        if self.fixed_boundary == 'non-loaded end (matrix)':
            return 0
        elif self.fixed_boundary == 'non-loaded end (reinf)':
            return 1
        elif self.fixed_boundary == 'loaded end (matrix)':
            return self.controlled_dof - 1

    controlled_dof = Property

    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    free_end_dof = Property

    def _get_free_end_dof(self):
        return self.n_e_x + 1

    k_max = Int(400,
                unit='mm',
                symbol='k_{\max}',
                desc='maximum number of iterations',
                ALG=True)

    tolerance = Float(1e-4,
                      unit='-',
                      symbol='\epsilon',
                      desc='required accuracy',
                      ALG=True)

    tloop = Property(Instance(TLoop),
                     depends_on='MAT,GEO,MESH,CS,TIME,ALG,BC')
    '''Algorithm controlling the time stepping.
    '''
    @cached_property
    def _get_tloop(self):
        k_max = self.k_max

        tolerance = self.tolerance
        return TLoop(ts=self.dots_grid, k_max=k_max,
                     tolerance=tolerance,
                     tline=self.tline,
                     bc_mngr=self.bcond_mngr)

    def get_d_ECid(self, vot):
        '''Get the displacements as a four-dimensional array 
        corresponding to element, material component, node, spatial dimension
        '''
        idx = self.tloop.get_time_idx(vot)
        d = self.tloop.U_record[idx]
        dof_ECid = self.tstepper.dof_ECid
        return d[dof_ECid]

    X_M = Property

    def _get_X_M(self):
        return self.tstepper.tse_integ.dots_list[0].X_M

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
        F_array = np.array(self.tloop.F_int_record, dtype=np.float_)
        return F_array[:, self.controlled_dof]

    def get_w_t(self):
        d_t = self.tloop.U_record
        dof_ECid = self.tstepper.dof_ECid
        d_t_ECid = d_t[:, dof_ECid]
        w_0 = d_t_ECid[:, 0, 1, 0, 0]
        w_L = d_t_ECid[:, -1, 1, -1, -1]
        return w_0, w_L

        U_array = np.array(self.tloop.U_record, dtype=np.float_)
        return U_array[:, (self.free_end_dof, self.controlled_dof)]

    def get_w(self, vot):
        '''Damage variables
        '''
        idx = self.tloop.get_time_idx(vot)
        w_Emd = self.tloop.w_record[idx]
        return w_Emd.flatten()

    def plot_u_C(self, ax, vot):
        X_M = self.X_M
        L = self.geometry.L_x
        u_C = self.get_u_C(vot).T
        ax.plot(X_M, u_C[0], linewidth=2, color='blue', label='matrix')
        ax.fill_between(X_M, 0, u_C[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, u_C[1], linewidth=2, color='orange', label='reinf')
        ax.fill_between(X_M, 0, u_C[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('displacement')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)

    def plot_eps_C(self, ax, vot):
        X_M = self.X_M
        L = self.geometry.L_x
        eps_C = self.get_eps_C(vot).T
        ax.plot(X_M, eps_C[0], linewidth=2, color='blue',)
        ax.fill_between(X_M, 0, eps_C[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, eps_C[1], linewidth=2, color='orange',)
        ax.fill_between(X_M, 0, eps_C[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('strain')
        ax.set_xlabel('bond length')

    def plot_sig_C(self, ax, vot):
        X_M = self.X_M
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
        ax.set_ylabel('stress flow')
        ax.set_xlabel('bond length')

    def plot_s(self, ax, vot):
        X_J = self.X_J
        s = self.get_s(vot)
        ax.fill_between(X_J, 0, s, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, s, linewidth=2, color='lightcoral')
        ax.set_ylabel('slip')
        ax.set_xlabel('bond length')

    def plot_sf(self, ax, vot):
        X_J = self.X_J
        sf = self.get_sf(vot)
        ax.fill_between(X_J, 0, sf, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, sf, linewidth=2, color='lightcoral')
        ax.set_ylabel('shear flow')
        ax.set_xlabel('bond length')

    def plot_w(self, ax, vot):
        X_J = self.X_J
        w = self.get_w(vot)
        ax.fill_between(X_J, 0, w, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, w, linewidth=2, color='lightcoral')
        ax.set_ylabel('damage')
        ax.set_xlabel('bond length')

    traits_view = View(Item('mats_eval_type'),)

    tree_view = traits_view

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW,
                     'load function': Viz2DLoadControlFunction,
                     }


def run_pullout_dp():
    po = PullOutModel(n_e_x=10, k_max=500)
    po.w_max = 0.01
    po.mats_eval_type = 'damage-plasticity'
    po.tline.step = 0.01
    po.geometry.L_x = 1.0
    po.loading_scenario.set(loading_type='monotonic')
    po.material.set(K=100000.0, gamma=-0.0)
    po.material.set(tau_bar=3.5, E_m=35000.0, E_f=170000.0, E_b=6700.0)
    po.material.omega_fn_type = 'li'
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=1000.0, plot_max=0.01)
    po.run()
#     Pu = po.rt_Pu
#     w = BMCSWindow(model=po)
#     po.add_viz2d('load function', 'load-time')
#     po.add_viz2d('F-w', 'load-displacement')
#     Pu.add_viz2d('diagram', 'Pu')
#     w.offline = False
#     w.finish_event = True
#     w.configure_traits()


def run_pullout_multilinear():
    po = PullOutModel(n_e_x=8, k_max=500)
    po.w_max = 20.0
    A_f = 16.44
    po.cross_section.A_f = A_f
    r = np.sqrt(A_f / np.pi)
    po.cross_section.P_b = 2 * r * np.pi
    po.mats_eval_type = 'multilinear'
    po.tline.step = 0.01
    po.geometry.L_x = 1.0
    po.loading_scenario.set(loading_type='monotonic')

    # todo
    po.material.set(s_data='0, 1.0, 5.0, 10.0, 20.0',
                    tau_data='0, 14.0, 16.5, 15.0, 7.0')
    po.material.update_bs_law = True
    Pu = po.rt_Pu
    w = BMCSWindow(model=po)
    Pu.add_viz2d('diagram')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    w.offline = False
    w.finish_event = True
    w.configure_traits()


def run_with_new_state():
    po = PullOutModel(n_e_x=100, k_max=1000, w_max=0.05)
    po.mats_eval_type = 'damage-plasticity'
    po.tline.step = 0.001
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=5)
    po.geometry.L_x = 45.0
    po.cross_section.set(A_f=64.0, P_b=28.0, A_m=28000.0)
    po.material.set(K=-0.0, gamma=-500, tau_bar=13.137)
    po.material.set(E_m=35000.0, E_f=170000.0, E_b=6700.0)
    po.material.omega_fn_type = 'li'
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=100.0, plot_max=0.01)
    po.run()
    po.rt_Pu.trace.configure_traits()


def run_pullout_multi(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      n_e_x=2, k_max=1000, w_max=0.1)
    po.mats_eval_type = 'multilinear'
    po.tline.step = 0.1
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 20.0',
                     tau_data='0, 800, 0, 0')
    po.mats_eval.update_bs_law = True
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function', 'load-time')
    po.add_viz2d('F-w', 'load-displacement')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')

    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    # run_pullout_multi()
    # run_with_new_state()
    run_pullout_dp()
