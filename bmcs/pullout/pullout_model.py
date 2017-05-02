'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.

'''

from bmcs.bond_slip.mats_bondslip import MATSBondSlipDP
from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.mats.mats_bondslip import MATSEvalFatigue
from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import BCDof, FEGrid, BCSlice, TStepper, TLoop
from ibvpy.core.bcond_mngr import BCondMngr
from traits.api import \
    Property, Instance, cached_property, \
    Bool, List, Float, Trait, Int
from traitsui.api import \
    View, Item
from view.plot2d import Viz2D, Vis2D
from view.window import BMCSModel, BMCSWindow

from bmcs.bond_slip.bond_material_params import MaterialParams
from ibvpy.core.tline import TLine
import numpy as np
from pullout import \
    CrossSection, Geometry, Viz2DPullOutFW, Viz2DPullOutField


class PullOutModel(BMCSModel, Vis2D):

    #=========================================================================
    # Tree node attributes
    #=========================================================================
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

    #=========================================================================
    # Interactive control of the time loop
    #=========================================================================
    def init(self):
        self.tloop.init()

    def eval(self):
        return self.tloop.eval()

    def pause(self):
        self.tloop.paused = True

    def stop(self):
        self.tloop.restart = True

    #=========================================================================
    # Test setup parameters
    #=========================================================================
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

    #=========================================================================
    # Discretization
    #=========================================================================
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

    #=========================================================================
    # Material model
    #=========================================================================
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

#     dots = Property(Instance(DOTSGridEval), depends_on='+input')
#
#     @cached_property
#     def _get_dots(self):
#         return DOTSGridEval(fets_eval=self.fets_eval,
#                             mats_eval=self.mats_eval,
#                             sdomain=self.fe_grid
#                             )
    tstepper = Property(Instance(TStepper),
                        depends_on='MAT,GEO,MESH,CS,ALG,BC')
    '''Objects representing the state of the model providing
    the predictor and corrector functionality needed for time-stepping
    algorithm.
    '''
    @cached_property
    def _get_tstepper(self):
        #self.fe_grid.dots = self.dots
        print 'new tstepper'
        ts = TStepper(
            sdomain=self.fe_grid,
            bcond_mngr=self.bcond_mngr
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
        print 'new tloop', self.tstepper
        return TLoop(tstepper=self.tstepper, k_max=k_max,
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

        U_array = np.array(self.tloop.U_record, dtype=np.float_)
        return U_array[:, (self.free_end_dof, self.controlled_dof)]

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
        ax.set_ylabel('displacement')
        ax.set_xlabel('bond length')
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
        ax.set_ylabel('strain')
        ax.set_xlabel('bond length')

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
        ax.set_ylabel('stress flow')
        ax.set_xlabel('bond length')

    def plot_s(self, ax, vot):
        X_J = self.tstepper.X_J
        s = self.get_s(vot)
        ax.fill_between(X_J, 0, s, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, s, linewidth=2, color='lightcoral')
        ax.set_ylabel('slip')
        ax.set_xlabel('bond length')

    def plot_sf(self, ax, vot):
        X_J = self.tstepper.X_J
        sf = self.get_sf(vot)
        ax.fill_between(X_J, 0, sf, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, sf, linewidth=2, color='lightcoral')
        ax.set_ylabel('shear flow')
        ax.set_xlabel('bond length')

    def plot_w(self, ax, vot):
        X_J = self.tstepper.X_J
        w = self.get_w(vot)
        ax.fill_between(X_J, 0, w, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, w, linewidth=2, color='lightcoral')
        ax.set_ylabel('damage')
        ax.set_xlabel('bond length')

    trait_view = View(Item('fets_eval'),
                      )

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW,
                     'load function': Viz2DLoadControlFunction,
                     }


def run_pullout():
    po = PullOutModel(n_e_x=100, k_max=500)
    po.tline.step = 0.01
    po.bcond_mngr.bcond_list[1].value = 0.01
    po.init()
    print po.tstepper.sdomain.dots.dots_list[0].dots_integ.state_array.shape


if __name__ == '__main__':
    run_pullout()
