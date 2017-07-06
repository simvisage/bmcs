'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.
'''

from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import \
    FEGrid, BCSlice, TStepper, TLoop, IMATSEval
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.core.tline import TLine
from ibvpy.fets.fets2D import FETS2D4Q
from ibvpy.mats.mats2D.mats2D_elastic import MATS2DElastic
from ibvpy.rtrace.rt_dof import RTDofGraph
from traits.api import \
    Property, Instance, cached_property, \
    Bool, List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow

import numpy as np


class CrossSection(BMCSLeafNode):
    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    b = Float(50.0,
              CS=True,
              auto_set=False, enter_set=True,
              desc='cross-section width [mm2]')
    h = Float(100.0,
              CS=True,
              auto_set=False, enter_set=True,
              desc='cross section height [mm2]')

    view = View(
        Item('h'),
        Item('b'),
    )

    tree_view = view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    L = Float(500.0,
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')

    view = View(
        Item('L_x'),
    )

    tree_view = view


class BendingTestModel(BMCSModel, Vis2D):

    #=========================================================================
    # Tree node attributes
    #=========================================================================
    node_name = 'bending test simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
            self.fixed_left_bc,
            self.fixed_right_bc,
            self.control_bc,
            self.rt_Pu
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
            self.fixed_left_bc,
            self.fixed_right_bc,
            self.control_bc,
            self.rt_Pu
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
    n_e_y = Int(4, auto_set=False, enter_set=True)

    w_max = Float(1, BC=True, auto_set=False, enter_set=True)

    free_end_dof = Property

    def _get_free_end_dof(self):
        return self.n_e_x + 1

    controlled_dof = Property

    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    fixed_dof = Property

    def _get_fixed_dof(self):
        return 0

    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type = Trait('elastic',
                           {'elastic': MATS2DElastic
                            },
                           MAT=True
                           )

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()

    @on_trait_change('BC,MAT,MESH')
    def reset_node_list(self):
        self._update_node_list()

    mats_eval = Instance(IMATSEval,
                         MAT=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    material = Property

    def _get_material(self):
        return self.mats_eval

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS2D4Q),
                         depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS2D4Q(h=self.cross_section.h,
                        b=self.cross_section.b,
                        mats_eval=self.mats_eval)

    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='BC,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bcond_mngr(self):
        bc_list = [self.fixed_left_bc,
                   self.fixed_right_bc,
                   self.control_bc]
        return BCondMngr(bcond_list=bc_list)

    fixed_left_bc = Property(depends_on='BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_left_bc(self):
        return BCSlice(var='u', value=0., dims=[0, 1],
                       slice=self.fe_grid[0, 0, 0, 0])

    fixed_right_bc = Property(depends_on='BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_right_bc(self):
        return BCSlice(var='u', value=0., dims=[1],
                       slice=self.fe_grid[-1, 0, -1, 0])

    control_bc = Property(depends_on='BC,GEO,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(var='u', value=self.w_max, dims=[1],
                       slice=self.fe_grid[self.n_e_x / 2, -1, :, -1])

    fe_grid = Property(Instance(FEGrid), depends_on='MAT,GEO,MESH,FE')
    '''Diescretization object.
    '''
    @cached_property
    def _get_fe_grid(self):
        # Element definition
        return FEGrid(coord_max=(self.geometry.L, self.cross_section.h),
                      shape=(self.n_e_x, self.n_e_y),
                      fets_eval=self.fets_eval)

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
        ts = TStepper(
            sdomain=self.fe_grid,
            bcond_mngr=self.bcond_mngr,
            rtrace_list=[self.rt_Pu,
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
        return TLoop(tstepper=self.tstepper, k_max=k_max,
                     tolerance=tolerance, debug=False,
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

    traits_view = View(Item('mats_eval_type'),)

    tree_view = traits_view


def run_bending3pt_elastic():
    po = BendingTestModel(n_e_x=10, k_max=500,
                          mats_eval_type='elastic')

    po.w_max = 0.01
    po.tline.step = 0.01
    po.geometry.L = 1.0
    po.loading_scenario.set(loading_type='monotonic')
    Pu = po.rt_Pu
    w = BMCSWindow(model=po)
    Pu.add_viz2d('diagram')

    print po.control_bc.slice
    print po.control_bc.slice.fe_grid
    po.run()
    return

    w.offline = False
    w.finish_event = True
    w.configure_traits()


if __name__ == '__main__':
    run_bending3pt_elastic()
    # run_with_new_state()
