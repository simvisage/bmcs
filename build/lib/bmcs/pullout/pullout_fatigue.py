'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.mats.mats_bondslip import MATSBondSlipFatigue, MATSBondSlipDP, MATSBondSlipMultiLinear, MATSBondSlipFRPDamage
from bmcs.mats.tloop_dp import TLoop
from bmcs.mats.tstepper_dp import TStepper
from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import BCDof, IMATSEval
from ibvpy.core.bcond_mngr import BCondMngr
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from reporter import Reporter, RInputRecord
from scipy import interpolate as ip
from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Int, Trait, on_trait_change
from traitsui.api import \
    View, Item, Group
from view.plot2d import Vis2D, Viz2D
from view.window import BMCSModel, BMCSWindow, TLine

import numpy as np
from .pullout import Viz2DPullOutFW, Viz2DPullOutField, \
    Viz2DEnergyPlot, Viz2DEnergyReleasePlot, \
    CrossSection, Geometry, PullOutModelBase


class PullOutModel(PullOutModelBase):

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
            Item('w_max', resizable=True, full_size=True),
            Group(
                Item('loading_scenario@', show_label=False),
            )
        )
    )

    tline = Instance(TLine, report=True)

    def _tline_default(self):
        # assign the parameters for solver and loading_scenario
        t_max = 1.0  # self.loading_scenario.t_max
        d_t = 0.02  # self.loading_scenario.d_t
        return TLine(min=0.0, step=d_t, max=t_max,
                     time_change_notifier=self.time_changed,
                     )

    loading_scenario = Instance(LoadingScenario,
                                report=True)

    def _loading_scenario_default(self):
        return LoadingScenario()

    cross_section = Instance(CrossSection, report=True)

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(Geometry, report=True)

    def _geometry_default(self):
        return Geometry()

    n_e_x = Int(20, MESH=True, auto_set=False, enter_set=True)

    w_max = Float(1, BC=True, auto_set=False, enter_set=True)

    controlled_dof = Property

    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    mats_eval_type = Trait('Fatigue',
                           {'damage-plasticity': MATSBondSlipDP,
                            'multilinear': MATSBondSlipMultiLinear,
                            'FRP-damage': MATSBondSlipFRPDamage,
                            'Fatigue': MATSBondSlipFatigue})

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()
        self._update_node_list()

    mats_eval = Instance(IMATSEval, report=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    material = Property

    def _get_material(self):
        return self.mats_eval

    fets_eval = Property(Instance(FETS1D52ULRHFatigue),
                         depends_on='CS')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRHFatigue(A_m=self.cross_section.A_m,
                                   P_b=self.cross_section.P_b,
                                   A_f=self.cross_section.A_f)

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
        return BCDof(node_name='pull-out force', var='u',
                     dof=self.controlled_dof, value=self.w_max,
                     time_function=self.loading_scenario)

    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='BC,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bcond_mngr(self):
        bc_list = [self.fixed_bc,
                   self.control_bc]
        return BCondMngr(bcond_list=bc_list)

    tstepper = Property(Instance(TStepper),
                        depends_on='MAT,GEO,MESH,CS,TIME,ALG,BC')
    '''Objects representing the state of the model providing
    the predictor and corrector functionality needed for time-stepping
    algorithm.
    '''
    @cached_property
    def _get_tstepper(self):
        return TStepper(node_name='Pull-out',
                        n_e_x=self.n_e_x,
                        mats_eval=self.mats_eval,
                        fets_eval=self.fets_eval,
                        L_x=self.geometry.L_x,
                        bcond_mngr=self.bcond_mngr
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

    def time_changed(self, time):
        if self.ui != None:
            self.ui.viz_sheet.time_changed(time)

    def time_range_changed(self, tmax):
        self.tline.max = tmax
        self.ui.viz_sheet.time_range_changed(tmax)

    def set_tmax(self, time):
        self.time_range_changed(time)

    def init(self):
        self.tloop.init()

    def eval(self):
        return self.tloop.eval()

    def pause(self):
        self.tloop.paused = True

    def stop(self):
        self.tloop.restart = True

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
        return self.tstepper.X_M

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

    def get_shear_integ(self):
        #         d_ECid = self.get_d_ECid(vot)
        #         s_Emd = np.einsum('Cim,ECid->Emd', self.tstepper.sN_Cim, d_ECid)
        #         idx = self.tloop.get_time_idx(vot)
        #         sf = self.tloop.sf_Em_record[idx]

        sf_t_Em = np.array(self.tloop.sf_Em_record)
        w_ip = self.fets_eval.ip_weights
        J_det = self.tstepper.J_det
        sN_Cim = self.tstepper.sN_Cim
        P_b = self.cross_section.P_b
        shear_integ = np.einsum('tEm,m,em->t', sf_t_Em, w_ip, J_det) * P_b
        return shear_integ

    def get_W_t(self):
        P_t = self.get_P_t()
        w_0, w_L = self.get_w_t()

        W_t = []
        for i, w in enumerate(w_L):
            W_t.append(np.trapz(P_t[:i + 1], w_L[:i + 1]))
        return W_t

    def get_U_bar_t(self):
        A = self.tstepper.A
        sig_t = np.array(self.tloop.sig_record)
        eps_t = np.array(self.tloop.eps_record)
        w_ip = self.fets_eval.ip_weights
        J_det = self.tstepper.J_det
        U_bar_t = np.einsum('m,Em,s,tEms,tEms->t',
                            w_ip, J_det, A, sig_t, eps_t)

        return U_bar_t / 2.0

    def get_dG_t(self):
        t = self.get_t()
        U_bar_t = self.get_U_bar_t()
        W_t = self.get_W_t()

        G = W_t - U_bar_t
        tck = ip.splrep(t, G, s=0, k=1)
        return ip.splev(t, tck, der=1)

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

    def get_wL_t(self):
        w_0, w_L = self.get_w_t()
        return w_L

    def get_omega(self, vot):
        '''Damage variables
        '''
        idx = self.tloop.get_time_idx(vot)
        w_Emd = self.tloop.omega_record[idx]
        return w_Emd.flatten()

    t = Property

    def _get_t(self):
        return self.get_t()

    def get_t(self):
        return np.array(self.tloop.t_record, dtype=np.float_)

    sig_tC = Property

    def _get_sig_tC(self):
        n_t = len(self.tloop.t_record)
        sig_tEmC = np.array(self.tloop.sig_EmC_record, dtype=np.float_)
        sig_tC = sig_tEmC.reshape(n_t, -1, 2)
        return sig_tC

    trait_view = View(Item('fets_eval'),
                      )

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW,
                     'load function': Viz2DLoadControlFunction,
                     'dissipation': Viz2DEnergyPlot,
                     'dissipation rate': Viz2DEnergyReleasePlot
                     }


def run_pullout_fatigue(*args, **kw):
    po = PullOutModel(n_e_x=200, k_max=500, w_max=1.0)
    po.tline.step = 0.001
    po.geometry.L_x = 42.0
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=20)
    po.loading_scenario.set(maximum_loading=19200)
    po.loading_scenario.set(unloading_ratio=0.1)
    po.loading_scenario.set(amplitude_type="constant")
    po.loading_scenario.set(loading_range="non-symmetric")
    po.cross_section.set(A_f=153.9, P_b=44, A_m=15400.0)
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function', 'Load-time')
    po.add_viz2d('F-w', 'Load-displacement')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'omega', plot_fn='omega')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


def test_reporter():
    po = PullOutModel(n_e_x=100, k_max=500, w_max=1.0)
    po.tline.step = 0.01
    po.geometry.L_x = 500.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function', 'Load-time')
    po.add_viz2d('F-w', 'Load-displacement')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'omega', plot_fn='omega')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    r = Reporter(report_items=[po, w.viz_sheet])
    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()


if __name__ == '__main__':
    run_pullout_fatigue()
    # test_reporter()
