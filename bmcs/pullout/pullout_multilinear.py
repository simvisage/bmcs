'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from bmcs.mats.mats_bondslip import MATSBondSlipDP, MATSBondSlipMultiLinear
from bmcs.mats.tloop_dp import TLoop
from bmcs.mats.tstepper_dp import TStepper
from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import BCDof, IMATSEval
from ibvpy.core.bcond_mngr import BCondMngr
from scipy import interpolate as ip
from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Int, Trait, on_trait_change
from traitsui.api import \
    View, Item, Group
from view.plot2d import Vis2D, Viz2D
from view.window import BMCSModel, BMCSWindow, TLine

import numpy as np
from pullout import Viz2DPullOutFW, Viz2DPullOutField, \
    CrossSection, Geometry, PullOutModelBase


class Viz2DEnergyPlot(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'line plot'

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d.get_t()
        U_bar_t = self.vis2d.get_U_bar_t()
        W_t = self.vis2d.get_W_t()
        ax.plot(t, U_bar_t, color='black', label='U(t)')
        ax.plot(t, W_t, color='black', label='W(t)')
        ax.fill_between(t, W_t, U_bar_t, facecolor='blue', alpha=0.2)
        ax.legend()


class Viz2DEnergyRatesPlot(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'line plot'

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d.get_t()

        dG = self.vis2d.get_dG_t()

        ax.plot(t, dG, color='black', label='dG/dt')
        ax.fill_between(t, 0, dG, facecolor='blue', alpha=0.2)
        ax.legend()


class PullOutModel(PullOutModelBase):

    mats_eval_type = Trait('multilinear',
                           {'damage-plasticity': MATSBondSlipDP,
                            'multilinear': MATSBondSlipMultiLinear},
                           MAT=True)

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
                     dof=self.fixed_dof, value=0.0)

    control_bc = Property(depends_on='BC,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCDof(node_name='pull-out displacement', var='u',
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

    k_max = Int(400,
                unit='$\mathrm{mm}$',
                symbol='$k_{\max}$',
                desc='Maximum number of iterations',
                ALG=True)

    tolerance = Float(1e-4,
                      unit='-',
                      symbol='$\epsilon$',
                      desc='Tolerance of residual',
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

    def get_w(self, vot):
        '''Damage variables
        '''
        idx = self.tloop.get_time_idx(vot)
        w_Emd = self.tloop.omega_record[idx]
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
        return np.min(u_C), np.max(u_C)

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
        return np.min(eps_C), np.max(eps_C)

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
        F_min = min(np.min(F_m), np.min(F_f))
        F_max = min(np.max(F_m), np.max(F_f))
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

    def plot_w(self, ax, vot):
        X_J = self.tstepper.X_J
        w = self.get_w(vot)
        ax.fill_between(X_J, 0, w, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, w, linewidth=2, color='lightcoral', label='bond')
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

    t = Property

    def _get_t(self):
        return self.get_t()

    def get_t(self):
        return np.array(self.tloop.t_record, dtype=np.float_)

    n_t = Property

    def _get_n_t(self):
        return len(self.tloop.t_record)

    sig_tC = Property

    def _get_sig_tC(self):
        n_t = self.n_t
        sig_tEmC = np.array(self.tloop.sig_EmC_record, dtype=np.float_)
        sig_tC = sig_tEmC.reshape(n_t, -1, 2)
        return sig_tC

    trait_view = View(Item('fets_eval'),
                      )

    viz2d_classes = {'field': Viz2DPullOutField,
                     'F-w': Viz2DPullOutFW,
                     'load function': Viz2DLoadControlFunction,
                     'dissipation': Viz2DEnergyPlot,
                     'dissipation rate': Viz2DEnergyRatesPlot
                     }


def run_pullout_multilinear(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      n_e_x=100, k_max=1000, w_max=2.0)
    po.tline.step = 0.01
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 20.0',
                     tau_data='0, 800, 0, 0')
    po.mats_eval.update_bs_law = True
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
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


def test_B():
    po = PullOutModel(n_e_x=1, k_max=500, w_max=1.5)
    po.tline.step = 1.0
    po.geometry.L_x = 1.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=1.0, P_b=1.0, A_m=1.0)
    po.mats_eval.set(E_m=1.0, E_f=1.0, E_b=0.0)
    po.mats_eval.set(gamma=0.0, K=15.0, tau_bar=45.0)
    po.mats_eval.omega_fn.set(alpha_2=1.0, plot_max=10.0)
    po.run()


if __name__ == '__main__':
    run_pullout_multilinear()
    # test_reporter()
    # test_B()
