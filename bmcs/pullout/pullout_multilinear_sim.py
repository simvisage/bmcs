'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

import copy
import time
from bmcs.time_functions import \
    Viz2DLoadControlFunction
from scipy import interpolate as ip
from traits.api import \
    Property
from traitsui.api import \
    View, Item
from view.plot2d.vis2d import Vis2D
from view.window import BMCSWindow

import numpy as np

from .pullout_sim import Viz2DPullOutFW, Viz2DPullOutField, \
    Viz2DEnergyPlot, Viz2DEnergyReleasePlot, \
    PullOutModelBase, PulloutResponse


class PullOutModel(PullOutModelBase):

    X_M = Property

    def _get_X_M(self):
        state = self.tstep.fe_domain[0]
        return state.xdomain.x_Ema[..., 0].flatten()

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

    def get_W_t(self):
        P_t = self.get_P_t()
        _, w_L = self.get_w_t()
        W_t = []
        for i, _ in enumerate(w_L):
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

    def xget_w_t(self):
        d_t = self.tloop.U_record
        dof_ECid = self.tstepper.dof_ECid
        d_t_ECid = d_t[:, dof_ECid]
        w_0 = d_t_ECid[:, 0, 1, 0, 0]
        w_L = d_t_ECid[:, -1, 1, -1, -1]
        return w_0, w_L

    def xget_wL_t(self):
        _, w_L = self.get_w_t()
        return w_L

    xt = Property

    def _xget_t(self):
        return self.get_t()

    def xget_t(self):
        return np.array(self.tloop.t_record, dtype=np.float_)

    xn_t = Property

    def x_get_n_t(self):
        return len(self.tloop.t_record)

    xsig_tC = Property

    def x_get_sig_tC(self):
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
                     'dissipation rate': Viz2DEnergyReleasePlot
                     }


def run_pullout_multilinear(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      title='Multi-linear bond slip law',
                      n_e_x=50, w_max=1.0)
    po.tloop.k_max = 1000
    po.tline.step = 0.05
    po.geometry.L_x = 100.0
    po.loading_scenario.trait_set(loading_type='monotonic')

#     po.cross_section.trait_set(A_f=100, P_b=1000, A_m=10000)
#     po.mats_eval.set(E_m=28000,
#                      E_f=170000,
#                      s_data='0, 0.1',
#                      tau_data='0, 100')
#    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.cross_section.set(A_f=153, P_b=44, A_m=15240.0)
    po.mats_eval.set(E_m=28000,
                     E_f=170000,
                     s_data='0, 0.1, 0.4, 4',
                     tau_data='0, 800, 0, 0')
    po.mats_eval.update_bs_law = True

    po.record['Pw'] = PulloutResponse()
    fw = Viz2DPullOutFW(name='Pw', vis2d=po.hist['Pw'])
    u_p = Viz2DPullOutField(plot_fn='u_p', vis2d=po)
    eps_p = Viz2DPullOutField(plot_fn='eps_p', vis2d=po)
    sig_p = Viz2DPullOutField(plot_fn='sig_p', vis2d=po)
    s = Viz2DPullOutField(plot_fn='s', vis2d=po)
    sf = Viz2DPullOutField(plot_fn='sf', vis2d=po)

    w = BMCSWindow(sim=po)
    w.viz_sheet.viz2d_list.append(fw)
    w.viz_sheet.viz2d_list.append(u_p)
    w.viz_sheet.viz2d_list.append(eps_p)
    w.viz_sheet.viz2d_list.append(sig_p)
    w.viz_sheet.viz2d_list.append(s)
    w.viz_sheet.viz2d_list.append(sf)

#    po.add_viz2d('load function', 'load-time')
#     po.add_viz2d('dissipation', 'dissipation')
#     po.add_viz2d('dissipation rate', 'dissipation rate')

#     w.viz_sheet.viz2d_dict['u_C'].visible = False
#     w.viz_sheet.viz2d_dict['load-time'].visible = False
#     w.viz_sheet.viz2d_dict['dissipation rate'].visible = False
#     w.viz_sheet.monitor_chunk_size = 10
#     w.viz_sheet.reference_viz2d_name = 'load-displacement'

#    w.run()
#    w.offline = True
#    time.sleep(1)
    w.configure_traits()


def run_31():
    w_max = 0.1  # [mm]
    d = 16.0  # [mm]
    E_f = 210000  # [MPa]
    E_m = 28000  # [MPa]
    A_f = (d / 2.) ** 2 * np.pi  # [mm^2]
    P_b = d * np.pi
    A_m = 100 * 100  # [mm^2]
    L_x = 5 * d
    pm = PullOutModel(mats_eval_type='multilinear',
                      n_e_x=50, w_max=w_max)
    pm.loading_scenario.loading_type = 'monotonic'
    pm.mats_eval.trait_set(E_m=E_m, E_f=E_f,
                           s_data='0, 0.001, 0.1',
                           tau_data='0, 5, 8')
    pm.mats_eval.update_bs_law = True
    pm.cross_section.trait_set(A_m=A_m, P_b=P_b, A_f=A_f)
    pm.geometry.L_x = L_x
    pm.record['Pw'] = PulloutResponse()
    fw = Viz2DPullOutFW(name='Pw', vis2d=pm.hist['Pw'])
    u_p = Viz2DPullOutField(plot_fn='u_p', vis2d=pm)
    eps_p = Viz2DPullOutField(plot_fn='eps_p', vis2d=pm)
    sig_p = Viz2DPullOutField(plot_fn='sig_p', vis2d=pm)
    s = Viz2DPullOutField(plot_fn='s', vis2d=pm)
    sf = Viz2DPullOutField(plot_fn='sf', vis2d=pm)

    w = BMCSWindow(sim=pm)
    w.viz_sheet.viz2d_list.append(fw)
    w.viz_sheet.viz2d_list.append(u_p)
    w.viz_sheet.viz2d_list.append(eps_p)
    w.viz_sheet.viz2d_list.append(sig_p)
    w.viz_sheet.viz2d_list.append(s)
    w.viz_sheet.viz2d_list.append(sf)
    w.configure_traits()


def run_pullout_multi(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      n_e_x=100, w_max=2.0)
    po.tloop.k_max = 1000
    po.tline.step = 0.02
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 4.0',
                     tau_data='0, 7.0, 0, 0')
    po.mats_eval.update_bs_law = True
#     po.run()

    po.record['Pw'] = PulloutResponse()
    fw = Viz2DPullOutFW(name='Pw', vis2d=po.hist['Pw'])
    u_p = Viz2DPullOutField(plot_fn='u_p', vis2d=po)
    eps_p = Viz2DPullOutField(plot_fn='eps_p', vis2d=po)
    sig_p = Viz2DPullOutField(plot_fn='sig_p', vis2d=po)
    s = Viz2DPullOutField(plot_fn='s', vis2d=po)
    sf = Viz2DPullOutField(plot_fn='sf', vis2d=po)
    w = BMCSWindow(sim=po)
    w.viz_sheet.viz2d_list.append(fw)
    w.viz_sheet.viz2d_list.append(u_p)
    w.viz_sheet.viz2d_list.append(eps_p)
    w.viz_sheet.viz2d_list.append(sig_p)
    w.viz_sheet.viz2d_list.append(s)
    w.viz_sheet.viz2d_list.append(sf)
#
#     w = BMCSWindow(sim=po)
#     po.add_viz2d('load function', 'load-time')
#     po.add_viz2d('F-w', 'load-displacement')
#     po.add_viz2d('field', 'u_C', plot_fn='u_C')
#     po.add_viz2d('dissipation', 'dissipation')
#     po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
#     po.add_viz2d('field', 's', plot_fn='s')
#     po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
#     po.add_viz2d('field', 'sf', plot_fn='sf')
#     po.add_viz2d('dissipation rate', 'dissipation rate')

    w.offline = False
    # w.finish_event = True
    w.configure_traits(*args, **kw)


def run_cb_multi(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      n_e_x=100, k_max=1000, w_max=2.0)
    po.fixed_boundary = 'clamped left'
    po.tline.step = 0.02
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 4.0',
                     tau_data='0, 70.0, 80, 90')
    po.mats_eval.update_bs_law = True
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function', 'load-time')
    po.add_viz2d('F-w', 'load-displacement')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


def run_po_paper2_4layers(*args, **kw):

    A_roving = 0.49
    h_section = 20.0
    b_section = 100.0
    n_roving = 11.0
    tt4_n_layers = 6
    A_f4 = n_roving * tt4_n_layers * A_roving
    A_c4 = h_section * b_section
    A_m4 = A_c4 - A_f4
    P_b4 = tt4_n_layers
    E_f = 180000.0
    E_m = 30000.0
    s_arr = np.array([0., 0.004, 0.0063, 0.0165,
                      0.0266, 0.0367, 0.0468, 0.057,
                      0.0671, 0.3,
                      1.0], dtype=np.float_)
    tau_arr = 0.7 * np.array([0., 40, 62.763, 79.7754,
                              63.3328, 53.0229, 42.1918,
                              28.6376, 17, 3, 1], dtype=np.float)
    po = PullOutModel(name='t33_pullout_multilinear',
                      n_e_x=100, k_max=1000, w_max=2.0)
    po.fixed_boundary = 'clamped left'
    po.loading_scenario.set(loading_type='monotonic')
    po.mats_eval.trait_set(E_f=E_f, E_m=E_m)
    po.mats_eval.s_tau_table = [
        s_arr, tau_arr
    ]
    po.cross_section.trait_set(A_f=A_f4, A_m=A_m4, P_b=P_b4)
    po.geometry.trait_set(L_x=500)
    po.trait_set(w_max=0.95, n_e_x=100)
    po.tline.trait_set(step=0.005)

    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function', 'load-time')
    po.add_viz2d('F-w', 'load-displacement')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_31()
    # run_pullout_multilinear()
    # run_po_paper2_4layers()
