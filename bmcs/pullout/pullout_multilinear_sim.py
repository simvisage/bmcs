'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

import time

from bmcs.time_functions import \
    Viz2DLoadControlFunction
from scipy import interpolate as ip
from traits.api import \
    Property
from traitsui.api import \
    View, Item
from view.window import BMCSWindow

import numpy as np

from .pullout_sim import Viz2DPullOutFW, Viz2DPullOutField, \
    Viz2DEnergyPlot, Viz2DEnergyReleasePlot, \
    PullOutModelBase, PulloutResponse


class PullOutModel(PullOutModelBase):

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
        _, w_L = self.get_w_t()
        return w_L

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
                     'dissipation rate': Viz2DEnergyReleasePlot
                     }


def run_pullout_multilinear(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      title='Multi-linear bond slip law',
                      n_e_x=20, k_max=1000, w_max=1.5)
    po.tline.step = 0.05
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 1.7',
                     tau_data='0, 70, 0, 0')
    po.mats_eval.update_bs_law = True
    po.record['Pw'] = PulloutResponse()
    fw = Viz2DPullOutFW(name='Pw', vis2d=po.hist['Pw'])

    po.run()
    time.sleep(4)
    w = BMCSWindow(model=po)
    w.viz_sheet.viz2d_list.append(fw)

#    po.add_viz2d('load function', 'load-time')
#     po.add_viz2d('F-w', 'load-displacement')
#     po.add_viz2d('field', 'u_C', plot_fn='u_C')
#     po.add_viz2d('dissipation', 'dissipation')
#     po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
#     po.add_viz2d('field', 's', plot_fn='s')
#     po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
#     po.add_viz2d('field', 'sf', plot_fn='sf')
#     po.add_viz2d('dissipation rate', 'dissipation rate')

#     w.viz_sheet.viz2d_dict['u_C'].visible = False
#     w.viz_sheet.viz2d_dict['load-time'].visible = False
#     w.viz_sheet.viz2d_dict['dissipation rate'].visible = False
#     w.viz_sheet.monitor_chunk_size = 10
#     w.viz_sheet.reference_viz2d_name = 'load-displacement'

    w.run()
    w.offline = True
    w.configure_traits(*args, **kw)


def run_pullout_multi(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      n_e_x=100, k_max=1000, w_max=2.0)
    po.tline.step = 0.02
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 4.0',
                     tau_data='0, 70.0, 0, 0')
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
    #w.finish_event = True
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
    s_arr = np.array([0., 0.004,  0.0063,  0.0165,
                      0.0266,  0.0367,  0.0468,  0.057,
                      0.0671,  0.3,
                      1.0], dtype=np.float_)
    tau_arr = 0.7 * np.array([0., 40,  62.763,  79.7754,
                              63.3328,  53.0229,  42.1918,
                              28.6376,  17,   3, 1], dtype=np.float)
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
    run_pullout_multilinear()
    # run_po_paper2_4layers()
