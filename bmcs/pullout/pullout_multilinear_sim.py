'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from view.window import BMCSWindow

import numpy as np

from .pullout_sim import Viz2DPullOutFW, Viz2DPullOutField, \
    Viz2DEnergyPlot, Viz2DEnergyReleasePlot, \
    PullOutModel, PulloutRecord


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

    po.record['Pw'] = PulloutRecord()
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
    pm.record['Pw'] = PulloutRecord()
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
    po.geometry.L_x = 500.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 4.0',
                     tau_data='0, 7.0, 0, 0')
    po.mats_eval.update_bs_law = True

    po.record['Pw'] = PulloutRecord()
    fw = Viz2DPullOutFW(name='Pw', vis2d=po.hist['Pw'])
    u_p = Viz2DPullOutField(plot_fn='u_p', vis2d=po)
    eps_p = Viz2DPullOutField(plot_fn='eps_p', vis2d=po)
    sig_p = Viz2DPullOutField(plot_fn='sig_p', vis2d=po)
    s = Viz2DPullOutField(plot_fn='s', vis2d=po)
    sf = Viz2DPullOutField(plot_fn='sf', vis2d=po)
    energy = Viz2DEnergyPlot(vis2d=po.hist['Pw'])
    dissipation = Viz2DEnergyReleasePlot(vis2d=po.hist['Pw'])
    w = BMCSWindow(sim=po)
    w.viz_sheet.viz2d_list.append(fw)
    w.viz_sheet.viz2d_list.append(u_p)
    w.viz_sheet.viz2d_list.append(eps_p)
    w.viz_sheet.viz2d_list.append(sig_p)
    w.viz_sheet.viz2d_list.append(s)
    w.viz_sheet.viz2d_list.append(sf)
    w.viz_sheet.viz2d_list.append(energy)
    w.viz_sheet.viz2d_list.append(dissipation)
    w.offline = False
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
    po.record['Pw'] = PulloutRecord()
    fw = Viz2DPullOutFW(name='Pw', vis2d=po.hist['Pw'])
    u_p = Viz2DPullOutField(plot_fn='u_p', vis2d=po)
    eps_p = Viz2DPullOutField(plot_fn='eps_p', vis2d=po)
    sig_p = Viz2DPullOutField(plot_fn='sig_p', vis2d=po)
    s = Viz2DPullOutField(plot_fn='s', vis2d=po)
    sf = Viz2DPullOutField(plot_fn='sf', vis2d=po)
    energy = Viz2DEnergyPlot(vis2d=po.hist['Pw'])
    dissipation = Viz2DEnergyReleasePlot(vis2d=po.hist['Pw'])

    w = BMCSWindow(sim=po)
    w.viz_sheet.viz2d_list.append(fw)
    w.viz_sheet.viz2d_list.append(u_p)
    w.viz_sheet.viz2d_list.append(eps_p)
    w.viz_sheet.viz2d_list.append(sig_p)
    w.viz_sheet.viz2d_list.append(s)
    w.viz_sheet.viz2d_list.append(sf)
    w.viz_sheet.viz2d_list.append(energy)
    w.viz_sheet.viz2d_list.append(dissipation)

    w.run()
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

    po.record['Pw'] = PulloutRecord()
    fw = Viz2DPullOutFW(name='Pw', vis2d=po.hist['Pw'])
    u_p = Viz2DPullOutField(plot_fn='u_p', vis2d=po)
    eps_p = Viz2DPullOutField(plot_fn='eps_p', vis2d=po)
    sig_p = Viz2DPullOutField(plot_fn='sig_p', vis2d=po)
    s = Viz2DPullOutField(plot_fn='s', vis2d=po)
    sf = Viz2DPullOutField(plot_fn='sf', vis2d=po)
    energy = Viz2DEnergyPlot(vis2d=po.hist['Pw'])
    dissipation = Viz2DEnergyReleasePlot(vis2d=po.hist['Pw'])
    w = BMCSWindow(sim=po)
    w.viz_sheet.viz2d_list.append(fw)
    w.viz_sheet.viz2d_list.append(u_p)
    w.viz_sheet.viz2d_list.append(eps_p)
    w.viz_sheet.viz2d_list.append(sig_p)
    w.viz_sheet.viz2d_list.append(s)
    w.viz_sheet.viz2d_list.append(sf)
    w.viz_sheet.viz2d_list.append(energy)
    w.viz_sheet.viz2d_list.append(dissipation)
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    # run_31()
    # run_pullout_multilinear()
    # run_pullout_multi()
    run_cb_multi()
    # run_po_paper2_4layers()
