'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

import numpy as np
from .pullout_sim import PullOutModel


def run_pullout_multilinear(*args, **kw):
    po = PullOutModel(name='t33_pullout_multilinear',
                      title='Multi-linear bond slip law',
                      n_e_x=50, w_max=1.0)
    po.tloop.k_max = 1000
    po.tline.step = 0.05
    po.geometry.L_x = 100.0
    po.loading_scenario.trait_set(loading_type='monotonic')
#    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.cross_section.set(A_f=153, P_b=44, A_m=15240.0)
    po.mats_eval_type = 'multilinear'
    po.mats_eval.set(E_m=28000,
                     E_f=170000,
                     s_data='0, 0.1, 0.4, 4',
                     tau_data='0, 800, 0, 0')
    po.mats_eval.update_bs_law = True
    w = po.get_window()
    w.run()
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
    w = po.get_window()
    w.run()
    w.configure_traits()

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
    w = po.get_window()
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
    w = po.get_window()
    w.run()
    w.configure_traits(*args, **kw)


def run_multilinear_bond_slip_law_sbr_jbielak():
    '''This is the verification of the calculation by Li. 
    '''

    po = PullOutModel(n_e_x=200, k_max=500, w_max=5.0)
    po.tline.step = 0.01
    po.loading_scenario.trait_set(loading_type='cyclic',
                                  amplitude_type='constant',
                                  loading_range='non-symmetric'
                                  )
    po.loading_scenario.trait_set(number_of_cycles=1,
                                  unloading_ratio=0.98,
                                  )
    po.geometry.set(L_x=100.0)
    po.cross_section.set(A_f=16.65, P_b=1.0, A_m=1543.35)
    po.mats_eval_type = 'multilinear'
    po.mats_eval.set(E_m=28480, E_f=170000)
    po.mats_eval.bs_law.set(
        xdata=[0.,  0.18965517,  0.37931034,  0.56896552,  0.75862069,
               0.94827586,  1.13793103,  1.32758621,  1.51724138,  1.70689655,
               1.89655172,  2.0862069,  2.27586207,  2.46551724,  2.65517241,
               2.84482759,  3.03448276,  3.22413793,  3.4137931,  3.60344828,
               3.79310345,  3.98275862,  4.17241379,  4.36206897,  4.55172414,
               4.74137931,  4.93103448,  5.12068966,  5.31034483,  5.5],
        ydata=np.array([0., 40.56519694, 43.86730345, 42.37807371,
                        43.5272407,  44.29410001,  46.04230264,  47.89711984,
                        50.03209956,  52.23118918,  54.40193739,  56.67975395,
                        58.97599182,  61.26809043,  63.60529275,  65.92661789,
                        68.22558581,  70.39763384,  72.49000557,  74.44268819,
                        76.16535426,  77.70806171,  79.20875264,  80.78660257,
                        82.08287581,  83.26309573,  84.31540923,  85.18093017,
                        85.99297513,  86.50752229], dtype=np.float_)
    )
    po.mats_eval.bs_law.replot()
    w = po.get_window()
    w.run()
    w.configure_traits()


def run_multilinear_bond_slip_law_epoxy_tvlach():
    '''This is the verification of the calculation by Li. 
    '''

    po = PullOutModel(n_e_x=200, k_max=500, w_max=0.15)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic',
                            amplitude_type='constant',
                            loading_range='non-symmetric'
                            )
    po.loading_scenario.set(number_of_cycles=1,
                            unloading_ratio=0.98,
                            )
    po.geometry.set(L_x=12.0)
    po.cross_section.set(A_f=2.2, P_b=1.0, A_m=10000.0 - 2.0)
    po.mats_eval_type = 'multilinear'
    po.mats_eval.set(E_m=49200.0, E_f=29500.0)
    po.mats_eval.bs_law.set(
        xdata=[0, 1e-6, 0.005, 0.035, 0.065, 0.095, 0.15],
        ydata=[0., 10.12901536,   39.9247595,
               84.22654625,  101.35300195,
               134.23784515, 158.97974139]
    )
    po.mats_eval.bs_law.replot()
    w = po.get_window()
    w.run()
    w.configure_traits()


if __name__ == '__main__':
    run_pullout_multilinear()
    # run_pullout_multi()
    # run_cb_multi()
    # run_po_paper2_4layers()
    # run_multilinear_bond_slip_law_sbr_jbielak()
    # run_multilinear_bond_slip_law_epoxy_tvlach()
