'''
Created on May 11, 2017

@author: rch
'''

from bmcs.course import lecture04 as l04
import numpy as np


def multilinear_bond_slip_law_sbr_jbielak():
    '''This is the verification of the calculation by Li. 
    '''

    po = l04.PullOutModel(n_e_x=200, k_max=500, w_max=5.0)
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
    po.run()
    l04.show(po)


def multilinear_bond_slip_law_epoxy_tvlach():
    '''This is the verification of the calculation by Li. 
    '''

    po = l04.PullOutModel(n_e_x=200, k_max=500, w_max=0.15)
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
    po.run()
    l04.show(po)


if __name__ == '__main__':
    multilinear_bond_slip_law_epoxy_tvlach()
