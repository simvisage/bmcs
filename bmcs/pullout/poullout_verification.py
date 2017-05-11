'''
Created on May 11, 2017

@author: rch
'''

import bmcs.pullout.lecture04 as lec
po = lec.PullOutModel(n_e_x=200, k_max=500, w_max=5.0)
po.tline.step = 0.01
po.loading_scenario.set(loading_type='cyclic',
                        amplitude_type='constant',
                        loading_range='non-symmetric'
                        )
po.loading_scenario.set(number_of_cycles=1,
                        unloading_ratio=0.98,
                        )
po.geometry.set(L_x=1.0)
po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
po.mats_eval
po.mats_eval.set(gamma=1.5, K=0.0, tau_bar=5.0)
po.mats_eval.omega_fn.set(alpha_1=1.0, alpha_2=1.0, plot_max=2.8)
po.run()
lec.show(po)
