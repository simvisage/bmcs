'''
Created on May 10, 2017

@author: rch
'''

if __name__ == '__main__':
    from bmcs.course import lecture04 as lec

    po = lec.PullOutModel(n_e_x=200, k_max=500, u_f0_max=2.5)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic',
                            amplitude_type='constant',
                            loading_range='non-symmetric'
                            )
    po.loading_scenario.set(number_of_cycles=2,
                            unloading_ratio=0.98,
                            )
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(gamma=25.0, K=0.0, tau_bar=2.5 * 9.0)
    po.mats_eval.omega_fn.set(alpha_1=1.0, alpha_2=2, plot_max=2.8)

    po.run()
    lec.show(po)
