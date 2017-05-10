'''
Created on May 9, 2017

@author: rch
'''

from view.window.bmcs_window import BMCSWindow

from bmcs.api import PullOutModel
import numpy as np


def example_01_anchorage_length():
    po = PullOutModel(n_e_x=100, k_max=500, w_max=1.5)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(gamma=0.0, K=15.0, tau_bar=45.0)
    po.mats_eval.omega_fn.set(alpha_2=1.0, plot_max=10.0)
    po.run()

    x = po.X_M
    t = po.t

    L_array = np.linspace(10, 2000, 10)
    sig_max_array = []
    for L in L_array:
        po.geometry.L_x = L
        po.run()
        sig_f = po.sig_tC[:, :, 1]
        sig_max = np.max(sig_f)
        sig_max_array.append(sig_max)

    w = BMCSWindow(model=po)
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'w', plot_fn='w')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
#
    w.offline = False
    w.finish_event = True

    w.configure_traits()

    import pylab
    pylab.plot(L_array, sig_max_array)
    pylab.show()


def example2_isotropic_hardening():
    '''Show the effect of strain softening represented 
    by kinematic or isotripic hardening.
    '''
    po = PullOutModel(n_e_x=10, k_max=50, w_max=0.15)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(gamma=0.0, K=-800.0, tau_bar=45.0)
    po.mats_eval.omega_fn.set(alpha_1=0.0, alpha_2=1.0, plot_max=10.0)
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
#
    w.offline = False
    w.finish_event = True

    w.configure_traits()


def example03_kinematic_hardening():
    '''Show the effect of strain softening represented 
    by negative kinematic hardening.

    @todo - fix - the values of stress upon loading should not 
    go to negative range.

    '''
    po = PullOutModel(n_e_x=10, k_max=50, w_max=0.05)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 1.0
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(K=0.0, gamma=-5000.0, tau_bar=45.0)
    po.mats_eval.omega_fn.set(alpha_1=0.0, alpha_2=1.0, plot_max=10.0)
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
#
    w.offline = False
    w.finish_event = True

    w.configure_traits()


def example04_damage():
    '''Show the effect of strain softening represented 
    by negative kinematic hardening.

    @todo - fix - the values of stress upon loading should not 
    go to negative range.

    @todo: motivate fracture energy

    '''
    po = PullOutModel(n_e_x=100, k_max=1000, w_max=0.01)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 1.0
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(K=100000.0, gamma=-0.0, tau_bar=45.0)
    po.mats_eval.omega_fn.set(alpha_1=1.0, alpha_2=1400.0, plot_max=0.01)
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
#
    w.offline = False
    w.finish_event = True

    w.configure_traits()


if __name__ == "__main__":
    # example_01_anchorage_length()()
    # example03_kinematic_hardening()()
    example2_isotropic_hardening()()
    # example04_damage()()
