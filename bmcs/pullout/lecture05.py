'''
Example script of bond - pullout evaluation.
'''

from view.window.bmcs_window import BMCSWindow

import numpy as np
from pullout_dp import PullOutModel


def show(po):
    w = BMCSWindow(model=po)
    po.add_viz2d('load function')
    po.add_viz2d('F-w')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'w', plot_fn='w')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')

    w.offline = False
    w.finish_event = True
    w.configure_traits()


def e51_isotropic_hardening():
    '''Show the effect of strain softening represented 
    by kinematic or isotripic hardening.
    '''
    po = PullOutModel(n_e_x=100, k_max=50, w_max=0.01)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.geometry.L_x = 1.0
    po.material.set(gamma=0.0, K=-1000.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=0.0, alpha_2=1.0, plot_max=10.0)
    po.run()

    show(po)


def e52_kinematic_hardening():
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
    po.material.set(K=0.0, gamma=-5000.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=0.0, alpha_2=1.0, plot_max=10.0)
    po.run()

    show(po)


def e53_damage_softening():
    '''Show the effect of strain softening represented 
    by negative kinematic hardening.

    @todo: motivate fracture energy

    '''
    po = PullOutModel(n_e_x=100, k_max=1000, w_max=0.08)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 1000.0
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.material.set(K=100000.0, gamma=-0.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=1000.0, plot_max=0.01)
    po.run()

    show(po)


def e54_damage_length_dependency():
    w_max = 0.07
    po = PullOutModel(n_e_x=80, k_max=1000, w_max=w_max)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.material.set(K=100000.0, gamma=-0.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=1000.0, plot_max=0.01)
    sig_f_max = 1600.00
    A_f = po.cross_section.A_f
    P_f_max = A_f * sig_f_max

    po.run()

    import pylab

    L_array = np.logspace(0, 3, 8)
    for L in L_array:
        po.geometry.L_x = L
        po.run()
        P = po.get_P_t()
        w0, wL = po.get_w_t()
        pylab.plot(wL, P, label='L=%f' % L)
    pylab.plot([0.0, w_max], [P_f_max, P_f_max], label='yarn strength')
    pylab.legend()
    pylab.show()
    show(po)


if __name__ == "__main__":
    #    e51_isotropic_hardening()
    #     e52_kinematic_hardening()
    # e53_damage_softening()
    e54_damage_length_dependency()
