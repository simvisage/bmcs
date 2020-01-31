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
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    w.offline = False
    w.finish_event = True
    w.configure_traits()


def e51_carbon_trc_pullout_test():
    '''Show the effect of strain softening represented 
    by negative kinematic hardening.

    @todo: motivate fracture energy.
    '''
    po = PullOutModel(n_e_x=100, k_max=1000, w_max=5.0)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 100.0
    po.cross_section.set(A_f=16.67, P_b=9.0, A_m=1540.0)
    po.material.set(K=0.0, gamma=1.5, tau_bar=5.0)
    po.material.set(E_m=30000.0, E_f=200000.0, E_b=12900.0)
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=1.0, plot_max=0.01)
    po.run()
    show(po)


def e52_frp_lap_joint_test():
    '''Show the effect of strain softening represented 
    by negative kinematic hardening.
    The parameters were taken from
    [Baena e. al. 2009]
    '''
    po = PullOutModel(n_e_x=100, k_max=1000, w_max=25.0)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 45.0
    po.cross_section.set(A_f=64.0, P_b=28.0, A_m=28000.0)
    po.material.set(K=-0.2, gamma=0.0, tau_bar=13.137)
    po.material.set(E_m=35000.0, E_f=170000.0, E_b=6700.0)
    po.material.omega_fn.set(alpha_1=0.0, alpha_2=1.0, plot_max=0.01)
    po.run()
    show(po)


def e51_isotropic_negative_hardening():
    '''Show the effect of strain softening represented 
    by kinematic or isotripic hardening.
    '''
    po = PullOutModel(n_e_x=100, k_max=500, w_max=0.1)
    po.tline.step = 0.005
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.geometry.L_x = 200.0
    po.material.set(gamma=0.0, K=-100.0, tau_bar=10.0)
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

    @todo: motivate fracture energy.
    '''
    po = PullOutModel(n_e_x=100, k_max=1000, w_max=0.62)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 100.0
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.material.set(K=100000.0, gamma=0.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=50.0, plot_max=0.62)
    po.run()
    show(po)


def e54_damage_length_dependency():
    w_max = 2.0
    po = PullOutModel(n_e_x=100, k_max=1000, w_max=w_max)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.geometry.L_x = 100.0
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.material.set(K=100000.0, gamma=0.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=50.0, plot_max=0.62)

    sig_f_max = 2400.00
    A_f = po.cross_section.A_f
    P_f_max = A_f * sig_f_max

    import pylab
    L_max = np.log10(350.0)
    L_array = np.logspace(0, L_max, 4)
    for L in L_array:
        po.geometry.L_x = L
        po.run()
        P = po.get_P_t()
        w0, wL = po.get_w_t()
        pylab.subplot(2, 1, 1)
        pylab.plot(wL, P, label='L=%f' % L)
        dG_bar = po.get_dG_t()
        pylab.subplot(2, 1, 2)
        pylab.plot(wL, dG_bar, label='L=%f' % L)

    pylab.subplot(2, 1, 1)
    pylab.plot([0.0, w_max], [P_f_max, P_f_max], label='yarn strength')
    pylab.legend()
    pylab.show()
    show(po)


def e55_damage_element_size():
    w_max = 0.1
    po = PullOutModel(n_e_x=80, k_max=1000, w_max=w_max)
    po.tline.step = 0.01
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=1)
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.material.set(K=100000.0, gamma=-0.0, tau_bar=45.0)
    po.material.omega_fn.set(alpha_1=1.0, alpha_2=1000.0, plot_max=0.01)
    po.geometry.L_x = 200.0

    import pylab

    n_e_array = [20, 40, 60, 80, 100]
    for n_e_x in n_e_array:
        po.n_e_x = n_e_x
        po.run()
        P = po.get_P_t()
        w0, wL = po.get_w_t()
        pylab.subplot(2, 2, 1)
        pylab.plot(wL, P, label='n_e=%d' % n_e_x)
        ax = pylab.subplot(2, 1, 2)
        po.plot_sf(ax, 1.0)
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    # e51_carbon_trc_pullout_test()
    e52_frp_lap_joint_test()
    # e51_isotropic_hardening()
    # e52_kinematic_hardening()
    # e53_damage_softening()
    # e54_damage_length_dependency()
    # e55_damage_element_size()
