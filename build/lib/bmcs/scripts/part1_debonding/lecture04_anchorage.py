'''
Example script of bond - pullout evaluation.
'''

from os.path import join

from bmcs.api import PullOutModel
from view.window.bmcs_window import BMCSWindow

import numpy as np


def get_pullout_model_carbon_concrete(w_max=5.0):
    '''Helper method to get the constructing the default
    configuration of the pullout model.
    '''
    '''Helper method to get the constructing the default
    configuration of the pullout model.
    '''
    po = PullOutModel(n_e_x=200, k_max=500, w_max=w_max)
    po.tline.step = 0.005
    po.loading_scenario.set(loading_type='cyclic',
                            amplitude_type='constant',
                            loading_range='non-symmetric'
                            )
    po.loading_scenario.set(number_of_cycles=1,
                            unloading_ratio=0.98,
                            )
    po.geometry.L_x = 100.0
    po.cross_section.set(A_f=16.67, P_b=9.0, A_m=1540.0)
    po.mats_eval_type = 'damage-plasticity'
    po.mats_eval.set(E_m=28480, E_f=170000)
    po.mats_eval.set(gamma=1.5, K=0.0, tau_bar=5.0)
    po.mats_eval.omega_fn_type = 'li'
    po.mats_eval.omega_fn.set(alpha_1=1.0, alpha_2=1, plot_max=2.8)
    return po


def show(po):
    w = BMCSWindow(sim=po)
    po.add_viz2d('load function', 'load-time')
    po.add_viz2d('F-w', 'load-displacement')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'omega', plot_fn='omega')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')

    w.offline = False
    w.finish_event = True
    w.configure_traits()


def e41_preconfigure_and_start_app():
    '''Fit the test responce of a textile carbon concrete cross section
    in shown in BMCS topic 3.3  
    '''
    po = get_pullout_model_carbon_concrete()
    po.run()
    show(po)


def e42_compare_two_simulations():
    '''Fit the test responce of a textile carbon concrete cross section
    in shown in BMCS topic 3.3  
    '''
    po = get_pullout_model_carbon_concrete(w_max=5.0)

    L_array = np.array([75, 100], dtype=np.float_)

    import pylab
    for L in L_array:
        po.geometry.L_x = L
        po.run()
        P, w0, wL = po.get_Pw_t()
        pylab.plot(wL, P, label='L=%d [mm]' % L)

    pylab.legend(loc=2)
    pylab.show()


def e43_study_length_dependence():
    w_max = 5.0
    po = get_pullout_model_carbon_concrete(w_max=w_max)
    po.loading_scenario.loading_type = 'monotonic'
    po.tline.step = 0.005

    L_array = np.array([100, 150, 200, 250, 300, 350],
                       dtype=np.float_)
    L_trait = po.geometry.traits()
    print(L_trait)
    L_trait['L_x'].range = L_array
    P_u_record = []
    for L in L_array:
        print('calculating length', L)
        po.geometry.L_x = L
        po.run()
        P, _, wL = po.get_Pw_t()
        P_u_record.append((L, P, wL))

    import pylab
    pylab.subplot(1, 2, 1)

    A_f = po.cross_section.A_f
    sig_f_max = 1600.00
    P_f_max = A_f * sig_f_max

    pylab.plot([0.0, w_max], [P_f_max, P_f_max], '-', label='yarn failure')

    max_P_list = []
    for L, P, u in P_u_record:
        pylab.plot(u, P, label='L=%d [mm]' % L)
        max_P_list.append(np.max(P))
    pylab.legend(loc=2)

    # plot the pullout force / length dependence
    pylab.subplot(1, 2, 2)
    pylab.plot([0.0, np.max(L_array)],
               [P_f_max, P_f_max], label='yarn failure')
    pylab.plot(L_array, max_P_list, 'o-')
    pylab.xlim(xmin=0)
    pylab.ylim(ymin=0)
    pylab.legend(loc=2)
    pylab.show()


if __name__ == "__main__":
    e41_preconfigure_and_start_app()
    # e42_compare_two_simulations()
    # e43_study_length_dependence()
