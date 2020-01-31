'''
Created on Jul 12, 2017

@author: rch
'''

from bmcs.bond_slip import BondSlipModel
from view.window import BMCSWindow


def construct_bond_slip_study(*args, **kw):
    bsm = BondSlipModel(name='t22_bond_slip_plasticity_kinem',
                        interaction_type='predefined',
                        material_model='plasticity',
                        n_steps=2000)
    bsm.title = 'Bond plasticity'
    bsm.desc = '''This example shows the response of bond material point 
with kinematic hardening with unloading and reloading.
'''

    w = BMCSWindow(model=bsm, n_cols=2)
    bsm.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
    bsm.add_viz2d('bond history', 'z-s', x_sv_name='s', y_sv_name='z')
    #bsm.add_viz2d('bond history', 'alpha-s', x_sv_name='s', y_sv_name='alpha')
    bsm.loading_scenario.set(loading_type='cyclic',
                             amplitude_type='constant'
                             )
    bsm.loading_scenario.set(number_of_cycles=2,
                             maximum_loading=0.005)
    bsm.material.set(gamma=2000, K=-0)
    bsm.run()
    return w


def run_bond_slip_model_p(*args, **kw):
    w = construct_bond_slip_study()
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_slip_model_p()
