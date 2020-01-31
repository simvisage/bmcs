'''
Created on Jul 12, 2017

@author: rch
'''

from bmcs.bond_slip import BondSlipModel
from view.window import BMCSWindow


def construct_bond_slip_study(*args, **kw):
    bsm = BondSlipModel(name='t21_bond_slip_damage',
                        interaction_type='predefined',
                        material_model='damage'
                        )
    bsm.title = 'Bond damage'
    bsm.desc = '''Single material point of a bond section is
loaded monotonically showing the bond deterioration
governed by a prescribed damage function $\omega$.
'''

    w = BMCSWindow(model=bsm, n_cols=2)
    bsm.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
    bsm.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
    bsm.add_viz2d('bond history', 'omega-s', x_sv_name='s', y_sv_name='omega')
    bsm.add_viz2d('bond history', 'kappa-s', x_sv_name='s', y_sv_name='kappa')
    bsm.loading_scenario.set(loading_type='cyclic',
                             amplitude_type='constant'
                             )
    bsm.loading_scenario.set(number_of_cycles=1,
                             maximum_loading=0.005,
                             unloading_ratio=0.5)
    bsm.material.omega_fn_type = 'jirasek'
    bsm.material.omega_fn.s_f = 0.003
    bsm.run()
    return w


def run_bond_slip_model_d(*args, **kw):
    w = construct_bond_slip_study(*args, **kw)
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_slip_model_d()
