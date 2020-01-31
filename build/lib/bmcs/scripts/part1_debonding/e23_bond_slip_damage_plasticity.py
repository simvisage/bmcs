'''
Created on Jul 12, 2017

@author: rch
'''

from bmcs.bond_slip import BondSlipModel
from view.window import BMCSWindow


class BondSlipDamagePlasticityStudy(BMCSWindow):
    title = 'Bond damage and plasticity'
    desc = '''This example shows the response of bond material point 
which enters simultaneously the yielding and damage regime. Damage is 
increasing during the first loading branch. Subsequently, the exhibits
material point exhibits kinematic hardening upon unloading and reloading.
'''

    def __init__(self, *args, **kw):
        self.model = BondSlipModel(name='e23_bond_slip_damage_plasticity',
                                   interaction_type='predefined',
                                   material_model='damage-plasticity',
                                   n_steps=100,)
        self.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
        self.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
        self.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
        self.add_viz2d('bond history', 'z-s', x_sv_name='s', y_sv_name='z')
        self.add_viz2d('bond history', 'alpha-s',
                       x_sv_name='s', y_sv_name='alpha')
        self.add_viz2d('bond history', 'omega-s',
                       x_sv_name='s', y_sv_name='omega')

        self.model.loading_scenario.set(loading_type='cyclic',
                                        amplitude_type='constant'
                                        )
        self.model.loading_scenario.set(number_of_cycles=3,
                                        maximum_loading=0.003,
                                        unloading_ratio=0.8,
                                        loading_range='symmetric')

        self.model.material.omega_fn_type = 'li'
        self.model.material.set(gamma=0, K=1000)
        self.model.material.omega_fn.set(alpha_1=1.0, alpha_2=2000)
        self.model.run()


def run_bond_slip_model_dp(*args, **kw):
    w = BondSlipDamagePlasticityStudy()
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_slip_model_dp()
