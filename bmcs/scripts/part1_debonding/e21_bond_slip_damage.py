'''
Created on Jul 12, 2017

@author: rch
'''

from bmcs.bond_slip import BondSlipModel
from view.window import BMCSWindow


class BondSlipDamageStudy(BMCSWindow):

    title = 'Bond damage'
    desc = '''Single material point of a bond section is
    loaded monotonically showing the bond deterioration
    governed by a prescribed damage function $\omega$.
    '''

    def __init__(self, *args, **kw):

        self.model = BondSlipModel(name='e21_bond_slip_damage',
                                   interaction_type='predefined',
                                   material_model='damage'
                                   )

        self.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
        self.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
        self.add_viz2d('bond history', 'omega-s',
                       x_sv_name='s', y_sv_name='omega')
        self.add_viz2d('bond history', 'kappa-s',
                       x_sv_name='s', y_sv_name='kappa')
        self.model.loading_scenario.set(loading_type='cyclic',
                                        amplitude_type='constant'
                                        )
        self.model.loading_scenario.set(number_of_cycles=1,
                                        maximum_loading=0.005,
                                        unloading_ratio=0.5)
        self.model.material.omega_fn_type = 'jirasek'
        self.model.material.omega_fn.s_f = 0.003
        self.model.run()


def run_bond_slip_model_d(*args, **kw):
    w = BondSlipDamageStudy(*args, **kw)
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_slip_model_d()
