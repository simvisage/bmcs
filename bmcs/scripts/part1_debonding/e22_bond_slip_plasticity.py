'''
Created on Jul 12, 2017

@author: rch
'''

from bmcs.bond_slip import BondSlipModel
from view.window import BMCSWindow


class BondSlipPlasticityStudy(BMCSWindow):

    title = 'Bond plasticity'
    desc = '''This example shows the response of bond material point 
    with kinematic hardening with unloading and reloading.
    '''

    def __init__(self, *args, **kw):
        self.model = BondSlipModel(name='e22_bond_slip_plasticity_kinem',
                                   interaction_type='predefined',
                                   material_model='plasticity',
                                   n_steps=200)
        self.add_viz2d('bond history', 's-t', x_sv_name='t', y_sv_name='s')
        self.add_viz2d('bond history', 'tau-s', x_sv_name='s', y_sv_name='tau')
        self.add_viz2d('bond history', 's_p-s', x_sv_name='s', y_sv_name='s_p')
        self.add_viz2d('bond history', 'z-s', x_sv_name='s', y_sv_name='z')
        #bsm.add_viz2d('bond history', 'alpha-s', x_sv_name='s', y_sv_name='alpha')
        self.model.loading_scenario.set(loading_type='cyclic',
                                        amplitude_type='constant'
                                        )
        self.model.loading_scenario.set(number_of_cycles=2,
                                        maximum_loading=0.005)
        self.model.material.set(gamma=2000, K=-0)
        self.model.run()


def run_bond_slip_model_p(*args, **kw):
    w = BondSlipPlasticityStudy()
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_slip_model_p()
