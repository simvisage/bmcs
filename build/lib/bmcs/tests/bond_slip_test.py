
import unittest

from bmcs.bond_slip.bond_slip_model import BondSlipModel


class TestBondSlip(unittest.TestCase):
    '''test for a cycle of different show cases.
    '''

    def setUp(self):
        '''Test combination ULS/SLS.
        '''
        self.bsm = BondSlipModel(interaction_type='predefined',
                                 material_model='damage-plasticity',
                                 n_steps=100,)

    def test_bond_damage_plasticity(self):
        bsm = self.bsm
        bsm.loading_scenario.set(loading_type='cyclic',
                                 amplitude_type='constant'
                                 )
        bsm.loading_scenario.set(maximum_loading=0.005)
        bsm.material.omega_fn_type = 'li'
        bsm.material.set(gamma=0, K=1000)
        bsm.material.omega_fn.set(alpha_1=1.0, alpha_2=2000)
        bsm.run()
        omega = bsm.get_sv_hist('omega')[-1, -1]
        self.assertAlmostEqual(omega, 0.961757180906)

    def tearDown(self):
        '''
        '''
        pass


if __name__ == "__main__":
    unittest.main()
