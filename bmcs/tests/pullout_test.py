
import unittest

from bmcs.pullout.pullout import PullOutModel


class TestPullOut(unittest.TestCase):
    '''test for a cycle of different show cases.
    '''

    def setUp(self):
        '''Test combination ULS/SLS.
        '''
        self.po = PullOutModel(n_e_x=100, k_max=500)
        self.po.tline.step = 0.1

    def test_pullout_model(self):
        '''  
        '''
        po = self.po
        po.tline.step = 0.1
        po.bcond_mngr.bcond_list[1].value = 0.01

        po.loading_scenario.set(loading_type='monotonic',
                                )
        po.run()

        u3 = po.tloop.U_record[-1, -3]
        self.assertAlmostEqual(u3, 0.00986152263039)

    def tearDown(self):
        '''
        '''
        pass


if __name__ == "__main__":
    unittest.main()
