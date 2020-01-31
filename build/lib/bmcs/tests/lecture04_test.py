
import unittest

import bmcs.pullout.lecture04 as l04


class TestLecture04(unittest.TestCase):
    '''test for a cycle of different show cases.
    '''

    def setUp(self):
        '''Test combination ULS/SLS.
        '''
        self.po = l04.get_pullout_model_carbon_concrete()

    def test_pullout_model(self):
        '''  
        '''
        po = self.po

        po.tline.step = 1.0
        po.loading_scenario.set(loading_type='monotonic')
        po.run()
        u3 = po.tloop.U_record[-1, -3]
        self.assertAlmostEqual(u3, 4.99853169651)

    def tearDown(self):
        '''
        '''
        pass


if __name__ == "__main__":
    unittest.main()
