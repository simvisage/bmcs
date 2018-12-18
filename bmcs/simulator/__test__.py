'''
Created on Dec 17, 2018

@author: rch
'''
import unittest
from .simulator import Simulator


class Test(unittest.TestCase):

    def test_sim_init(self):
        sim = Simulator()
        sim.run()
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
