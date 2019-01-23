'''
Created on Dec 17, 2018

@author: rch
'''
from traits.api import Interface


class ITStep(Interface):

    def make_iter(self):
        '''Perform one iteration step.
        '''

    def make_incr(self):
        '''Update the control, primary and state variables..
        '''
