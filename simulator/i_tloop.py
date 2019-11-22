'''
Created on Dec 17, 2018

'''
from traits.api import Interface


class ITLoop(Interface):
    '''Loop performing the simulation of a time-stepping model  
    along the specified time line and recording its history.
    '''

    def __call__(self):
        '''Implementation of the loop.
        '''
