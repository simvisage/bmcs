'''
Created on Dec 17, 2018

@author: rch
'''
from traits.api import Interface


class IHist(Interface):

    def add_timestep(self, t):
        '''Add time step to history
        '''

    def get_time_idx_arr(self, t):
        '''Get the index corresponding to visual time
        '''

    def get_time_idx(self, t):
        '''Get an index of a time step given the time
        '''
