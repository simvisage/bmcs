'''
Created on 01.03.2017

@author: abaktheer
'''

from mpl_toolkits.axisartist.axis_artist import UnimplementedException
from scipy.interpolate import interp1d
from traits.api import HasStrictTraits, Range, Float

import numpy as np


class TimeFunctionFactory(HasStrictTraits):
    
    def deliver(self):
        raise UnimplementedException, 'deliver method not implemented'
    

class Monotonic(TimeFunctionFactory):
    
    maximum_loading = Float(1.0)
    number_of_increments = Float(50)
    t_max = Float(1.0)
     
    def _get_time_function(self):
        d_levels = np.linspace(0, self.maximum_loading, 2)
        d_levels[0] = 0
        d_levels.reshape(-1, 2)[:, 0] *= 0
        d_history = d_levels.flatten()
        d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])
        
        t_arr = np.linspace(0, self.t_max, len(self.d_arr))
        return interp1d(t_arr,d_arr)

class TFCyclicIncreasedSymmetric(TimeFunctionFactory):
    
    number_of_cycles = Float(10.0)
    maximum_loading = Float(1.0)
    number_of_increments = Float(20)
    t_max = Float(1.0)
    
    def _get_time_function(self):
        d_levels = np.linspace(0, self.maximum_loading, self.number_of_cycles * 2)
        d_levels.reshape(-1, 2)[:, 0] *= -1
        d_history = d_levels.flatten()
        d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])
        t_arr = np.linspace(0, self.t_max, len(self.d_arr))
        return interp1d(t_arr,d_arr)

class TFCyclicIncreasedNonsymmetric(TimeFunctionFactory):  
    
    number_of_cycles = Float(10.0)
    maximum_loading = Float(1.0)
    number_of_increments = Float(20)
    t_max = Float(1.0)
    
    def _get_time_function(self):
        d_levels = np.linspace(0, self.maximum_loading, self.number_of_cycles * 2)
        d_levels.reshape(-1, 2)[:, 0] *= 0
        d_history = d_levels.flatten()
        d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])
        t_arr = np.linspace(0, self.t_max, len(self.d_arr))
        return interp1d(t_arr,d_arr)
 
class TFCyclicConstantSymmetric(TimeFunctionFactory):
    
    number_of_cycles = Float(10.0)
    maximum_loading = Float(1.0)
    number_of_increments = Float(20) 
    t_max = Float(1.0)
    
    def _get_time_function(self):
        d_levels = np.linspace(0, self.maximum_loading, self.number_of_cycles * 2)
        d_levels.reshape(-1, 2)[:, 0] = -self.maximum_loading
        d_levels[0] = 0
        d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
        d_history = d_levels.flatten()
        d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])
        t_arr = np.linspace(0, self.t_max, len(self.d_arr))
        return interp1d(t_arr,d_arr)
              
class TFCyclicConstantNonsymmetric(TimeFunctionFactory): 
    
    number_of_cycles = Float(10.0)
    maximum_loading = Float(1.0)
    unloading_ratio = Range(0., 1., value=0.5)
    number_of_increments = Float(20)
    t_max = Float(1.0) 
    
    def _get_time_function(self):
        d_1 = np.zeros(1)
        d_2 = np.linspace(0, self.maximum_loading, self.number_of_cycles * 2)
        d_2.reshape(-1, 2)[:, 0] = self.maximum_loading
        d_2.reshape(-1, 2)[:, 1] = self.maximum_loading * \
                self.unloading_ratio
        d_history = d_2.flatten()
        d_arr = np.hstack((d_1, d_history))
        t_arr = np.linspace(0, self.t_max, len(self.d_arr))
        return interp1d(t_arr,d_arr)
           
#class Custom(TimeFunctionFactory):
    
                
                
                 
if __name__ == '__main__':





    
    
    
    
    
    