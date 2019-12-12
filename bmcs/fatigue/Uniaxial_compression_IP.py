


from bmcs.time_functions import \
    LoadingScenario
    
from ibvpy.bcond import BCDof

from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from ibvpy.mats.mats3D.mats3D_microplane import \
    MATS3DMplCSDEEQ, MATS3DMplDamageEEQ
        
from simulator.api import Simulator, XDomainSinglePoint

import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    
    
    material_model = MATS3DMplCSDEEQ()
    

    s = Simulator(
        domains=[(XDomainSinglePoint(), material_model)]
    )
    
    bc = BCDof(
        var='u', dof=0, value=-0.05,
        time_function=LoadingScenario()
    )
    
    
#------------------------------------------------------------------------------ 
# cyclic loading
#------------------------------------------------------------------------------ 
    #bc.time_function.trait_set(loading_type='cyclic', number_of_cycles = 2, amplitude_type = 'constant' , unloading_ratio = 0.5)
    
#------------------------------------------------------------------------------ 
# monotonic loading
#------------------------------------------------------------------------------ 
    bc.time_function.trait_set(loading_type='monotonic')
    
    s.tline.step = 0.01
    s.tloop.k_max = 1000
    s.bc = [bc]
    s.run()

    
#------------------------------------------------------------------------------ 
# Extracting data
#------------------------------------------------------------------------------ 
    U_t = s.hist.U_t
    F_t = s.hist.F_t
    t = s.hist.t
    state_vars = s.hist.state_vars
    a = s.hist.record_dict.values()
    
    #print(F_t)
    #print(U_t)
    print (t)
    #print(state_vars)
    #print(np.array([sv['w_T_Emn'] for sv in state_vars[0]]))
    
    
#------------------------------------------------------------------------------ 
# plotting
#------------------------------------------------------------------------------     
    plt.subplot(111)

    plt.plot(-U_t[:,0], -F_t[:,0], 'k', linewidth=1, alpha=1.0)
    
    #plt.plot(-U_t[:,1], -F_t[:,0], 'k', linewidth=1, alpha=1.0)

    plt.show()

    