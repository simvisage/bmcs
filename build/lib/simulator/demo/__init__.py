'''
..todo:: compare with the current ibvpy.core.tloop and generalize the NRTLoop

..todo:: prepare the next example using sympy to allow for general functions

..todo:: make the `boundary condition' time dependent

..todo:: interface of a model - find an optimum handling of state transitions 
        with R, dR and state_update
        
..todo:: test the scenario - no convergence - and restart with modified tie step

..todo:: Handle the END event of the simulator in case 
that the calculation thread is still running. In such a case,
the request should be confirmed by the user action.

..todo:: Handle the resume state - should it recalculate the 
model staet after the pause event?


'''