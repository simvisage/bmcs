'''
Created on Dec 17, 2018

@author: rch
'''

from traits.api import Interface, Type


class IModel(Interface):
    r'''
    Interface of a time-stepping model capturing the state
    of a modeled object at the particular instance of time 
    supplying the predictor and corrector interface.

    There are three categories of parameters 

    * model parameters capturing the

      * geometry
      * boundary conditions
      * material characteristics
      * type of material behavior

    * control variable u that is associated with 
      state ``s`` 

    * for the given state response variables ``r(u,s)`` and eventually
      their derivatives 

    To support an iterative scheme with prediction based 
    on discrete the model can store several time steps  
    needed to make predictions for the next step.

    A model communicates with the time loop through their 
    common interface which depends on the particular type 
    of time-stepping scheme. In case of a time integration 
    scheme the control variable is represented by displacements,
    in case of an optimization problem, the function supplying 
    the values and gradients of response variables is required.
    In case of fragmentation problem, the model supplies 
    the array of load factors need to induce a crack in 
    a material point. 
    '''
    tloop_type = Type

    tstep_type = Type

    def get_corr_pred(self, U_k, t_n):
        '''Return the value and the derivative of a function
        '''

    def init_state(self):
        '''Initialize state.
        '''

    def get_state(self):
        '''Get the current control and primary variables.
        '''

    def update_state(self, U_k, t_n):
        '''Update the control, primary and state variables..
        '''

    def record_state(self):
        '''Provide the current state for history recording.
        '''

    def get_R(self, u, t):
        '''Get the value of the residuum.
        '''

    def get_dR(self, u, t):
        '''Get the gradient.
        '''
