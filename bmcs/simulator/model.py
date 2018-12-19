'''
Created on Dec 17, 2018

@author: rch
'''

from traits.api import \
    provides, Type
from view.ui.bmcs_tree_node import \
    BMCSTreeNode
from .i_model import IModel
from .tloop import TLoop


@provides(IModel)
class Model(BMCSTreeNode):
    '''
    Base class of a time-stepping model capturing the state
    of a modeled object at the particular instance of time 
    supplying the predictor and corrector interface.

    The model includes the geometry, boundary conditions, 
    material characteristics, type of material behavior

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

    tloop_type = Type(TLoop)
    '''Type of time loop to be used with the model
    '''

    # declare the control variables

    # declare the response variables

    def get_R(self, u, t):
        '''Get the value of the residuum.
        '''

    def get_dR(self, u, t):
        '''Get the gradient.
        '''

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
