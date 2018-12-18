'''
Created on Dec 17, 2018

@author: rch
'''

from traits.api import provides
from view.ui.bmcs_tree_node import \
    BMCSTreeNode
from .i_model import IModel


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
    # declare the control variables

    # declare the response variables

    def get_R(self, u, t):
        '''Get the value of the residuum.
        '''

    def get_dR(self, u, t):
        '''Get the gradient.
        '''

    def init_state(self):
        '''Initialize state.
        '''

    def record_state(self, u, t):
        '''Record state in history.
        '''
