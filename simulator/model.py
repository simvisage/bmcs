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
from .tstep import TStep


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

    tstep_type = Type(TStep)
    '''Type of the time step to be used with the model
    '''

    def init_state(self):
        pass

    def record_state(self):
        pass
