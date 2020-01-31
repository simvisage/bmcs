
from traits.api import Array, Int, List, Interface, Type
from ibvpy.core.i_tstepper_eval import ITStepperEval

#-------------------------------------------------------------------
# IFETSEval - interface for fe-numerical quadrature
#-------------------------------------------------------------------

class IFElem( Interface ):
    '''Finite Element interface for spatial resolution and integration.
    '''
    dof_r = List
    
    geo_r = List
    
    n_nodal_dofs = Int
    

class IFETSEval( IFElem, ITStepperEval ):
    '''Interface for time steppers.
    Unified interface for spatial and temporal resolution and integration.
    '''
    #        
    #    The class of the domain time-stepper must be harmonized with the 
    #    integration scheme over a finite elements. Essentially, two types are 
    #    distinguished:
    #    - regular integration implemented in DOTSEval
    #    - irregular integration (mainteinance of integration points for
    #      each element within the domain is required) implemented in XDOTSEval
    #
    dots_class = Type

    def adjust_spatial_context_for_point( self, sctx ):
        '''
        Method gets called prior to the evaluation at the material point level.
        
        The method can be used for dimensionally reduced elements that 
        are using higher-dimensional material models.
        '''        
        pass
    
    def get_corr_pred(self, sctx, u, du, tn, tn1, u_avg, 
                      B_mtx_grid, J_det_grid,
                      ip_coords, ip_weights ):   
        '''Return the corrector and predictor for the next step.
        ''' 
        pass