from enthought.traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, \
     Instance, Int, Trait, Range, HasTraits, on_trait_change, Event, \
     implements, Dict, Property, cached_property, Delegate

from enthought.traits.ui.api import \
     Item, View, HSplit, VSplit, VGroup, Group, Spring

# Chaco imports
from enthought.chaco.chaco_plot_editor import \
     ChacoPlotEditor, \
     ChacoPlotItem
from enthought.enable.component_editor import \
     ComponentEditor
from enthought.chaco.tools.api import \
     PanTool, SimpleZoom
from enthought.chaco.api import \
     Plot, AbstractPlotData, ArrayPlotData

#from dacwt import DAC

from numpy import \
     array, ones, zeros, outer, inner, transpose, dot, frompyfunc, \
     fabs, sqrt, linspace, vdot, identity, tensordot, \
     sin as nsin, meshgrid, float_, ix_, \
     vstack, hstack, sqrt as arr_sqrt, eye

from math import pi as Pi, cos, sin, exp, sqrt as scalar_sqrt

from scipy.linalg import eig, inv

from ibvpy.core.tstepper import \
     TStepper as TS

from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval

from ibvpy.api import RTrace, RTraceGraph, RTraceArraySnapshot


#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------

class MATS2DConduction( MATS2DEval ):
    '''
    Elastic Model.
    '''

    implements( IMATSEval )

    #---------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #---------------------------------------------------------------------------
  
    stress_state  = Enum("plane_stress","plane_strain","rotational_symetry")
   
    #---------------------------------------------------------------------------
    # Material parameters 
    #---------------------------------------------------------------------------

    k   = Float( 1.,
                 label = "k",
                 desc = "conduction",
                 auto_set = False )
    
    D_mtx = Property(Array, depends_on ='k')
    @cached_property
    def _get_D_mtx(self):
        return self.k * eye(2)

    
    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event                     

    #---------------------------------------------------------------------------------------------
    # View specification
    #---------------------------------------------------------------------------------------------

    view_traits = View( VSplit( Item('k'),
                                ),
                        resizable = True
                        )

    #-----------------------------------------------------------------------------------------------
    # Private initialization methods
    #-----------------------------------------------------------------------------------------------


 
    #-----------------------------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-----------------------------------------------------------------------------------------------

    def new_cntl_var(self):
        return zeros( 3, float_ )

    def new_resp_var(self):
        return zeros( 3, float_ )

        
    #-----------------------------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-----------------------------------------------------------------------------------------------

    def get_corr_pred( self, sctx, eps_app_eng, d_eps, tn, tn1 ):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        
        # You print the stress you just computed and the value of the apparent E

        return  dot(self.D_mtx,eps_app_eng) , self.D_mtx
 
    #---------------------------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #---------------------------------------------------------------------------------------------
    

    #---------------------------------------------------------------------------------------------
    # Response trace evaluators
    #---------------------------------------------------------------------------------------------

    def get_sig_norm( self, sctx, eps_app_eng ):
        sig_eng, D_mtx = self.get_corr_pred( sctx, eps_app_eng, 0, 0, 0 )
        return array( [ scalar_sqrt( sig_eng[0]**2 + sig_eng[1]**2 ) ] )
    

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait( Dict )
    def _rte_dict_default(self):
        return {'sig_app'  : self.get_sig_app,
                'eps_app'  : self.get_eps_app,
                'sig_norm' : self.get_sig_norm,
                'strain_energy' : self.get_strain_energy}

if __name__ == '__main__':
    #--------------------------------------------------------------------------------
    # Example using the mats2d_explore 
    #--------------------------------------------------------------------------------
    from ibvpy.mats.mats2D.mats2D_explore import MATS2DExplore
    mats2D_explore = \
    MATS2DExplore( mats2D_eval = MATS2DElastic(),
                    rtrace_list = [ RTraceGraph(name = 'strain 0 - stress 0',
                                                   var_x = 'eps_app', idx_x = 0,
                                                   var_y = 'sig_app', idx_y = 0,
                                                   update_on = 'update' ),
                                    RTraceGraph(name = 'strain 0 - strain 1',
                                                   var_x = 'eps_app', idx_x = 0,
                                                   var_y = 'eps_app', idx_y = 1,
                                                   update_on = 'update' ),
                                    RTraceGraph(name = 'stress 0 - stress 1',
                                                   var_x = 'sig_app', idx_x = 0,
                                                   var_y = 'sig_app', idx_y = 1,
                                                   update_on = 'update' ),
                                    RTraceGraph(name = 'time - sig_norm',
                                                   var_x = 'time', idx_x = 0,
                                                   var_y = 'sig_norm', idx_y = 0,
                                                   update_on = 'update' )

                                    ])
        
    mats2D_explore.tloop.eval()
    #mme.configure_traits( view = 'traits_view_begehung' )
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = mats2D_explore )
    ibvpy_app.main()
    

