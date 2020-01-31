
from enthought.traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, \
     Instance, Int, Trait, Range, HasTraits, on_trait_change, Event, \
     implements, Dict, Property, cached_property, Delegate, List

from util.traits.either_type import \
    EitherType

from enthought.traits.ui.api import \
    Item, View, HSplit, VSplit, VGroup, Group, Spring

from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.core.tloop import TLoop, TLine
from ibvpy.core.tstepper import TStepper
from ibvpy.core.sdomain import SDomain
from ibvpy.core.scontext import SContext
from ibvpy.core.ibv_model import IBVModel
from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_domain import FEDomain
from numpy import pi as Pi, array, zeros, c_, mgrid
from mats2D.mats2D_explore import MATS2DExplore
from mats1D.mats1D_explore import MATS1DExplore
from mats1D5.mats1D5_explore import MATS1D5Explore

class FEUnitElem( FEDomain ):
    '''Unit volume for non-local or regularized material models.
    '''
    mats_eval = IMATSEval

    def new_scontext( self ):
        '''Spatial context factory'''
        sctx = SContext()
        nd = self.mats_eval.n_dims
        x, y = mgrid[0:nd + 1, 0:nd + 1]
        sctx.X_reg = c_[ x.flatten(), y.flatten() ]

        state_arr_size = self.mats_eval.get_state_array_size()
        sctx.mats_state_array = zeros( state_arr_size, 'float_' )
        return sctx

class MATSExplore( IBVModel ):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    dim = EitherType( names = ['1D', '1D5', '2D', '3D' ],
                      klasses = [MATS1DExplore,
                                 MATS1D5Explore,
                                 MATS2DExplore,
                                 #MATS3DExplore
                                 ] )

    max_load = Float( 5.0, enter_set = True, auto_set = False )

    time_function = Callable( lambda t: t )

    n_steps = Float( 30, enter_set = True, auto_set = False )

    tolerance = Float( 1e-5, enter_set = True, auto_set = False )

    n_iterations = Float( 10, enter_set = True, auto_set = False )

    n_restarts = Float( 0, enter_set = True, auto_set = False )

    def _dim_default( self ):
        return MATS2DExplore( explorer = self )

    def _dim_changed( self ):
        self.dim.explorer = self
        self.dim._mats_eval_changed()

    def _tloop_default( self ):

        self.ts = TStepper( 
                        tse = self.dim.mats_eval,
                        sdomain = FEUnitElem( mats_eval = self.dim.mats_eval )
                       )

        # Put the time-stepper into the time-loop
        #
        n_steps = self.n_steps

        tloop = TLoop( tstepper = self.ts,
                       KMAX = 100, RESETMAX = self.n_restarts,
                       tolerance = 1e-8,
                       tline = TLine( min = 0.0, step = 1.0 / n_steps, max = 1.0 ) )

        return tloop

    traits_view = View( 
                        Item( 'dim@', show_label = False ),
                              resizable = True,
                              width = 1.0,
                              height = 1.0,
                              scrollable = True,
                              )

if __name__ == '__main__':
    from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import \
        MATS2DMicroplaneDamage

    from ibvpy.mats.matsXD.matsXD_cmdm.matsXD_cmdm_phi_fn import \
         PhiFnStrainHardeningLinear

    phi_fn = PhiFnStrainHardeningLinear( alpha = 0.5, beta = 0.7 )
    explorer = MATSExplore( dim = MATS2DExplore( mats_eval = \
                                                 MATS2DMicroplaneDamage( n_mp = 30,
                                                                         phi_fn = phi_fn ) ) )

    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = explorer )
    ibvpy_app.main()
