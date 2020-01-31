
from enthought.traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, \
     Instance, Int, Trait, Range, HasStrictTraits, on_trait_change, Event, \
     implements, Dict, Property, cached_property, Delegate, List, WeakRef

from util.traits.either_type import \
    EitherType

from enthought.traits.ui.api import \
    Item, View, HSplit, VSplit, VGroup, Group, Spring

from ibvpy.api import BCDof, RTrace, TStepper
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.core.tloop import TLoop, TLine
from ibvpy.core.sdomain import SDomain
from ibvpy.core.scontext import SContext
from ibvpy.api import BCDof
from ibvpy.core.ibv_model import IBVModel
from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import MATS2DMicroplaneDamage
from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_domain import FEDomain

from ibvpy.mats.matsXD.matsXD_explore import MATSXDExplore

class MATS2DExplore( MATSXDExplore ):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    mats_eval = EitherType( klasses = [MATS2DElastic, 
                                       MATS2DMicroplaneDamage,
                                       MATS2DScalarDamage ] )

    def _mats_eval_default(self):
        return MATS2DElastic()
