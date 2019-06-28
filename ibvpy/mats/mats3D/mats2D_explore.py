
from ibvpy.api import BCDof
from ibvpy.api import BCDof, RTrace, TStepper
from ibvpy.core.ibv_model import IBVModel
from ibvpy.core.scontext import SContext
from ibvpy.core.sdomain import SDomain
from ibvpy.core.tloop import TLoop, TLine
from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import MATS2DMicroplaneDamage
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
from ibvpy.mats.matsXD.matsXD_explore import MATSXDExplore
from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mesh.fe_domain import FEDomain
from ibvpy.mesh.fe_grid import FEGrid
from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, \
     Instance, Int, Trait, Range, HasStrictTraits, on_trait_change, Event, \
     Dict, Property, cached_property, Delegate, List, WeakRef
from traitsui.api import \
    Item, View, HSplit, VSplit, VGroup, Group, Spring
from util.traits.either_type import \
    EitherType


class MATS2DExplore(MATSXDExplore):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    mats_eval = EitherType(klasses=[MATS2DElastic,
                                       MATS2DMicroplaneDamage,
                                       MATS2DScalarDamage ])

    def _mats_eval_default(self):
        return MATS2DElastic()
