

from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import MATS2DMicroplaneDamage
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.mats.mats2D.mats2D_sdamage.vmats2D_sdamage import MATS2DScalarDamage
from ibvpy.mats.matsXD.matsXD_explore import MATSXDExplore
from util.traits.either_type import \
    EitherType


class MATS2DExplore(MATSXDExplore):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    mats_eval = EitherType(klasses=[MATS2DElastic,
                                    MATS2DMicroplaneDamage,
                                    MATS2DScalarDamage])

    def _mats_eval_default(self):
        return MATS2DElastic()
