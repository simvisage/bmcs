
from traits.api import Trait

from ibvpy.mats.mats3D.mats3D_cmdm.mats3D_cmdm import MATS3DMicroplaneDamage
from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic
from ibvpy.mats.mats3D.mats3D_microplane import MATS3DMplCSDEEQ
from ibvpy.mats.mats3D.mats3D_plastic.mats3D_desmorat import MATS3DDesmorat
from ibvpy.mats.mats3D.mats3D_sdamage.vmats3D_sdamage import MATS3DScalarDamage
from ibvpy.mats.matsXD.matsXD_explore import MATSXDExplore
from util.traits.either_type import \
    EitherType


class MATS3DExplore(MATSXDExplore):
    '''
    Simulate the loading histories of a material point in 3D space.
    '''

    mats_eval = EitherType(klasses=[MATS3DElastic,
                                    MATS3DDesmorat,
                                    MATS3DMplCSDEEQ,
                                    MATS3DMicroplaneDamage,
                                    MATS3DScalarDamage])

    def _mats_eval_default(self):
        return MATS3DScalarDamage()

    def _mats_eval_changed(self):
        pass