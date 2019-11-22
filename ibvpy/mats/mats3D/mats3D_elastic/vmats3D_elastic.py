
from ibvpy.mats.mats3D.mats3D_eval import \
    MATS3DEval
from traits.api import Constant


class MATS3DElastic(MATS3DEval):
    '''Elastic Model.
    '''
    n_dims = Constant(3)
