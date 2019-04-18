
from traits.api import Instance, Int
from .fets_eval import FETSEval

from .i_fets_eval import IFETSEval


class XFETSEval(FETSEval):
    '''Base class of an eXtended or enriched element.
    '''
    parent_fets_eval = Instance(IFETSEval)
    
    n_parents = Int(desc='number of parents included in the evaluation')
