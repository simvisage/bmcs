

from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, Interface, implements, \
     Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
     on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property, Dict

from fets_eval import FETSEval
from i_fets_eval import IFETSEval

class XFETSEval( FETSEval ):
    '''Base class of an eXtended or enriched element.
    '''
    parent_fets_eval = Instance( IFETSEval )
    
    n_parents = Int( desc = 'number of parents included in the evaluation')