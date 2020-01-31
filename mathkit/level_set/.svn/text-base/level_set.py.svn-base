
    
from enthought.traits.api import \
    HasTraits, List, Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, Tuple, \
    Int, Float, implements, Delegate, Interface

from enthought.traits.ui.api import \
    View, Item

from ibvpy.core.i_sdomain import \
    ISDomain
    
from ibvpy.core.sdomain import \
    SDomain

from numpy import \
    array, unique, min, max, mgrid, ogrid, c_, alltrue, repeat, ix_, \
    arange, ones, zeros, multiply, sort, index_exp, indices, add, hstack, \
    frompyfunc, where

from ibvpy.plugins.mayavi.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid

from ibvpy.mesh.cell_grid.cell_spec import CellSpec, GridCell
from ibvpy.mesh.cell_grid.cell_array import CellView, ICellView, CellArray, ICellArraySource

from math import sin

class ILevelSetFn(Interface):
   def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        raise NotImplementedError
    
class SinLSF(HasTraits):
    implements(ILevelSetFn)
    a = Float(1.5, enter_set = True, auto_set = False )
    b = Float(2.0, enter_set = True, auto_set = False )
    
    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return y - ( sin( self.b * x ) + self.a )
    
class PlaneLSF(HasTraits):
    implements(ILevelSetFn)
    a = Float(.5, enter_set = True, auto_set = False )
    b = Float(2.0, enter_set = True, auto_set = False )
    c = Float(-2.5, enter_set = True, auto_set = False )
    
    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return self.a * x + self.b * y + self.c
    
    
class ElipseLSF(HasTraits):
    implements(ILevelSetFn)
    a = Float(.5, enter_set = True, auto_set = False )
    b = Float(2.0, enter_set = True, auto_set = False )
    c = Float(-2.5, enter_set = True, auto_set = False )
    
    def level_set_fn(self, x, y):
        '''Level set function evaluation.
        '''
        return self.a * x*x + self.b * y*y - self.c
    
    