
#-- Imports --------------------------------------------------------------------

from os.path \
    import join, dirname
    
from numpy \
    import sqrt
    
from numpy.random \
    import random

from traits.api \
    import HasTraits, Property, Array, Any, Event, \
    on_trait_change, Instance, WeakRef, Int, Str, Bool, Trait
    
from traitsui.api \
    import View, Item, TabularEditor, HSplit, Group
    
from etsproxy.traits.ui.menu \
    import NoButtons, CancelButton
    
from etsproxy.traits.ui.tabular_adapter \
    import TabularAdapter

from etsproxy.pyface.image_resource \
    import ImageResource
    
from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid    

#-- Constants ------------------------------------------------------------------

import etsproxy.traits.ui.api

#-- Tabular Adapter Definition -------------------------------------------------

class CoordArrayAdapter ( TabularAdapter ):

    columns = Property
    def _get_columns(self):
        return [('node', 'index'),
                ('x', 0),
                ('y', 1),
                ('z', 2) ]
        
    font        = 'Courier 10'
    alignment   = 'right'
    format      = '%g'
    index_text  = Property
    
    def _get_index_text ( self ):
        return str( self.row )
        
#-- Tabular Editor Definition --------------------------------------------------

coord_array_editor = TabularEditor(
    adapter = CoordArrayAdapter( ),
)

class ElemView(HasTraits):
    '''Get the element numbers.
    '''
    elem_num = Int(-1)
    elem_coords = Array
    view = View( Item( 'elem_num', style = 'readonly' ),
                 Item('elem_coords', editor = coord_array_editor, style = 'readonly' )
                 )
    

#-- Tabular Adapter Definition -------------------------------------------------

class ElemArrayAdapter ( TabularAdapter ):

    columns = Property
    def _get_columns(self):
        data = getattr( self.object, self.name )
        if len( data.shape ) != 2:
            raise ValueError('element node array must be two-dimensional')
        n_columns = getattr( self.object, self.name ).shape[1]

        cols = [ (str(i), i ) for i in range( n_columns ) ]
        return [ ('element', 'index') ] + cols
        
    font        = 'Courier 10'
    alignment   = 'right'
    format      = '%d'
    index_text  = Property
#    index_image = Property
    
    def _get_index_text ( self ):
        return str( self.row )
        
    def x_get_index_image ( self ):
        x, y, z = self.item
        if sqrt( (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 ) <= 0.25:
            return 'red_flag'
        return None

#-- Tabular Editor Definition --------------------------------------------------

elem_array_editor = TabularEditor(
    adapter = ElemArrayAdapter(),
    selected_row = 'current_row',
)

#-- ElemArrayView Class Definition -------------------------------------------------

class ElemArrayView ( HasTraits ):

    data = Array
    rt_domain = WeakRef
    
    
    mvp_elem_labels = Trait( MVPointLabels )
    def _mvp_elem_labels_default(self):
        return MVPointLabels( name = 'Geo node numbers', 
                                  points = self._get_current_elem_coords,
                                  scalars = self._get_current_elem_numbers,
                                  #color = (0.254902,0.411765,0.882353)
                                  color = (0.15,0.85,0.45))
                                 
        
    mvp_elem_geo = Trait( MVPolyData )
    def _mvp_elem_geo_default(self):
        return MVPolyData( name = 'Geo node numbers', 
                               points = self._get_current_elem_coords,
                               lines = self._get_current_elem_lines,
                               #color = (0.254902,0.411765,0.882353))
                               color= (0,55,0,75,0.0))
        
    
    show_elem = Bool(True)
            
    def _get_current_elem_coords(self):
        return self.rt_domain._get_elem_coords( self.current_row )
    
    def _get_current_elem_numbers(self):
        return self.data[self.current_row]
    
    def _get_current_elem_lines(self):
        line_map = self.rt_domain.grid_cell_spec.cell_lines
        return line_map
    
    current_row = Int(-1)
    @on_trait_change('current_row')
    def redraw(self):
        if self.show_elem:
            self.mvp_elem_labels.redraw('label_scalars')
            self.mvp_elem_geo.redraw()
    
    elem_view = Instance( ElemView )
    def _elem_view_default(self):
        return ElemView()
    
    @on_trait_change('current_row')
    def _display_current_row(self):
        if self.current_row != -1:
            self.elem_view.elem_num = self.current_row
            elem_coords = self.rt_domain._get_elem_coords( self.current_row )
            self.elem_view.elem_coords = elem_coords
    
    view = View(
                HSplit(
                       Item( 'data', editor = elem_array_editor, 
                             show_label = False, style = 'readonly' ),
                        Group(Item( 'elem_view@', show_label = False ),
                              Item( 'show_elem' , label = 'SHOW')
                             )),
        title     = 'Array Viewer',
        width     = 0.6,
        height    = 0.4,
        resizable = True,
        buttons   = [CancelButton]
    )
    
# Run the demo (if invoked from the command line):
if __name__ == '__main__':

    from traits.api import Button
    from .mdomain import MGridDomain

    rt_domain = MGridDomain( shape = (20,20,1) )
    demo = ElemArrayView( data = rt_domain._get_elnode_map(),
                          rt_domain = rt_domain  )
    demo.configure_traits()
    
