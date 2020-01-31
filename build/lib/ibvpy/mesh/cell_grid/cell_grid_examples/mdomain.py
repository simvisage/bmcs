    
from traits.api import \
    HasTraits, List, Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, Tuple, \
    Int, Float, String

from traits.api import Enum

from traitsui.api import \
    View, Item

from ibvpy.core.i_sdomain import \
    ISDomain
    
from ibvpy.core.sdomain import \
    SDomain

from numpy import \
    array, unique, min, max, mgrid, ogrid, c_, alltrue, repeat, ix_, \
    arange, ones, zeros, multiply, sort, index_exp

from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid


from .mcell import MGridCellSpec, MGridCell
from .elem_array_view import ElemArrayView
from functools import reduce


class MGridDomain(SDomain):
    '''
    Get an array with element node coordinates
    '''

    # Grid cell template - gets repeated according to the 
    # within the grid geometry specification
    #
    grid_cell = Property( depends_on = 'grid_cell_spec' )
    @cached_property
    def _get_grid_cell(self):
        return MGridCell( grid_cell_spec = self.grid_cell_spec )


    # Grid geometry specification
    #
    coord_min = Array( Float, value = [ 0., 0., 0.] )
    coord_max = Array( Float, value = [ 1., 1., 1.] )
    
    grid_cell_spec = Instance( MGridCellSpec )
    def _grid_cell_spec_default(self):
        return MGridCellSpec()
    
    # Remark[rch]: 
    # beware - the Int type is not regarded as a normal int
    # within an array and must be first converted to int array
    #
    # Had we defined int as the dtype of an array, there would
    # be errors during editing
    #
    shape   = Array( Int, value = [ 1, 1, 1 ] )
    
    # Derived specifier for element grid shape
    # It converts the Int array to int so that it can be
    # used by general numpy operators
    #
    egrid_shape = Property( depends_on = 'shape' )
    @cached_property
    def _get_egrid_shape(self):
        return self.shape.astype(int)

    #-------------------------------------------------------------------------
    # Shaping and slicing methods for construction and orientation
    #-------------------------------------------------------------------------
    def _get_grid_shape(self):
        '''
        Get the grid shape for the full index and point grids 
        '''
        cell_shape = self.grid_cell_spec.get_cell_shape().astype(int)
        egrid_shape = self.egrid_shape
        return multiply( cell_shape - 1, egrid_shape  ) + 1

    def _get_grid_size(self):
        '''Get the size of the full index and point grids
        '''
        shape = self._get_grid_shape()
        return reduce( lambda i,j: i*j, shape )
    
    def _get_igrid_slices(self):
        '''Get slices defining the index grid
        '''
        sub_cell_shape = self.grid_cell_spec.get_cell_shape() - 1
        elem_grid_shape = self.egrid_shape
        return tuple( [ slice( 0, c*g+1 ) 
                        for c,g in zip(sub_cell_shape,elem_grid_shape) ] )

    def _get_icell_slices(self):
        '''Get slices extracting the first cell from the index grid
        '''
        cell_shape = self.grid_cell_spec.get_cell_shape()
        return tuple( [ slice( 0,c )
                        for c in cell_shape ] )        

    def _get_icell(self):
        '''Get the node map within a cell of a 1-3 dimensional grid
        
        The enumeration of nodes within a single cell is managed by the
        self.grid_cell. This must be adapted to the global enumeration of the grid.
        
        The innermost index runs over the z-axis. Thus, the index of the points
        on the z axis is [0,1,...,shape[2]-1]. The following node at the y axis has 
        the number [shape[2], shape[2]+1, ..., shape[2]*2].

        '''
        igrid = arange( self._get_grid_size() ).reshape( self._get_grid_shape( ))
        # extract the first cell from the igrid
        icell_slices = self._get_icell_slices()
        icell = igrid[ icell_slices ]
        return icell

    #-------------------------------------------------------------------------
    # Generation methods for geometry and index maps
    #-------------------------------------------------------------------------

    def _get_pgrid_slices(self):
        '''Get the slices to be used for the mgrid tool 
        to generate the point grid.
        '''
        ndims = self.grid_cell_spec.get_n_dims()
        shape = self._get_grid_shape()
        return tuple( [ slice( float(self.coord_min[i]),
                               float(self.coord_max[i]),
                               complex(0,shape[i]) ) 
                        for i in range(ndims ) ] )

    pgrid = Property( depends_on = 'grid_cell_spec.+,shape,coord_min,coord_max' )
    @cached_property
    def _get_pgrid(self):
        '''
        Construct the point grid underlying the mesh grid structure.
        '''
        return mgrid[ self._get_pgrid_slices() ]

    def _get_pgrid_dims(self):
        return self._get_grid_shape()

    def _get_points(self):
        return c_[ tuple([ x.flatten() for x in self.pgrid ]) ]

    def _get_tpoints(self):
        '''Get the points in with deepest index along the x axes.
        '''
        tp = self.pgrid.swapaxes(1,3)
        return c_[ tuple([ x.flatten() for x in tp ]) ]
      
      
        
    # Base node grid
    #
    def _get_base_node_array(self):
        '''
        Construct the base node grid.
        '''
        # get the cell shape - number of cell points without the next base point
        subcell_shape = self.grid_cell_spec.get_cell_shape().astype(int) - 1
        # get the element grid shape (number of elements in each dimension)
        egrid_shape = self.egrid_shape

        # get the index offset between two neighboring points on the x-axis  
        z_offset = subcell_shape[2]
        # get the index offset between two neighboring points on the y-axis  
        y_offset = ( egrid_shape[2] * subcell_shape[2] + 1) * subcell_shape[1]
        # get the index offset between two neighboring points on the z-axis  
        x_offset = ( egrid_shape[2] * subcell_shape[2] + 1 ) * \
                   ( egrid_shape[1] * subcell_shape[1] + 1 ) * \
                     subcell_shape[0]

        # grid shape (shape + 1)
        gshape = egrid_shape + 1
        
        # generate points along an axis respecting the above offsets 
        xi_offsets = x_offset * arange( gshape[0] )
        yi_offsets = y_offset * arange( gshape[1] )
        zi_offsets = z_offset * arange( gshape[2] )

        # expand the dimension of offsets and sum them up. using broadcasting
        # this generates the grid of the base nodes (the last node is cut away 
        # as it does not hold any element
        igrid = xi_offsets[:-1,None,None] + \
                yi_offsets[None,:-1,None] + \
                zi_offsets[None,None,:-1]

        # return the indexes of the base nodes
        return sort( igrid.flatten() )

    def _get_elnode_map(self):
        '''
        Construct an array with 
        '''
        icell = self._get_icell()
        node_map   = icell.flatten()[ self.grid_cell.node_map ]
        base_nodes = self._get_base_node_array()

        # Use broadcasting to construct the node map for all elements
        #
        elnode_map = base_nodes[:,None] + node_map[None,:]
        return elnode_map

    def _get_expanded_elem_coords(self):
        iexp = index_exp[ self._get_elnode_map() ]
        return self._get_points()[iexp]

    def _get_elem_coords(self, elnum):
        iexp = index_exp[ self._get_elnode_map()[ elnum ] ]
        return self._get_points()[iexp]

    def _get_epoints(self):
        '''Get element node numbers
        
        Extract the indices of all points that are connected to an element.
        '''
        return self._get_points()[ ix_( self._get_epoint_numbers( ) ) ]
        
    def _get_epoint_numbers(self):
        # flatten and unique
        elnodes = self._get_elnode_map()
        return unique( elnodes.flatten() )

    show_elem_array = Button
    def _show_elem_array_fired(self):
        elem_array = self._get_elnode_map()
        self.show_array = ElemArrayView( data = elem_array, 
                                         rt_domain = self )
        self.show_array.configure_traits( kind = 'live' )
        
    #-----------------------------------------------------------------
    # Visualization related methods
    #-----------------------------------------------------------------

    mvp_pgrid = Trait( MVStructuredGrid )
    def _mvp_pgrid_default(self):
        return MVStructuredGrid( name = 'Point grid', 
                                  dims = self._get_pgrid_dims,
                                  points = self._get_tpoints )
    
    mvp_mgrid_ngeo_labels = Trait( MVPointLabels )
    def _mvp_mgrid_ngeo_labels_default(self):
        return MVPointLabels( name = 'Geo node numbers', 
                                  points = self._get_epoints,
                                  scalars = self._get_epoint_numbers)


    

    refresh_button = Button('Draw')
    @on_trait_change('refresh_button')
    def redraw(self):
        '''
        '''
        self.mvp_pgrid.redraw( )        
        #self.mvp_mgrid_ngeo_labels.redraw( 'label_scalars' )        

    #------------------------------------------------------------------
    # UI - related methods
    #------------------------------------------------------------------
    traits_view = View(Item('grid_cell_spec'),
                       Item('shape@'),
                       Item('coord_min'),
                       Item('coord_max'),
                       Item('refresh_button'),
                       Item('show_elem_array'),
                       resizable = True,
                       scrollable = True,
                       height = 0.5,
                       width = 0.5)        
    
if __name__ == '__main__':
    mgd = MGridDomain( shape = (2,1,1) )
    
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = mgd )
    ibvpy_app.main()
    
    
    