
from traits.api import \
    HasTraits, List, Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, Tuple, \
    Int, Float

from traitsui.api import \
    View, Item

from ibvpy.core.i_sdomain import \
    ISDomain

from ibvpy.core.sdomain import \
    SDomain

from numpy import \
    array, unique, min, max, mgrid, ogrid, c_, repeat, ix_, \
    arange, ones, zeros, multiply, sort, allclose, index_exp

from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels

class CellSpec( HasTraits ):
    '''
    '''
    node_coords = Array( float, value = [[-1, -1, 0],
                         [ 1, -1, 0],
                         [ 1, 1, 0],
                         [ 1, 1, 1],
                         [-1, 1, -1],
                         [-1 / 2., 1, 0],
                         [ 0.  , 1, 0],
                         [ 1 / 2., 1, 0]] )


#    xnode_coords = List( [[-1,0,-1],
#                          [ 1,0,-1],
#                          [ 1,0, 1],
#                          [-1,0, 1]] )
#
#    node_coords = List( [[-1,-1,-1],
#                         [ 1, 0, 0],
#                         [ 1, 1, 1],
#                         ] )

    traits_view = View( Item( 'node_coords', style = 'readonly' ),
                       resizable = True,
                       height = 0.5,
                       width = 0.5 )

    _node_array = Property( Array( 'float_' ), depends_on = 'node_coords' )
    @cached_property
    def _get__node_array( self ):
        '''Get the node array as float_
        '''
        # check that the nodes are equidistant
        return array( self.node_coords, 'float_' )

    n_dims = Property( depends_on = 'node_coords' )
    @cached_property
    def _get_n_dims( self ):
        '''Get the number of dimension of the cell
        '''
        return self._node_array.shape[1]

    def get_cell_shape( self ):
        '''Get the shape of the cell grid.
        '''
        cell_shape = ones( 3, dtype = int )
        ndims = self.n_dims
        narray = self._node_array
        cell_shape[0:ndims] = array( [len( unique( narray[:, i] ) )
                                      for i in range( ndims ) ], dtype = int )
        cell_shape = array( [len( unique( narray[:, i] ) )
                                      for i in range( ndims ) ], dtype = int )
        return cell_shape

    def get_cell_slices( self ):
        '''Get slices for the generation of the cell grid.
        '''
        ndims = self.n_dims
        narray = self._node_array
        return tuple( [ slice( 
                              min( narray[:, i] ),
                              max( narray[:, i] ),
                              complex( 0, len( unique( narray[:, i] ) ) ),
                              )
                        for i in range( ndims ) ] )

    #-------------------------------------------------------------------
    # Visualization-related specification
    #-------------------------------------------------------------------
    cell_lines = Array( int, value = [[0, 1], [1, 2], [2, 0]] )
    cell_faces = Array( int, value = [[0, 1, 2]] )


class GridCell( SDomain ):
    '''
    A single mgrid cell for geometrical representation of the domain.
    
    Based on the grid_cell_spec attribute, 
    the node distribution is determined.
    
    '''
    # Everything depends on the grid_cell_specification
    #
    grid_cell_spec = Instance( CellSpec )
    def _grid_cell_spec_default( self ):
        return CellSpec()

    # Generated grid cell coordinates as they come from mgrid.
    # The dimensionality of the mgrid comes from the 
    # grid_cell_spec_attribute
    #
    grid_cell_coords = Property( depends_on = 'grid_cell_spec' )
    @cached_property
    def _get_grid_cell_coords( self ):
        grid_cell = mgrid[ self.grid_cell_spec.get_cell_slices() ]
        return c_[ tuple( [ x.flatten() for x in grid_cell ] ) ]

    n_nodes = Property( depends_on = 'grid_cell_spec' )
    @cached_property
    def _get_n_nodes( self ):
        '''Return the number of all nodes within the cell.
        '''
        return self.grid_cell_coords.shape[0]

    # Node map lists the active nodes within the grid cell
    # in the specified order
    #
    node_map = Property( Array( int ), depends_on = 'grid_cell_spec' )
    @cached_property
    def _get_node_map( self ):
        n_map = []
        for node in self.grid_cell_spec._node_array:
            for idx, grid_cell_node in enumerate( self.grid_cell_coords ):
                if allclose( node , grid_cell_node , atol = 1.0e-3 ):
                    n_map.append( idx )
                    continue
        return array( n_map, int )

    #-----------------------------------------------------------------
    # Visualization related methods
    #-----------------------------------------------------------------
    mvp_mgrid_ngeo_labels = Trait( MVPointLabels )
    def _mvp_mgrid_ngeo_labels_default( self ):
        return MVPointLabels( name = 'Geo node numbers',
                                  points = self._get_points,
                                  scalars = self._get_node_distribution )

    refresh_button = Button( 'Draw' )
    @on_trait_change( 'refresh_button' )
    def redraw( self ):
        '''
        '''
        self.mvp_mgrid_ngeo_labels.redraw( 'label_scalars' )

    def _get_points( self ):
        points = self.grid_cell_coords[ ix_( self.node_map ) ]
        shape = points.shape
        if shape[1] < 3:
            _points = zeros( ( shape[0], 3 ), dtype = float )
            _points[:, 0:shape[1]] = points
            return _points
        else:
            return points

    def _get_node_distribution( self ):
        #return arange(len(self.node_map))
        n_points = self.grid_cell_coords.shape[0]
        full_node_map = ones( n_points, dtype = float ) * -1.
        full_node_map[ ix_( self.node_map ) ] = arange( len( self.node_map ) )
        return full_node_map

    def __getitem__( self, idx ):
        # construct the full boolean map of the grid cell'        
        node_bool_map = repeat( False, self.n_nodes ).reshape( self.grid_cell_spec.get_cell_shape() )
        # put true at the sliced positions         
        node_bool_map[idx] = True
        # extract the used nodes using the node map
        node_selection = node_bool_map.flatten()[ self.node_map ]
        return node_selection

    #------------------------------------------------------------------
    # UI - related methods
    #------------------------------------------------------------------
    traits_view = View( Item( 'grid_cell_spec' ),
                       Item( 'refresh_button' ),
                       Item( 'node_map' ),
                       resizable = True,
                       height = 0.5,
                       width = 0.5 )

if __name__ == '__main__':

    cc = CellSpec( node_coords = [[-1, -1],
                                   [1, -1],
                                   [0, 0],
                                   [1, 1],
                                   [-1, 1]] )



    mgc = GridCell( grid_cell_spec = cc )

    print('cell_shape')
    print(cc.get_cell_shape())

    print('grid_cell_coords')
    print(mgc.grid_cell_coords)

    print('node_map')
    print(mgc.node_map)

    print('node distribution')
    print(mgc._get_node_distribution())

    node_bool_map = repeat( False, mgc.n_nodes ).reshape( cc.get_cell_shape() )
    print('*** construct the full boolean map of the grid cell')
    print('node_bool_map')
    print(node_bool_map)

    node_bool_map[:, -1] = True
    print('*** make the slice')
    print('node_bool_map[:,-1]')
    print(node_bool_map)

    print('*** apply the node map to the full boolean sliced array')
    print('node_map[ node_bool_map]')
    node_selection = mgc[:, -1]
    print(node_selection)

    print('*** get the numbers of nodes within the slice')
    print('node_map[ node_bool_map]')
    print(mgc.node_map[ node_selection])

    print('*** get the point coordinates within the selection')
    print('points')
    print(mgc._get_points()[ node_selection ])

