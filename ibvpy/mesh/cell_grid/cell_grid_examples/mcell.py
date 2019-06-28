 
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
    arange, ones, zeros, multiply, sort

from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels
  
class MGridCellSpec(HasTraits):
    '''
    '''
    
    geo_type = Enum("Triangle", "Diamond")

    
    node_coords = List( [[-1,   -1,  0],
                         [ 1,   -1,  0],
                         [ 1,    1,  0],
                         [ 1,    1,  1],
                         [-1,    1, -1],
                         [-1/2., 1,  0],
                         [ 0.  , 1,  0],
                         [ 1/2., 1,  0]] )

    xnode_coords = List( [[-1,0,-1],
                          [ 1,0,-1],
                          [ 1,0, 1],
                          [-1,0, 1]] )
    #-------------------------------------------------------------------
    # Visualization-related specification
    #-------------------------------------------------------------------
    cell_lines = Array( int, value = [[0,1],[1,2],[2,0]])

    @on_trait_change('geo_type')
    def _reset_node_coords(self):
        if self.geo_type == "Triangle":
            # three points selected like triangle 
            self.node_coords =  [[-1,-1,-1],
                                 [ 1, 0, 0],
                                 [ 1, 1, 1],
                                 ]
            # Triangle: 
            self.cell_lines = [[0,1],[1,2],[2,0]]
            cell_faces = [[0,1,2]]
        else:
            #Diamond
            self.node_coords = [[0,   -1,   -1],   
                                [1,    0,   -1],   
                                [0,    1,   -1],   
                                [-1,   0,   -1],
        
                                [0,   -1,    1],
                                [1,    0,    1],
                                [0,    1,    1],
                                [-1,   0,    1],
                                
                                [1,   -1,    0],
                                [-1,  -1,    0],
                                
                                [1,    1,    0],
                                [-1,   1,    0],
                                ]
            # Diamand
            self.cell_lines =  [[0,1],[1,2],[2,3],[3,0], 
                           [4,5],[5,6],[6,7],[7,4], 
                           [0,8],[8,4],[4,9],[9,0], 
                           [2,10],[10,6],[6,11],[11,2],
                           [9,3],[3,11],[11,7],[7,9],
                           [8,1],[1,10],[10,5],[5,8] ]
            
            cell_faces = [[0,3,2,1],[4,5,6,7], [0,8,4,9], [2,10,6,11], [9,3,11,7], [8,1,10,5] ]


    traits_view = View(Item('geo_type'),
                       Item('node_coords',style='readonly'),
                       Item('cell_lines',style='readonly'),
                       resizable = True,
                       height = 0.5,
                       width = 0.5)

    _node_array = Property( Array('float_'), depends_on = 'node_coords' )
    @cached_property
    def _get__node_array(self):
        '''Get the node array as float_
        '''
        # check that the nodes are equidistant
        return array( self.node_coords, 'float_' )

    def get_n_dims(self):
        '''Get the number of dimension of the cell
        '''
        return self._node_array.shape[1]
    
    def get_cell_shape(self):
        '''Get the shape of the cell grid.
        '''
        npoints = self._node_array.shape[1]
        narray = self._node_array

        return array( [len( unique( narray[:,i] ) )
                       for i in range(npoints) ], dtype = int )

    def get_cell_slices(self):
        '''Get slices for the generation of the cell grid.
        '''
        ndims = self.get_n_dims()
        narray = self._node_array
        return tuple( [ slice( 
                              min( narray[:,i] ),
                              max( narray[:,i] ),
                              complex(0,len(unique(narray[:,i]))), 
                              ) 
                        for i in range(ndims) ] )

class MGridCell(SDomain):
    '''
    A single mgrid cell for geometrical representation of the domain.
    
    Based on the grid_cell_spec attribute, 
    the node distribution is determined.
    
    '''
    # Everything depends on the grid_cell_specification
    #
    grid_cell_spec = Instance( MGridCellSpec )
    def _grid_cell_spec_default(self):
        return MGridCellSpec()
    
    
    # Generated grid cell coordinates as they come from mgrid.
    # The dimensionality of the mgrid comes from the 
    # grid_cell_spec_attribute
    #
    grid_cell_coords = Property( depends_on = 'grid_cell_spec')
    @cached_property
    def _get_grid_cell_coords(self):
        grid_cell = mgrid[ self.grid_cell_spec.get_cell_slices() ]
        return c_[ tuple([ x.flatten() for x in grid_cell ]) ]

    # Node map lists the active nodes within the grid cell
    # in the specified order
    #
    node_map = Property( Array(int), depends_on = 'grid_cell_spec' )
    @cached_property
    def _get_node_map(self):
        n_map = []
        for node in self.grid_cell_spec._node_array:
            for idx, grid_cell_node in enumerate( self.grid_cell_coords ):
                if alltrue( node == grid_cell_node ):
                    n_map.append( idx )
                    continue
        return array(n_map,int)

    #-----------------------------------------------------------------
    # Visualization related methods
    #-----------------------------------------------------------------
    mvp_mgrid_ngeo_labels = Trait( MVPointLabels )
    def _mvp_mgrid_ngeo_labels_default(self):
        return MVPointLabels( name = 'Geo node numbers',
                                  points = self._get_points,
                                  scalars = self._get_node_distribution,
                                  color = (153,204,0))
        
     
    refresh_button = Button('Draw')
    @on_trait_change('refresh_button')
    def redraw(selfMGridCell):
        '''
        '''
        self.mvp_mgrid_ngeo_labels.redraw( 'label_scalars' )        

    def _get_points(self):
        #points = self.grid_cell_coords[ ix_(self.node_map) ]

        points = self.grid_cell_coords 
        print(points)
        shape = points.shape
        if shape[1] < 3:
            _points = zeros( (shape[0],3), dtype = float )
            _points[:,0:shape[1]] = points
            return _points
        else:
            return points

    def _get_node_distribution(self):
        #return arange(len(self.node_map))
        n_points = self.grid_cell_coords.shape[0]
        full_node_map = ones(n_points,dtype=float) * -1.
        full_node_map[ ix_(self.node_map) ] = arange(len(self.node_map))
        print(full_node_map)
        return full_node_map

    #------------------------------------------------------------------
    # UI - related methods
    #------------------------------------------------------------------
    traits_view = View(Item('grid_cell_spec'),
                       Item('refresh_button'),
                       Item('node_map'),
                       resizable = True,
                       height = 0.5,
                       width = 0.5)        

class MGridPntCell(MGridCell):
    '''
    '''
    
class MGridDofCell(MGridCell):
    '''
    '''


if __name__ == '__main__':
    
    mgc = MGridCell( grid_cell_spec = MGridCellSpec() )
    
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = mgc )
    ibvpy_app.main()

