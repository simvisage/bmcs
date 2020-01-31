
from functools import reduce

from numpy import \
    array, mgrid, c_, \
    arange, zeros, multiply, sort, index_exp, add, \
    frompyfunc
from traits.api import \
    Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, Tuple, \
    Int, Float, provides, Delegate, Callable
from traitsui.api import \
    View, Item

from ibvpy.core.sdomain import \
    SDomain
from ibvpy.plugins.mayavi_util.pipelines import \
    MVStructuredGrid

from .cell_array import CellView, CellArray, ICellArraySource
from .cell_grid_slice import CellGridSlice
from .cell_spec import CellSpec, GridCell


#--------------------------------------------------------------------------
# CellGrid
#--------------------------------------------------------------------------
@provides(ICellArraySource)
class CellGrid(SDomain):
    '''
    Manage an array of cells defined within a structured grid.

    The distinction between the coordinate information supplied 
    in the arrays is done using the following naming convention:

    point - geometric points within the regular grid

    node - point specified in the grid_cell specification

    vertex - node with topological function (corner nodes)

    base_node - the first node of an element

    point_idx_grid ... enumeration of points within the point_grid

    cell_idx_grid ... enumeration of cells in the grid maps the 
                      topological index to the flattened index
                      ( ix, iy, iz ) -> ii

    cell_node_map ... array mapping the cell idx to the  point idx 
                      representing a node of the cell

    '''

    # Everything depends on the grid_cell_specification
    # defining the distribution of nodes within the cell.
    #
    grid_cell_spec = Instance(CellSpec)

    def _grid_cell_spec_default(self):
        return CellSpec()

    n_dims = Delegate('grid_cell_spec')

    # Grid cell template - gets repeated according to the
    # within the grid geometry specification
    #
    grid_cell = Property(depends_on='grid_cell_spec.+')

    @cached_property
    def _get_grid_cell(self):
        return GridCell(grid_cell_spec=self.grid_cell_spec)

    # Grid geometry specification
    #
    coord_min = Array(Float, value=[0., 0., 0.])
    coord_max = Array(Float, value=[1., 1., 1.])

    # Remark[rch]:
    # beware - the Int type is not regarded as a normal int
    # within an array and must be first converted to int array
    #
    # Had we defined int as the dtype of an array, there would
    # be errors during editing
    #
    shape = Array(Int, value=[1, 1, 1])

    # Derived specifier for element grid shape
    # It converts the Int array to int so that it can be
    # used by general numpy operators
    #
    cell_idx_grid_shape = Property(Array(int), depends_on='shape')

    @cached_property
    def _get_cell_idx_grid_shape(self):
        return array(self.shape, dtype=int)

    cell_idx_grid_size = Property(Int, depends_on='shape')

    @cached_property
    def _get_cell_idx_grid_size(self):
        return reduce(lambda x, y: x * y, self.cell_idx_grid_shape)

    # Grid with the enumeration of the cells respecting
    # the dimensionality. This grid is used to implement
    # the mapping between the cells and nodes.
    #
    cell_idx_grid = Property(Array(int), depends_on='shape')

    @cached_property
    def _get_cell_idx_grid(self):
        return arange(self.cell_idx_grid_size).reshape(self.cell_idx_grid_shape)

    def __getitem__(self, idx):
        '''High level access and slicing to the cells within the grid.

        The return value is a tuple with 
        1. array of cell indices
        2. array of nodes for each element
        3. array of coordinates for each node.
        '''
        return CellGridSlice(cell_grid=self, grid_slice=idx)

    #-------------------------------------------------------------------------
    # Shape and size characteristics used for both the idx_grid and point_grid
    #-------------------------------------------------------------------------
    def _get_point_grid_shape(self):
        '''Get the grid shape for the full index and point grids.

        This is the background grid. Some of the nodes can be unused by the
        cells depending on the specification in the grid_cell_spec.
        '''
        cell_shape = self.grid_cell_spec.get_cell_shape().astype(int)
        cell_idx_grid_shape = self.cell_idx_grid_shape
        return multiply(cell_shape - 1, cell_idx_grid_shape) + 1

    point_grid_size = Property(depends_on='shape,grid_cell_spec.node_coords')

    def _get_point_grid_size(self):
        '''Get the size of the full index and point grids
        '''
        shape = self._get_point_grid_shape()
        return reduce(lambda i, j: i * j, shape)

    #-------------------------------------------------------------------------
    # point_idx_grid - shaping and slicing methods for construction and orientation
    # point_idx_grid represents the enumeration of the nodes of the cell grid. It
    # serves for constructing the mappings between cells and nodes.
    #-------------------------------------------------------------------------
    def _get_point_idx_grid_slices(self):
        '''Get slices defining the index grid in a format suitable 
        for mgrid generation.
        '''
        subcell_shape = self.grid_cell_spec.get_cell_shape() - 1
        cell_idx_grid_shape = self.cell_idx_grid_shape
        return tuple([slice(0, c * g + 1)
                      for c, g in zip(subcell_shape, cell_idx_grid_shape)])

    point_idx_grid = Property(
        Array, depends_on='shape,grid_cell_spec.node_coords')

    @cached_property
    def _get_point_idx_grid(self):
        '''Get the index numbers of the points within the grid
        '''
        return arange(self.point_grid_size).reshape(self._get_point_grid_shape())

    #-------------------------------------------------------------------------
    # Unit cell enumeration - used as a template for enumerations in 3D
    #-------------------------------------------------------------------------
    idx_cell_slices = Property(Tuple)

    def _get_idx_cell_slices(self):
        '''Get slices extracting the first cell from the point index grid
        '''
        cell_shape = self.grid_cell_spec.get_cell_shape()
        return tuple([slice(0, c)
                      for c in cell_shape])

    idx_cell = Property(Array)

    def _get_idx_cell(self):
        '''Get the node map within a cell of a 1-3 dimensional grid

        The enumeration of nodes within a single cell is managed by the
        self.grid_cell. This must be adapted to the global enumeration of the grid.

        The innermost index runs over the z-axis. Thus, the index of the points
        on the z axis is [0,1,...,shape[2]-1]. The following node at the y axis has 
        the number [shape[2], shape[2]+1, ..., shape[2]*2].
        '''
        return self.point_idx_grid[self.idx_cell_slices]

    def get_cell_idx(self, offset):
        '''Get the address of the cell within the cell grid.
        '''
        idx_tuple = zeros((self.n_dims,), dtype='int_')
        roof = offset
        for i, n in enumerate(self.shape[-1:0:-1]):
            idx = roof / (n)
            roof -= idx * n
            idx_tuple[i] = idx
        idx_tuple[self.n_dims - 1] = roof
        return tuple(idx_tuple)

    def get_cell_offset(self, idx_tuple):
        '''Get the index of the cell within the flattened list.
        '''
        return self.cell_idx_grid[idx_tuple]

    #-------------------------------------------------------------------------
    # vertex slices
    #-------------------------------------------------------------------------
    vertex_slices = Property

    def _get_vertex_slices(self):
        cell_shape = self.grid_cell_spec.get_cell_shape()
        return tuple([slice(0, None, c - 1)
                      for c in cell_shape])

    #-------------------------------------------------------------------------
    # point_grid - shaping and slicing methods for construction and orientation
    #-------------------------------------------------------------------------
    point_x_grid_slices = Property

    def _get_point_x_grid_slices(self):
        '''Get the slices to be used for the mgrid tool 
        to generate the point grid.
        '''
        ndims = self.n_dims
        shape = self._get_point_grid_shape()
        return tuple([slice(float(self.coord_min[i]),
                            float(self.coord_max[i]),
                            complex(0, shape[i]))
                      for i in range(ndims)])

    #-------------------------------------------------------------------------
    # Geometry transformation
    #-------------------------------------------------------------------------
    geo_transform = Callable

    geo_transform_vfn = Property

    def _get_geo_transform_vfn(self):
        vfn_shell_stb = frompyfunc(self.geo_transform, 2, 3)

    #-------------------------------------------------------------------------
    # Point coordinates x - is parametric, X - is global
    #-------------------------------------------------------------------------
    point_x_grid = Property(
        depends_on='grid_cell_spec.+,shape,coord_min,coord_max')

    @cached_property
    def _get_point_x_grid(self):
        '''
        Construct the point grid underlying the mesh grid structure.
        '''
        return mgrid[self._get_point_x_grid_slices()]

    point_X_grid = Property

    def _get_point_X_grid(self):
        '''
        Construct the point grid underlying the mesh grid structure.
        '''
        x_dim_shape = self.point_x_grid.shape[1:]
        return array([self.point_X_arr[:, i].reshape(x_dim_shape)
                      for i in range(self.n_dims)], dtype='float_')

    point_x_arr = Property

    def _get_point_x_arr(self):
        return c_[tuple([x.flatten() for x in self.point_x_grid])]

    point_X_arr = Property(
        depends_on='grid_cell_spec,shape,coord_min,coord_max, geo_transform')

    @cached_property
    def _get_point_X_arr(self):
        '''Get the (n,3) array with point coordinates.
        '''
        # If the geo transform has been set, perform the mapping
        #
        if self.geo_transform:
            return self.geo_transform(self.point_x_arr)
        else:
            return self.point_x_arr

    #-------------------------------------------------------------------------
    # Vertex manipulations
    #-------------------------------------------------------------------------
    vertex_idx_grid = Property

    def _get_vertex_idx_grid(self):
        '''
        Construct the base node grid. Base node has the lowest number within a cell.
        All relevant cell_node numbers can be derived just be adding an array
        of relative node offsets within the cell to the base node.
        (see the method get_cell_nodes( cell_num)  below)  
        '''
        # get the cell shape - number of cell points without the next base
        # point
        subcell_shape = self.grid_cell_spec.get_cell_shape().astype(int) - 1
        # get the element grid shape (number of elements in each dimension)
        cell_idx_grid_shape = self.cell_idx_grid_shape

        # The following code determines the offsets between two neighbouring nodes
        # along each axis. It loops over the axes so that also 1- and 2- dimensional
        # grids are included. For 3D grid - the following code shows what's happening
        #
        # 1) get the index offset between two neighboring points on the z-axis
        #
        # z_offset = subcell_shape[2]
        #
        # 2) get the index offset between two neighboring points on the y-axis
        #
        # y_offset = ( cell_idx_grid_shape[2] * subcell_shape[2] + 1) * subcell_shape[1]
        #
        # 3) get the index offset between two neighboring points on the x-axis
        #
        # x_offset = ( cell_idx_grid_shape[2] * subcell_shape[2] + 1 ) * \
        #            ( cell_idx_grid_shape[1] * subcell_shape[1] + 1 ) * \
        #            subcell_shape[0]

        offsets = zeros(self.n_dims, dtype='int_')

        for i in range(self.n_dims - 1, -1, -1):
            offsets[i] = subcell_shape[i]
            for j in range(i + 1, self.n_dims):
                offsets[i] *= (cell_idx_grid_shape[j] * subcell_shape[j] + 1)

        # grid shape (shape + 1)
        gshape = cell_idx_grid_shape + 1

        # Determine the offsets of all base nodes on each axis by multiplying
        # the respective offset with a point enumeration on that axis
        #
        # In 3D corresponds to the following
        #
        #        xi_offsets = x_offset * arange( gshape[0] )
        #        yi_offsets = y_offset * arange( gshape[1] )
        #        zi_offsets = z_offset * arange( gshape[2] )

        all_offsets = [offsets[n] * arange(gshape[n])
                       for n in range(self.n_dims)]

        # Construct the broadcastable slices used for the construction of the
        # base node index grid. In 3D this corresponds to the following:
        #
        # Expand the dimension of offsets and sum them up. using broadcasting
        # this generates the grid of the base nodes (the last node is cut away
        # as it does not hold any element
        #        idx_grid = xi_offsets[:-1,None,None] + \
        #                   yi_offsets[None,:-1,None] + \
        #                   zi_offsets[None,None,:-1]

        slices = []
        for i in range(self.n_dims):
            s = [None for j in range(self.n_dims)]
            s[i] = slice(None)
            slices.append(tuple(s))

        vertex_offsets = [all_offsets[i][slices[i]]
                          for i in range(self.n_dims)]
        vertex_idx_grid = reduce(add, vertex_offsets)

        # return the indexes of the vertex nodes
        return vertex_idx_grid

    vertex_idx_arr = Property

    def _get_vertex_idx_arr(self):
        '''Get the index of nodes at the vertex of the cells
        of the cells. The result is a sorted flat array. The base node 
        position within the returned array defines the index of the cell. 
        '''
        vertex_idx_grid = self.vertex_idx_grid
        return sort(vertex_idx_grid.flatten())

    vertex_x_grid = Property

    def _get_vertex_x_grid(self):
        return array([x_grid[self.vertex_slices] for x_grid in self.point_x_grid])

    vertex_X_grid = Property

    def _get_vertex_X_grid(self):
        return array([X_grid[self.vertex_slices] for X_grid in self.point_X_grid])

    vertex_x_arr = Property

    def _get_vertex_x_arr(self):
        return c_[tuple([x.flatten() for x in self.vertex_x_grid])]

    vertex_X_arr = Property

    def _get_vertex_X_arr(self):
        return c_[tuple([X.flatten() for X in self.vertex_X_grid])]

    #-------------------------------------------------------------------------
    # Cell manipulations
    #-------------------------------------------------------------------------
    base_nodes = Property

    def _get_base_nodes(self):
        '''Get the index of nodes that are the bottom left front vertexs
        of the cells. The result is a sorted flat array. The base node 
        position within the returned array defines the index of the cell. 
        '''
        vertex_idx_grid = self.vertex_idx_grid
        cutoff_last = [slice(0, -1) for i in range(self.n_dims)]
        base_node_grid = vertex_idx_grid[tuple(cutoff_last)]
        return sort(base_node_grid.flatten())

    cell_node_map = Property(depends_on='shape,grid_cell_spec.+')

    @cached_property
    def _get_cell_node_map(self):
        '''
        Construct an array with the mapping between elements and nodes. 
        Returns the dof for [ cell_idx, node, dim ]
        '''
        idx_cell = self.idx_cell
        node_map = idx_cell.flatten()[self.grid_cell.node_map]
        base_nodes = self.base_nodes

        # Use broadcasting to construct the node map for all elements
        #
        cell_node_map = base_nodes[:, None] + node_map[None, :]
        return cell_node_map

    cell_grid_node_map = Property(depends_on='shape,grid_cell_spec.+')

    @cached_property
    def _get_cell_grid_node_map(self):
        '''
        Return the dof for [ cell_x, cell_y, node, dim ]
        where 
        - cell_x - is the cell index in the first dimension
        - cell_y - is the cell index in the second dimension
        - node - is the node index within the cell
        - dim - is the index of the dof within the node
        '''
        new_shape = tuple(self.shape) + self.cell_node_map.shape[1:]
        return self.cell_node_map.reshape(new_shape)

    def get_cell_point_x_arr(self, cell_idx):
        '''Return the node coordinates included in the cell cell_idx. 
        '''
        iexp = index_exp[self.cell_node_map[cell_idx]]
        return self.point_x_arr[iexp]

    def get_cell_point_X_arr(self, cell_idx):
        '''Return the node coordinates included in the cell cell_idx. 
        '''
        iexp = index_exp[self.cell_node_map[cell_idx]]
        return self.point_X_arr[iexp]

    #-------------------------------------------------------------------------
    # @todo - candidate for deletion - the slicing operator [] is doing this more generally
    #-------------------------------------------------------------------------
    boundary_slices = Property(depends_on='grid_cell_spec.+')

    @cached_property
    def _get_boundary_slices(self):
        '''Get the slices to get the boundary nodes.
        '''
        # slices must correspond to the dimensions
        slices = []
        for i in range(self.n_dims):
            s_low = [slice(None) for j in range(self.n_dims)]
            s_low[i] = 0
            s_high = [slice(None) for j in range(self.n_dims)]
            s_high[i] = -1
            slices.append(s_low)
            slices.append(s_high)
        return slices

    #--------------------------------------------------------------------------
    # Wrappers exporting the grid date to mayavi pipelines
    #--------------------------------------------------------------------------\

    def get_cell_mvpoints(self, cell_idx):
        '''Return the node coordinates included in the cell cell_idx. 
        In case that the grid is in reduced dimensions - blow it up with zeros.
        '''
        points = self.get_cell_point_X_arr(cell_idx)

        # augment the points to be 3D
        if self.n_dims < 3:
            mvpoints = zeros((points.shape[0], 3), dtype='float_')
            mvpoints[:, :self.n_dims] = points
            return mvpoints
        else:
            return points

    def _get_mvpoints_grid_shape(self):
        '''Shape of point grid in 3D.
        The information is needed by the mayavi pipeline to use the
        StructuredGrid source - tvtk class.
        '''
        shape = self._get_point_grid_shape()
        return tuple(list(shape) + [1 for n in range(3 - len(shape))])

    def _get_mvpoints(self, swap=True):
        '''Get the points in with deepest index along the x axes.
        This format is required when putting the point data to the 
        vtk structured grid.
        '''
        point_X_grid = self.point_X_grid
        if swap == True:
            point_X_grid = point_X_grid.swapaxes(1, self.n_dims)

        point_X_arr = c_[tuple([X.flatten() for X in point_X_grid])]

        # augment the points to be 3D
        if self.n_dims < 3:
            mv_points = zeros((point_X_arr.shape[0], 3), dtype='float_')
            mv_points[:, :self.n_dims] = point_X_arr
            return mv_points
        else:
            return point_X_arr

    #-----------------------------------------------------------------
    # Visualization-related methods
    #-----------------------------------------------------------------

    mvp_point_grid = Trait(MVStructuredGrid)

    def _mvp_point_grid_default(self):
        return MVStructuredGrid(name='Point grid',
                                dims=self._get_mvpoints_grid_shape,
                                points=self._get_mvpoints)

    refresh_button = Button('Draw')

    @on_trait_change('refresh_button')
    def redraw(self):
        '''Redraw the point grid.
        '''
        self.mvp_point_grid.redraw()

    cell_array = Button('Browse cell array')

    def _cell_array_fired(self):
        cell_array = self.cell_node_map
        self.show_array = CellArray(data=cell_array,
                                    cell_view=CellView(cell_grid=self))
        self.show_array.configure_traits(kind='live')

    #------------------------------------------------------------------
    # UI - related methods
    #------------------------------------------------------------------
    traits_view = View(Item('grid_cell_spec'),
                       Item('shape@'),
                       Item('coord_min'),
                       Item('coord_max'),
                       Item('refresh_button'),
                       Item('cell_array'),
                       resizable=True,
                       scrollable=True,
                       height=0.5,
                       width=0.5)


if __name__ == '__main__':

    from numpy import sin

    mgd = CellGrid(shape=(2, 1),
                   geo_transform=lambda p: sin(p),
                   grid_cell_spec=CellSpec(node_coords=[[-1, -1],
                                                        [1, -1],
                                                        [0, 0],
                                                        [1, 1],
                                                        [-1, 1]]))

    print('--------- Point index grid specification -------')

    print('point grid shape')
    print(mgd._get_point_grid_shape())
    print('point grid size')
    print(mgd.point_grid_size)
    print('point idx grid')
    print(mgd.point_idx_grid)

    print('--------- Point coordinates --------------')

    print('point x grid')
    print(mgd.point_x_grid)
    print('point X grid')
    print(mgd.point_X_grid)
    print('point x arr')
    print(mgd.point_x_arr)
    print('point X arr')
    print(mgd.point_X_arr)

    print('--------- Vertex index grid specification --------------')

    print('vertex nodes')
    print(mgd.vertex_idx_arr)
    print('vertex_idx_grid')
    print(mgd.vertex_idx_grid)

    print('--------- Vertex coordinates --------------')

    print('vertex_x_grid')
    print(mgd.vertex_x_grid)
    print('vertex_X_grid')
    print(mgd.vertex_X_grid)
    print('vertex_x_arr')
    print(mgd.vertex_x_arr)
    print('vertex_X_arr')
    print(mgd.vertex_X_arr)

    print('--------- Cell node mapping   --------------')

    print('cell node map')
    print(mgd.cell_node_map)

    print('--------- Index transformations ------------')

    print('cell offset for ( 1, 0  )')
    offset = mgd.get_cell_offset((1, 0))
    print(offset)

    print('cell idx - should be ( 1, 0 )')
    idx = mgd.get_cell_idx(offset)
    print(idx)

    print('--------- Vertex idx array -----------------')

    print('get the geometry for visualization')
    print(mgd._get_mvpoints())
    print(mgd._get_mvpoints_grid_shape())

    print('base nodes')
    print(mgd.base_nodes)
