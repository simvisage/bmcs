
from traits.api import \
    HasTraits, WeakRef, Tuple, Property, cached_property, Int, Float, Bool, Any


class CellGridSlice(HasTraits):
    '''General implementation of a slice within the FEGrid
    '''
    cell_grid = WeakRef()  # 'ibvpy.mesh.cell_grid.cell_grid.CellGrid' )

    grid_slice = Any

    idx_tuple = Property(
        Tuple, depends_on='grid_slice,cell_grid.+changed_structure')

    @cached_property
    def _get_idx_tuple(self):
        ''' Classify the index specification
            By default, the idx is assigned to element index
            to handle the case of a single integer index
            '''
        idx1 = self.grid_slice
        # The default for node index is the universal slice
        idx2 = slice(None, None)
        n_dims = self.cell_grid.n_dims
        # If the index is an iterable
        if hasattr(self.grid_slice, '__iter__'):
            # Get the first n_dims of indexes for the cell grid
            idx1 = self.grid_slice[:n_dims]
            # If there are more indexes than n_dims save them for
            # use within the cell to identify the nodes.
            if len(self.grid_slice) > n_dims:
                idx2 = self.grid_slice[n_dims:]
        return (idx1, idx2)

    elem_grid = Property

    def _get_elem_grid(self):
        idx1, _ = self.idx_tuple
        return self.cell_grid.cell_idx_grid[idx1]

    elems = Property

    def _get_elems(self):
        # get the cells affected by the slice @todo - rename cell_grid
        # attribute within CellGrid
        return self.elem_grid.flatten()

    # Get the node map associated with the sliced elements
    #
    nodes = Property

    def _get_nodes(self):
        # get the node map associated with the sliced elements
        idx1, idx2 = self.idx_tuple

        sliced_cell_node_map = self.cell_grid.cell_node_map[self.elems]

        # get the sliced nodes for the sliced cells
        return sliced_cell_node_map[:, self.cell_grid.grid_cell[idx2]]

    points = Property

    def _get_points(self):
        # get the coordinates for the sliced coords
        return self.cell_grid.point_X_arr[self.nodes]

    # Global coordinates of nodes involved in the slice
    # Structured element by element
    #
    point_X_arr = Property

    def _get_point_X_arr(self):
        return self.cell_grid.point_X_arr[self.nodes]

    # Parametric coordinates of nodes involved in the slice
    # Structured element by element
    #
    point_x_arr = Property

    def _get_point_x_arr(self):
        return self.cell_grid.point_x_arr[self.nodes]


if __name__ == '__main__':

    from .cell_grid import CellGrid
    from .cell_spec import CellSpec

    cell_grid = CellGrid(shape=(2, 8),
                         grid_cell_spec=CellSpec(node_coords=[[-1, -1],
                                                              [1, -1],
                                                              [0, 0],
                                                              [1, 1],
                                                              [-1, 1]]))

    print('cell_grid')
    print(cell_grid.cell_idx_grid)

    print('cell_grid points')
    print(cell_grid.point_idx_grid)

    print('grid_cell')
    print(cell_grid.grid_cell)

    print('grid_cell[:,-1]')
    print(cell_grid.grid_cell[:, -1])

    print('sliced cells')
    cell_grid_slice = cell_grid[:, -1, :, -1]

    print('elemes')
    print(cell_grid_slice.elems)

    print('nodes')
    print(cell_grid_slice.nodes)

    print('global coords')
    print(cell_grid_slice.point_X_arr)

    print('local coords')
    print(cell_grid_slice.point_x_arr)

    print('cell grid')
    print(cell_grid_slice.cell_grid)

    print('get a single element - the first one - cell grid[0,0]')
    print(cell_grid[0].elems)

    print('get all nodes of the boundary elements in second-direction')
    print(cell_grid[:, -1, ...].nodes)
