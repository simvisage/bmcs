
from traits.api import \
    HasTraits, List, Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, \
    Int, Float, DelegatesTo, provides, WeakRef, Bool
from traitsui.api import \
    TabularEditor
from traitsui.api import \
    View, Item, Group
from traitsui.tabular_adapter import \
    TabularAdapter

from ibvpy.core.sdomain import \
    SDomain
from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels
import numpy as np

from .cell_array import CellView, CellArray, ICellArraySource
from .cell_grid import CellGrid
from .cell_grid_slice import CellGridSlice


#--------------------------------------------------------------------------
# DofGrid
#--------------------------------------------------------------------------
@provides(ICellArraySource)
class DofCellGrid(SDomain):

    '''
    Get an array with element Dof numbers
    '''
    cell_grid = Instance(CellGrid)

    get_cell_point_X_arr = DelegatesTo('cell_grid')
    get_cell_mvpoints = DelegatesTo('cell_grid')
    cell_node_map = DelegatesTo('cell_grid')
    get_cell_offset = DelegatesTo('cell_grid')

    # offset of dof within domain list
    #
    dof_offset = Int(0)

    # number of degrees of freedom in a single node
    #
    n_nodal_dofs = Int(3)
    #-------------------------------------------------------------------------
    # Generation methods for geometry and index maps
    #-------------------------------------------------------------------------
    n_dofs = Property(depends_on='cell_grid.shape,n_nodal_dofs,dof_offset')

    def _get_n_dofs(self):
        '''
        Get the total number of DOFs
        '''
        unique_cell_nodes = np.unique(self.cell_node_map.flatten())
        n_unique_nodes = len(unique_cell_nodes)
        return n_unique_nodes * self.n_nodal_dofs

    dofs = Property(depends_on='cell_grid.shape,n_nodal_dofs,dof_offset')

    @cached_property
    def _get_dofs(self):
        '''
        Construct the point grid underlying the mesh grid structure.
        '''
        cell_node_map = self.cell_node_map

        unique_cell_nodes = np.unique(cell_node_map.flatten())
        n_unique_nodes = len(unique_cell_nodes)

        n_nodal_dofs = self.n_nodal_dofs
        n_nodes = self.cell_grid.point_grid_size
        node_dof_array = np.repeat(-1, n_nodes *
                                   n_nodal_dofs).reshape(n_nodes, n_nodal_dofs)

        # Enumerate the DOFs in the mesh. The result is an array with n_nodes rows
        # and n_nodal_dofs columns
        #
        # A = array( [[ 0, 1 ],
        #             [ 2, 3 ],
        #             [ 4, 5 ]] );
        #
        node_dof_array[np.index_exp[unique_cell_nodes]] = \
            np.arange(
                n_unique_nodes * n_nodal_dofs).reshape(n_unique_nodes,
                                                       n_nodal_dofs)

        # add the dof_offset before returning the array
        #
        node_dof_array += self.dof_offset
        return node_dof_array

    dofs_Ia = Property()

    def _get_dofs_Ia(self):
        return self.dofs

    def _get_doffed_nodes(self):
        '''
        Get the indices of nodes containing DOFs. 
        '''
        cell_node_map = self.cell_node_map

        unique_cell_nodes = np.unique(cell_node_map.flatten())

        n_nodes = self.cell_grid.point_grid_size
        doffed_nodes = np.repeat(-1, n_nodes)

        doffed_nodes[np.index_exp[unique_cell_nodes]] = 1
        return np.where(doffed_nodes > 0)[0]

    #-----------------------------------------------------------------
    # Elementwise-representation of dofs
    #-----------------------------------------------------------------

    cell_dof_map = Property(depends_on='cell_grid.shape,n_nodal_dofs')

    def _get_cell_dof_map(self):
        return self.dofs[np.index_exp[self.cell_grid.cell_node_map]]

    dof_Eid = Property
    '''Mapping of Element, Node, Dimension -> DOF 
    '''

    def _get_dof_Eid(self):
        return self.cell_dof_map

    cell_grid_dof_map = Property(depends_on='cell_grid.shape,n_nodal_dofs')

    def _get_cell_grid_dof_map(self):
        return self.dofs[np.index_exp[self.cell_grid.cell_grid_node_map]]

    def get_cell_dofs(self, cell_idx):
        return self.cell_dof_map[cell_idx]

    elem_dof_map = Property(depends_on='cell_grid.shape,n_nodal_dofs')

    @cached_property
    def _get_elem_dof_map(self):
        el_dof_map = np.copy(self.cell_dof_map)
        tot_shape = el_dof_map.shape[0]
        n_entries = el_dof_map.shape[1] * el_dof_map.shape[2]
        elem_dof_map = el_dof_map.reshape(tot_shape, n_entries)
        return elem_dof_map

    def __getitem__(self, idx):
        '''High level access and slicing to the cells within the grid.

        The return value is a tuple with 
        1. array of cell indices
        2. array of nodes for each element
        3. array of coordinates for each node.
        '''
        dgs = DofGridSlice(dof_grid=self, grid_slice=idx)
        return dgs

    #-----------------------------------------------------------------
    # Spatial queries for dofs
    #-----------------------------------------------------------------

    def _get_dofs_for_nodes(self, nodes):
        '''Get the dof numbers and associated coordinates
        given the array of nodes.
        '''
        doffed_nodes = self._get_doffed_nodes()
#         print 'nodes'
#         print nodes
#         print 'doffed_nodes'
#         print doffed_nodes
        intersect_nodes = np.intersect1d(
            nodes, doffed_nodes, assume_unique=False)
        return (self.dofs[np.index_exp[intersect_nodes]],
                self.cell_grid.point_X_arr[np.index_exp[intersect_nodes]])

    def get_boundary_dofs(self):
        '''Get the boundary dofs and the associated coordinates
        '''
        nodes = [self.cell_grid.point_idx_grid[s]
                 for s in self.cell_grid.boundary_slices]
        dofs, coords = [], []
        for n in nodes:
            d, c = self._get_dofs_for_nodes(n)
            dofs.append(d)
            coords.append(c)
        return (np.vstack(dofs), np.vstack(coords))

    def get_all_dofs(self):
        nodes = self.cell_grid.point_idx_grid[...]
        return self._get_dofs_for_nodes(nodes)

    def get_left_dofs(self):
        nodes = self.cell_grid.point_idx_grid[0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_right_dofs(self):
        nodes = self.cell_grid.point_idx_grid[-1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_top_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, -1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, 0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_front_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, :, -1]
        return self._get_dofs_for_nodes(nodes)

    def get_back_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, :, 0]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_left_dofs(self):
        nodes = self.cell_grid.point_idx_grid[0, 0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_front_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, 0, -1]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_back_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, 0, 0]
        return self._get_dofs_for_nodes(nodes)

    def get_top_left_dofs(self):
        nodes = self.cell_grid.point_idx_grid[0, -1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_right_dofs(self):
        nodes = self.cell_grid.point_idx_grid[-1, 0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_top_right_dofs(self):
        nodes = self.cell_grid.point_idx_grid[-1, -1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[0] % 2 == 1:
            slice_middle_x = self.cell_grid.point_idx_grid.shape[0] / 2
            nodes = self.cell_grid.point_idx_grid[slice_middle_x, 0, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_bottom_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in x-direction')

    def get_top_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[0] % 2 == 1:
            slice_middle_x = self.cell_grid.point_idx_grid.shape[0] / 2
            nodes = self.cell_grid.point_idx_grid[slice_middle_x, -1, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_top_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in x-direction')

    def get_left_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[1] % 2 == 1:
            slice_middle_y = self.cell_grid.point_idx_grid.shape[1] / 2
            nodes = self.cell_grid.point_idx_grid[0, slice_middle_y, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_left_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in y-direction')

    def get_right_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[1] % 2 == 1:
            slice_middle_y = self.cell_grid.point_idx_grid.shape[1] / 2
            nodes = self.cell_grid.point_idx_grid[-1, slice_middle_y, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_right_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in y-direction')

    def get_left_front_bottom_dof(self):
        nodes = self.cell_grid.point_idx_grid[0, 0, -1]
        return self._get_dofs_for_nodes(nodes)

    def get_left_front_middle_dof(self):
        if self.cell_grid.point_idx_grid.shape[1] % 2 == 1:
            slice_middle_y = self.cell_grid.point_idx_grid.shape[1] / 2
            nodes = self.cell_grid.point_idx_grid[0, slice_middle_y, -1]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_left_middle_front_dof:'
                  ' the method is only defined for an odd number of dofs in y-direction')

    #-----------------------------------------------------------------
    # Visualization related methods
    #-----------------------------------------------------------------

    refresh_button = Button('Draw')

    @on_trait_change('refresh_button')
    def redraw(self):
        '''Redraw the point grid.
        '''
        self.cell_grid.redraw()

    dof_cell_array = Button

    def _dof_cell_array_fired(self):
        cell_array = self.cell_grid.cell_node_map
        self.show_array = CellArray(data=cell_array,
                                    cell_view=DofCellView(cell_grid=self))
        self.show_array.current_row = 0
        self.show_array.configure_traits(kind='live')
    #------------------------------------------------------------------
    # UI - related methods
    #------------------------------------------------------------------
    traits_view = View(Item('n_nodal_dofs'),
                       Item('dof_offset'),
                       Item('cell_grid@', show_label=False),
                       Item('refresh_button', show_label=False),
                       Item('dof_cell_array', show_label=False),
                       resizable=True,
                       scrollable=True,
                       height=0.5,
                       width=0.5)


class DofGridSlice(CellGridSlice):

    dof_grid = WeakRef(DofCellGrid)

    def __init__(self, dof_grid, **args):
        self.dof_grid = dof_grid
        super(DofGridSlice, self).__init__(**args)

    cell_grid = Property()

    def _get_cell_grid(self):
        return self.dof_grid.cell_grid

    dofs = Property

    def _get_dofs(self):
        _, idx2 = self.idx_tuple
        return self.dof_grid.cell_dof_map[
            np.ix_(
                self.elems,
                self.cell_grid.grid_cell[idx2]
            )
        ]

#-----------------------------------------------------------------------
# View a single cell instance
#-----------------------------------------------------------------------

#-- Tabular Adapter Definition -------------------------------------------


class DofTabularAdapter (TabularAdapter):

    columns = Property

    def _get_columns(self):
        data = getattr(self.object, self.name)
        if len(data.shape) > 2:
            raise ValueError('point array must be 1-2-3-dimensional')

        n_columns = 0
        if len(data.shape) == 2:
            n_columns = data.shape[1]

        cols = [(str(i), i) for i in range(n_columns)]
        return [('node', 'index')] + cols

#    columns = Property
#    def _get_columns(self):
#        return [('node', 'index'),
#                ('x', 0),
#                ('y', 1),
#                ('z', 2) ]
#
    font = 'Courier 10'
    alignment = 'right'
    format = '%d'
    index_text = Property

    def _get_index_text(self):
        return str(self.row)

#-- Tabular Editor Definition --------------------------------------------


dof_tabular_editor = TabularEditor(
    adapter=DofTabularAdapter(),
)


class DofCellView(CellView):

    '''View a single cell instance.
    '''
    # implements(ICellView)

    elem_dofs = Array

    def set_cell_traits(self):
        '''Set the trait values for the current cell_idx
        '''
        self.elem_dofs = self.cell_grid.get_cell_dofs(self.cell_idx)

    #---------------------------------------------------------------------
    # Visualize
    #---------------------------------------------------------------------
    draw_cell = Bool(False)

    view = View(
        Item('cell_idx', style='readonly',
             resizable=False, label='Cell index'),
        Group(Item('elem_dofs',
                   editor=dof_tabular_editor,
                   show_label=False,
                   resizable=True,
                   style='readonly')),
        Item('draw_cell', label='show DOFs')
    )

    # register the pipelines for plotting labels and geometry
    #
    mvp_elem_labels = Trait(MVPointLabels)

    def _mvp_elem_labels_default(self):
        return MVPointLabels(name='Geo node numbers',
                             points=self._get_cell_mvpoints,
                             vectors=self._get_cell_labels,
                             color=(0.0, 0.411765, 0.882353))

    mvp_elem_geo = Trait(MVPolyData)

    def _mvp_elem_geo_default(self):
        return MVPolyData(name='Geo node numbers',
                          points=self._get_elem_points,
                          lines=self._get_elem_lines,
                          color=(0.254902, 0.411765, 0.882353))

    def _get_cell_mvpoints(self):
        return self.cell_grid.get_cell_mvpoints(self.cell_idx)

    def _get_cell_labels(self):
        cell_dofs = self.cell_grid.get_cell_dofs(self.cell_idx)
        shape = cell_dofs.shape
        if shape[1] < 3:
            cd = np.zeros((shape[0], 3))
            cd[:, :shape[1]] = cell_dofs
            return cd
        else:
            return cell_dofs

    def redraw(self):
        if self.draw_cell:
            self.mvp_elem_labels.redraw(label_mode='label_vectors')


if __name__ == '__main__':

    from .cell_spec import CellSpec

#    cell_grid = CellGrid( shape = (1,1),
#                    grid_cell_spec = CellSpec( node_coords = [[-1,-1],[1,-1],[1,1],[-1,1]] ) )
#    dof_grid = DofCellGrid( cell_grid = cell_grid )
#
#    print 'dofs'
#    print dof_grid.dofs
#    print 'idx_grid'
#    print dof_grid.cell_grid.idx_grid

    dof_grid = DofCellGrid(cell_grid=CellGrid(shape=(1, 1, 1)),
                           dof_offset=1000)
    print('idx_grid')
    print(dof_grid.cell_grid.point_idx_grid)
    print('base node array')
    print(dof_grid.cell_grid.base_nodes)
    print('left')
    print(dof_grid.get_left_dofs())
    print('right')
    print(dof_grid.get_right_dofs())
    print('bottom')
    print(dof_grid.get_bottom_dofs())
    print('top')
    print(dof_grid.get_top_dofs())
    print('back')
    print(dof_grid.get_back_dofs())
    print('front')
    print(dof_grid.get_front_dofs())

    print('boundary')
    print(dof_grid.get_boundary_dofs())

    cell_grid = CellGrid(grid_cell_spec=CellSpec(node_coords=[[-1, -1],
                                                              [1, -1],
                                                              [0, 0],
                                                              [1, 1],
                                                              [-1, 1]]),
                         shape=(2, 3))

    cell_grid = CellGrid(grid_cell_spec=CellSpec(node_coords=[[-1, -1], [1, -1], [0, 0], [1, 1], [-1, 1]]),
                         coord_max=(2., 3.),
                         shape=(2, 3))

    dof_grid = DofCellGrid(cell_grid=cell_grid,
                           n_nodal_dofs=2,
                           dof_offset=2000)

    print('node_grid_shape')
    print(dof_grid.cell_grid.cell_idx_grid_shape)
    print('node_grid')
    print(dof_grid.cell_grid.cell_idx_grid)
#    print 'cell_grid_right_elem_dof_map'
    print(dof_grid.elem_dof_map[dof_grid.cell_grid.cell_idx_grid[-1, :]])
    print('elem_dof_map')
    print(dof_grid.elem_dof_map)
    print('cell_dof_map')
    print(dof_grid.cell_dof_map[0])
    print('idx_grid')
    print(dof_grid.cell_grid.point_idx_grid)
    print('base node array')
    print(dof_grid.cell_grid.base_nodes)

    print('x_max dofs')
    print(dof_grid[:, -1, :, -1].elems)

    print(dof_grid[:, -1, :, -1].dofs)

#    print 'all'
#    print dof_grid[...]
#    print 'left'
#    print dof_grid.get_left_dofs()
#    print 'right'
#    print dof_grid.get_right_dofs()
#    print 'bottom'
#    print dof_grid.get_bottom_dofs()
#    print 'top'
#    print dof_grid.get_top_dofs()
#
#    print 'boundary'
#    print dof_grid.get_boundary_dofs()

#
#    from ibvpy.plugins.ibvpy_app import IBVPyApp
#    ibvpy_app = IBVPyApp( ibv_resource = dof_grid )
#    ibvpy_app.main()
#
