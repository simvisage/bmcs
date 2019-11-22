
from math import sin

from numpy import \
    array, unique, min, max, mgrid, ogrid, c_, alltrue, repeat, ix_, \
    arange, ones, zeros, multiply, sort, index_exp, frompyfunc, where, \
    zeros_like, sign, sometrue, delete, ma
from traits.api import \
    HasTraits, List, Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, Tuple, \
    Int, Float, provides, WeakRef, Bool, Any, Interface, \
    DelegatesTo, Bool, Callable
from traitsui.api import \
    TabularEditor
from traitsui.api import \
    View, Item, Group
from traitsui.tabular_adapter import \
    TabularAdapter

from ibvpy.core.i_sdomain import \
    ISDomain
from ibvpy.core.sdomain import \
    SDomain
from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid
from mathkit.level_set.level_set import ILevelSetFn, SinLSF, PlaneLSF, ElipseLSF

from .cell_array import CellView, ICellView, CellArray, ICellArraySource
from .cell_grid import CellGrid
from .cell_grid_slice import CellGridSlice


#--------------------------------------------------------------------------
# GeoCellGrid
#--------------------------------------------------------------------------
@provides(ICellArraySource)
class GeoCellGrid(SDomain):

    '''
    Get an array with element node coordinates
    '''
    cell_grid = Instance(CellGrid)

    #-------------------------------------------------------------------------
    # Generation methods for geometry and index maps
    #-------------------------------------------------------------------------
    elem_X_map = Property(depends_on='cell_grid.+')

    @cached_property
    def _get_elem_X_map(self):
        iexp = index_exp[self.cell_grid.cell_node_map]
        return self.cell_grid.point_X_arr[iexp]

    elem_x_map = Property(depends_on='cell_grid.+')

    @cached_property
    def _get_elem_x_map(self):
        iexp = index_exp[self.cell_grid.cell_node_map]
        return self.cell_grid.point_x_arr[iexp]

    def __getitem__(self, idx):
        '''High level access and slicing to the cells within the grid.

        The return value is a tuple with 
        1. array of cell indices
        2. array of nodes for each element
        3. array of coordinates for each node.
        '''
        return GeoGridSlice(geo_grid=self, grid_slice=idx)

    get_cell_point_X_arr = DelegatesTo('cell_grid')
    get_cell_point_x_arr = DelegatesTo('cell_grid')
    get_cell_mvpoints = DelegatesTo('cell_grid')
    cell_node_map = DelegatesTo('cell_grid')
    point_X_grid = DelegatesTo('cell_grid')
    point_x_grid = DelegatesTo('cell_grid')
    point_X_arr = DelegatesTo('cell_grid')
    point_x_arr = DelegatesTo('cell_grid')
    n_dims = DelegatesTo('cell_grid')

    def get_cell_node_labels(self, cell_idx):
        iexp = index_exp[self.cell_grid.cell_node_map[cell_idx]]
        return self.cell_grid.points[iexp]

    #-----------------------------------------------------------------
    # Level set interaction methods
    #-----------------------------------------------------------------

    def get_ls_mask(self, ls_mask_function):
        '''Return a boolean array indicating the masked entries of the level set.
        '''
        X_pnt = self.cell_grid.vertex_X_grid
        vect_fn = frompyfunc(ls_mask_function, self.n_dims, 1)
        ls_mask = vect_fn(*X_pnt)
        return ls_mask

    def _get_transiting_edges_1d(self, ls_function, ls_mask_function=None):

        X_pnt = self.cell_grid.vertex_X_grid

        vect_fn = frompyfunc(ls_function, self.n_dims, 1)

        ls = vect_fn(*X_pnt)

        x_edges = where(ls[:-1] * ls[1:] <= 0)
        return x_edges

    def _get_transiting_edges(self, ls_function, ls_limits=None):

        X_pnt = self.cell_grid.vertex_X_grid

        vect_fn = frompyfunc(ls_function, self.n_dims, 1)
        ls = vect_fn(*X_pnt)

        x_edges = where(ls[:-1, :] * ls[1:, :] <= 0)
        y_edges = where(ls[:, :-1] * ls[:, 1:] <= 0)

        ii, jj = x_edges
        # Get element numbers for each dimension separately
        # for each entry in x_edges
        e_idx = []
        for i, j in zip(ii, jj):
            if j < self.cell_grid.shape[1]:
                e_idx.append([i, j])
            if j > 0:
                e_idx.append([i, j - 1])

        ii, jj = y_edges
        for i, j in zip(ii, jj):
            if i < self.cell_grid.shape[0]:
                e_idx.append([i, j])
            if i > 0:
                e_idx.append([i - 1, j])

        if e_idx == []:
            return e_idx
        else:
            e_exp = array(e_idx, dtype=int).transpose()
            return (e_exp[0, :], e_exp[1, :])

    def get_intersected_cells(self, ls_function, ls_mask_function=None):

        if self.n_dims == 1:
            e_idx = self._get_transiting_edges_1d(
                ls_function, ls_mask_function)
        else:
            e_idx = self._get_transiting_edges(ls_function, ls_mask_function)
        return unique(self.cell_grid.cell_idx_grid[e_idx])

    def get_negative_cells(self, ls_function):
        vect_fn = frompyfunc(ls_function, self.n_dims, 1)
        X_pnt = self.cell_grid.vertex_X_grid
        ls = vect_fn(*X_pnt)

        cutoff_slices = [slice(0, -1) for i in range(self.n_dims)]

        ls = ls[cutoff_slices]

        neg_idx = where(ls < 0)
        negative = unique(self.cell_grid.cell_idx_grid[neg_idx])

        intersected = unique(self.get_intersected_cells(ls_function))

        remaining = list(negative)
        for i in intersected:
            try:
                remaining.remove(i)
            except:
                ValueError

        remaining = array(remaining, dtype=int)
        return remaining

    #-------------------------------------------------------------------------
    # Visualization of level sets
    #-------------------------------------------------------------------------

    def get_mvscalars(self):
        return self.level_set_grid.swapaxes(0, self.cell_grid.n_dims - 1).flatten()

    def _get_ielem_points(self):
        #icells = self.get_elem_intersection()
        icells = self.elem_intersection
        mvpoints = []
        for cell_idx in icells:
            mvpoints += list(self.get_cell_mvpoints(cell_idx))
        return array(mvpoints, dtype='float_')

    def _get_ielem_polys(self):
        #ncells = len( self.get_elem_intersection() )
        ncells = len(self.elem_intersection)
        return arange(ncells * 4).reshape(ncells, 4)

    #-----------------------------------------------------------------
    # Visualization related methods
    #-----------------------------------------------------------------

    refresh_button = Button('Draw')

    @on_trait_change('refresh_button')
    def redraw(self):
        '''Redraw the point grid.
        '''
        self.cell_grid.redraw()

    geo_cell_array = Button

    def _geo_cell_array_fired(self):
        elem_array = self.cell_grid.cell_node_map
        self.show_array = CellArray(data=elem_array,
                                    cell_view=GeoCellView(cell_grid=self))
        self.show_array.current_row = 0
        self.show_array.configure_traits(kind='live')

    #-----------------------------------------------------------------
    # Visualization of level sets related methods
    #-----------------------------------------------------------------

    mvp_point_grid = Trait(MVStructuredGrid)

    def _mvp_point_grid_default(self):
        return MVStructuredGrid(name='Point grid',
                                dims=self.cell_grid._get_mvpoints_grid_shape,
                                points=self.cell_grid._get_mvpoints,
                                scalars=self.get_mvscalars)

    mvp_intersect_elems = Trait(MVPolyData)

    def _mvp_intersect_elems_default(self):
        return MVPolyData(name='Intersected elements',
                          points=self._get_ielem_points,
                          polys=self._get_ielem_polys)

    ls_refresh_button = Button('Draw Levelset')

    @on_trait_change('ls_refresh_button')
    def ls_redraw(self):
        '''Redraw the point grid.
        '''
        self.mvp_point_grid.redraw()
        self.mvp_intersect_elems.redraw()

    #------------------------------------------------------------------
    # UI - related methods
    #------------------------------------------------------------------
    traits_view = View(Item('cell_grid@', show_label=False),
                       Item('refresh_button', show_label=False),
                       Item('ls_refresh_button', show_label=False),
                       Item('geo_cell_array', show_label=False),
                       resizable=True,
                       scrollable=True,
                       height=0.5,
                       width=0.5)


class GeoGridSlice(CellGridSlice):

    geo_grid = WeakRef(GeoCellGrid)

    def __init__(self, geo_grid, **args):
        self.geo_grid = geo_grid
        super(GeoGridSlice, self).__init__(**args)

    cell_grid = Property(depends_on='geo_grid.+changed_structure')

    @cached_property
    def _get_cell_grid(self):
        return self.geo_grid.cell_grid

    point_X_arr = Property

    def _get_point_X_arr(self):
        idx1, idx2 = self.idx_tuple
        return self.geo_grid.elem_X_map[ix_(self.elems, self.cell_grid.grid_cell[idx2])]

    point_x_arr = Property

    def _get_point_x_arr(self):
        idx1, idx2 = self.idx_tuple
        return self.geo_grid.elem_x_map[ix_(self.elems, self.cell_grid.grid_cell[idx2])]

#-- Tabular Adapter Definition -------------------------------------------


class CoordTabularAdapter (TabularAdapter):

    columns = Property

    def _get_columns(self):
        data = getattr(self.object, self.name)
        if len(data.shape) > 2:
            raise ValueError('point array must be 1-2-3-dimensional')

        n_columns = 0
        array_attrib = getattr(self.object, self.name)
        if len(array_attrib.shape) == 2:
            n_columns = array_attrib.shape[1]

        cols = [(str(i), i) for i in range(n_columns)]
        return [('node', 'index')] + cols

    font = 'Courier 10'
    alignment = 'right'
    format = '%g'
    index_text = Property

    def _get_index_text(self):
        return str(self.row)

#-- Tabular Editor Definition --------------------------------------------


coord_tabular_editor = TabularEditor(
    adapter=CoordTabularAdapter(),
)

#-----------------------------------------------------------------------
# View a single cell instance
#-----------------------------------------------------------------------


class GeoCellView(CellView):

    '''View a single cell instance.
    '''
    # implements(ICellView)

    cell_X_arr = Array

    cell_x_arr = Array

    def set_cell_traits(self):
        self.cell_X_arr = self.cell_grid.get_cell_point_X_arr(self.cell_idx)
        self.cell_x_arr = self.cell_grid.get_cell_point_x_arr(self.cell_idx)

    #-----------------------------------------------------------------------
    # Visualization
    #-----------------------------------------------------------------------
    draw_cell = Bool(False)

    view = View(Item('cell_idx', style='readonly',
                     label='Cell index', resizable=False),
                Group(Item('cell_X_arr',
                           editor=coord_tabular_editor,
                           style='readonly',
                           show_label=False)),
                Item('draw_cell', label='show geometrical nodes')
                )

    # register the pipelines for plotting labels and geometry
    #
    mvp_cell_node_labels = Trait(MVPointLabels)

    def _mvp_cell_node_labels_default(self):
        return MVPointLabels(name='Geo node numbers',
                             points=self._get_cell_mvpoints,
                             scalars=self._get_cell_node_labels,
                             color=(0.254902, 0.411765, 0.882353))

    mvp_cell_geo = Trait(MVPolyData)

    def _mvp_cell_geo_default(self):
        return MVPolyData(name='Geo node numbers',
                          points=self._get_cell_points,
                          lines=self._get_cell_lines,
                          color=(0.254902, 0.411765, 0.882353))

    def redraw(self):
        if self.draw_cell:
            self.mvp_cell_node_labels.redraw()

    #-----------------------------------------------------------------------
    # Private methods
    #-----------------------------------------------------------------------
    def _get_cell_mvpoints(self):
        return self.cell_grid.get_cell_mvpoints(self.cell_idx)

    def _get_cell_node_labels(self):
        return self.cell_grid.get_cell_node_labels(self.cell_idx)

    def _get_cell_lines(self):
        return self.cell_grid.grid_cell_spec.cell_lines


if __name__ == '__main__':

    from .cell_spec import CellSpec, GridCell

    def demo_1d():

        # Get the intersected element of a one dimensional grid
        cell_grid = CellGrid(grid_cell_spec=CellSpec(node_coords=[[-1.0],
                                                                  [0.0],
                                                                  [1.0]],
                                                     ),
                             shape=(5,), coord_max=[1.])
        geo_grid = GeoCellGrid(cell_grid=cell_grid)

        def ls_function(x): return x - 0.5

        print('vertex grid')
        print(cell_grid.vertex_X_grid)
        print('cell_grid')
        print(cell_grid.cell_idx_grid)

        print('intersected')
        print(geo_grid.get_intersected_cells(ls_function))
        print('negative')
        print(geo_grid.get_negative_cells(ls_function))

    def demo_2d():

        # Get the intersected element of a one dimensional grid
        cell_grid = CellGrid(grid_cell_spec=CellSpec(node_coords=[[-1, -1],
                                                                  [-1, 0],
                                                                  [-1, 1],
                                                                  [1, -1],
                                                                  [1, 0],
                                                                  [1, 1]]
                                                     ),
                             shape=(5, 2), coord_max=[1., 1.])
        geo_grid = GeoCellGrid(cell_grid=cell_grid)

        print('vertex_X_grid', end=' ')
        print(geo_grid.cell_grid.vertex_X_grid)

        def ls_function(x, y): return x - 0.5

        def ls_mask_function(x, y): return y <= 0.75

        print('cell_grid')
        print(cell_grid.cell_idx_grid)

        print('intersected')
        print(geo_grid.get_intersected_cells(ls_function, ls_mask_function))
        print('negative')
        print(geo_grid.get_negative_cells(ls_function))

    def demo_3d():

        geo_grid = GeoCellGrid(cell_grid=CellGrid(shape=(2, 1, 1),
                                                  coord_max=[3., 3., 3.]))

        print('elem_X_map')
        print(geo_grid.elem_X_map)
        print('cell_grid_shape')
        print(geo_grid.cell_grid.cell_idx_grid.shape)
        print('index')
        print(geo_grid.cell_grid.cell_idx_grid[0, 0, 0])
        print('first element - direct access')
        print(geo_grid.elem_X_map[0])
        print('first element - mapped access')
        print(geo_grid[0, 0, 0])

        print('x_max dofs')
        print(geo_grid[:, -1, :, -1].elems)

        print('X_grid')
        print(geo_grid[:, -1, :, -1].point_x_arr)

        print('X_arr')
        print(geo_grid[:, -1, :, -1].point_X_arr)

        print('point_grid')
        print(geo_grid.point_X_grid)

    demo_1d()
    demo_2d()
    demo_3d()
