
from ibvpy.core.sdomain import \
    SDomain
from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid
from mathkit.level_set.level_set import ILevelSetFn, SinLSF
from numpy import \
    array, unique, min, max, mgrid, ogrid, c_, alltrue, repeat, ix_, \
    arange, ones, zeros, multiply, sort, index_exp, frompyfunc, where
from traits.api import \
    Instance, Array, Int, on_trait_change, Property, cached_property, \
    List, Button, HasTraits, WeakRef, Float, Delegate, \
    Callable, Enum, Trait
from traitsui.api import View, Item, HSplit, Group, TabularEditor

from etsproxy.traits.ui.tabular_adapter import TabularAdapter

from .cell_grid.cell_array import ICellView, CellView, CellArray, ICellArraySource
from .cell_grid.cell_grid import CellGrid
from .cell_grid.cell_spec import CellSpec
from .cell_grid.dof_grid import DofCellGrid, DofCellView
from .cell_grid.geo_grid import GeoCellGrid, GeoCellView
from .fe_grid import FEGrid, MElem


class FELevelSetDomain(SDomain):
    '''
    '''
    source_domain = Instance(FEGrid)
    ls_value = Enum('pos', 'neg', 'trans')
    ls_function = Callable

    #-----------------------------------------------------------------
    # Level set interaction methods
    #-----------------------------------------------------------------

    level_set_grid = Property(Array, depends_on='a,b,cell_grid,coord_max')

    def _get_level_set_grid(self):
        X, Y = self.source_domain.geo_grid.cell_grid.point_grid
        vect_fn = frompyfunc(self.ls_function, 2, 1)
        values = vect_fn(X, Y)
        return array(values, dtype='float_')

    def _get_transiting_edges(self):
        ls = self.level_set_grid
        x_edges = where(ls[:-1, :] * ls[1:, :] <= 0)
        y_edges = where(ls[:, :-1] * ls[:, 1:] <= 0)

        ii, jj = x_edges
        # Get element numbers for each dimension separately
        # for each entry in x_edges 
        e_idx = []
        shape = self.source_domain.geo_grid.cell_grid.shape
        for i, j in zip(ii, jj):
            if j < shape[1]:
                e_idx.append([i, j])
            if j > 0:
                e_idx.append([i, j - 1])

        ii, jj = y_edges
        for i, j in zip(ii, jj):
            if i < shape[0]:
                e_idx.append([i, j])
            if i > 0:
                e_idx.append([i - 1, j])

        if e_idx == []:
            return e_idx
        else:
            e_exp = array(e_idx, dtype=int).transpose()
            return (e_exp[0, :], e_exp[1, :])

    elem_intersection = Property(Array)

    @cached_property
    def _get_elem_intersection(self):
        e_idx = self._get_transiting_edges()
        return unique(self.source_domain.geo_grid.get_elem_grid()[ e_idx ])

    #--------------------------------------------------------------------------------
    # Visualiyation of level sets
    #--------------------------------------------------------------------------------

    def get_mvscalars(self):
        return self.level_set_grid.swapaxes(0, self.source_domain.geo_grid.cell_grid.n_dims - 1).flatten()

    def _get_ielem_points(self):
        icells = self.elem_intersection
        mvpoints = []
        for cell_idx in icells:
            mvpoints += list(self.source_domain.geo_grid.get_cell_mvpoints(cell_idx))
        return array(mvpoints, dtype='float_')

    def _get_ielem_polys(self):
        ncells = len(self.elem_intersection)
        return arange(ncells * 4).reshape(ncells, 4)

    #-----------------------------------------------------------------
    # Visualization of level sets related methods
    #-----------------------------------------------------------------

    mvp_intersect_elems = Trait(MVPolyData)

    def _mvp_intersect_elems_default(self):
        return MVPolyData(name='Intersected elements',
                                  points=self._get_ielem_points,
                                  polys=self._get_ielem_polys)

    refresh_button = Button('Draw Levelset')

    @on_trait_change('refresh_button')
    def redraw(self):
        '''Redraw the point grid.
        '''
        self.mvp_intersect_elems.redraw()

    #-----------------------------------------------------------------
    # DofMap related stuff
    #-----------------------------------------------------------------

    n_nodal_dofs = Delegate('source_domain')

    # @todo - change to property so that the extension elements are counted as well.
    #
    n_dofs = Delegate('source_domain')

    elem_dof_map = Property(Array,
                             depends_on=' ls_function,source_domain.+')

    def _get_elem_dof_map(self):
        '''Get the dof map for the elements involved in the level set.
        '''
        elem_dof_map = self.source_domain.elem_dof_map[self.elem_intersection]

    elements = Property(List, depends_on=\
                       'ls_function,source_domain.+')

    @cached_property
    def _get_elements(self):
        # # - need a separate enumeration of nodes - define separate n_extension_dofs
        return [MElem(dofs=dofs, point_X_arr=point_X_arr, point_x_arr=point_x_arr)
                for dofs, point_X_arr, point_x_arr
                in zip(self.source_domain.elem_dof_map[self.elem_intersection],
                        self.source_domain.elem_X_map[self.elem_intersection],
                        self.source_domain.elem_X_map[self.elem_intersection]
                        ) ]

    shape = Property

    def _get_shape(self):
        return len(self.elements)

    #------------------------------------------------------------------
    # UI - related methods
    #------------------------------------------------------------------
    traits_view = View(Item('source_domain@', show_label=False),
                       Item('refresh_button', show_label=False),
                       Item('ls_value', show_label=False),
                       resizable=True,
                       scrollable=True,
                       height=0.5,
                       width=0.5)


if __name__ == '__main__':

    fe_domain = FEGrid(coord_min=(-2., -2., 0),
                        coord_max=(2., 2., 0),
                        shape=(10, 10),
                        geo_r=[[-1, -1],
                                        [-1, 1],
                                        [ 1, 1],
                                        [ 1, -1]],
                          dof_r=[[-1, -1],
                                        [-1, 1],
                                        [ 1, 1],
                                        [ 1, -1]])

    ls_domain = FELevelSetDomain(source_domain=fe_domain,
                                  ls_function=lambda x, y: x ** 2 + y ** 2 - 1.)

    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp(ibv_resource=ls_domain)
    ibvpy_app.main()
