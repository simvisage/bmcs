
from traits.api import \
    HasStrictTraits, WeakRef, Any, Property, cached_property, provides
from .i_fe_grid_slice import IFEGridSlice


@provides(IFEGridSlice)
class FEGridIdxSlice(HasStrictTraits):
    '''General implementation of a slice within the FEGrid
    '''

    fe_grid = WeakRef('ibvpy.mesh.fe_grid.FEGrid')

    grid_slice = Any

    def __repr__(self):
        return repr(self.grid_slice)

    dof_grid_slice = Property(depends_on='fe_grid.dof_grid')

    @cached_property
    def _get_dof_grid_slice(self):
        return self.fe_grid.dof_grid[self.grid_slice]

    geo_grid_slice = Property(depends_on='fe_grid.geo_grid')

    @cached_property
    def _get_geo_grid_slice(self):
        return self.fe_grid.geo_grid[self.grid_slice]

    elem_grid = Property

    def _get_elem_grid(self):
        return self.dof_grid_slice.elem_grid

    elems = Property

    def _get_elems(self):
        return self.dof_grid_slice.elems

    dof_nodes = Property

    def _get_dof_nodes(self):
        return self.dof_grid_slice.nodes

    dofs = Property

    def _get_dofs(self):
        return self.dof_grid_slice.dofs

    dof_X = Property

    def _get_dof_X(self):
        return self.dof_grid_slice.point_X_arr

    geo_nodes = Property

    def _get_geo_nodes(self):
        return self.geo_grid_slice.nodes

    geo_x = Property

    def _get_geo_x(self):
        return self.geo_grid_slice.point_x_arr

    geo_X = Property

    def _get_geo_X(self):
        return self.geo_grid_slice.point_X_arr


if __name__ == '__main__':

    from ibvpy.fets.fets2D import FETS2D4Q
    from ibvpy.mesh.fe_grid import FEGrid
    fets_sample = FETS2D4Q()

    fe_domain = FEGrid(coord_max=(2., 3., ),
                       shape=(2, 3),
                       inactive_elems=[3],
                       fets_eval=fets_sample)

    fe_slice = fe_domain[:, -1, :, -1]
    print('elems')
    print(fe_slice.elems)
    print('dof_nodes')
    print(fe_slice.dof_nodes)
    print('dofs')
    print(fe_slice.dofs)
    print('dof coords')
    print(fe_slice.dof_X)
    print('geo_nodes')
    print(fe_slice.geo_nodes)
    print('geo_X')
    print(fe_slice.geo_X)
    print('parametric points')
    print(fe_slice.geo_x)
