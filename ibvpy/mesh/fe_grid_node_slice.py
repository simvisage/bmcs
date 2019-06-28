
from traits.api import \
    HasStrictTraits, WeakRef, Any, Property, provides
from .i_fe_grid_slice import IFENodeSlice


class ISliceProperty(HasStrictTraits):

    fe_grid = WeakRef()

    def __getitem__(self, idx):
        return FEGridNodeSlice(fe_grid=self.fe_grid, grid_slice=idx)


@provides(IFENodeSlice)
class FEGridNodeSlice(HasStrictTraits):
    '''General implementation of a slice within the FEGrid
    '''

    fe_grid = WeakRef('ibvpy.mesh.fe_grid.FEGrid')

    grid_slice = Any

    def __repr__(self):
        return repr(self.grid_slice)

    dof_nodes = Property

    def _get_dof_nodes(self):
        return self.fe_grid.dof_grid.cell_grid.point_idx_grid[self.grid_slice]

    dofs = Property

    def _get_dofs(self):
        return self.fe_grid.dof_grid.dofs_Ia[self.dof_nodes, :]

    dof_X = Property

    def _get_dof_X(self):
        return self.fe_grid.dof_grid.cell_grid.point_X_arr[self.dof_nodes, :]

    geo_nodes = Property

    def _get_geo_nodes(self):
        return self.fe_grid.geo_grid.cell_grid.point_idx_grid[self.grid_slice]

    geo_x = Property

    def _get_geo_x(self):
        return self.fe_grid.geo_grid.cell_grid.point_x_arr[self.geo_nodes, :]

    geo_X = Property

    def _get_geo_X(self):
        return self.fe_grid.geo_grid.cell_grid.point_X_arr[self.geo_nodes, :]


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
