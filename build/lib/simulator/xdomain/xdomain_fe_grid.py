
from ibvpy.fets.i_fets_eval import IFETSEval
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
from simulator.i_xdomain import IXDomain
from traits.api import \
    Property, cached_property, \
    provides, Callable, \
    Tuple, Int, Type, Array, Float, Instance, Bool
from view.ui.bmcs_tree_node import BMCSTreeNode

import numpy as np


@provides(IXDomain)
class XDomainFEGrid(BMCSTreeNode):

    hidden = Bool(False)
    #=========================================================================
    # Type and shape specification of state variables representing the domain
    #=========================================================================
    U_var_shape = Property(Int)

    def _get_U_var_shape(self):
        return self.mesh.n_dofs

    vtk_expand_operator = Property

    def _get_vtk_expand_operator(self):
        return self.fets.vtk_expand_operator

    K_type = Type(SysMtxArray)

    state_var_shape = Property(Tuple)

    def _get_state_var_shape(self):
        return (self.mesh.n_active_elems, self.fets.n_m,)

    #=========================================================================
    # Input parameters
    #=========================================================================
    coord_min = Array(Float, value=[0., 0., 0.], GEO=True)
    '''Grid geometry specification - min corner point
    '''
    coord_max = Array(Float, value=[1., 1., 1.], MESH=True)
    '''Grid geometry specification - max corner point
    '''
    shape = Array(Int, value=[1, 1, 1], MESH=True)
    '''Number of elements in the individual dimensions
    '''
    geo_transform = Callable
    '''Geometry transformation
    '''
    integ_factor = Float(1.0, input=True, CS=True)
    '''Integration factor used to multiply the integral
    '''
    fets = Instance(IFETSEval, input=True, FE=True)
    '''Finite element type
    '''

    dim_u = Int(2)

    Diff1_abcd = Array(np.float_, input=True)
    '''Symmetric operator distributing the first order
    derivatives of the shape functions into the 
    tensor field
    '''

    def _Diff1_abcd_default(self):
        delta = np.identity(self.dim_u)
        # symmetrization operator
        Diff1_abcd = 0.5 * (
            np.einsum('ac,bd->abcd', delta, delta) +
            np.einsum('ad,bc->abcd', delta, delta)
        )
        return Diff1_abcd

    #=========================================================================
    # Finite element discretization respecting the FE definition
    #=========================================================================
    mesh = Property(Instance(FEGrid), depends_on='MESH,GEO')

    @cached_property
    def _get_mesh(self):
        return FEGrid(coord_min=self.coord_min,
                      coord_max=self.coord_max,
                      shape=self.shape,
                      geo_transform=self.geo_transform,
                      fets_eval=self.fets)

    x_Eia = Property(depends_on='MESH,GEO,CS,FE')

    def _get_x_Eia(self):
        x_Ia = self.mesh.X_Id
        I_Ei = self.mesh.I_Ei
        x_Eia = x_Ia[I_Ei, :]
        return x_Eia

    x_Ema = Property(depends_on='MESH,GEO,CS,FE')

    def _get_x_Ema(self):
        return np.einsum(
            'im,Eia->Ema', self.fets.N_im, self.x_Eia
        )

    o_Ia = Property(depends_on='MESH,GEO,CS,FE')

    @cached_property
    def _get_o_Ia(self):
        x_Ia = self.mesh.X_Id
        n_I, _ = x_Ia.shape
        n_a = self.mesh.n_nodal_dofs
        do = self.mesh.dof_offset
        return do + np.arange(n_I * n_a, dtype=np.int_).reshape(-1, n_a)

    o_Eia = Property(depends_on='MESH,GEO,CS,FE')

    @cached_property
    def _get_o_Eia(self):
        I_Ei = self.mesh.I_Ei
        return self.o_Ia[I_Ei]

    B1_Einabc = Property(depends_on='MESH,GEO,CS,FE')
    '''Kinematic mapping between displacement and strain in every
    visualization point
    '''

    @cached_property
    def _get_B1_Einabc(self):
        inv_J_Enar = np.linalg.inv(self.J_Enar)
        return np.einsum(
            'abcd,imr,Eidr->Eimabc',
            self.Diff1_abcd, self.fets.dN_inr, inv_J_Enar
        )

    I_Ei = Property(depends_on='MESH,GEO,CS,FE')
    '''[element, node] -> global node
    '''

    def _get_I_Ei(self):
        return self.mesh.I_Ei

    det_J_Em = Property(depends_on='MESH,GEO,CS,FE')
    '''Jacobi matrix in integration points
    '''

    def _get_det_J_Em(self):
        return np.linalg.det(self.J_Emar)

    J_Emar = Property(depends_on='MESH,GEO,CS,FE')
    '''Jacobi matrix in integration points
    '''
    @cached_property
    def _get_J_Emar(self):
        return np.einsum(
            'imr,Eia->Emar', self.fets.dN_imr, self.x_Eia
        )

    J_Enar = Property(depends_on='MESH,GEO,CS,FE')
    '''Jacobi matrix in nodal points
    '''
    @cached_property
    def _get_J_Enar(self):
        return np.einsum(
            'inr,Eia->Enar',
            self.fets.dN_inr, self.x_Eia
        )

    #=========================================================================
    # Conversion between linear algebra objects and field variables
    #=========================================================================
    B1_Eimabc = Property(depends_on='MESH,GEO,CS,FE')
    '''Kinematic mapping between displacements and strains in every
    integration point.
    '''
    @cached_property
    def _get_B1_Eimabc(self):
        inv_J_Emar = np.linalg.inv(self.J_Emar)
        return np.einsum(
            'abcd,inr,Eidr->Einabc',
            self.Diff1_abcd, self.fets.dN_imr, inv_J_Emar
        )

    B_Eimabc = Property(depends_on='MESH,GEO,CS,FE')
    '''Kinematic mapping between displacements and strains in every
    integration point.
    '''
    @cached_property
    def _get_B_Eimabc(self):
        return self.B1_Eimabc

    BB_Emicjdabef = Property(depends_on='MESH,GEO,CS,FE')
    '''Quadratic form of the kinematic mapping.
    '''

    def _get_BB_Emicjdabef(self):
        return np.einsum(
            '...Eimabc,...Ejmefd, Em, m->...Emicjdabef',
            self.B_Eimabc, self.B_Eimabc, self.det_J_Em, self.fets.w_m
        )

    n_dofs = Property

    def _get_n_dofs(self):
        return self.mesh.n_dofs

    def map_U_to_field(self, U):
        n_c = self.fets.n_nodal_dofs
        #U_Ia = U.reshape(-1, n_c)
        U_Eia = U[self.o_Eia]
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab',
            self.B_Eimabc, U_Eia
        )
        return eps_Emab

    def map_field_to_F(self, sig_Emab):
        _, n_i, _, _, _, n_c = self.B_Eimabc.shape
        f_Eic = self.integ_factor * np.einsum(
            'm,Eimabc,Emab,Em->Eic',
            self.fets.w_m, self.B_Eimabc, sig_Emab, self.det_J_Em
        )
        f_Ei = f_Eic.reshape(-1, n_i * n_c)
        o_E = self.o_Eia.reshape(-1, n_i * n_c)
        return o_E.flatten(), f_Ei.flatten()

    def map_field_to_K(self, D_Emabef):
        K_Eicjd = self.integ_factor * np.einsum(
            'Emicjdabef,Emabef->Eicjd',
            self.BB_Emicjdabef, D_Emabef
        )
        _, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_Eij = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        o_Ei = self.o_Eia.reshape(-1, n_i * n_c)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=o_Ei)

    debug_cell_data = Bool(False)
    # @todo - comment this procedure`

    def get_vtk_cell_data(self, position, point_offset, cell_offset):
        if position == 'nodes':
            subcell_offsets, subcell_lengths, subcells, subcell_types = \
                self.fets.vtk_node_cell_data
        elif position == 'int_pnts':
            subcell_offsets, subcell_lengths, subcells, subcell_types = \
                self.fets.vtk_ip_cell_data

        if self.debug_cell_data:
            print('subcell_offsets')
            print(subcell_offsets)
            print('subcell_lengths')
            print(subcell_lengths)
            print('subcells')
            print(subcells)
            print('subcell_types')
            print(subcell_types)

        n_subcells = subcell_types.shape[0]
        n_cell_points = self.n_cell_points
        subcell_size = subcells.shape[0] + n_subcells

        if self.debug_cell_data:
            print('n_cell_points', n_cell_points)
            print('n_cells', self.n_cells)

        vtk_cell_array = np.zeros((self.n_cells, subcell_size), dtype=int)

        idx_cell_pnts = np.repeat(True, subcell_size)

        if self.debug_cell_data:
            print('idx_cell_pnts')
            print(idx_cell_pnts)

        idx_cell_pnts[subcell_offsets] = False

        if self.debug_cell_data:
            print('idx_cell_pnts')
            print(idx_cell_pnts)

        idx_lengths = idx_cell_pnts == False

        if self.debug_cell_data:
            print('idx_lengths')
            print(idx_lengths)

        point_offsets = np.arange(self.n_cells) * n_cell_points
        point_offsets += point_offset

        if self.debug_cell_data:
            print('point_offsets')
            print(point_offsets)

        vtk_cell_array[:, idx_cell_pnts] = point_offsets[
            :, None] + subcells[None, :]
        vtk_cell_array[:, idx_lengths] = subcell_lengths[None, :]

        if self.debug_cell_data:
            print('vtk_cell_array')
            print(vtk_cell_array)

        n_active_cells = self.mesh.n_active_elems

        if self.debug_cell_data:
            print('n active cells')
            print(n_active_cells)

        cell_offsets = np.arange(n_active_cells, dtype=int) * subcell_size
        cell_offsets += cell_offset
        vtk_cell_offsets = cell_offsets[:, None] + subcell_offsets[None, :]

        if self.debug_cell_data:
            print('vtk_cell_offsets')
            print(vtk_cell_offsets)

        vtk_cell_types = np.zeros(
            self.n_cells * n_subcells, dtype=int
        ).reshape(self.n_cells, n_subcells)
        vtk_cell_types += subcell_types[None, :]

        if self.debug_cell_data:
            print('vtk_cell_types')
            print(vtk_cell_types)

        return (vtk_cell_array.flatten(),
                vtk_cell_offsets.flatten(),
                vtk_cell_types.flatten())

    n_cells = Property(Int)

    def _get_n_cells(self):
        '''Return the total number of cells'''
        return self.mesh.n_active_elems

    n_cell_points = Property(Int)

    def _get_n_cell_points(self):
        '''Return the number of points defining one cell'''
        return self.fets.n_vtk_r


if __name__ == '__main__':
    from ibvpy.fets.fets1D5.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
    xd = XDomainFEGrid(coord_max=(1,), shape=(1,),
                       dim_u=2, fets=FETS1D52ULRHFatigue())
    print(xd.BB_Emicjdabef.shape)
    print(xd.get_vtk_cell_data('nodes', 0, 0))
