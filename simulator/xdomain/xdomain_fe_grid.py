
from traits.api import \
    Property, cached_property, \
    provides, Callable, \
    Tuple, Int, Type, Array, Float, Instance

from ibvpy.fets.i_fets_eval import IFETSEval
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
import numpy as np
from simulator.i_xdomain import IXDomain
from view.ui.bmcs_tree_node import BMCSTreeNode


@provides(IXDomain)
class XDomainFEGrid(BMCSTreeNode):

    #=========================================================================
    # Type and shape specification of state variables representing the domain
    #=========================================================================
    U_var_shape = Property(Int)

    vtk_expand_operator = Property

    def _get_vtk_expand_operator(self):
        return self.fets.vtk_expand_operator

    K_type = Type(SysMtxArray)

    def _get_U_var_shape(self):
        return self.mesh.n_dofs

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

    D1_abcd = Array(np.float_, input=True)
    '''Symmetric operator distributing the 
    derivatives of the shape functions into the 
    tensor field
    '''

    def _D1_abcd_default(self):
        delta = np.identity(self.dim_u)
        # symmetrization operator
        D1_abcd = 0.5 * (
            np.einsum('ac,bd->abcd', delta, delta) +
            np.einsum('ad,bc->abcd', delta, delta)
        )
        return D1_abcd

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

    #=========================================================================
    # Differential operators on FE approximation of the field variables
    #=========================================================================
    cached_grid_values = Property(Tuple,
                                  depends_on='MESH,GEO,CS,FE')

    @cached_property
    def _get_cached_grid_values(self):
        x_Ia = self.mesh.X_Id
        n_I, n_a = x_Ia.shape
        dof_Ia = np.arange(n_I * n_a, dtype=np.int_).reshape(n_I, -1)
        I_Ei = self.mesh.I_Ei
        x_Eia = x_Ia[I_Ei, :]
        dof_Eia = dof_Ia[I_Ei]
        J_Emar = np.einsum(
            'imr,Eia->Emar', self.fets.dN_imr, x_Eia
        )
        J_Enar = np.einsum(
            'inr,Eia->Enar', self.fets.dN_inr, x_Eia
        )
        det_J_Em = np.linalg.det(J_Emar)
        return (dof_Eia, x_Eia, dof_Ia, I_Ei,
                det_J_Em, J_Emar, J_Enar)

    B1_Einabc = Property()
    '''Kinematic mapping between displacement and strain in every
    visualization point
    '''

    @cached_property
    def _get_B1_Einabc(self):
        inv_J_Enar = np.linalg.inv(self.J_Enar)
        return np.einsum(
            'abcd,imr,Eidr->Eimabc',
            self.D1_abcd, self.fets.dN_inr, inv_J_Enar
        )

    dof_Eia = Property()
    '''Mapping [element, node, direction] -> degree of freedom.
    '''

    def _get_dof_Eia(self):
        return self.cached_grid_values[0]

    x_Eia = Property()
    '''Mapping [element, node, direction] -> value of coordinate.
    '''

    def _get_x_Eia(self):
        return self.cached_grid_values[1]

    dof_Ia = Property()
    '''[global node, direction] -> degree of freedom
    '''

    def _get_dof_Ia(self):
        return self.cached_grid_values[2]

    I_Ei = Property()
    '''[element, node] -> global node
    '''

    def _get_I_Ei(self):
        return self.cached_grid_values[3]

    det_J_Em = Property()
    '''Jacobi matrix in integration points
    '''

    def _get_det_J_Em(self):
        return self.cached_grid_values[4]

    J_Emar = Property()
    '''Jacobi matrix in integration points
    '''

    def _get_J_Emar(self):
        return self.cached_grid_values[5]

    J_Enar = Property()
    '''Jacobi matrix in nodal points
    '''

    def _get_J_Enar(self):
        return self.cached_grid_values[6]

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
            self.D1_abcd, self.fets.dN_imr, inv_J_Emar
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
            'Eimabc,Ejmefd, Em, m->Emicjdabef',
            self.B_Eimabc, self.B_Eimabc, self.det_J_Em, self.fets.w_m
        )

    def map_U_to_field(self, U):
        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]
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
        dof_E = self.dof_Eia.reshape(-1, n_i * n_c)
        F_int = np.bincount(dof_E.flatten(), weights=f_Ei.flatten())
        return F_int

    def map_field_to_K(self, D_Emabef):
        K_Eicjd = self.integ_factor * np.einsum(
            'Emicjdabef,Emabef->Eicjd', self.BB_Emicjdabef, D_Emabef
        )
        _, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_Eij = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_Ei = self.dof_Eia.reshape(-1, n_i * n_c)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=dof_Ei)


if __name__ == '__main__':
    from ibvpy.fets.fets1D5.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
    xd = XDomainFEGrid(coord_max=(1,), shape=(1,),
                       dim_u=2, fets=FETS1D52ULRHFatigue())
    print(xd.BB_Emicjdabef.shape)
