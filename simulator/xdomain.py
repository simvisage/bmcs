
from traits.api import \
    HasStrictTraits, Instance, Property, \
    provides, Enum, Float, on_trait_change, \
    Interface, Tuple, Int, Type

from ibvpy.dots.vdots_grid import \
    DOTSGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
from mathkit.tensor import EPS, DELTA
import numpy as np
from view.ui import BMCSLeafNode

from .i_xdomain import IXDomain


DD = np.hstack([DELTA, np.zeros_like(DELTA)])
EEPS = np.hstack([np.zeros_like(EPS), EPS])
GAMMA = np.einsum(
    'ik,jk->kij', DD, DD
) + np.einsum(
    'ikj->kij', np.fabs(EEPS)
)
GAMMA_inv = np.einsum(
    'ik,jk->kij', DD, DD
) + 0.5 * np.einsum(
    'ikj->kij', np.fabs(EEPS)
)

GG = np.einsum(
    'mij,nkl->mnijkl', GAMMA_inv, GAMMA_inv
)


@provides(IXDomain)
class XDomainSinglePoint(BMCSLeafNode):

    U_var_shape = (6,)

    state_var_shape = (1,)

    def map_U_to_field(self, eps_eng):
        return np.einsum(
            'kij,...k->...ij', GAMMA, eps_eng
        )[np.newaxis, ...]

    def map_field_to_F(self, eps_tns):
        return np.einsum(
            'kij,...ij->...k', GAMMA_inv, eps_tns
        )

    def map_field_to_K(self, tns4):
        K_mij = np.einsum(
            'mnijkl,...ijkl->...mn', GG, tns4
        )
        dof_Ei = np.arange(6, dtype=np.int_)[np.newaxis, ...]
        return SysMtxArray(mtx_arr=K_mij, dof_map_arr=dof_Ei)


@provides(IXDomain)
class XDomainFEGrid(DOTSGrid):

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
        n_E, n_i, n_m, n_a, n_b, n_c = self.B_Eimabc.shape
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
        n_E, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_Eij = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_Ei = self.dof_Eia.reshape(-1, n_i * n_c)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=dof_Ei)
