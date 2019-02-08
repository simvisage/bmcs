
from traits.api import \
    Property, cached_property, \
    provides, \
    Array

from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
import numpy as np
from simulator.api import IXDomain

from .xdomain_fe_grid import XDomainFEGrid


@provides(IXDomain)
class XDomainFEInterface(XDomainFEGrid):

    vtk_expand_operator = Array(np.float_)

    def _vtk_expand_operator_default(self):
        return np.identity(3)

    D0_abc = Array(np.float_)

    def _D0_abc_default(self):
        D3D_33 = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]], np.float_)
        D2D_11 = np.array([[0, 0],
                           [0, 1]], np.float_)
        return np.einsum('ab,cc->abc', D3D_33, D2D_11)

    D1_abcd = Array(np.float)

    def _D1_abcd_default(self):
        delta = np.vstack([np.identity(2), np.zeros((1, 2), dtype=np.float_)])
        return 0.5 * (
            np.einsum('ac,bd->abcd', delta, delta) +
            np.einsum('ad,bc->abcd', delta, delta)
        )

    N_Eimabc = Property(depends_on='+input')

    @cached_property
    def _get_N_Eimabc(self):
        x_Eia = self.x_Eia
        N_Eimabc = np.einsum(
            'abc,im->Eimabc',
            self.D0_abc, self.fets.N_im
        )
        return N_Eimabc

    NN_Emicjdabef = Property(depends_on='+input')

    @cached_property
    def _get_NN_Emicjdabef(self):
        NN_Emicjdabef = np.einsum(
            'Eimabc,Ejmefd, Em, m->Emicjdabef',
            self.N_Eimabc, self.N_Eimabc, self.det_J_Em, self.fets.w_m
        )
        return NN_Emicjdabef

    def map_U_to_field(self, U):
        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]

        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab',
            self.N_Eimabc, U_Eia
        )
        return eps_Emab

    def map_field_to_F(self, sig_Emab):
        _, n_i, _, _, _, n_c = self.B_Eimabc.shape
        f_Eic = self.integ_factor * np.einsum(
            'm,Eimabc,Emab,Em->Eic',
            self.fets.w_m, self.N_Eimabc, sig_Emab, self.det_J_Em
        )
        f_Ei = f_Eic.reshape(-1, n_i * n_c)
        dof_E = self.dof_Eia.reshape(-1, n_i * n_c)
        F_int = np.bincount(dof_E.flatten(), weights=f_Ei.flatten())
        return F_int

    def map_field_to_K(self, D_Emabef):
        K_Eicjd = self.integ_factor * np.einsum(
            'Emicjdabef,Emabef->Eicjd',
            self.NN_Emicjdabef, D_Emabef
        )
        _, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_Eij = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_Ei = self.dof_Eia.reshape(-1, n_i * n_c)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=dof_Ei)
