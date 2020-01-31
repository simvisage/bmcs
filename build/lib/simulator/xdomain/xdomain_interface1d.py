
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
from simulator.i_xdomain import IXDomain
from traits.api import \
    provides, Int, \
    Array, Property, cached_property
import numpy as np
from .xdomain_fe_grid import XDomainFEGrid


@provides(IXDomain)
class XDomainFEInterface1D(XDomainFEGrid):

    A = Property()
    '''array containing the A_m, L_b, A_f
    '''

    def _get_A(self):
        return np.array([self.fets.A_m,
                         self.fets.P_b,
                         self.fets.A_f])

    DELTA_p = Array()

    def _DELTA_p_default(self):
        return np.array([-1, 1], np.float_)

    o_Epia = Property(depends_on='+input')
    '''For a given element, layer, node number and dimension
    return the dof number
    '''
    @cached_property
    def _get_o_Epia(self):
        dof_Eipd = self.mesh.dof_grid.cell_dof_map[..., np.newaxis]
        return np.einsum('Eipd->Epid', dof_Eipd)

    dim_u = Int(2)

    B_Eimabc = Property(depends_on='MESH,GEO,CS,FE')
    '''Kinematic mapping between displacements and strains in every
    integration point.
    '''
    @cached_property
    def _get_B_Eimabc(self):

        fets_eval = self.fets
        mesh = self.mesh
        n_s = 3

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_ip = fets_eval.n_gp
        n_e = mesh.n_active_elems
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        inv_J_Emar = np.linalg.inv(self.J_Emar)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None, :])
        dNr = 0.5 * geo_r[:, :, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('Eidf,inf->Eind', inv_J_Emar, dNr)

        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :, :, B_N_n_rows, B_N_n_cols] = (B_factors[None, None, :] *
                                              Nx[:, :, N_idx])
        B[:, :, :, B_dN_n_rows, B_dN_n_cols] = dNx[:, :, :, dN_idx]
        return B

    def _get_BB_Emicjdabef(self):
        return np.einsum(
            's, ...Emisd,...Emjtf, Em, m->...Emidjfst',
            self.A, self.B_Eimabc, self.B_Eimabc, self.det_J_Em, self.fets.w_m
        )

    def map_U_to_field(self, U):
        U_Eia = U[self.o_Eia]
        s_Emr = np.einsum(
            'Emisd,Eid->Ems',
            self.B_Eimabc, U_Eia
        )
        return s_Emr

    def map_field_to_K(self, D_Emabef):
        K_Eicjd = self.integ_factor * np.einsum(
            '...Emirjsab,...Emab->Eirjs',
            self.BB_Emicjdabef, D_Emabef
        )
        _, _, n_i, _, n_a = self.B_Eimabc.shape
        n_o = n_i * n_a
        K_Eij = K_Eicjd.reshape(-1, n_o, n_o)
        o_Ei = self.o_Eia.reshape(-1, n_o)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=o_Ei)

    def map_field_to_F(self, sig_Emab):
        _, _, n_i, _, n_a = self.B_Eimabc.shape
        f_Eic = self.integ_factor * np.einsum(
            's,m,Emisd,Ems,Em->Eid',
            self.A, self.fets.w_m, self.B_Eimabc, sig_Emab, self.det_J_Em
        )
        n_o = n_i * n_a
        f_Ei = f_Eic.reshape(-1, n_o)
        o_E = self.o_Eia.reshape(-1, n_o)
        return o_E.flatten(), f_Ei.flatten()
