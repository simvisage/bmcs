
from ibvpy.mesh.i_fe_grid_slice import IFENodeSlice
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
from mathkit.tensor import EPS
from simulator.api import IXDomain
from traits.api import \
    provides, Tuple, \
    Array, Property, Instance, cached_property

import numpy as np

from .xdomain_transform import XDomainFEGridTransform


@provides(IXDomain)
class XDomainFEInterface(XDomainFEGridTransform):

    #=========================================================================
    # CRIMINAL - Change
    #=========================================================================
    state_var_shape = Property(Tuple)

    def _get_state_var_shape(self):
        n_elems = len(self.o_Epia)
        return (n_elems, self.fets.n_m,)

    vtk_expand_operator = Array(np.float_)

    def _vtk_expand_operator_default(self):
        return np.identity(3)

    DELTA_p = Array()

    def _DELTA_p_default(self):
        return np.array([-1, 1], np.float_)

    I = Instance(IFENodeSlice)
    J = Instance(IFENodeSlice)

    o_Epia = Property(depends_on='changed')

    @cached_property
    def _get_o_Epia(self):
        o_Ia = self.I.dofs
        o_Ja = self.J.dofs
        o_piEa = np.array([[o_Ia[:-1], o_Ia[1:]],
                           [o_Ja[:-1], o_Ja[1:]]])
        return np.einsum('piEa->Epia', o_piEa)

    o_Eia = Property(depends_on='changed')
    '''flatten the first two dimensions consistently woth o_pEia
    '''
    @cached_property
    def _get_o_Eia(self):
        flat_bnd = (-1,) + self.o_Epia.shape[2:]
        return self.o_Epia.reshape(flat_bnd)

    X_pEia = Property(depends_on='changed')

    @cached_property
    def _get_X_pEia(self):
        X_Ia = self.I.geo_X
        X_Ja = self.J.geo_X
        X_piEa = np.array([[X_Ia[:-1], X_Ia[1:]],
                           [X_Ja[:-1], X_Ja[1:]]])
        return np.einsum('piEa->pEia', X_piEa)

    x_Eia = Property(depends_on='changed')

    @cached_property
    def _get_x_Eia(self):
        DELTA_pp = np.identity(2)
        # Correct the FETS interface - to provide shape functions
        # at integration points.
        return np.einsum(
            'pp,mi,pEma->Eia',
            DELTA_pp, self.fets.N_mi_geo, self.X_pEia
        ) / 2.0

    det_J_Em = Property(depends_on='MESH,GEO,CS,FE')
    '''Get the determinant of the local jacobi matrix
    '''
    @cached_property
    def _get_det_J_Em(self):
        r_Eia = np.einsum(
            'Emra,Eia->Eir',
            self.T_Emra[..., :self.x_Eia.shape[-1]], self.x_Eia
        )
        J_Emar = np.einsum(
            'imr,Eia->Emar', self.fets.dN_imr, r_Eia[..., :1]
        )
        return np.linalg.det(J_Emar)

    T_Emra = Property(depends_on='MESH,GEO,CS,FE')

    @cached_property
    def _get_T_Emra(self):

        x_Eia = self.x_Eia
        dx_Emas = np.einsum(
            'ims,Eia->Emas', self.fets.dN_imr, x_Eia
        )
        DELTA_12 = np.array([[1, 0]], np.float_)
        dx2_Emar = np.einsum('sr,Emas->Emar',
                             DELTA_12, dx_Emas
                             )
        # expansion tensor
        DELTA23_ab = np.array([[1, 0, 0],
                               [0, 1, 0]], dtype=np.float_)

        dx3_Emar = np.einsum(
            'ik,...ij,jl->...kl',
            DELTA23_ab, dx2_Emar, DELTA23_ab
        )
        dx3_Emar[..., 2, 2] = 1.0

        m_0_Ema = dx3_Emar[..., 0]

        m_1_Ema = np.einsum(
            '...i,...j,ijk->...k',
            dx3_Emar[..., 2], m_0_Ema, EPS
        )
        m_2_Ema = np.einsum(
            '...i,...j,ijk->...k',
            m_0_Ema, m_1_Ema, EPS)
        M_rEma = np.array([m_0_Ema, m_1_Ema, m_2_Ema])
        M_Emra = np.einsum('rEma->Emra', M_rEma)
        norm_M_Emra = np.sqrt(
            np.einsum('...ij,...ij->...i', M_Emra, M_Emra)
        )[..., np.newaxis]

        T_Emra = M_Emra / norm_M_Emra
        return T_Emra

    B_Eimabc = Property(depends_on='MESH,GEO,CS,FE')
    '''Kinematic mapping between displacements and strains in every
    integration point.
    '''
    @cached_property
    def _get_B_Eimabc(self):
        return np.einsum(
            'p,Emra,im->Empira',
            self.DELTA_p, self.T_Emra[..., :2, :2], self.fets.N_im
        )

    def _get_BB_Emicjdabef(self):
        return np.einsum(
            '...Empira,...Emqjsb, Em, m->...Empirqjsab',
            self.B_Eimabc, self.B_Eimabc, self.det_J_Em, self.fets.w_m
        )

    n_dofs = Property

    def _get_n_dofs(self):
        return 0

    def map_U_to_field(self, U):
        U_Epia = U[self.o_Epia]
        s_Emr = np.einsum(
            'Empira,Epia->Emr',
            self.B_Eimabc, U_Epia
        )
        return s_Emr

    def map_field_to_K(self, D_Emabef):
        K_Eicjd = self.integ_factor * np.einsum(
            '...Empirqjsab,...Emrs->Epiaqjb',
            self.BB_Emicjdabef, D_Emabef
        )
        _, _, n_p, n_i, _, n_a = self.B_Eimabc.shape
        n_o = n_p * n_i * n_a
        K_Eij = K_Eicjd.reshape(-1, n_o, n_o)
        o_Ei = self.o_Epia.reshape(-1, n_o)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=o_Ei)

    def map_field_to_F(self, sig_Emab):
        _, _, n_p, n_i, _, n_a = self.B_Eimabc.shape
        f_Eic = self.integ_factor * np.einsum(
            'm,Empira,Emr,Em->Epia',
            self.fets.w_m, self.B_Eimabc, sig_Emab, self.det_J_Em
        )
        n_o = n_p * n_i * n_a
        f_Ei = f_Eic.reshape(-1, n_o)
        o_E = self.o_Eia.reshape(-1, n_o)
        return o_E.flatten(), f_Ei.flatten()
