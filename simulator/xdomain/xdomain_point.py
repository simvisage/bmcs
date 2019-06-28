
from traits.api import \
    provides

from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
from mathkit.tensor import EPS, DELTA
import numpy as np
from simulator.i_xdomain import IXDomain
from view.ui import BMCSLeafNode


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
