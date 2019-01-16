from traits.api import \
    provides, HasStrictTraits, Property, Array, cached_property

from bmcs.simulator import \
    Simulator, TLoop, IXDomain
from bmcs.simulator.xdomain import \
    DD, EEPS, XDomainFEGrid
from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from mathkit.tensor import EPS, DELTA
import numpy as np

from .interaction_scripts import run_rerun_test


@provides(IXDomain)
class XDomainAxiSym(XDomainFEGrid):

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

    cached_grid_values2 = Property(depends_on='+input')

    @cached_property
    def _get_cached_grid_values2(self):
        x_Ia = self.mesh.X_Id
        I_Ei = self.mesh.I_Ei
        x_Eia = x_Ia[I_Ei, :]
        J_Emar = np.einsum(
            'imr,Eia->Emar', self.fets.dN_imr, x_Eia
        )
        det_J_Em = np.linalg.det(J_Emar)
        r_Em = np.einsum(
            'im,Eic->Emc',
            self.fets.N_im, x_Eia
        )[..., 1]
        N_Eimabc = np.einsum(
            'abc,im, Em->Eimabc',
            self.D0_abc, self.fets.N_im, 1. / r_Em
        )
        print('x_mc', r_Em)
#         B_Einabc = np.einsum(
#             'abc,in->inabc',
#             self.D0_abc, self.fets.N_in
#         )
        BB_Emicjdabef = np.einsum(
            'Eimabc,Ejmefd, Em, m->Emicjdabef',
            N_Eimabc, N_Eimabc, det_J_Em, self.fets.w_m
        )
        return (BB_Emicjdabef, N_Eimabc)  # , B_Einabc)

    def map_U_to_field2(self, U):
        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]

        BB_Emicjdabef, N_Eimabc = self.cached_grid_values2
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab',
            N_Eimabc, U_Eia
        )
        return eps_Emab


U_Eic = np.array([[[0.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 0.0],
                   [0.0, 1.0]]], dtype=np.float_)

xdomain = XDomainAxiSym(L_x=1, L_y=1,
                        integ_factor=0,
                        n_x=1, n_y=1,
                        fets=FETS2D4Q())

eps_ij = xdomain.map_U_to_field(U_Eic.flatten())
print(eps_ij)

print(xdomain.cached_grid_values2[1])
eps_ij = xdomain.map_U_to_field2(U_Eic.flatten())
print(eps_ij)
