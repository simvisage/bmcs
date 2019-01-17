from traits.api import \
    provides, Property, Array, cached_property

from bmcs.simulator import \
    Simulator, IXDomain
from bmcs.simulator.xdomain import \
    DD, EEPS, XDomainFEGrid
from ibvpy.bcond import BCSlice
from ibvpy.fets import FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DScalarDamage
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
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

    N_Eimabc = Property(depends_on='+input')

    @cached_property
    def _get_N_Eimabc(self):
        x_Eia = self.x_Eia
        r_Em = np.einsum(
            'im,Eic->Emc',
            self.fets.N_im, x_Eia
        )[..., 1]
        N_Eimabc = np.einsum(
            'abc,im, Em->Eimabc',
            self.D0_abc, self.fets.N_im, 1. / r_Em
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
            self.B_Eimabc + self.N_Eimabc, U_Eia
        )
        return eps_Emab

    def map_field_to_F(self, sig_Emab):
        n_E, n_i, n_m, n_a, n_b, n_c = self.B_Eimabc.shape
        f_Eic = self.integ_factor * np.einsum(
            'm,Eimabc,Emab,Em->Eic',
            self.fets.w_m, self.B_Eimabc + self.N_Eimabc, sig_Emab, self.det_J_Em
        )
        f_Ei = f_Eic.reshape(-1, n_i * n_c)
        dof_E = self.dof_Eia.reshape(-1, n_i * n_c)
        F_int = np.bincount(dof_E.flatten(), weights=f_Ei.flatten())
        return F_int

    def map_field_to_K(self, D_Emabef):
        K_Eicjd = self.integ_factor * np.einsum(
            'Emicjdabef,Emabef->Eicjd',
            self.BB_Emicjdabef + self.NN_Emicjdabef, D_Emabef
        )
        n_E, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_Eij = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_Ei = self.dof_Eia.reshape(-1, n_i * n_c)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=dof_Ei)


xdomain = XDomainAxiSym(x_0=(0, 0),
                        L_x=1, L_y=1,
                        integ_factor=1,
                        n_x=10, n_y=10,
                        fets=FETS2D4Q())

m = MATS3DDesmorat()

s = Simulator(
    model=m,
    xdomain=xdomain
)

fixed_y = BCSlice(slice=xdomain.mesh[:, 0, :, 0],
                  var='u', dims=[1], value=0)
fixed_x = BCSlice(slice=xdomain.mesh[0, :, 0, :],
                  var='u', dims=[0], value=0)
control_bc = BCSlice(slice=xdomain.mesh[-1, :, -1, :],
                     var='u', dims=[1], value=-0.0003)
s.tstep.bcond_mngr.bcond_list = [
    fixed_x,
    fixed_y,
    control_bc
]

s.run()
s.join()
print(s.tstep.U_n)
