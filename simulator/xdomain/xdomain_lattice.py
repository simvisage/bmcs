
from mathkit.matrix_la import SysMtxArray, DenseMtx, SysMtxAssembly
from mathkit.tensor import EPS, DELTA
from mpl_toolkits import mplot3d
from simulator.i_xdomain import IXDomain
from traits.api import \
    provides, Property, cached_property, Array, \
    Int, Tuple
from view.ui import BMCSLeafNode

import matplotlib.pyplot as plt
import numpy as np


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
class XDomainLattice(BMCSLeafNode):

    X_Ia = Array(dtype=np.float_, MESH=True)

    I_Li = Array(dtype=np.int_, MESH=True)

    U_var_shape = Property(Int)

    def _get_U_var_shape(self):
        return len(self.X_Ia.shape) * 3 * 2

    state_var_shape = Property(Tuple)

    def _get_state_var_shape(self):
        return (len(self.I_Li),)

    nT_Lba = Property(depends_on='MESH')

    @cached_property
    def _get_nT_Lba(self):
        X_Lia = self.X_Ia[I_Li]
        Tu_La = X_Lia[..., 1, :] - X_Lia[..., 0, :]
        I = np.fabs(Tu_La[..., 0]) > np.fabs(Tu_La[..., 2])
        Tvv_La = np.c_[0 * Tu_La[..., 0], -Tu_La[..., 2], Tu_La[..., 1]]
        Tvv_La[I, :] = np.c_[-Tu_La[I, 1], Tu_La[I, 0], 0 * Tu_La[I, 0]]
        Tw_La = np.einsum('abc,...a,...b->...c', EPS, Tu_La, Tvv_La)
        Tv_La = np.einsum('abc,...a,...b->...c', EPS, Tw_La, Tu_La)
        T_Lba = np.einsum('...bla->...lba', np.array([Tu_La, Tv_La, Tw_La]))
        norm_T_lb = 1. / np.sqrt(np.einsum(
            '...lba,...lba->...lb', T_Lba, T_Lba)
        )
        nT_Lba = np.einsum('...lb,...lba->...lba', norm_T_lb, T_Lba)
        return nT_Lba

    B_Lipac = Property(depends_on='MESH')

    @cached_property
    def _get_B_Lipac(self):
        DELTA2 = np.identity(2)
        X_Lia = X_Ia[I_Li]
        Xm_La = np.einsum('ii,Lia->La', DELTA2, X_Lia) / 2
        Xm_Lia = Xm_La[..., np.newaxis, :]
        dXm_Lia = Xm_Lia - X_Lia
        switch_sign = np.array([-1, 1], dtype=np.float_)
        S_Liac = np.einsum('i,ii,...Lac->...Liac',
                           switch_sign, DELTA2, self.nT_Lba)
        return np.einsum('pLiac->Lipac', np.array(
            [S_Liac, np.einsum('Liab,bcd,Lid->Liac',
                               S_Liac, EPS, dXm_Lia)]
        ))

    B_Lipabjqcd = Property(depends_on='MESH,GEO,CS')

    @cached_property
    def _get_B_Lipabjqcd(self):
        return np.einsum(
            '...Lipab,...Ljqcd->...Lipabjqcd',
            self.B_Lipac, self.B_Lipac
        )

    o_Ipa = Property(depends_on='MESH')

    @cached_property
    def _get_o_Ipa(self):
        n_I, _ = X_Ia.shape
        n_a = 3
        n_p = 2
        return np.arange(n_I * n_p * n_a, dtype=np.int_).reshape(-1, n_p, n_a)

    o_Lipa = Property(depends_on='MESH,GEO,CS,FE')

    @cached_property
    def _get_o_Lipa(self):
        I_Li = self.I_Li
        return self.o_Ipa[I_Li]

    def map_U_to_field(self, U):
        U_Lipa = U[self.o_Lipa]
        return np.einsum('Lipac,Lipc->La', self.B_Lipac, U_Lipa)

    def map_field_to_F(self, sig_La):
        n_p, _, n_i, _, n_c = self.B_Lipac.shape
        f_Lipa = np.einsum('Lipac,La->Lipc', self.B_Lipac, sig_La)
        f_Li = f_Lipa.reshape(-1, n_i * n_p * n_c)
        o_L = self.o_Lipa.reshape(-1, n_i * n_p * n_c)
        return o_L.flatten(), f_Li.flatten()

    def map_field_to_K(self, D_Lbc):
        print('D\n', D_Lbc)
        K_Lipbjqd = np.einsum(
            'Lipabjqcd,Lbc->Lipajqd', self.B_Lipabjqcd, D_Lbc
        )
        _, n_i, n_p, n_c, _, _, _ = K_Lipbjqd.shape
        K_Lij = K_Lipbjqd.reshape(-1, n_i * n_p * n_c, n_i * n_p * n_c)
        o_Li = self.o_Lipa.reshape(-1, n_i * n_p * n_c)
        return SysMtxArray(mtx_arr=K_Lij, dof_map_arr=o_Li)


def test01_spring():
    X_Ia = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
        ], dtype=np.float_
    )

    I_Li = np.array(
        [
            [0, 1],
        ], dtype=np.int_
    )

    U_Ia = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.float_
    )

    Phi_Ia = np.array(
        [
            [0, 0, 0.0 * np.pi],
            [0, 0, 0.0 * np.pi],
        ], dtype=np.float_
    )

    return X_Ia, I_Li, U_Ia, Phi_Ia


def test02_quad():
    X_Ia = np.array(
        [
            [0, 0, 0],
            [0, 2, 0],
            [-1, 1, 0],
            [1, 1, 0]
        ], dtype=np.float_
    )

    I_Li = np.array(
        [
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3]
        ], dtype=np.int_
    )

    U_Ia = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float_
    )

    Phi_Ia = np.array(
        [
            [0, 0, 0.1 * np.pi],
            [0, 0, -0.1 * np.pi],
            [0, 0, 0.1 * np.pi],
            [0.3 * np.pi, 0, 0.1 * np.pi]
        ], dtype=np.float_
    )

    return X_Ia, I_Li, U_Ia, Phi_Ia


def test03_tetrahedron():
    X_Ia = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.3, 0.3, 1.0]
        ], dtype=np.float_
    )

    I_Li = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3]
        ], dtype=np.int_
    )

    U_Ia = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float_
    )

    Phi_Ia = np.array(
        [
            [0, 0, 0.1 * np.pi],
            [0, 0, -0.1 * np.pi],
            [0, 0, 0.1 * np.pi],
            [0.3 * np.pi, 0, 0.1 * np.pi]
        ], dtype=np.float_
    )

    return X_Ia, I_Li, U_Ia, Phi_Ia


def test04_pyramide():
    X_Ia = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float_
    )

    I_Li = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3]
        ], dtype=np.int_
    )

    U_Ia = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float_
    )

    Phi_Ia = np.array(
        [
            [0, 0, 0.1 * np.pi],
            [0, 0, -0.1 * np.pi],
            [0, 0, 0.1 * np.pi],
            [0.3 * np.pi, 0, 0.1 * np.pi]
        ], dtype=np.float_
    )

    return X_Ia, I_Li, U_Ia, Phi_Ia


def plot3d(X_Ia, I_Li):
    X_Lia = X_Ia[I_Li]
    X_Lai = np.einsum('Lia->Lai', X_Lia)
    plt.figure()
    ax = plt.axes(projection="3d")
    for X_ai in X_Lai:
        x_line, y_line, z_line = X_ai
        ax.plot3D(x_line, y_line, z_line, 'gray')
    x_points, y_points, z_points = np.einsum('Ia->aI', X_Ia)
    ax.scatter3D(x_points, y_points, z_points, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_u3d(X_Lai, XU_Lai, XUm_Lai, XUIm_Lai):
    # def plot_u3d(X_Lia, XU_Lia):

    plt.figure()
    ax = plt.axes(projection="3d")
    for X_ai, XU_ai, XUm_ai in zip(X_Lai, XU_Lai, XUm_Lai):
        #    for X_ai, XU_ai in zip(X_Lai, XU_Lai):
        x_line, y_line, z_line = X_ai
        ax.plot3D(x_line, y_line, z_line, 'gray')
        x_line, y_line, z_line = XU_ai
        ax.plot3D(x_line, y_line, z_line, 'green')
        x_line, y_line, z_line = XUm_ai
        ax.plot3D(x_line, y_line, z_line, 'red')
    for XUIm_ai in XUIm_Lai:
        x_line, y_line, z_line = XUIm_ai
        ax.plot3D(x_line, y_line, z_line, 'blue')
#     x_points, y_points, z_points = np.einsum('Ia->aI', XU_Ia)
#     ax.scatter3D(x_points, y_points, z_points, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


if __name__ == '__main__':
    X_Ia, I_Li, U_Ia, Phi_Ia = test03_tetrahedron()
    xdomain = XDomainLattice(X_Ia=X_Ia, I_Li=I_Li)
    U_pLia = np.array([U_Ia[I_Li], Phi_Ia[I_Li]])
    U = np.einsum('pLia->Lipa', U_pLia).flatten()
    eps_La = xdomain.map_U_to_field(U)
    D_Lab = 1. * np.array(
        [
            [
                [100000.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]
            ]
        ], dtype=np.float_)
    sig_La = np.einsum('...ab,...b->...a', D_Lab, eps_La)
    L_O, F_Lo = xdomain.map_field_to_F(sig_La)
    F_int = np.bincount(L_O, weights=F_Lo)

    k_mtx_array = xdomain.map_field_to_K(D_Lab)
    K = SysMtxAssembly()
    K.sys_mtx_arrays.append(k_mtx_array)
    K_dense = DenseMtx(assemb=K)
    K.register_constraint(a=0,  u_a=0.)  # node 1
    K.register_constraint(a=1,  u_a=0.)  #
    K.register_constraint(a=2,  u_a=0.)  #

    K.register_constraint(a=3,  u_a=0.)  #
    K.register_constraint(a=4,  u_a=0.)  #
    K.register_constraint(a=5,  u_a=0.)  #

#    K.register_constraint(a=6,  u_a=0.)  # node 2
    K.register_constraint(a=7,  u_a=0.)  #
    K.register_constraint(a=8,  u_a=0.)  #

    K.register_constraint(a=9,  u_a=0.)  #
    K.register_constraint(a=10,  u_a=0.)  #
    K.register_constraint(a=11,  u_a=0.)  #

    K.register_constraint(a=12,  u_a=0.)  # node 3
#    K.register_constraint(a=13,  u_a=0.)  #
    K.register_constraint(a=14,  u_a=0.)  #

    K.register_constraint(a=15,  u_a=0.)  #
    K.register_constraint(a=16,  u_a=0.)  #
    K.register_constraint(a=17,  u_a=0.)  #

#    K.register_constraint(a=20,  u_a=-.3)  # clamped end
    K.register_constraint(a=21,  u_a=0.0)  # clamped end
    K.register_constraint(a=22,  u_a=0.3)  # clamped end
    K.register_constraint(a=23,  u_a=0.0)  # clamped end
    F_ext = np.zeros_like(F_int)
    F_ext[20] = -10
    K.apply_constraints(F_ext)
    print(F_ext)
    K_dense = DenseMtx(assemb=K)
    print('Rank 2', np.linalg.matrix_rank(K_dense.mtx))
    U, pd = K.solve()
    U_Ia = U[xdomain.o_Ipa][..., 0, :]
    Phi_Ia = U[xdomain.o_Ipa][..., 1, :]

    X_Lia = X_Ia[I_Li]
    X_aiL = np.einsum('Lia->aiL', X_Lia)
    X_Lai = np.einsum('Lia->Lai', X_Lia)
    DELTA2 = np.identity(2)
    Xm_La = np.einsum('ii,Lia->La', DELTA2, X_Lia) / 2
    Xm_Lia = Xm_La[..., np.newaxis, :]
    dXm_Lia = Xm_Lia - X_Lia

    Um_trans_Lia = U_Ia[I_Li]
    Phi_Lia = Phi_Ia[I_Li]
    Um_rot_Lia = np.einsum('abc,...b,...c->...a',
                           EPS, Phi_Lia, dXm_Lia)
    Um_Lia = Um_trans_Lia + Um_rot_Lia

    #---
    Um_trans_aiL = np.einsum('Lia->aiL', Um_trans_Lia)
    XU_aiL = X_aiL + Um_trans_aiL
    XUm_aiL = np.einsum('Lia->aiL', (Xm_Lia + Um_Lia))
    XUIm_aiL = np.concatenate(
        [
            np.einsum('iaL->aiL',
                      np.array([XU_aiL[:, 0, ...], XUm_aiL[:, 0, ...]])),
            np.einsum('iaL->aiL',
                      np.array([XU_aiL[:, 1, ...], XUm_aiL[:, 1, ...]]))
        ], axis=-1
    )
    #----
    XU_Lai = np.einsum('aiL->Lai', XU_aiL)
    XUm_Lai = np.einsum('aiL->Lai', XUm_aiL)
    XUIm_Lai = np.einsum('aiL->Lai', XUIm_aiL)
#    plot3d(X_Ia[I_Li])
#    XU_Lia = X_Ia[I_Li] + U_Ia[I_Li]
    plot_u3d(X_Lai, XU_Lai, XUm_Lai, XUIm_Lai)
    plt.show()
