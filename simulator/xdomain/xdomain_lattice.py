
from mathkit.matrix_la import SysMtxArray, DenseMtx, SysMtxAssembly
from mathkit.tensor import EPS
from mpl_toolkits import mplot3d
from simulator.i_xdomain import IXDomain
from traits.api import \
    Property, Array, Int, HasStrictTraits, \
    cached_property, Tuple, provides, Instance, DelegatesTo, \
    Constant, Bool
from view.ui import BMCSLeafNode

import matplotlib.pyplot as plt
import numpy as np
import traitsui.api as ui


class LatticeTessellation(BMCSLeafNode):
    '''Lattice tessellation
    '''
    node_name = 'tessellation'

    X_Ia = Array(np.float_,
                 MESH=True,
                 input=True,
                 unit=r'm',
                 symbol=r'$X_{Ia}$',
                 auto_set=False, enter_set=True,
                 desc='node coordinates')
    I_Li = Array(np.int_,
                 MESH=True,
                 input=True,
                 unit='-',
                 symbol='$I_{Li}$',
                 auto_set=False, enter_set=True,
                 desc='connectivity')
    view = ui.View(
        ui.Item('X_Ia'),
        ui.Item('I_La'),
    )

    dof_offset = Int(0)

    n_dofs = Property()

    def _get_n_dofs(self):
        return len(self.X_Ia) * 6

    tree_view = view

    vtk_expand_operator = Array(np.float_, value=np.identity(3))

    n_nodal_dofs = Constant(3)


@provides(IXDomain)
class XDomainLattice(BMCSLeafNode):

    hidden = Bool(False)

    #=========================================================================
    # Methods needed by XDomain to chain the subdomains
    #=========================================================================
    dof_offset = DelegatesTo('mesh')

    def set_next(self, next_):
        self.mesh.next_grid = next_.mesh

    def set_prev(self, prev):
        self.mesh.prev_grid = prev.mesh

    mesh = Instance(LatticeTessellation)

    X_Ia = DelegatesTo('mesh')
    I_Li = DelegatesTo('mesh')
    n_dofs = DelegatesTo('mesh')
    vtk_expand_operator = DelegatesTo('mesh')
    n_nodal_dofs = DelegatesTo('mesh')

    U_var_shape = Property(Int)

    def _get_U_var_shape(self):
        return len(self.X_Ia.shape) * 3 * 2

    state_var_shape = Property(Tuple)

    def _get_state_var_shape(self):
        return (len(self.I_Li),)

    nT_Lba = Property(depends_on='MESH')

    @cached_property
    def _get_nT_Lba(self):
        X_Lia = self.X_Ia[self.I_Li]
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
        X_Lia = self.X_Ia[self.I_Li]
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
        n_I, _ = self.X_Ia.shape
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
        K_Lipbjqd = np.einsum(
            'Lipabjqcd,Lac->Lipbjqd', self.B_Lipabjqcd, D_Lbc
        )
        _, n_i, n_p, n_c, _, _, _ = K_Lipbjqd.shape
        K_Lij = K_Lipbjqd.reshape(-1, n_i * n_p * n_c, n_i * n_p * n_c)
        o_Li = self.o_Lipa.reshape(-1, n_i * n_p * n_c)
        return SysMtxArray(mtx_arr=K_Lij, dof_map_arr=o_Li)

    Xm_Lia = Property

    def _get_Xm_Lia(self):
        X_Lia = self.X_Ia[self.I_Li]
        DELTA2 = np.identity(2)
        Xm_La = np.einsum('ii,Lia->La', DELTA2, X_Lia) / 2
        return Xm_La[..., np.newaxis, :]

    dXm_Lia = Property

    def _get_dXm_Lia(self):
        X_Lia = self.X_Ia[self.I_Li]
        return self.Xm_Lia - X_Lia

    def get_vis3d(self):
        X_Lia = self.X_Ia[self.I_Li]
        X_aiL = np.einsum('Lia->aiL', X_Lia)
        X_Lai = np.einsum('Lia->Lai', X_Lia)
        Um_trans_Lia = U_Ia[I_Li]
        Phi_Lia = Phi_Ia[I_Li]
        Um_rot_Lia = np.einsum('abc,...b,...c->...a',
                               EPS, Phi_Lia, self.dXm_Lia)
        Um_Lia = Um_trans_Lia + Um_rot_Lia
        #---
        Um_trans_aiL = np.einsum('Lia->aiL', Um_trans_Lia)
        XU_aiL = X_aiL + Um_trans_aiL
        XUm_aiL = np.einsum('Lia->aiL', (self.Xm_Lia + Um_Lia))
        XUIm_aiL = np.concatenate(
            [
                np.einsum('iaL->aiL',
                          np.array([XU_aiL[:, 0, ...], XUm_aiL[:, 0, ...]])),
                np.einsum('iaL->aiL',
                          np.array([XU_aiL[:, 1, ...], XUm_aiL[:, 1, ...]]))
            ], axis=-1
        )
        XU_Lai = np.einsum('aiL->Lai', XU_aiL)
        XUm_Lai = np.einsum('aiL->Lai', XUm_aiL)
        XUIm_Lai = np.einsum('aiL->Lai', XUIm_aiL)

        return X_Lai, XU_Lai, XUm_Lai, XUIm_Lai


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

    fixed_dofs = [0, 1, 2, 3, 4, 5,
                  8, 9, 10, 11]
    control_dofs = []
    return X_Ia, I_Li, U_Ia, Phi_Ia, fixed_dofs, control_dofs


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

    fixed_dofs = []
    control_dofs = []

    return X_Ia, I_Li, U_Ia, Phi_Ia, fixed_dofs, control_dofs


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

    fixed_dofs = [0, 1, 2, 3, 4, 5,
                  7, 8, 9, 10, 11,
                  12, 14, 15, 16, 17,
                  21, 23]

    control_dofs = [22]

    return X_Ia, I_Li, U_Ia, Phi_Ia, fixed_dofs, control_dofs


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

    fixed_dofs = []
    control_dofs = []
    return X_Ia, I_Li, U_Ia, Phi_Ia, fixed_dofs, control_dofs


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
    X_Ia, I_Li, U_Ia, Phi_Ia, fixed_dofs, control_dofs = test03_tetrahedron()
    lt = LatticeTessellation(X_Ia=X_Ia, I_Li=I_Li)
    xdomain = XDomainLattice(mesh=lt)
    U_pLia = np.array([U_Ia[I_Li], Phi_Ia[I_Li]])
    U = np.einsum('pLia->Lipa', U_pLia).flatten()
    eps_La = xdomain.map_U_to_field(U)

    from ibvpy.mats.mats3D_ifc import MATS3DIfcElastic
    mats = MATS3DIfcElastic()
    sig_La, D_Lab = mats.get_corr_pred(eps_La, 1.0)
    L_O, F_Lo = xdomain.map_field_to_F(sig_La)
    F_int = np.bincount(L_O, weights=F_Lo)

    k_mtx_array = xdomain.map_field_to_K(D_Lab)
    K = SysMtxAssembly()
    K.sys_mtx_arrays.append(k_mtx_array)
    K_dense = DenseMtx(assemb=K)
    print(K_dense)

    for dof in fixed_dofs:
        K.register_constraint(a=dof, u_a=0)

    K.register_constraint(a=22,  u_a=0.3)  # clamped end
    F_ext = np.zeros_like(F_int)
    F_ext[20] = -100000
    K.apply_constraints(F_ext)
    # print(F_ext)
    #K_dense = DenseMtx(assemb=K)
    print('Rank 2', np.linalg.matrix_rank(K_dense.mtx))
    U, pd = K.solve()
    U_Ia = U[xdomain.o_Ipa][..., 0, :]
    Phi_Ia = U[xdomain.o_Ipa][..., 1, :]

    eps_La = xdomain.map_U_to_field(U)
    sig_La, D_Lab = mats.get_corr_pred(eps_La, 1.0)
    L_O, F_Lo = xdomain.map_field_to_F(sig_La)
    F_int = np.bincount(L_O, weights=F_Lo)
    #print('F_int 2\n', F_int)

    #Fu = np.einsum('ij,j->i', K_dense.mtx, U)
    # print(Fu)

    X_Lai, XU_Lai, XUm_Lai, XUIm_Lai = xdomain.get_vis3d()
    plot_u3d(X_Lai, XU_Lai, XUm_Lai, XUIm_Lai)
    plt.show()
