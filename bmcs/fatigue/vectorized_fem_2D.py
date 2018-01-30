'''
Created on Jan 24, 2018

This script demonstrates the looples implementation
of the finite element code for multiphase continuum.
Example (2D discretization)

@author: rch, abaktheer
'''

from ibvpy.api import FEGrid, FETSEval
from mathkit.matrix_la import \
    SysMtxArray, SysMtxAssembly
from mayavi.scripts import mayavi2
from tvtk.api import \
    tvtk

import numpy as np
import sympy as sp
import traits.api as tr
from tvtk.tvtk_classes import tvtk_helper


#========================================
# Tensorial operators
#========================================
# Identity tensor
delta = np.identity(2)
print 'delta', delta

# symetrization operator
I_sym_abcd = 0.5 * \
    (np.einsum('ac,bd->abcd', delta, delta) +
     np.einsum('ad,bc->abcd', delta, delta))
print 'I_sym_abcd', I_sym_abcd
print 'I_sym_abcd.shape', I_sym_abcd.shape

#=================================================
# 4 nodes iso-parametric quadrilateral element
#=================================================

# generate shape functions with sympy
xi_1 = sp.symbols('xi_1')
xi_2 = sp.symbols('xi_2')

#=========================================================================
# Finite element specification
#=========================================================================

N_xi_i = sp.Matrix([(1.0 - xi_1) * (1.0 - xi_2) / 4.0,
                    (1.0 + xi_1) * (1.0 - xi_2) / 4.0,
                    (1.0 + xi_1) * (1.0 + xi_2) / 4.0,
                    (1.0 - xi_1) * (1.0 + xi_2) / 4.0], dtype=np.float_)


dN_xi_ir = sp.Matrix(((-(1.0 / 4.0) * (1.0 - xi_2), -(1.0 / 4.0) * (1.0 - xi_1)),
                      ((1.0 / 4.0) * (1.0 - xi_2), -
                       (1.0 / 4.0) * (1.0 + xi_1)),
                      ((1.0 / 4.0) * (1.0 + xi_2), (1.0 / 4.0) * (1.0 + xi_1)),
                      (-(1.0 / 4.0) * (1.0 + xi_2), (1.0 / 4.0) * (1.0 - xi_1))), dtype=np.float_)

#dN_xi_ia = N_xi_i.diff('xi_1')

print 'N_xi_i', N_xi_i
print 'dN_xi_ir', dN_xi_ir
print 'dN_xi_ia.shape', dN_xi_ir.shape


class FETS2D4u4x(FETSEval):
    dof_r = tr.Array(np.float_,
                     value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    geo_r = tr.Array(np.float_,
                     value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    vtk_r = tr.Array(np.float_,
                     value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    n_nodal_dofs = 2
    vtk_r = tr.Array(np.float_, value=[[-1, -1], [1, -1], [1, 1],
                                       [-1, 1]])
    vtk_cells = [[0, 1, 2, 3]]
    vtk_cell_types = 'Quad'

    # numerical integration points (IP) and weights
    xi_m = tr.Array(np.float_,
                    value=[[-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                           [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
                           ])

    w_m = tr.Array(value=[1, 1, 1, 1], dtype=np.float_)

    shape_function_values = tr.Property(tr.Tuple)
    '''The values of the shape functions and their derivatives at the IPs
    '''
    @tr.cached_property
    def _get_shape_function_values(self):
        N_mi = np.array([N_xi_i.subs(zip([xi_1, xi_2], xi))
                         for xi in self.xi_m], dtype=np.float_)
        N_im = np.einsum('mi->im', N_mi)
        dN_mir = np.array([dN_xi_ir.subs(zip([xi_1, xi_2], xi))
                           for xi in self.xi_m], dtype=np.float_).reshape(4, 4, 2)
        dN_nir = np.array([dN_xi_ir.subs(zip([xi_1, xi_2], xi))
                           for xi in self.vtk_r], dtype=np.float_).reshape(4, 4, 2)
        dN_imr = np.einsum('mir->imr', dN_mir)
        dN_inr = np.einsum('nir->inr', dN_nir)
        return (N_im, dN_imr, dN_inr)

    N_im = tr.Property()
    '''Shape function values in integration poindots.
    '''

    def _get_N_im(self):
        return self.shape_function_values[0]

    dN_imr = tr.Property()
    '''Shape function derivatives in integration poindots.
    '''

    def _get_dN_imr(self):
        return self.shape_function_values[1]

    dN_inr = tr.Property()
    '''Shape function derivatives in visualization poindots.
    '''

    def _get_dN_inr(self):
        return self.shape_function_values[2]


class MATSElastic2D(tr.HasStrictTraits):
    # -----------------------------------------------------------------------------------------------------
    # Construct the fourth order elasticity tensor for the plane stress case (shape: (2,2,2,2))
    # -----------------------------------------------------------------------------------------------------
    E = tr.Float(28000.0, input=True)
    nu = tr.Float(0.2, input=True)
    # first Lame paramter

    def _get_lame_params(self):
        la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2 + 2 * self.nu)
        return la, mu

    # elasticity matrix (shape: (3,3))
    D_ab = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_ab(self):
        D_ab = np.zeros([3, 3])
        E = self.E
        nu = self.nu
        D_ab[0, 0] = E / (1.0 - nu * nu)
        D_ab[0, 1] = E / (1.0 - nu * nu) * nu
        D_ab[1, 0] = E / (1.0 - nu * nu) * nu
        D_ab[1, 1] = E / (1.0 - nu * nu)
        D_ab[2, 2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)
        return D_ab

    map2d_ijkl2a = tr.Array(np.int_, value=[[[[0, 0],
                                              [0, 0]],
                                             [[2, 2],
                                              [2, 2]]],
                                            [[[2, 2],
                                              [2, 2]],
                                             [[1, 1],
                                                [1, 1]]]])
    map2d_ijkl2b = tr.Array(np.int_, value=[[[[0, 2],
                                              [2, 1]],
                                             [[0, 2],
                                              [2, 1]]],
                                            [[[0, 2],
                                              [2, 1]],
                                             [[0, 2],
                                                [2, 1]]]])

    D_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_abef(self):
        return self.D_ab[self.map2d_ijkl2a, self.map2d_ijkl2b]

    def get_corr_pred(self, eps_Emab, deps_Emab, t):
        D_Emabef = self.D_abef[None, None, :, :, :, :]
        sig_Emab = np.einsum('...abef,...ef->...ab',
                             D_Emabef, eps_Emab)

        return D_Emabef, sig_Emab


class DOTSGrid(tr.HasStrictTraits):
    L_x = tr.Float(200, input=True)
    L_y = tr.Float(100, input=True)
    n_x = tr.Int(100, input=True)
    n_y = tr.Int(30, input=True)
    fets = tr.Instance(FETSEval, input=True)
    mats = tr.Instance(MATSElastic2D, input=True)
    mesh = tr.Property(tr.Instance(FEGrid), depends_on='+input')

    @tr.cached_property
    def _get_mesh(self):
        return FEGrid(coord_max=(self.L_x, self.L_y),
                      shape=(self.n_x, self.n_y),
                      fets_eval=self.fets)

    cached_grid_values = tr.Property(tr.Tuple,
                                     depends_on='+input')

    @tr.cached_property
    def _get_cached_grid_values(self):
        x_Ia = self.mesh.X_Id
        # print 'x_Ia', x_Ia

        n_I, n_a = x_Ia.shape
        dof_Ia = np.arange(n_I * n_a, dtype=np.int_).reshape(n_I, -1)
        # print 'dof_Ia', dof_Ia

        I_Ei = self.mesh.I_Ei
        # print 'I_Ei', I_Ei

        x_Eia = x_Ia[I_Ei, :]
        # print 'x_Eia', x_Eia

        dof_Eia = dof_Ia[I_Ei]
        # print 'dof_Eia', dof_Eia

        x_Ema = np.einsum('im,Eia->Ema', self.fets.N_im, x_Eia)
        # print 'x_Ema', x_Ema

        J_Emar = np.einsum('imr,Eia->Emar', self.fets.dN_imr, x_Eia)
        J_Enar = np.einsum('inr,Eia->Enar', self.fets.dN_inr, x_Eia)

        det_J_Em = np.linalg.det(J_Emar)
        # print 'det(J_Em)', det_J_Em

        inv_J_Emar = np.linalg.inv(J_Emar)
        inv_J_Enar = np.linalg.inv(J_Enar)
        # print 'inv(J_Emar)', inv_J_Emar

        B_Eimabc = np.einsum('abcd,imr,Eidr->Eimabc',
                             I_sym_abcd, self.fets.dN_imr, inv_J_Emar)
        # print 'eps_Emab', eps_Emab
        B_Einabc = np.einsum('abcd,inr,Eidr->Einabc',
                             I_sym_abcd, self.fets.dN_inr, inv_J_Enar)

        BB_Emicjdabef = np.einsum('Eimabc,Ejmefd, Em, m->Emicjdabef',
                                  B_Eimabc, B_Eimabc, det_J_Em,
                                  self.fets.w_m)

        return (BB_Emicjdabef, B_Eimabc,
                dof_Eia, x_Eia, dof_Ia, I_Ei,
                B_Einabc, det_J_Em)

    BB_Emicjdabef = tr.Property()
    '''
    '''

    def _get_BB_Emicjdabef(self):
        return self.cached_grid_values[0]

    B_Eimabc = tr.Property()
    '''.
    '''

    def _get_B_Eimabc(self):
        return self.cached_grid_values[1]

    dof_Eia = tr.Property()
    '''.
    '''

    def _get_dof_Eia(self):
        return self.cached_grid_values[2]

    x_Eia = tr.Property()
    '''.
    '''

    def _get_x_Eia(self):
        return self.cached_grid_values[3]

    dof_Ia = tr.Property()
    '''.
    '''

    def _get_dof_Ia(self):
        return self.cached_grid_values[4]

    I_Ei = tr.Property()
    '''.
    '''

    def _get_I_Ei(self):
        return self.cached_grid_values[5]

    B_Einabc = tr.Property()
    '''.
    '''

    def _get_B_Einabc(self):
        return self.cached_grid_values[6]

    det_J_Em = tr.Property()
    '''.
    '''

    def _get_det_J_Em(self):
        return self.cached_grid_values[7]

    def get_corr_pred(self, U, dU, t):

        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]
        eps_Emab = np.einsum('Eimabc,Eic->Emab', self.B_Eimabc, U_Eia)
        dU_Ia = U.reshape(-1, n_c)
        dU_Eia = dU_Ia[self.I_Ei]
        deps_Emab = np.einsum('Eimabc,Eic->Emab', self.B_Eimabc, dU_Eia)
        D_Emabef, sig_Emab = self.mats.get_corr_pred(eps_Emab, deps_Emab, t)
        K_Eicjd = np.einsum('Emicjdabef,Emabef->Eicjd',
                            self.BB_Emicjdabef, D_Emabef)
        n_E, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_E = K_Eicjd.reshape(n_E, n_i * n_c, n_j * n_d)
        dof_E = dots.dof_Eia.reshape(n_E, n_i * n_c)
        K_subdomain = SysMtxArray(mtx_arr=K_E, dof_map_arr=dof_E)

        f_Eic = np.einsum('m,Eimabc,Emab,Em->Eic',
                          self.fets.w_m, self.B_Eimabc, sig_Emab,
                          self.det_J_Em)
        f_Ei = f_Eic.reshape(-1, n_i * n_c)
        F_dof = np.bincount(dof_E.flatten(), weights=f_Ei.flatten())
        F_int = F_dof

        return K_subdomain, F_int


mats2d_elastic = MATSElastic2D()
fets2d_4u_4q = FETS2D4u4x()
dots = DOTSGrid(fets=fets2d_4u_4q,
                mats=mats2d_elastic)

K = SysMtxAssembly()
U = np.zeros((dots.mesh.n_dofs,), dtype=np.float_)
dU = np.copy(U)
K_subdomain, F_int = dots.get_corr_pred(U, dU, 0)
K.sys_mtx_arrays.append(K_subdomain)
# print 'K', K
print '-----------------------------------------'
fixed_dofs = dots.mesh[0, :, 0, :].dofs.flatten()
for dof in fixed_dofs:
    K.register_constraint(dof, 0)
F = np.zeros((dots.dof_Ia.size))
K.apply_constraints(F)
# print F.shape
loaded_dofs = dots.mesh[-1, :, -1, :].dofs.flatten()
for dof in loaded_dofs:
    F[dof] = 1000.0
U = K.solve(F)
# print 'U', U

n_c = fets2d_4u_4q.n_nodal_dofs
d_Ia = U.reshape(-1, n_c)
d_Eia = d_Ia[dots.I_Ei]
eps_Enab = np.einsum('Einabc,Eic->Enab', dots.B_Einabc, d_Eia)
# print 'eps_Emab', eps_Enab

sig_Enab = np.einsum('abef,Emef->Emab', mats2d_elastic.D_abef, eps_Enab)
# print 'sig_Emab', sig_Enab

delta23_ab = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=np.float_)

cell_class = tvtk.Quad().cell_type
n_E, n_i, n_a = dots.x_Eia.shape
n_Ei = n_E * n_i
points = np.einsum('Ia,ab->Ib', dots.x_Eia.reshape(-1, n_c), delta23_ab)
ug = tvtk.UnstructuredGrid(points=points)
ug.set_cells(cell_class, np.arange(n_Ei).reshape(n_E, n_i))

vectors = np.einsum('Ia,ab->Ib', d_Eia.reshape(-1, n_c), delta23_ab)
ug.point_data.vectors = vectors
ug.point_data.vectors.name = 'displacement'
# Now view the data.
warp_arr = tvtk.DoubleArray(name='displacement')
warp_arr.from_array(vectors)
ug.point_data.add_array(warp_arr)

eps_Encd = tensors = np.einsum('...ab,ac,bd->...cd',
                               eps_Enab, delta23_ab, delta23_ab)
tensors = eps_Encd[:, :, [0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0]].reshape(-1, 6)
tensors = eps_Encd.reshape(-1, 9)
ug.point_data.tensors = tensors
ug.point_data.tensors.name = 'strain'


@mayavi2.standalone
def view():
    from mayavi.sources.vtk_data_source import VTKDataSource
    from mayavi.modules.outline import Outline
    from mayavi.modules.surface import Surface
    from mayavi.modules.vectors import Vectors
    from mayavi.filters.api import WarpVector, ExtractTensorComponents

    mayavi.new_scene()
    # The single type one
    src = VTKDataSource(data=ug)
    mayavi.add_source(src)
    warp_vector = WarpVector()
    mayavi.add_filter(warp_vector, src)
    surface = Surface()
    mayavi.add_filter(surface, warp_vector)

    etc = ExtractTensorComponents()
    mayavi.add_filter(etc, warp_vector)
    surface2 = Surface()
    mayavi.add_filter(surface2, etc)
    etc.filter.scalar_mode = 'component'

    lut = etc.children[0]
    lut.scalar_lut_manager.show_scalar_bar = True
    lut.scalar_lut_manager.show_legend = True
    lut.scalar_lut_manager.scalar_bar.height = 0.8
    lut.scalar_lut_manager.scalar_bar.width = 0.17
    lut.scalar_lut_manager.scalar_bar.position = np.array([0.82,  0.1])


if __name__ == '__main__':
    view()
