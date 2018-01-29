'''
Created on Jan 24, 2018

This script demonstrates the looples implementation
of the finite element code for multiphase continuum.
Example (2D discretization)

@author: rch, abaktheer
'''

from ibvpy.api import FEGrid, FETSEval
from ibvpy.fets import fets2D
from mathkit.matrix_la import \
    SysMtxArray, SysMtxAssembly
from mayavi.modules.surface import Surface
from mayavi.scripts import mayavi2
from tvtk.api import \
    tvtk

import numpy as np
import pylab as p
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
# numerical integration points (IP) and weights
xi_m = np.array([[-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                 [1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                 [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                 [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
                 ], dtype=np.float_
                )

w_m = np.array([1, 1, 1, 1], dtype=np.float_)

xi_n = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float_)
vtk_cells = [[0, 1, 2, 3]]
vtk_cell_types = 'Quad'

print 'xi_m', xi_m
print 'xi_m.shape', xi_m.shape
print 'w_m', w_m
print '*************************'

# the values of the shape functions and their derivatives at the IPs
N_mi = np.array([N_xi_i.subs(zip([xi_1, xi_2], xi))
                 for xi in xi_m], dtype=np.float_)
N_im = np.einsum('mi->im', N_mi)
dN_mir = np.array([dN_xi_ir.subs(zip([xi_1, xi_2], xi))
                   for xi in xi_m], dtype=np.float_).reshape(4, 4, 2)
dN_nir = np.array([dN_xi_ir.subs(zip([xi_1, xi_2], xi))
                   for xi in xi_n], dtype=np.float_).reshape(4, 4, 2)
dN_imr = np.einsum('mir->imr', dN_mir)
dN_inr = np.einsum('nir->inr', dN_nir)


print 'N_im', N_im
print 'N_im.shape', N_im.shape
print 'dN_ima', dN_imr
print 'dN_ima.shape', dN_imr.shape
print '*************************'


class FETS2D4u4x(FETSEval):
    dof_r = tr.Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    geo_r = tr.Array(value=[[-1, -1], [1, -1], [1, 1], [-1, 1]])
    vtk_r = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    n_nodal_dofs = 2


# Discretization
L_x, L_y = 200, 100
mesh = FEGrid(coord_max=(L_x, L_y),
              shape=(100, 30),
              fets_eval=FETS2D4u4x())

x_Ia = mesh.X_Id
# print 'x_Ia', x_Ia

n_I, n_a = x_Ia.shape
dof_Ia = np.arange(n_I * n_a, dtype=np.int_).reshape(n_I, -1)
# print 'dof_Ia', dof_Ia

I_Ei = mesh.I_Ei
# print 'I_Ei', I_Ei

x_Eia = x_Ia[I_Ei, :]
# print 'x_Eia', x_Eia

dof_Eia = dof_Ia[I_Ei]
# print 'dof_Eia', dof_Eia

x_Ema = np.einsum('im,Eia->Ema', N_im, x_Eia)
# print 'x_Ema', x_Ema

J_Emar = np.einsum('imr,Eia->Emar', dN_imr, x_Eia)
J_Enar = np.einsum('inr,Eia->Enar', dN_inr, x_Eia)

det_J_Em = np.linalg.det(J_Emar)
# print 'det(J_Em)', det_J_Em

inv_J_Emar = np.linalg.inv(J_Emar)
inv_J_Enar = np.linalg.inv(J_Enar)
# print 'inv(J_Emar)', inv_J_Emar

B_Eimabc = np.einsum('abcd,imr,Eidr->Eimabc', I_sym_abcd, dN_imr, inv_J_Emar)
# print 'eps_Emab', eps_Emab
B_Einabc = np.einsum('abcd,inr,Eidr->Einabc', I_sym_abcd, dN_inr, inv_J_Enar)

BB_Emicjdabef = np.einsum('Eimabc,Ejmefd, Em, m->Emicjdabef',
                          B_Eimabc, B_Eimabc, det_J_Em, w_m)
# print 'BB_Emicjdabef', BB_Emicjdabef

# -----------------------------------------------------------------------------------------------------
# Construct the fourth order elasticity tensor for the plane stress case (shape: (2,2,2,2))
# -----------------------------------------------------------------------------------------------------
E = 28000.0
nu = 0.2
# first Lame paramter
la = E * nu / ((1 + nu) * (1 - 2 * nu))
# second Lame parameter (shear modulus)
mu = E / (2 + 2 * nu)

# elasticity matrix (shape: (3,3))
D_ab = np.zeros([3, 3])
D_ab[0, 0] = E / (1.0 - nu * nu)
D_ab[0, 1] = E / (1.0 - nu * nu) * nu
D_ab[1, 0] = E / (1.0 - nu * nu) * nu
D_ab[1, 1] = E / (1.0 - nu * nu)
D_ab[2, 2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)

map2d_ijkl2a = np.array([[[[0, 0],
                           [0, 0]],
                          [[2, 2],
                           [2, 2]]],
                         [[[2, 2],
                           [2, 2]],
                          [[1, 1],
                             [1, 1]]]])
map2d_ijkl2b = np.array([[[[0, 2],
                           [2, 1]],
                          [[0, 2],
                           [2, 1]]],
                         [[[0, 2],
                           [2, 1]],
                          [[0, 2],
                             [2, 1]]]])

D_abef = D_ab[map2d_ijkl2a, map2d_ijkl2b]
# print 'D_stress', D_abef
# print 'D_stress', D_abef.shape

K_Eicjd = np.einsum('Emicjdabef,abef->Eicjd',
                    BB_Emicjdabef, D_abef)
# print 'K_Eicjd', K_Eicjd.shape
n_E, n_i, n_c, n_j, n_d = K_Eicjd.shape
K_E = K_Eicjd.reshape(n_E, n_i * n_c, n_j * n_d)
# print 'K_Eicjd', K_E
dof_E = dof_Eia.reshape(n_E, n_i * n_c)
K_subdomain = SysMtxArray(mtx_arr=K_E, dof_map_arr=dof_E)
K = SysMtxAssembly()
K.sys_mtx_arrays.append(K_subdomain)
# print 'K', K
print '-----------------------------------------'
fixed_dofs = mesh[0, :, 0, :].dofs.flatten()
for dof in fixed_dofs:
    K.register_constraint(dof, 0)
F = np.zeros((dof_Ia.size))
K.apply_constraints(F)
# print F.shape
loaded_dofs = mesh[-1, :, -1, :].dofs.flatten()
for dof in loaded_dofs:
    F[dof] = 1000.0
U = K.solve(F)
# print 'U', U

d_Ia = U.reshape(-1, n_c)
d_Eia = d_Ia[I_Ei]
eps_Enab = np.einsum('Einabc,Eic->Enab', B_Einabc, d_Eia)
# print 'eps_Emab', eps_Enab

sig_Enab = np.einsum('abef,Emef->Emab', D_abef, eps_Enab)
# print 'sig_Emab', sig_Enab

delta23_ab = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=np.float_)

cell_class = tvtk.Quad().cell_type
n_E, n_i, n_a = x_Eia.shape
n_Ei = n_E * n_i
points = np.einsum('Ia,ab->Ib', x_Eia.reshape(-1, n_c), delta23_ab)
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
