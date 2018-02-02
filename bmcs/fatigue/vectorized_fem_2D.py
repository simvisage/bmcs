
import os
from ibvpy.api import \
    FEGrid, FETSEval, TLine, BCSlice
from ibvpy.core.bcond_mngr import BCondMngr
from mathkit.matrix_la import \
    SysMtxArray, SysMtxAssembly
from mayavi import mlab
from mayavi.filters import extract_tensor_components
from mayavi.filters.api import WarpVector, ExtractTensorComponents
from mayavi.modules.outline import Outline
from mayavi.modules.surface import Surface
from mayavi.modules.vectors import Vectors
from traits.has_traits import HasStrictTraits
from tvtk.api import \
    tvtk, write_data

from ibvpy.mats.mats2D.mats2D_sdamage.vmats2D_sdamage import \
    MATS2D, MATS2DScalarDamage
import numpy as np
import sympy as sp
import traits.api as tr
from tvtk.tvtk_classes import tvtk_helper

'''
Created on Jan 24, 2018

This script demonstrates the looples implementation
of the finite element code for multiphase continuum.
Example (2D discretization)

@author: rch, abaktheer
'''


#========================================
# Tensorial operators
#========================================
# Identity tensor
delta = np.identity(2)
# symetrization operator
I_sym_abcd = 0.5 * \
    (np.einsum('ac,bd->abcd', delta, delta) +
     np.einsum('ad,bc->abcd', delta, delta))
# expansion tensor
delta23_ab = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=np.float_)

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

    n_m = tr.Property

    def _get_n_m(self):
        return len(self.w_m)

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


class MATSElastic2D(MATS2D):

    def get_corr_pred(self, eps_Emab, deps_Emab, t_n, t_n1,
                      update_state, s_Emg):
        D_Emabef = self.D_abef[None, None, :, :, :, :]
        sig_Emab = np.einsum('...abef,...ef->...ab', D_Emabef, eps_Emab)
        return D_Emabef, sig_Emab


class DOTSGrid(tr.HasStrictTraits):
    '''Domain time steppsr on a grid mesh
    '''
    L_x = tr.Float(200, input=True)
    L_y = tr.Float(100, input=True)
    n_x = tr.Int(100, input=True)
    n_y = tr.Int(30, input=True)
    fets = tr.Instance(FETSEval, input=True)
    mats = tr.Instance(MATS2D, input=True)
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
        n_I, n_a = x_Ia.shape
        dof_Ia = np.arange(n_I * n_a, dtype=np.int_).reshape(n_I, -1)
        I_Ei = self.mesh.I_Ei
        x_Eia = x_Ia[I_Ei, :]
        dof_Eia = dof_Ia[I_Ei]
        x_Ema = np.einsum('im,Eia->Ema', self.fets.N_im, x_Eia)
        J_Emar = np.einsum('imr,Eia->Emar', self.fets.dN_imr, x_Eia)
        J_Enar = np.einsum('inr,Eia->Enar', self.fets.dN_inr, x_Eia)
        det_J_Em = np.linalg.det(J_Emar)
        inv_J_Emar = np.linalg.inv(J_Emar)
        inv_J_Enar = np.linalg.inv(J_Enar)
        B_Eimabc = np.einsum('abcd,imr,Eidr->Eimabc',
                             I_sym_abcd, self.fets.dN_imr, inv_J_Emar)
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

    def get_corr_pred(self, U, dU, t_n, t_n1,
                      update_state, s_Emg):
        '''Get the corrector and predictor for the given increment
        of unknown 
        '''
        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]
        eps_Emab = np.einsum('Eimabc,Eic->Emab', self.B_Eimabc, U_Eia)
        dU_Ia = dU.reshape(-1, n_c)
        dU_Eia = dU_Ia[self.I_Ei]
        deps_Emab = np.einsum('Eimabc,Eic->Emab', self.B_Eimabc, dU_Eia)
        D_Emabef, sig_Emab = self.mats.get_corr_pred(eps_Emab, deps_Emab,
                                                     t_n, t_n1,
                                                     update_state, s_Emg)
        K_Eicjd = np.einsum('Emicjdabef,Emabef->Eicjd',
                            self.BB_Emicjdabef, D_Emabef)
        n_E, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_E = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_E = dots.dof_Eia.reshape(-1, n_i * n_c)
        K_subdomain = SysMtxArray(mtx_arr=K_E, dof_map_arr=dof_E)
        f_Eic = np.einsum('m,Eimabc,Emab,Em->Eic',
                          self.fets.w_m, self.B_Eimabc, sig_Emab,
                          self.det_J_Em)
        f_Ei = f_Eic.reshape(-1, n_i * n_c)
        F_dof = np.bincount(dof_E.flatten(), weights=f_Ei.flatten())
        F_int = F_dof
        norm_F_int = np.linalg.norm(F_int)
        return K_subdomain, F_int, norm_F_int


class TimeLoop(HasStrictTraits):

    tline = tr.Instance(TLine)
    '''Time line object specifying the start, end, time step and current time
    '''

    def _tline_default(self):
        return TLine(min=0.0, max=1.0, step=1.0)

    ts = tr.Instance(DOTSGrid)
    '''State object delivering the predictor and corrector
    '''

    bc_mngr = tr.Instance(BCondMngr, ())
    '''Boundary condition manager
    '''

    bc_list = tr.List([])
    '''List of boundary conditions
    '''

    def _bc_list_changed(self):
        print 'setting boundary conditions'
        self.bc_mngr.bcond_list = self.bc_list

    step_tolerance = tr.Float(1e-8)
    '''Time step tolerance
    '''

    KMAX = tr.Int(300)
    '''Maximum number of iterations
    '''

    tolerance = tr.Float(1e-3)
    '''Tolerance of the residuum norm 
    '''

    t_n1 = tr.Float(0, input=True)
    '''Target time for the next increment
    '''

    t_n = tr.Float(0, input=True)
    '''Time of the last equilibrium state 
    '''

    d_t = tr.Float(0, input=True)
    '''Current time increment size
    '''

    def eval(self):

        update_state = False

        s_Emg = np.zeros((self.ts.mesh.n_active_elems,
                          self.ts.fets.n_m,
                          self.ts.mats.get_state_array_size()), dtype=np.float_)

        print 'setting up boundary conditions'
        tloop.bc_mngr.setup(None)

        K = SysMtxAssembly()
        self.bc_mngr.apply_essential(K)
        U_n = np.zeros((dots.mesh.n_dofs,), dtype=np.float_)
        dU = np.copy(U_n)
        U_k = np.copy(U_n)

        while (self.t_n1 - self.tline.max) <= self.step_tolerance:

            print 'current time %f' % self.t_n1,

            self.d_t = self.tline.step

            k = 0
            step_flag = 'predictor'

            while k < self.KMAX:

                K.reset_mtx()  # zero the stiffness matrix
                K_arr, F_int, n_F_int = self.ts.get_corr_pred(
                    U_k, dU, self.t_n, self.t_n1, update_state, s_Emg)
                if update_state:
                    update_state = False

                K.sys_mtx_arrays.append(K_arr)
                F_int *= -1  # in-place sign change of the internal forces
                self.bc_mngr.apply(step_flag, None, K, F_int,
                                   self.t_n, self.t_n1)
                K.apply_constraints(F_int)
                if n_F_int == 0.0:
                    n_F_int = 1.0
                norm = np.linalg.norm(F_int, ord=None)  # / n_F_int
                if norm < self.tolerance:  # convergence satisfied
                    print 'converged in %d iterations' % (k + 1)
                    update_state = True
                    break  # update_switch -> on
                dU = K.solve()
                U_k += dU
                k += 1
                step_flag = 'corrector'

            U_n = np.copy(U_k)
            self.t_n = self.t_n1
            self.record_response(U_k, self.t_n, s_Emg)
            self.t_n1 = self.t_n + self.d_t
            self.tline.val = min(self.t_n, self.tline.max)

        return U_n

    ug = tr.WeakRef
    write_dir = tr.Directory

    def record_response(self, U, t, s_Emg):
        n_c = self.ts.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[dots.I_Ei]
        eps_Enab = np.einsum('Einabc,Eic->Enab', dots.B_Einabc, U_Eia)
        sig_Enab = np.einsum('abef,Emef->Emab', dots.mats.D_abef, eps_Enab)

        U_vector_field = np.einsum('Ia,ab->Ib',
                                   U_Eia.reshape(-1, n_c), delta23_ab)
        self.ug.point_data.vectors = U_vector_field
        self.ug.point_data.vectors.name = 'displacement'
        eps_Encd = np.einsum('...ab,ac,bd->...cd',
                             eps_Enab, delta23_ab, delta23_ab)
        eps_Encd_tensor_field = eps_Encd.reshape(-1, 9)
        self.ug.point_data.tensors = eps_Encd_tensor_field
        self.ug.point_data.tensors.name = 'strain'
        fname = os.path.join(self.write_dir, 'step_%008.4f' % t)
        write_data(self.ug, fname.replace('.', '_'))


def mlab_view(dataset):
    #     fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
    #                       figure=dataset.class_name[3:])
    engine = mlab.get_engine()
    scene = engine.scenes[0]
    scene.scene.z_plus_view()
    src = mlab.pipeline.add_dataset(dataset)
    warp_vector = mlab.pipeline.warp_vector(src)
    surf = mlab.pipeline.surface(warp_vector)

    etc = ExtractTensorComponents()
    engine.add_filter(etc, warp_vector)
    surface2 = Surface()
    engine.add_filter(surface2, etc)
    etc.filter.scalar_mode = 'component'

    lut = etc.children[0]
    lut.scalar_lut_manager.show_scalar_bar = True
    lut.scalar_lut_manager.show_legend = True
    lut.scalar_lut_manager.scalar_bar.height = 0.8
    lut.scalar_lut_manager.scalar_bar.width = 0.17
    lut.scalar_lut_manager.scalar_bar.position = np.array([0.82,  0.1])


mats2d = MATS2DScalarDamage(
    # stiffness='algorithmic',
    stiffness='secant',
    epsilon_0=0.03,
    epsilon_f=0.5
)
fets2d_4u_4q = FETS2D4u4x()
xdots = DOTSGrid(L_x=600, L_y=100, n_x=51, n_y=10, fets=fets2d_4u_4q,
                 mats=mats2d)
dots = DOTSGrid(L_x=1, L_y=1, n_x=10, n_y=5, fets=fets2d_4u_4q,
                mats=mats2d)
if __name__ == '__main__':
    tloop = TimeLoop(tline=TLine(min=0, max=1, step=0.1),
                     ts=dots)
    if False:
        tloop.bc_list = [BCSlice(slice=dots.mesh[0, :, 0, :],
                                 var='u', dims=[0, 1], value=0),
                         BCSlice(slice=dots.mesh[25, -1, :, -1],
                                 var='u', dims=[1], value=-50),
                         BCSlice(slice=dots.mesh[-1, :, -1, :],
                                 var='u', dims=[0, 1], value=0)
                         ]
    tloop.bc_list = [BCSlice(slice=dots.mesh[0, :, 0, :],
                             var='u', dims=[0], value=0),
                     BCSlice(slice=dots.mesh[0, 0, 0, 0],
                             var='u', dims=[1], value=0),
                     BCSlice(slice=dots.mesh[-1, :, -1, :],
                             var='u', dims=[1], value=0.2)
                     ]

    cell_class = tvtk.Quad().cell_type
    #U = tloop.eval()
    n_c = fets2d_4u_4q.n_nodal_dofs
    n_i = 4
    #eps_Enab = np.einsum('Einabc,Eic->Enab', dots.B_Einabc, U_Eia)
    #sig_Enab = np.einsum('abef,Emef->Emab', dots.mats.D_abef, eps_Enab)
    n_E, n_i, n_a = dots.x_Eia.shape
    n_Ei = n_E * n_i
    points = np.einsum('Ia,ab->Ib', dots.x_Eia.reshape(-1, n_c), delta23_ab)
    ug = tvtk.UnstructuredGrid(points=points)
    ug.set_cells(cell_class, np.arange(n_E * n_i).reshape(n_E, n_i))

    vectors = np.zeros_like(points)
    ug.point_data.vectors = vectors
    ug.point_data.vectors.name = 'displacement'
    warp_arr = tvtk.DoubleArray(name='displacement')
    warp_arr.from_array(vectors)
    ug.point_data.add_array(warp_arr)

    home = os.path.expanduser('~')
    target_dir = os.path.join(home, 'simdb', 'simdata')
    tloop.set(ug=ug, write_dir=target_dir)
    U = tloop.eval()

#     eps_Encd = np.einsum('...ab,ac,bd->...cd',
#                          eps_Enab, delta23_ab, delta23_ab)
#     tensors = eps_Encd.reshape(-1, 9)
#     ug.point_data.tensors = tensors
#     ug.point_data.tensors.name = 'strain'
#
#     mlab_view(ug)
#
#     def save_data():
#         """Save the files"""
#         home = os.path.expanduser('~')
#         target_dir = os.path.join(home, 'simdb', 'simdata')
#         for i, (u, eps) in enumerate(zip(tloop.U_Ib_record,
#                                          tloop.eps_Enab_record)):
#             ug.point_data.vectors = u
#             ug.point_data.tensors = eps
#             write_data(ug, os.path.join(target_dir, 'step_%02d' % i))
#
#     save_data()
