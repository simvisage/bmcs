
'''
Created on Jan 24, 2018

This script demonstrates the looples implementation
of the finite element code for multiphase continuum.
Example (2D discretization)

@author: rch, abaktheer
'''
import os

from ibvpy.api import \
    FEGrid, FETSEval, TLine, BCSlice
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.fets import \
    FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2D, MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage
from mathkit.matrix_la import \
    SysMtxArray, SysMtxAssembly
from mayavi import mlab
from mayavi.filters.api import ExtractTensorComponents
from traits.has_traits import HasStrictTraits
from tvtk.api import \
    tvtk, write_data

import numpy as np
import sympy as sp
import traits.api as tr
from tvtk.tvtk_classes import tvtk_helper


delta = np.identity(2)
# symetrization operator
I_sym_abcd = 0.5 * \
    (np.einsum('ac,bd->abcd', delta, delta) +
     np.einsum('ad,bc->abcd', delta, delta))
# expansion tensor
delta23_ab = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=np.float_)


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
        x_Ema = np.einsum(
            'im,Eia->Ema', self.fets.N_im, x_Eia
        )
        J_Emar = np.einsum(
            'imr,Eia->Emar', self.fets.dN_imr, x_Eia
        )
        J_Enar = np.einsum(
            'inr,Eia->Enar', self.fets.dN_inr, x_Eia
        )
        det_J_Em = np.linalg.det(J_Emar)
        inv_J_Emar = np.linalg.inv(J_Emar)
        inv_J_Enar = np.linalg.inv(J_Enar)
        B_Eimabc = np.einsum(
            'abcd,imr,Eidr->Eimabc', I_sym_abcd, self.fets.dN_imr, inv_J_Emar
        )
        B_Einabc = np.einsum(
            'abcd,inr,Eidr->Einabc', I_sym_abcd, self.fets.dN_inr, inv_J_Enar
        )
        BB_Emicjdabef = np.einsum(
            'Eimabc,Ejmefd, Em, m->Emicjdabef', B_Eimabc, B_Eimabc,
            det_J_Em, self.fets.w_m
        )
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
    '''
    '''

    def _get_x_Eia(self):
        return self.cached_grid_values[3]

    dof_Ia = tr.Property()
    '''
    '''

    def _get_dof_Ia(self):
        return self.cached_grid_values[4]

    I_Ei = tr.Property()
    '''
    '''

    def _get_I_Ei(self):
        return self.cached_grid_values[5]

    B_Einabc = tr.Property()
    '''
    '''

    def _get_B_Einabc(self):
        return self.cached_grid_values[6]

    det_J_Em = tr.Property()
    '''
    '''

    def _get_det_J_Em(self):
        return self.cached_grid_values[7]

    state_arrays = tr.Property(tr.Dict(tr.Str, tr.Array),
                               depends_on='fets, mats')
    '''Dictionary of state arrays.
    The entry names and shapes are defined by the material
    model.
    '''
    @tr.cached_property
    def _get_state_arrays(self):
        return {
            name: np.zeros((self.mesh.n_active_elems, self.fets.n_m,)
                           + mats_sa_shape, dtype=np.float_)
            for name, mats_sa_shape
            in self.mats.state_array_shapes.items()
        }

    def get_corr_pred(self, U, dU, t_n, t_n1, update_state):
        '''Get the corrector and predictor for the given increment
        of unknown .
        '''
        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab', self.B_Eimabc, U_Eia
        )
        dU_Ia = dU.reshape(-1, n_c)
        dU_Eia = dU_Ia[self.I_Ei]
        deps_Emab = np.einsum(
            'Eimabc,Eic->Emab', self.B_Eimabc, dU_Eia
        )
        D_Emabef, sig_Emab = self.mats.get_corr_pred(
            eps_Emab, deps_Emab, t_n, t_n1, update_state,
            **self.state_arrays
        )
        K_Eicjd = np.einsum(
            'Emicjdabef,Emabef->Eicjd', self.BB_Emicjdabef, D_Emabef
        )
        n_E, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_E = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_E = dots.dof_Eia.reshape(-1, n_i * n_c)
        K_subdomain = SysMtxArray(mtx_arr=K_E, dof_map_arr=dof_E)
        f_Eic = np.einsum(
            'm,Eimabc,Emab,Em->Eic', self.fets.w_m, self.B_Eimabc, sig_Emab,
            self.det_J_Em
        )
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
    '''Boundary condition manager.
    '''

    bc_list = tr.List([])
    '''List of boundary conditions.
    '''

    def _bc_list_changed(self):
        self.bc_mngr.bcond_list = self.bc_list

    step_tolerance = tr.Float(1e-8)
    '''Time step tolerance.
    '''

    KMAX = tr.Int(300)
    '''Maximum number of iterations.
    '''

    tolerance = tr.Float(1e-3)
    '''Tolerance of the residuum norm. 
    '''

    t_n1 = tr.Float(0, input=True)
    '''Target time for the next increment.
    '''

    t_n = tr.Float(0, input=True)
    '''Time of the last equilibrium state. 
    '''

    d_t = tr.Float(0, input=True)
    '''Current time increment size.
    '''

    def eval(self):

        update_state = False

        tloop.bc_mngr.setup(None)

        K = SysMtxAssembly()
        self.bc_mngr.apply_essential(K)
        U_n = np.zeros((dots.mesh.n_dofs,), dtype=np.float_)
        dU = np.copy(U_n)
        U_k = np.copy(U_n)
        F_ext = np.zeros_like(U_n)

        while (self.t_n1 - self.tline.max) <= self.step_tolerance:

            print 'current time %f' % self.t_n1,

            self.d_t = self.tline.step

            k = 0
            step_flag = 'predictor'

            while k < self.KMAX:

                K.reset_mtx()
                K_arr, F_int, n_F_int = self.ts.get_corr_pred(
                    U_k, dU, self.t_n, self.t_n1, update_state
                )
                if update_state:
                    update_state = False

                K.sys_mtx_arrays.append(K_arr)

                F_ext[:] = 0.0
                self.bc_mngr.apply(step_flag, None, K, F_ext,
                                   self.t_n, self.t_n1)
                R = F_ext - F_int
                K.apply_constraints(R)
                if n_F_int == 0.0:
                    n_F_int = 1.0
                norm = np.linalg.norm(R, ord=None)  # / n_F_int
                if norm < self.tolerance:  # convergence satisfied
                    print 'converged in %d iterations' % (k + 1)
                    update_state = True
                    self.F_int_record.append(F_int)
                    self.U_record.append(np.copy(U_k))
                    break  # update_switch -> on
                dU = K.solve()
                U_k += dU
                k += 1
                step_flag = 'corrector'

            U_n = np.copy(U_k)
            self.t_n = self.t_n1
            self.record_response(U_k, self.t_n)
            self.t_n1 = self.t_n + self.d_t
            self.tline.val = min(self.t_n, self.tline.max)

        return U_n

    ug = tr.WeakRef
    write_dir = tr.Directory
    F_int_record = tr.List(tr.Array(np.float_))
    U_record = tr.List(tr.Array(np.float_))
    F_ext_record = tr.List(tr.Array(np.float_))
    t_record = tr.List(np.float_)
    record_dofs = tr.Array(np.int_)

    def record_response(self, U, t):
        n_c = self.ts.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[dots.I_Ei]
        eps_Enab = np.einsum('Einabc,Eic->Enab', dots.B_Einabc, U_Eia)
        sig_Enab = np.einsum('abef,Emef->Emab', dots.mats.D_abcd, eps_Enab)

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
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                      figure=dataset.class_name[3:])
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


mats2d = MATS2DMplDamageEEQ(
    # stiffness='secant',
    epsilon_0=0.03,
    epsilon_f=1.9 * 1000
)


mats2d = MATS2DScalarDamage(
    stiffness='algorithmic',
    epsilon_0=0.03,
    epsilon_f=1.9 * 1000
)

mats2d = MATS2DElastic(
)

fets2d_4u_4q = FETS2D4Q()
dots = DOTSGrid(L_x=600, L_y=100, n_x=51, n_y=10,
                fets=fets2d_4u_4q, mats=mats2d)
xdots = DOTSGrid(L_x=4, L_y=1, n_x=40, n_y=10,
                 fets=fets2d_4u_4q, mats=mats2d)
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
    if True:
        tloop.bc_list = [BCSlice(slice=dots.mesh[0, 0, 0, 0],
                                 var='u', dims=[1], value=0),
                         BCSlice(slice=dots.mesh[-1, 0, -1, 0],
                                 var='u', dims=[1], value=0),
                         BCSlice(slice=dots.mesh[25, -1, :, -1],
                                 var='u', dims=[1], value=-50),
                         BCSlice(slice=dots.mesh[25, -1, :, -1],
                                 var='u', dims=[0], value=0)
                         ]
    if False:
        tloop.bc_list = [BCSlice(slice=dots.mesh[0, :, 0, :],
                                 var='u', dims=[0], value=0),
                         BCSlice(slice=dots.mesh[0, 0, 0, 0],
                                 var='u', dims=[1], value=0),
                         BCSlice(slice=dots.mesh[-1, :, -1, :],
                                 var='u', dims=[1], value=0.01)
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

    record_dofs = dots.mesh[25, -1, :, -1].dofs[:, :, 1].flatten()
    Fd_int_t = np.array(tloop.F_int_record)
    Ud_t = np.array(tloop.U_record)
    import pylab as p
    F_int_t = -np.sum(Fd_int_t[:, record_dofs], axis=1)
    U_t = -Ud_t[:, record_dofs[0]]
    t_arr = np.array(tloop.t_record, dtype=np.float_)
    p.plot(U_t, F_int_t)
    p.show()
