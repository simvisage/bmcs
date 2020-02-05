import sys

from ibvpy.api import BCDof
from ibvpy.fets.fets_eval import FETSEval, IFETSEval
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from scipy.interpolate import interp1d
from traits.api import provides, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List

import matplotlib.pyplot as plt
import numpy as np


class MATSEval(HasTraits):

    '''for monotonic pull-out response'''

    E_m = Float(28484., tooltip='Stiffness of the matrix',
                auto_set=False, enter_set=False)

    E_f = Float(170000., tooltip='Stiffness of the fiber',
                auto_set=False, enter_set=False)

    slip = List
    bond = List

    def b_s_law(self, x):
        return np.sign(x) * np.interp(np.abs(x), self.slip, self.bond)

    def G(self, x):
        d = np.diff(self.bond) / np.diff(self.slip)
        d = np.append(d, d[-1])
        G = interp1d(
            np.array(self.slip), d, kind='zero', fill_value=(0, 0), bounds_error=False)
#         y = np.zeros_like(x)
#         y[x < self.slip[0]] = d[0]
#         y[x > self.slip[-1]] = d[-1]
#         x[x < self.slip[0]] = self.slip[-1] + 10000.
#         y[x <= self.slip[-1]] = G(x[x <= self.slip[-1]])
#         return np.sign(x) * y
        return G(np.abs(x))

    n_e_x = Float

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1):
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        D[:, :, 1, 1] = self.G(eps[:, :, 1])

        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig
        sig[:, :, 1] = self.b_s_law(eps[:, :, 1])

        return sig, D

    n_s = Constant(3)


@provides(IFETSEval)
class FETS1D52ULRH(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    debug_on = True

    A_m = Float(120. * 13. - 9. * 1.85, desc='matrix area [mm2]')
    A_f = Float(9. * 1.85, desc='reinforcement area [mm2]')
    L_b = Float(1., desc='perimeter of the bond interface [mm]')

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_nodal_dofs = Int(2)

    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    n_dof_r = Property
    '''Number of node positions associated with degrees of freedom. 
    '''
    @cached_property
    def _get_n_dof_r(self):
        return len(self.dof_r)

    n_e_dofs = Property
    '''Number of element degrees
    '''
    @cached_property
    def _get_n_dofs(self):
        return self.n_nodal_dofs * self.n_dof_r

    def _get_ip_coords(self):
        offset = 1e-6
        return np.array([[-1 + offset, 0., 0.], [1 - offset, 0., 0.]])

    def _get_ip_weights(self):
        return np.array([1., 1.], dtype=float)

    # Integration parameters
    #
    ngp_r = 2

    def get_N_geo_mtx(self, r_pnt):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        r = r_pnt[0]
        N_mtx = np.array([[0.5 - r / 2., 0.5 + r / 2.]])
        return N_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.
        '''
        return np.array([[-1. / 2, 1. / 2]])

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        return self.get_N_geo_mtx(r_pnt)

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)


class TStepper(HasTraits):

    '''Time stepper object for non-linear Newton-Raphson solver.
    '''

    mats_eval = Instance(MATSEval, arg=(), kw={})  # material model

    fets_eval = Instance(FETS1D52ULRH, arg=(), kw={})  # element formulation

    A = Property(depends_on='fets_eval.A_f, fets_eval.A_m, fets_eval.L_b')
    '''array containing the A_m, L_b, A_f
    '''
    @cached_property
    def _get_A(self):
        return np.array([self.fets_eval.A_m, self.fets_eval.L_b, self.fets_eval.A_f])

    # number of elements
    n_e_x = Float(20.)

    # specimen length
    L_x = Float(75.)

    domain = Property(depends_on='n_e_x, L_x')
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Element definition
        domain = FEGrid(coord_max=(self.L_x,),
                        shape=(self.n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

    bc_list = List(Instance(BCDof))

    J_mtx = Property(depends_on='n_e_x, L_x')
    '''Array of Jacobian matrices.
    '''
    @cached_property
    def _get_J_mtx(self):
        fets_eval = self.fets_eval
        domain = self.domain
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)
        # [ n_e, n_geo_r, n_dim_geo ]
        elem_x_map = domain.elem_X_map
        # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
        J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
        return J_mtx

    J_det = Property(depends_on='n_e_x, L_x')
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property(depends_on='n_e_x, L_x')
    '''The B matrix
    '''
    @cached_property
    def _get_B(self):
        '''Calculate and assemble the system stiffness matrix.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain

        n_s = mats_eval.n_s

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_ip = fets_eval.n_gp
        n_e = domain.n_active_elems
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        J_inv = np.linalg.inv(self.J_mtx)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None, :])
        dNr = 0.5 * geo_r[:, :, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('eidf,inf->eind', J_inv, dNr)

        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :, :, B_N_n_rows, B_N_n_cols] = (B_factors[None, None, :] *
                                              Nx[:, :, N_idx])
        B[:, :, :, B_dN_n_rows, B_dN_n_cols] = dNx[:, :, :, dN_idx]

        return B

    def apply_essential_bc(self):
        '''Insert initial boundary conditions at the start up of the calculation.. 
        '''
        self.K = SysMtxAssembly()
        for bc in self.bc_list:
            bc.apply_essential(self.K)

    def apply_bc(self, step_flag, K_mtx, F_ext, t_n, t_n1):
        '''Apply boundary conditions for the current load increement
        '''
        for bc in self.bc_list:
            bc.apply(step_flag, None, K_mtx, F_ext, t_n, t_n1)

    def get_corr_pred(self, step_flag, d_U, eps, sig, t_n, t_n1):
        '''Function calculationg the residuum and tangent operator.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain
        elem_dof_map = domain.elem_dof_map
        n_e = domain.n_active_elems
        n_dof_r, n_dim_dof = self.fets_eval.dof_r.shape
        n_nodal_dofs = self.fets_eval.n_nodal_dofs
        n_el_dofs = n_dof_r * n_nodal_dofs
        # [ i ]
        w_ip = fets_eval.ip_weights

        d_u_e = d_U[elem_dof_map]
        #[n_e, n_dof_r, n_dim_dof]
        d_u_n = d_u_e.reshape(n_e, n_dof_r, n_nodal_dofs)
        #[n_e, n_ip, n_s]
        d_eps = np.einsum('einsd,end->eis', self.B, d_u_n)

        # update strain
        eps += d_eps

        # material response state variables at integration point
        sig, D = mats_eval.get_corr_pred(eps, d_eps, sig, t_n, t_n1)

        # system matrix
        self.K.reset_mtx()
        Ke = np.einsum('i,s,einsd,eist,eimtf,ei->endmf',
                       w_ip, self.A, self.B, D, self.B, self.J_det)

        self.K.add_mtx_array(
            Ke.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)

        # internal forces
        # [n_e, n_n, n_dim_dof]
        Fe_int = np.einsum('i,s,eis,einsd,ei->end',
                           w_ip, self.A, sig, self.B, self.J_det)
        F_int = -np.bincount(elem_dof_map.flatten(), weights=Fe_int.flatten())
        self.apply_bc(step_flag, self.K, F_int, t_n, t_n1)
        return F_int, self.K, eps, sig


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.01)
    t_max = Float(1.0)
    k_max = Int(200)
    tolerance = Float(1e-4)

    def eval(self):

        self.ts.apply_essential_bc()

        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s
        U_record = np.zeros(n_dofs)
        F_record = np.zeros(n_dofs)
        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))

        sf_record = np.zeros(2 * n_e)  # shear flow

        sig_f_record = np.zeros(2 * n_e)
        sig_m_record = np.zeros(2 * n_e)

        while t_n1 <= self.t_max:
            t_n1 = t_n + self.d_t
            k = 0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k < self.k_max:
                R, K, eps, sig = self.ts.get_corr_pred(
                    step_flag, d_U_k, eps, sig, t_n, t_n1)

                F_ext = -R
                K.apply_constraints(R)
                d_U_k, _ = K.solve()
                d_U += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    F_record = np.vstack((F_record, F_ext))
                    U_k += d_U
                    U_record = np.vstack((U_record, U_k))
                    sf_record = np.vstack((sf_record, sig[:, :, 1].flatten()))

                    sig_m_record = np.vstack(
                        (sig_m_record, sig[:, :, 0].flatten()))
                    sig_f_record = np.vstack(
                        (sig_f_record, sig[:, :, 2].flatten()))

                    break
                k += 1
                if k == self.k_max:
                    print(self.ts.mats_eval.bond)
                    print('nonconvergence')
                step_flag = 'corrector'

            t_n = t_n1
        return U_record, F_record, sf_record, sig_m_record, sig_f_record


if __name__ == '__main__':

    #=========================================================================
    # nonlinear solver
    #=========================================================================
    # initialization

    ts = TStepper()

    ts.L_x = 1

    ts.n_e_x = 20

#     ts.mats_eval.slip = [0.0, 0.09375, 0.505, 0.90172413793103456, 1.2506896551724138, 1.5996551724137933, 1.9486206896551728, 2.2975862068965522, 2.6465517241379315,
#                          2.9955172413793107, 3.34448275862069, 3.6934482758620693, 4.0424137931034485, 4.3913793103448278, 4.7403448275862079, 5.0893103448275863, 5.4382758620689664, 5.7000000000000002]
#
#     ts.mats_eval.bond = [0.0, 43.05618551913318, 40.888629416715574, 49.321970730383285, 56.158143245133338, 62.245706611484323, 68.251000923721875, 73.545464379399633, 79.032738465995692,
# 84.188949455670524, 87.531858162376921, 91.532666285021264,
# 96.66808302759236, 100.23305856244875, 103.01090365681807,
# 103.98920712455558, 104.69444418370917, 105.09318577617957]

    ts.mats_eval.slip = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ts.mats_eval.bond = [0., 10., 30., 35., 20., 10.]

    n_dofs = ts.domain.n_dofs

#     tf = lambda t: 1 - np.abs(t - 1)

    ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=0.5)]

    tl = TLoop(ts=ts)
#
#     U_record, F_record, sf_record, sig_m_record, sig_f_record = tl.eval()
#     n_dof = 2 * ts.domain.n_active_elems + 1
#     plt.plot(U_record[:, n_dof], F_record[:, n_dof],
#              marker='.', label='numerical')

    ts.L_x = 200
    U_record, F_record, sf_record, sig_m_record, sig_f_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof], label='loaded')
    plt.plot(U_record[:, 1], F_record[:, n_dof], label='free')

    np.savetxt('D:\\loaded.txt', np.vstack((
        U_record[:, n_dofs - 1], F_record[:, n_dofs - 1])))

    np.savetxt('D:\\free.txt',  np.vstack((
        U_record[:, 1], F_record[:, n_dofs - 1])))

    plt.xlabel('displacement [mm]')
    plt.ylabel('pull-out force [N]')
#     plt.ylim(0, 20000)
    plt.legend(loc='best')

    plt.show()
