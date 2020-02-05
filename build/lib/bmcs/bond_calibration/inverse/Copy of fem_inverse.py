'''
Created on 19.01.2016

@author: Yingxiong
'''
from envisage.ui.workbench.api import WorkbenchApplication
from mayavi.sources.api import VTKDataSource, VTKFileReader
from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
from ibvpy.api import BCDof
from ibvpy.fets.fets_eval import FETSEval, IFETSEval
from ibvpy.mats.mats1D import MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import MATS1D5Bond
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.optimize import newton, brentq, bisect, minimize_scalar


class MATSEval(HasTraits):

    E_m = Float(10, tooltip='Stiffness of the matrix',
                auto_set=False, enter_set=False)

    E_f = Float(10, tooltip='Stiffness of the fiber',
                auto_set=False, enter_set=False)

    slip = List([0.])

    bond = List([0.])

#     b_s_law = Property
#
#     def _get_b_s_law(self):
#         return interp1d(self.slip, self.bond)
# return np.interp(self.slip, self.bond)
    def b_s_law(self, x):
        return np.interp(x, self.slip, self.bond)

    def G(self, x):
        d = np.diff(self.bond) / np.diff(self.slip)
        d = np.append(d, d[-1])
        G = interp1d(np.array(self.slip), d, kind='zero')
        y = np.zeros_like(x)
        y[x < self.slip[0]] = d[0]
        y[x > self.slip[-1]] = d[-1]
        x[x < self.slip[0]] = self.slip[-1] + 10000.
        y[x <= self.slip[-1]] = G(x[x <= self.slip[-1]])
        return y
#     G = Property
#
#     def _get_G(self):
#         d = np.diff(self.bond) / np.diff(self.slip)
#         d = np.append(d, np.nan)
#         return interp1d(self.slip, d, kind='zero')

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1):
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

#         d = np.diff(self.bond) / np.diff(self.slip)
#         d = np.append(d, np.nan)

#         G = interp1d(np.array(self.slip) * (1. + 1e-8), d, kind='zero')
#         print d
#         print G(np.array([[0.035, 0.0035], [0.0035, 0.0035]]))
#         print self.slip
#         a = eps[:,:, 1]
#         print np.amax(a)
#         try:
        D[:, :, 1, 1] = self.G(eps[:,:, 1])
#         except:
#             print np.array(self.slip) * (1. + 1e-4)
#             print eps[:, :, 1]
#             sys.exit()
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

        sig[:, :, 1] = self.b_s_law(eps[:,:, 1])
        return sig, D

    n_s = Constant(3)


class FETS1D52ULRH(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    implements(IFETSEval)

    debug_on = True

    A_m = Float(1., desc='matrix area [mm2]')
    A_f = Float(1., desc='reinforcement area [mm2]')
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

    mats_eval = Property(Instance(MATSEval))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_mats_eval(self):
        return MATSEval()

    fets_eval = Property(Instance(FETS1D52ULRH))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRH()

    A = Property()
    '''array containing the A_m, L_b, A_f
    '''
    @cached_property
    def _get_A(self):
        return np.array([self.fets_eval.A_m, self.fets_eval.L_b, self.fets_eval.A_f])

    domain = Property(Instance(FEGrid))
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Number of elementsx
        n_e_x = 4
        # length
        L_x = 40.0
        # Element definition
        domain = FEGrid(coord_max=(L_x,),
                        shape=(n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

    bc_list = List(Instance(BCDof))

    J_mtx = Property
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

    J_det = Property
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property
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
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None,:])
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
        B[:, :,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                              Nx[:, :, N_idx])
        B[:, :,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]

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
#         if np.any(sig) == np.nan:
#             sys.exit()

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
    tolerance = Float(1e-8)
    w_arr = Array
    pf_arr = Array

    def pf(self, tau_i, eps, sig, t_n, d_t):
        '''evaluate the pull-out force according to tau_i
        '''
        eps_temp = np.copy(eps)
        sig_temp = np.copy(sig)
        step_flag = 'predictor'
        d_U_k = np.zeros(n_dofs)
        self.ts.mats_eval.bond[-1] = tau_i
        k = 0
        while k < self.k_max:
            R, K, eps_temp, sig_temp = ts.get_corr_pred(
                step_flag, d_U_k, eps_temp, sig_temp, t_n, t_n + d_t)
            F_ext = -R
            K.apply_constraints(R)
            d_U_k = K.solve()
            k += 1
            if k == self.k_max:
                print('pf non-convergence')
            step_flag = 'corrector'
            if np.linalg.norm(R) < self.tolerance:
                return F_ext[-1]

    def regularization(self, eps, sig, t_n, d_t, i):
        '''regularization
        '''
        eps_temp = np.copy(eps)
        sig_temp = np.copy(sig)

#         tau_i = 0.
#
#         pf = [0.5 * (self.pf_arr[i - 1] + self.pf_arr[i]),
#               self.pf_arr[i], 0.5 * (self.pf_arr[i] + self.pf_arr[i + 1])]
#
#         for k, j in enumerate([0.5, 1.0, 1.5]):
#             dw = self.w_arr[i] - self.w_arr[i - 1]
#
#             self.ts.mats_eval.slip.append(self.w_arr[i - 1] + j * dw)
#             self.ts.mats_eval.bond.append(0.)
#
#             print j
#
#             tau = lambda tau_i: self.pf(
#                 tau_i, eps_temp, sig_temp, t_n + (j - 0.5) * self.d_t, 0.5 * self.d_t) - pf[k]
#             tau_i += brentq(tau, 0., 6., xtol=1e-16)
#
#             eps_temp, sig_temp = self.update_eps_sig(
#                 eps_temp, sig_temp, t_n + (j - 0.5) * self.d_t, 0.5 * self.d_t)
#
#         del self.ts.mats_eval.slip[-3:]
#         del self.ts.mats_eval.bond[-3:]
#
#         return tau_i / 3.
        self.ts.mats_eval.slip.append(self.w_arr[i])
        self.ts.mats_eval.bond.append(0.)

        tau = lambda tau_i: self.pf(
            tau_i, eps_temp, sig_temp, t_n, self.d_t) - self.pf_arr[i]

#         print tau(0.)
#         print tau(6.)
        try:
            tau_i = brentq(tau, 0., 20., xtol=1e-16)
        except:
            tau_i = 0
        return tau_i

    def update_eps_sig(self, eps, sig, t_n, t_n1):
        step_flag = 'predictor'
        d_U_k = np.zeros(n_dofs)
        k = 0
        while k < self.k_max:
            R, K, eps, sig = self.ts.get_corr_pred(
                step_flag, d_U_k, eps, sig, t_n, t_n1)
            F_ext = -R
            K.apply_constraints(R)
            d_U_k = K.solve()
            k += 1
            step_flag = 'corrector'
            if np.linalg.norm(R) < self.tolerance:
                return eps, sig

    def eval(self):

        self.ts.apply_essential_bc()

        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s

        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        i = 0.

        while t_n1 <= self.t_max:
            i += 1.
            print(i)
            t_n1 = t_n + self.d_t

#             self.ts.mats_eval.slip.append(self.w_arr[i])
#             self.ts.mats_eval.bond.append(0.)
#             t_n1 = t_n + self.d_t
#
#             tau = lambda tau_i: self.pf(
#                 tau_i, eps, sig, t_n, self.d_t) - self.pf_arr[i]
#
#             tau_i = brentq(tau, 0., 6., xtol=1e-16)
#
#             print tau_i

            tau_i = self.regularization(eps, sig, t_n, self.d_t, i)

            print(tau_i)

#             self.ts.mats_eval.slip.append(self.w_arr[i])
#             self.ts.mats_eval.bond.append(tau_i)
            eps, sig = self.update_eps_sig(eps, sig, t_n, t_n1)

#             step_flag = 'predictor'
#             d_U_k = np.zeros(n_dofs)
# self.ts.mats_eval.bond[-1] = tau_i
#             k = 0
#             while k < self.k_max:
#                 R, K, eps, sig = ts.get_corr_pred(
#                     step_flag, d_U_k, eps, sig, t_n, t_n1)
#                 F_ext = -R
#                 K.apply_constraints(R)
#                 d_U_k = K.solve()
# d_U += d_U_k
#                 if np.linalg.norm(R) < self.tolerance:
#                     print F_ext[-1]
#                     print self.pf_arr[i]
#                     print '===='
#                     break
#                 k += 1
#                 step_flag = 'corrector'

            # regularization
#             if i % 2.0 == 0.:
# print i
# print self.ts.mats_eval.slip[-3:]
#                 b_avg = np.mean(self.ts.mats_eval.bond[-2:])
#                 s_avg = np.mean(self.ts.mats_eval.slip[-2:])
#                 del self.ts.mats_eval.bond[-2:]
#                 del self.ts.mats_eval.slip[-2:]
#                 self.ts.mats_eval.bond.append(b_avg)
#                 self.ts.mats_eval.slip.append(s_avg)

            t_n = t_n1
        return self.ts.mats_eval.slip, self.ts.mats_eval.bond

if __name__ == '__main__':

    #=========================================================================
    # nonlinear solver
    #=========================================================================
    # initialization

    ts = TStepper()

    n_dofs = ts.domain.n_dofs

    tf = lambda t: 1 - np.abs(t - 1)

    ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=3.0)]

    w_arr, pf_arr = np.loadtxt('D:\\1.txt')
#     w_arr, pf_arr = np.loadtxt('D:\\1.txt', delimiter=',').T

    intep = interp1d(w_arr, pf_arr)

    w_arr = np.linspace(0, 3.0, 101)

    pf_arr = intep(w_arr)

    tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr)

    slip, bond = tl.eval()

    plt.plot(slip, bond, label='solved')
    x = np.linspace(0., 3.0, 1000)
    y = np.zeros_like(x)
    y[x < 1.05] = 0.1 * x[x < 1.05] - 0.05 * x[x < 1.05] ** 2
    y[x > 1.05] = 0.1 * 1.05 - 0.05 * \
        1.05 ** 2 - 0.005 * (x[x > 1.05] - 1.05)

#     y[x < 1.01] = 1. * x[x < 1.01] - 0.5 * x[x < 1.01] ** 2
#     y[x > 1.01] = 1. * 1.01 - 0.5 * \
#         1.01 ** 2 - 0.01 * (x[x > 1.01] - 1.01)
#
    plt.plot(x, y, 'k--', label='original')
    plt.xlabel('slip')
    plt.ylabel('bond')
    plt.legend()
    plt.show()
