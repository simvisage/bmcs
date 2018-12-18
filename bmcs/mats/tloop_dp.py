'''
Created on 12.01.2016

@author: Yingxiong
'''
from traits.api import \
    Int, HasStrictTraits, Instance, \
    Float, \
    Array, List, Bool

from ibvpy.core.tline import TLine
import numpy as np

from .tstepper_dp import TStepper


class TLoop(HasStrictTraits):

    ts = Instance(TStepper)
    tline = Instance(TLine)
    d_t = Float(0.005)
    t_max = Float(1.0)
    k_max = Int(1000)
    tolerance = Float(1e-4)
    # Tolerance in the time variable to end the iteration.
    step_tolerance = Float(1e-8)

    omega_record = List([])
    kappa_record = List([])
    U_record = Array(dtype=np.float_)
    F_record = Array(dtype=np.float_)
    sf_Em_record = List
    t_record = List
    eps_record = List
    eps_p_record = List
    sig_record = List
    sig_EmC_record = List
    D_record = List

    paused = Bool(False)
    restart = Bool(True)

    def reset_sv_hist(self):
        n_dofs = self.ts.sdomain.n_dofs
        n_e = self.ts.sdomain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = 3
        sig_rec = np.zeros((n_e, n_ip, 2))
        sf_rec = np.zeros((n_e, n_ip))
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        self.omega_record = [np.zeros((n_e, n_ip))]
        self.kappa_record = [np.zeros((n_e, n_ip))]
        self.U_record = np.zeros(n_dofs)
        self.F_record = np.zeros(n_dofs)
        t_n = self.tline.val
        self.t_record = [t_n]
        self.eps_record = [np.zeros_like(eps)]
        self.eps_p_record = [np.zeros_like(eps)]
        self.sig_record = [np.zeros_like(sig)]
        self.sig_EmC_record = [np.copy(sig_rec)]
        self.sf_Em_record = [np.copy(sf_rec)]
        D = np.zeros((n_e, n_ip, n_s, n_s))
        self.D_record = [np.zeros_like(D)]

    def init(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = 0
            self.reset_sv_hist()
            self.restart = False

    def eval(self):

        self.ts.apply_essential_bc()

        self.d_t = self.tline.step
        t_n = self.tline.val
        t_n1 = t_n
        n_dofs = self.ts.sdomain.n_dofs
        n_e = self.ts.sdomain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = 3

        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        D = np.zeros((n_e, n_ip, n_s, n_s))

        xs_pi = np.zeros((n_e, n_ip))
        alpha = np.zeros((n_e, n_ip))
        z = np.zeros((n_e, n_ip))
        omega = np.zeros((n_e, n_ip))
        kappa = np.zeros((n_e, n_ip))

        while (t_n1 - self.tline.max) <= self.step_tolerance and \
                not (self.restart or self.paused):
            k = 0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k <= self.k_max and \
                    not (self.restart or self.paused):

                R, F_int, K, eps, sig, xs_pi, alpha, z, kappa, omega, D, eps_p = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, eps, sig, t_n, t_n1, xs_pi, alpha, z, kappa, omega)

                K.apply_constraints(R)
                d_U_k, pos_def = K.solve()
                d_U += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    self.F_record = np.vstack((self.F_record, F_int))
                    U_k += d_U
                    self.t_record.append(t_n1)
                    self.U_record = np.vstack((self.U_record, U_k))
                    self.eps_record.append(np.copy(eps))
                    eps_p_full = np.zeros_like(eps)
                    eps_p_full[:, :, 1] = eps_p
                    self.eps_p_record.append(np.copy(eps_p_full))
                    self.sig_record.append(np.copy(sig))
                    self.sig_EmC_record.append(sig[:, :, (0, 2)])
                    self.sf_Em_record.append(np.copy(sig[:, :, 1]))
                    self.omega_record.append(np.copy(omega))
                    self.kappa_record.append(np.copy(kappa))
                    self.D_record.append(np.copy(D))
                    break
                k += 1
                step_flag = 'corrector'

            if k >= self.k_max:
                print(' ----------> No convergence for the time step %f', t_n1)
                break
            if self.restart or self.paused:
                print('interrupted iteration')
                break
            t_n = t_n1
            t_n1 = t_n + self.d_t
            self.tline.val = min(t_n, self.tline.max)

        return

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        x = self.t_record
        idx = np.array(np.arange(len(x)), dtype=np.float_)
        t_idx = np.interp(vot, x, idx)
        return np.array(t_idx + 0.5, np.int_)

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))


if __name__ == '__main__':

    from ibvpy.api import BCDof

    ts = TStepper()

    n_dofs = ts.sdomain.n_dofs
    ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                  BCDof(var='f', dof=n_dofs - 1, value=5)]

    tl = TLoop(ts=ts)

    tl.eval()
