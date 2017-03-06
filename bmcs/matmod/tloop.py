'''
Created on 12.01.2016

@author: Yingxiong
'''
from traits.api import Int, HasTraits, Instance, \
    Float

import numpy as np
from tstepper import TStepper


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.005)
    t_max = Float(1.0)
    k_max = Int(1000)
    tolerance = Float(1e-4)

    def eval(self):

        self.ts.apply_essential_bc()

        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = 3

        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        D = np.zeros((n_e, n_ip, 3, 3))

        xs_pi = np.zeros((n_e, n_ip))
        alpha = np.zeros((n_e, n_ip))
        z = np.zeros((n_e, n_ip))
        w = np.zeros((n_e, n_ip))
        w_last = np.zeros((n_e, n_ip))
        w_record = [np.zeros((n_e, n_ip))]

        U_record = np.zeros(n_dofs)
        F_record = np.zeros(n_dofs)
        sf_record = np.zeros(2 * n_e)
        t_record = [t_n]
        eps_record = [np.zeros_like(eps)]
        sig_record = [np.zeros_like(sig)]
        D_record = [np.zeros_like(D)]

        while t_n1 <= self.t_max - self.d_t:
            t_n1 = t_n + self.d_t
            k = 0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k <= self.k_max:
                # if k == self.k_max:  # handling non-convergence
                #                     scale *= 0.5
                # print scale
                #                     t_n1 = t_n + scale * self.d_t
                #                     k = 0
                #                     d_U = np.zeros(n_dofs)
                #                     d_U_k = np.zeros(n_dofs)
                #                     step_flag = 'predictor'
                #                     eps = eps_r
                #                     sig = sig_r
                #                     alpha = alpha_r
                #                     q = q_r
                #                     kappa = kappa_r

                R, F_int, K, eps, sig, xs_pi, alpha, z, w, D = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, eps, sig, t_n, t_n1, xs_pi, alpha, z, w)

                K.apply_constraints(R)
                # print 'r=', np.linalg.norm(R)
                d_U_k = K.solve()
                d_U += d_U_k
#                 print 'r', np.linalg.norm(R)
                if np.linalg.norm(R) < self.tolerance:
                    F_record = np.vstack((F_record, F_int))
                    U_k += d_U
                    U_record = np.vstack((U_record, U_k))
                    sf_record = np.vstack((sf_record, sig[:, :, 1].flatten()))
                    eps_record.append(np.copy(eps))
                    sig_record.append(np.copy(sig))
                    t_record.append(t_n1)
                    w_record.append(np.copy(w))
                    w_last = w
                    D_record.append(np.copy(D))
                    # print 'eps=',eps
                    # print'D=', D
                    break
                k += 1
                step_flag = 'corrector'


            if k >= self.k_max:
                print ' ----------> No Convergence any more'
                break
            print t_n1
            print 'K=', k
            t_n = t_n1
            # for i in range(1 ,len(D_record)):
            # print'D=', D_record#.flatten()#[-1:-1][-1, :, 1 , 1].flatten()

            # print'K=',k
            # print'D_record=',D_record
        return (U_record, F_record, sf_record, np.array(t_record),
                eps_record, sig_record, w_record, D_record)

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from ibvpy.api import BCDof

    ts = TStepper()

    n_dofs = ts.domain.n_dofs
    # print'n_dofs', n_dofs

#     tf = lambda t: 1 - np.abs(t - 1)
#     ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
# BCDof(var='u', dof=n_dofs - 1, value=2.5, time_function=tf)]
    #load = np.array([0,3,0,2])
    #U = np.zeros(n_dofs)
    #F = np.zeros(n_dofs)

    # for i in range(0, len(load)):
    ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                  BCDof(var='f', dof=n_dofs - 1, value=5)]

    tl = TLoop(ts=ts)

    (U_record, F_record, sf_record, t_record,
     eps_record, sig_record,  w_record, D_record) = tl.eval()
    #U_record = np.vstack((U, U_record))
    #F_record = np.vstack((F, F_record))

    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof] * 2, F_record[:, n_dof] / 1000., marker='.')
    plt.xlabel('displacement')
    plt.ylabel('force')
    plt.show()
