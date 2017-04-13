'''
Created on 12.01.2016

@author: Yingxiong
'''
from traits.api import Int, HasStrictTraits, Instance, \
    Float, Property, property_depends_on, Callable, \
    Array, List
from traitsui.api import View, Item
from view.ui import BMCSLeafNode

import numpy as np
from tstepper import TStepper


class TLine(BMCSLeafNode):

    '''
    Time line for the control parameter.

    This class sets the time-range of the computation - the start and stop time.
    val is the value of the current time.

    TODO - the info page including the number of load steps
    and estimated computation time.

    TODO - the slide bar is not read-only. How to include a real progress bar?
    '''
    node_name = 'time range'

    min = Float(0.0)
    max = Float(1.0)
    step = Float(0.1)
    val = Float(0.0)

    def _val_changed(self):
        if self.time_change_notifier:
            self.time_change_notifier(self.val)

    time_change_notifier = Callable

    tree_view = View(
        Item('min'),
        Item('max'),
        Item('step'),
        Item('val', style='readonly')
    )


class TLoop(HasStrictTraits):

    ts = Instance(TStepper)
    tline = Instance(TLine)
    d_t = Float(0.005)
    t_max = Float(1.0)
    k_max = Int(1000)
    tolerance = Float(1e-4)
    # Tolerance in the time variable to end the iteration.
    step_tolerance = Float(1e-8)

    w_record = List([])
    U_record = Array(dtype=np.float_)
    F_record = Array(dtype=np.float_)
    sf_record = Array(dtype=np.float_)
    t_record = List
    eps_record = List
    sig_record = List
    D_record = List

    def eval(self):

        self.ts.apply_essential_bc()

        self.d_t = self.tline.step
        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.sdomain.n_dofs
        n_e = self.ts.sdomain.n_active_elems
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

        self.w_record = [np.zeros((n_e, n_ip))]
        self.U_record = np.zeros(n_dofs)
        self.F_record = np.zeros(n_dofs)
        self.sf_record = np.zeros(2 * n_e)
        self.t_record = [t_n]
        self.eps_record = [np.zeros_like(eps)]
        self.sig_record = [np.zeros_like(sig)]
        self.D_record = [np.zeros_like(D)]

        while (t_n1 - self.tline.max) <= self.step_tolerance:
            k = 0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k <= self.k_max:

                R, F_int, K, eps, sig, xs_pi, alpha, z, w, D = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, eps, sig, t_n, t_n1, xs_pi, alpha, z, w)

                K.apply_constraints(R)
                d_U_k = K.solve()
                d_U += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    self.F_record = np.vstack((self.F_record, F_int))
                    U_k += d_U
                    self.U_record = np.vstack((self.U_record, U_k))
                    self.sf_record = np.vstack(
                        (self.sf_record, sig[:, :, 1].flatten()))
                    self.eps_record.append(np.copy(eps))
                    self.sig_record.append(np.copy(sig))
                    self.t_record.append(t_n1)
                    self.w_record.append(np.copy(w))
                    self.D_record.append(np.copy(D))
                    # print 'eps=',eps
                    # print'D=', D
                    break
                k += 1
                step_flag = 'corrector'

            if k >= self.k_max:
                print ' ----------> No Convergence any more'
                break
            t_n = t_n1
            t_n1 = t_n + self.d_t
            self.tline.val = min(t_n, self.tline.max)

        return

    def get_time_idx(self, vot):
        x = self.t_record
        idx = np.array(np.arange(len(x)), dtype=np.float_)
        return int(np.interp(vot, x, idx))

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from ibvpy.api import BCDof

    ts = TStepper()

    n_dofs = ts.sdomain.n_dofs
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

    n_dof = 2 * ts.sdomain.n_active_elems + 1
    plt.plot(U_record[:, n_dof] * 2, F_record[:, n_dof] / 1000., marker='.')
    plt.xlabel('displacement')
    plt.ylabel('force')
    plt.show()
