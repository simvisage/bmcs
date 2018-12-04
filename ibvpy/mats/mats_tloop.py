'''
Created on 12.01.2016

@author: Yingxiong
'''
from ibvpy.core.tline import TLine
from traits.api import Int, HasStrictTraits, Instance, \
    Float, on_trait_change, Callable, \
    Array, List, Bool, Property, cached_property, Str, Dict
from traitsui.api import View, Item
from view.ui import BMCSLeafNode

from mats3D.mats3D_explore import MATS3DExplore
#from matsXD.matsXD_explore import MATSXDExplore
import numpy as np


class TLoop(HasStrictTraits):

    ts = Instance(MATS3DExplore)

    tline = Instance(TLine)
    d_t = Float(0.005)
    t_max = Float(1.0)
    k_max = Int(1000)
    tolerance = Float(1e-4)
    # Tolerance in the time variable to end the iteration.
    step_tolerance = Float(1e-8)

    t_record = List
    U = Array(dtype=np.float_)
    F = Array(dtype=np.float_)
    K = Array(dtype=np.float_)
    state = Array(dtype=np.float_)
    F_record = List
    U_record = List
    state_record = List
    K_record = List

    paused = Bool(False)
    restart = Bool(True)

    def reset_sv_hist(self):
        n_comps = 6
        n_dofs = n_comps
        sa_shape = self.ts.state_array_shapes
        self.state = np.zeros(sa_shape)
        self.U = np.zeros((n_dofs,))
        self.F = np.zeros((n_dofs,))
        t_n = self.tline.val
        self.t_record = [t_n]
        self.U_record = [np.zeros_like(self.eps)]
        self.F_record = [np.copy(self.sig)]
        self.state_record = [np.copy(self.state)]
        self.K = np.zeros((n_dofs, n_dofs))
        self.K_record = [np.zeros_like(self.K)]

    def init(self):
        print 'INIT'
        if self.paused:
            self.paused = False
        if self.restart:
            print 'RESET TIME'
            self.tline.val = 0
            self.reset_sv_hist()
            self.restart = False

    def eval(self):

        self.ts.bcond_mngr.apply_essential(self.K)

        self.d_t = self.tline.step
        t_n = self.tline.val
        t_n1 = t_n

        U_k = self.U_k

        while (t_n1 - self.tline.max) <= self.step_tolerance and \
                not (self.restart or self.paused):

            k = 0
            step_flag = 'predictor'
            d_U = np.zeros_like(U_k)
            d_U_k = np.zeros_like(U_k)
            while k <= self.k_max and \
                    not (self.restart or self.paused):

                R, F_int, K = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, t_n, t_n1, self.state
                )

                K.apply_constraints(R)
                d_U_k = K.solve()
                d_U += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    U_k += d_U
                    self.t_record.append(t_n1)
                    self.U_record.append(U_k)
                    self.F_record.append(F_int)
                    self.state_record.append(np.copy(self.state))
                    self.K_record.append(np.copy(K))
                    break
                k += 1
                step_flag = 'corrector'

            if k >= self.k_max:
                print ' ----------> No Convergence any more'
                break
            if self.restart or self.paused:
                print 'interrupted iteration'
                break
            t_n = t_n1
            t_n1 = t_n + self.d_t
            self.tline.val = min(t_n, self.tline.max)

        return U_k

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

    ts = MATS3DExplore()

    ts.bcond_mngr.bcond_list = [BCDof(var='f', dof=0, value=1.0)]

    tl = TLoop(ts=ts)

    U_k = tl.eval()
