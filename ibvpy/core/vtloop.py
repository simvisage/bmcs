'''
Created on Feb 8, 2018

@author: rch
'''

from ibvpy.core.i_tstepper_eval import ITStepperEval
from mathkit.matrix_la import \
    SysMtxAssembly
import numpy as np
import traits.api as tr

from .bcond_mngr import BCondMngr
from .tline import TLine


class TimeLoop(tr.HasStrictTraits):

    tline = tr.Instance(TLine)
    '''Time line object specifying the start, end, time step and current time
    '''

    def _tline_default(self):
        return TLine(min=0.0, max=1.0, step=1.0)

    ts = tr.Instance(ITStepperEval)
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

    k_max = tr.Int(300)
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

    paused = tr.Bool(False)
    restart = tr.Bool(True)
    stop = tr.Bool(False)

    def init(self):
        self.stop = False
        if self.paused:
            self.paused = False
            return
        if self.restart:
            self.tline.val = 0
            # self.setup()
            self.restart = False
        self.bc_mngr.setup(None)
        for rt in self.response_traces:
            rt.setup(self)

    algorithmic = tr.Bool(True)

    def eval(self):
        update_state = False
        K = SysMtxAssembly()
        self.bc_mngr.apply_essential(K)
        U_n = np.zeros((self.ts.mesh.n_dofs,), dtype=np.float_)
        dU = np.copy(U_n)
        U_k = np.copy(U_n)
        F_ext = np.zeros_like(U_n)
        algorithmic = self.algorithmic
        pos_def = True

        while (self.t_n1 - self.tline.max) <= self.step_tolerance:

            print('current time %f' % self.t_n1, end=' ')
            self.d_t = self.tline.step
            k = 0
            step_flag = 'predictor'

            while k < self.k_max:

                if self.stop:
                    return U_n

                K.reset_mtx()
                K_arr, F_int, n_F_int = self.ts.get_corr_pred(
                    U_k, dU, self.t_n, self.t_n1, update_state,
                    algorithmic
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
                if norm < self.tolerance:
                    print('converged in %d iterations' % (k + 1))
                    update_state = True
                    self.record_response(U_k, F_int, self.t_n1)
                    break
                dU, pos_def = K.solve(check_pos_def=algorithmic)
                if algorithmic and not pos_def:
                    algorithmic = False
                    print('switched to secant')
                    continue
                U_k += dU
                k += 1
                step_flag = 'corrector'

            U_n = np.copy(U_k)
            self.t_n = self.t_n1
            self.t_n1 = self.t_n + self.d_t
            self.tline.val = min(self.t_n, self.tline.max)

        return U_n

    write_dir = tr.Directory
    F_int_record = tr.List(tr.Array(np.float_))
    U_record = tr.List(tr.Array(np.float_))
    F_ext_record = tr.List(tr.Array(np.float_))
    t_record = tr.List

    response_traces = tr.List()

    def record_response(self, U_k, F_int, t):

        self.F_int_record.append(np.copy(F_int))
        self.U_record.append(np.copy(U_k))
        self.t_record.append(self.t_n1)

        for rt in self.response_traces:
            rt.update(U_k, t)
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
