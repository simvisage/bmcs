'''
Created on 12.01.2016

@author: Yingxiong
'''
from traits.api import Int, HasStrictTraits, Instance, \
    Float, Event, \
    Array, List, Bool, Property, cached_property, Str, Dict
from traitsui.api import View, Item

from ibvpy.core.tline import TLine
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import numpy as np
from view.ui import BMCSLeafNode

from .mats3D.mats3D_explore import MATS3DExplore


#from matsXD.matsXD_explore import MATSXDExplore
class TLoop(HasStrictTraits):

    ts = Instance(MATS3DExplore)

    tline = Instance(TLine, ())
    d_t = Float(0.005)
    t_max = Float(1.0)
    k_max = Int(50)
    tolerance = Float(1e-4)
    step_tolerance = Float(1e-8)

    t_record = List
    U_n = Array(dtype=np.float_)
    K = Array(dtype=np.float_)
    state = Array(dtype=np.float_)
    F_record = List
    U_record = List
    state_record = List

    K = Instance(SysMtxAssembly)

    paused = Bool(False)
    restart = Bool(True)

    def setup(self):
        n_comps = 6
        n_dofs = n_comps
        self.U_n = np.zeros((n_dofs,))
        t_n = self.tline.val
        self.t_record = [t_n]
        self.U_record = [np.zeros_like(self.U_n)]
        self.F_record = [np.copy(self.U_n)]
        self.state_record = [np.copy(self.state)]
        # Set up the system matrix
        #
        self.K = SysMtxAssembly()
        self.ts.bcond_mngr.apply_essential(self.K)

    state_changed = Event
    state_arrays = Property(Dict(Str, Array),
                            depends_on='state_changed')
    '''Dictionary of state arrays.
    The entry names and shapes are defined by the material
    model.
    '''
    @cached_property
    def _get_state_arrays(self):
        sa_shapes = self.ts.state_array_shapes
        print('state array generated', sa_shapes)
        return {
            name: np.zeros(mats_sa_shape, dtype=np.float_)[np.newaxis, ...]
            for name, mats_sa_shape
            in list(sa_shapes.items())
        }

    def init(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = 0
            self.state_changed = True
            self.setup()
            self.restart = False

    def eval(self):
        # Reset the system matrix (constraints are preserved)
        #
        self.d_t = self.tline.step
        F_ext = np.zeros_like(self.U_n)
        t_n = self.tline.val
        t_n1 = t_n
        U_n = self.U_n
        while (t_n1 - self.tline.max) <= self.step_tolerance and \
                not (self.restart or self.paused):
            k = 0
            print('load factor', t_n1, end=' ')
            step_flag = 'predictor'
            U_k = np.copy(U_n)
            d_U_k = np.zeros_like(U_k)
            while k <= self.k_max and \
                    not (self.restart or self.paused):

                self.K.reset_mtx()

                K_mtx, F_int = self.ts.get_corr_pred(
                    U_k, d_U_k, t_n, t_n1,
                    step_flag,
                    **self.state_arrays
                )

                self.K.add_mtx(K_mtx)

                # Prepare F_ext by zeroing it
                #
                F_ext[:] = 0.0

                # Assemble boundary conditions in K and self.F_ext
                #
                self.ts.bcond_mngr.apply(
                    step_flag, None, self.K, F_ext, t_n, t_n1)

                # Return the system matrix assembly K and the residuum
                #
                R = F_ext - F_int

                self.K.apply_constraints(R)
                d_U_k, pos_def = self.K.solve()
                U_k += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    U_n[:] = U_k[:]
                    self.t_record.append(t_n1)
                    self.U_record.append(U_k)
                    self.F_record.append(F_int)
                    self.state_record.append(np.copy(self.state))
                    break
                k += 1
                step_flag = 'corrector'

            if k >= self.k_max:
                print(' ----------> no convergence')
                break
            else:
                print('(', k, ')')
            if self.restart or self.paused:
                print('interrupted iteration')
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
    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import \
        MATS3DElastic
    from ibvpy.mats.mats3D.mats3D_plastic.mats3D_desmorat import \
        MATS3DDesmorat
    from ibvpy.mats.mats3D import \
        MATS3DMplCSDEEQ
    ts = MATS3DExplore(
        mats_eval=MATS3DDesmorat()
    )

    ts.bcond_mngr.bcond_list = [BCDof(var='u', dof=0, value=0.001)]

    tl = TLoop(ts=ts, tline=TLine(step=0.1))
    tl.init()
    U_k = tl.eval()

    print(U_k)

    print('U', tl.U_record)
    print('F', tl.F_record)
