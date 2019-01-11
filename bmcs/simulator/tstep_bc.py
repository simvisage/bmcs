
from traits.api import \
    Instance, Property, cached_property, Enum, Float, on_trait_change

from bmcs.simulator.tstep_state import TStepState
from ibvpy.core.bcond_mngr import BCondMngr
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import numpy as np


class TStepBC(TStepState):

    # Boundary condition manager
    #
    bcond_mngr = Instance(BCondMngr)

    def _bcond_mngr_default(self):
        return BCondMngr()

    step_flag = Enum('predictor', 'corrector')
    '''Step flag to control the inclusion of essential BC'
    '''
    @on_trait_change('model_structure_changed')
    def _reset(self):
        self.step_flag = 'predictor'

    t_n = Float(0.0)
    '''Fundamental state time used for time dependent essential BC'
    '''

    K = Property(
        Instance(SysMtxAssembly),
        depends_on='model_structure_changed'
    )
    '''System matrix with registered essencial boundary conditions.
    '''
    @cached_property
    def _get_K(self):
        K = SysMtxAssembly()
        self.bcond_mngr.apply_essential(K)
        return K

    _corr_pred = Property(depends_on='primary_var_changed,t_n1')

    @cached_property
    def _get__corr_pred(self):
        print('U_k', self.U_k)
        U_k_field = self.model.map_vector_to_field(self.U_k)
        sig_k, D_k = self.model.get_corr_pred(
            U_k_field, self.t_n1,
            **self.state_vars
        )
        K_k = self.model.do_map_tns4_to_tns2(D_k)
        print('K_k', K_k)
        self.K.add_mtx(K_k)
        F_ext = np.zeros_like(self.U_k)
        self.bcond_mngr.apply(
            self.step_flag, None, self.K, F_ext, self.t_n, self.t_n1
        )
        F_int = self.model.map_field_to_vector(sig_k).flatten()
        print('F_int', F_int)
        R = F_ext - F_int
        self.K.apply_constraints(R)
        return R, self.K

    def _get_R_norm(self):
        R = self.R
        return np.sqrt(np.einsum('...i,...i', R, R))

    def make_iter(self):
        '''Perform a single iteration
        '''
        d_U_k, pos_def = self.K.solve()
        self.U_k[:] += d_U_k
        self.primary_var_changed = True
        self.step_flag = 'corrector'

    def make_incr(self):
        '''Update the control, primary and state variables..
        '''
        self.U_n[:] = self.U_k[:]
        U_k_field = self.model.map_vector_to_field(self.U_k)
        self.model.update_state(
            U_k_field, self.t_n1,
            **self.state_vars
        )
        self.hist.record_timestep(
            self, self.U_n, self.state_vars
        )
        self.t_n = self.t_n1
        self.step_flag = 'predictor'
