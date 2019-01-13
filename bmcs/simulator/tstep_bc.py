
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
        self.K.reset_mtx()
        # Get the field representation of the primary variable
        U_k_field = self.model.map_U_to_field(self.U_k)
        sig_k, D_k = self.model.get_corr_pred(
            U_k_field, self.t_n1, **self.state_vars
        )
        K_k = self.model.map_field_to_K(D_k)
        self.K.add_mtx(K_k)
        F_ext = np.zeros_like(self.U_k)
        self.bcond_mngr.apply(
            self.step_flag, None, self.K, F_ext, self.t_n, self.t_n1
        )
        F_int = self.model.map_field_to_F(sig_k).flatten()
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
        U_k_field = self.model.map_U_to_field(self.U_k)
        self.model.update_state(
            U_k_field, self.t_n1,
            **self.state_vars
        )
        self.hist.record_timestep(
            self, self.U_n, self.state_vars
        )
        self.t_n = self.t_n1
        self.step_flag = 'predictor'

    def __str__(self):
        s = '\nt_n: %g, t_n1: %g' % (self.t_n, self.t_n1)
        s += '\nU_n' + str(self.U_n)
        s += '\nU_k' + str(self.U_k)
        return s
