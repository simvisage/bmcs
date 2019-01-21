
import copy

from traits.api import \
    Instance, Property, cached_property, Enum, on_trait_change, \
    List, Dict, WeakRef, DelegatesTo

from ibvpy.core.bcond_mngr import \
    BCondMngr
from mathkit.matrix_la.sys_mtx_assembly import \
    SysMtxAssembly
import numpy as np

from .tstep_state import TStepState
from .xdomain import IXDomain


class TStepBC(TStepState):

    xdomain = DelegatesTo('sim')

    # Boundary condition manager
    #
    bcond_mngr = Property(Instance(BCondMngr),
                          depends='bc,bc_items, model_structure_changed')

    @cached_property
    def _get_bcond_mngr(self):
        return BCondMngr(bcond_list=self.sim.bc)

    state_k = Dict
    '''State variables within the current iteration step
    '''

    record = DelegatesTo('sim')

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
        self.bcond_mngr.setup(None)
        self.bcond_mngr.apply_essential(K)
        return K

    _corr_pred = Property(depends_on='primary_var_changed,t_n1')

    @cached_property
    def _get__corr_pred(self):
        self.K.reset_mtx()
        # Get the field representation of the primary variable
        U_k_field = self.xdomain.map_U_to_field(self.U_k)
        self.state_k = copy.deepcopy(self.state_n)
        sig_k, D_k = self.model.get_corr_pred(
            U_k_field, self.t_n1, **self.state_k
        )
        K_k = self.xdomain.map_field_to_K(D_k)
        self.K.sys_mtx_arrays.append(K_k)
        F_ext = np.zeros_like(self.U_k)
        self.bcond_mngr.apply(
            self.step_flag, None, self.K, F_ext, self.t_n, self.t_n1
        )
        F_int = self.xdomain.map_field_to_F(sig_k).flatten()
        R = F_ext - F_int
        self.K.apply_constraints(R)
        return R, self.K, F_int

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
        for name, s_k in self.state_k.items():
            self.state_n[name] = s_k
        self.hist.record_timestep(self.t_n1, self.U_n, self.F_k, self.state_n)
        self.t_n = self.t_n1
        self.step_flag = 'predictor'

    def __str__(self):
        s = '\nt_n: %g, t_n1: %g' % (self.t_n, self.t_n1)
        s += '\nU_n' + str(self.U_n)
        s += '\nU_k' + str(self.U_k)
        return s
