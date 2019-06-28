
import copy

from ibvpy.core.bcond_mngr import \
    BCondMngr
from mathkit.matrix_la.sys_mtx_assembly import \
    SysMtxAssembly
from traits.api import \
    HasStrictTraits, provides, \
    Instance, Property, cached_property, Enum, on_trait_change, \
    DelegatesTo, WeakRef, Event, Array, Float

import numpy as np

from .domain_state import DomainState
from .i_tstep import ITStep
from .xdomain.xdomain import XDomain


@provides(ITStep)
class TStepBC(HasStrictTraits):

    sim = WeakRef

    model = DelegatesTo('sim')

    hist = DelegatesTo('sim')

    primary_var_changed = Event

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.fe_domain
    #=========================================================================
    #
    #=========================================================================

    t_n = Float(0.0)
    '''Fundamental state time used for time dependent essential BC'
    '''
    t_n1 = Float(0.0)
    '''Target time value of the control variable.
    '''
    U_n = Property(Array(np.float_), depends_on='model_structure_changed')
    '''Fundamental value of the primary (control variable)
    '''
    @cached_property
    def _get_U_n(self):
        return np.zeros_like(self.U_k)

    '''Current fundamental value of the primary variable.
    '''
    U_k = Property(Array(np.float_), depends_on='model_structure_changed')
    '''Fundamental value of the primary (control variable)
    '''
    @cached_property
    def _get_U_k(self):

        U_var_shape = self.fe_domain.U_var_shape
        return np.zeros(U_var_shape, dtype=np.float_).flatten()

    model_structure_changed = Event

    # Boundary condition manager
    #
    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='bc,bc_items,model_structure_changed')

    @cached_property
    def _get_bcond_mngr(self):
        return BCondMngr(bcond_list=self.sim.bc)

    def init_state(self):
        '''Initialize state.
        '''
        self.t_n = 0.0
        self.t_n1 = 0.0
        self.model_structure_changed = True

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

    fe_domain = Property(depends_on='model_structure_changed')

    @cached_property
    def _get_fe_domain(self):
        domains = [
            DomainState(tstep=self, xdomain=xdomain, tmodel=tmodel)
            for xdomain, tmodel in self.sim.domains
        ]
        return XDomain(domains)

    corr_pred = Property(depends_on='primary_var_changed,t_n1')

    @cached_property
    def _get_corr_pred(self):
        self.K.reset_mtx()
        f_Eis, K_ks, dof_Es = np.array(
            [s.get_corr_pred(self.U_k, self.t_n, self.t_n1)
             for s in self.fe_domain]
        ).T
        self.K.sys_mtx_arrays = list(K_ks)  # improve
        F_ext = np.zeros_like(self.U_k)
        self.bcond_mngr.apply(
            self.step_flag, None, self.K, F_ext, self.t_n, self.t_n1
        )
        F_int = np.bincount(
            np.hstack(np.hstack(dof_Es)),
            weights=np.hstack(np.hstack(f_Eis))
        )
        R = F_ext - F_int
        self.K.apply_constraints(R)
        return R, self.K, F_int

    def make_iter(self):
        '''Perform a single iteration
        '''
        d_U_k, _ = self.K.solve()
        self.U_k[:] += d_U_k
        self.primary_var_changed = True
        self.step_flag = 'corrector'

    def make_incr(self):
        '''Update the control, primary and state variables..
        '''
        self.U_n[:] = self.U_k[:]
        states = [d.record_state() for d in self.fe_domain]
        self.hist.record_timestep(self.t_n1, self.U_k, self.F_k, states)
        self.t_n = self.t_n1
        self.step_flag = 'predictor'

    def __str__(self):
        s = '\nt_n: %g, t_n1: %g' % (self.t_n, self.t_n1)
        s += '\nU_n' + str(self.U_n)
        s += '\nU_k' + str(self.U_k)
        return s

    R_norm = Property

    def _get_R_norm(self):
        R = self.R
        return np.sqrt(np.einsum('...i,...i', R, R))

    R = Property

    def _get_R(self):
        R, _, _ = self.corr_pred
        return R.flatten()

    dR = Property

    def _get_dR(self):
        _, dR, _ = self.corr_pred
        return dR

    F_k = Property

    def _get_F_k(self):
        _, _, F_k = self.corr_pred
        return F_k

    record = DelegatesTo('sim')
