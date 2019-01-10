'''
'''

from traits.api import \
    HasStrictTraits, WeakRef, \
    Property, Float, Instance, DelegatesTo, \
    cached_property, Event, Dict, Str, Array, provides

import numpy as np

from .i_hist import IHist
from .i_model import IModel
from .i_tstep import ITStep


@provides(ITStep)
class TStepState(HasStrictTraits):
    '''Time step with managed and configurable state variables.
    '''
    model = Instance(IModel)

    hist = WeakRef(IHist)

    primary_var_changed = Event

    t_n1 = Float(0.0, auto_set=False, enter_set=True)
    '''Target time value of the control variable.
    '''
    U_n = Property(Array(np.float_), depends_on='model_structure_changed')
    '''Fundamental value of the primary (control variable)
    '''
    @cached_property
    def _get_U_n(self):
        U_var_shape = self.model.U_var_shape
        return np.zeros(U_var_shape, dtype=np.float_).flatten()

    '''Current fundamental value of the primary variable.
    '''
    U_k = Property(Array(np.float_), depends_on='model_structure_changed')
    '''Fundamental value of the primary (control variable)
    '''
    @cached_property
    def _get_U_k(self):
        U_var_shape = self.model.U_var_shape
        return np.zeros(U_var_shape, dtype=np.float_).flatten()

    F_0 = Float(1.0, auto_set=False, enter_set=True)
    '''Target value of a function (load).
    '''

    model_structure_changed = Event

    state_vars = Property(Dict(Str, Array),
                          depends_on='model_structure_changed')
    '''Dictionary of state arrays.
    The entry names and shapes are defined by the material
    model.
    '''
    @cached_property
    def _get_state_vars(self):
        sa_shapes = self.model.state_var_shapes
        print('state array generated', sa_shapes)
        return {
            name: np.zeros(mats_sa_shape, dtype=np.float_)[np.newaxis, ...]
            for name, mats_sa_shape
            in list(sa_shapes.items())
        }

    def init_state(self):
        '''Initialize state.
        '''
        self.t_n1 = 0.0
        self.model_structure_changed = True

    def record_state(self):
        '''Provide the current state for history recording.
        '''
        pass

    _corr_pred = Property(depends_on='primary_var_changed,t_n1')

    @cached_property
    def _get__corr_pred(self):
        U_k_r = self.U_k.reshape(self.model.U_var_shape)
        F, dF = self.model.get_corr_pred(
            U_k_r, self.t_n1,
            **self.state_vars
        )
        F_t = self.F_0 * self.t_n1

        return F_t - F, dF

    R = Property

    def _get_R(self):
        R, dR = self._corr_pred
        return R.flatten()

    dR = Property

    def _get_dR(self):
        R, dR = self._corr_pred
        return dR

    R_norm = Property

    def _get_R_norm(self):
        R = self.R
        return np.sqrt(R * R)

    def make_iter(self):
        '''Perform a single iteration
        '''
        d_U = self.R / self.dR[:, 0]
        self.U_k[:] += d_U[:]
        self.primary_var_changed = True

    def make_incr(self):
        '''Update the control, primary and state variables..
        '''
        self.U_n[:] = self.U_k[:]
        self.model.update_state(
            self.U_k, self.t_n1,
            **self.state_vars
        )
        self.hist.record_timestep(
            self, self.U_n, self.state_vars
        )
