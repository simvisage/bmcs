'''
'''

from traits.api import \
    HasStrictTraits, WeakRef, \
    Property, Float, DelegatesTo, \
    cached_property, Event, Dict, Str, Array, provides
import numpy as np
from .i_hist import IHist
from .i_model import IModel
from .i_tstep import ITStep


@provides(ITStep)
class TStepDemo(HasStrictTraits):
    '''Time step with managed and configurable state variables.
    '''

    sim = WeakRef

    model = DelegatesTo('sim')

    hist = DelegatesTo('sim')

    primary_var_changed = Event

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
        U_var_shape = self.xdomain.U_var_shape
        return np.zeros(U_var_shape, dtype=np.float_).flatten()

    '''Current fundamental value of the primary variable.
    '''
    U_k = Property(Array(np.float_), depends_on='model_structure_changed')
    '''Fundamental value of the primary (control variable)
    '''
    @cached_property
    def _get_U_k(self):
        U_var_shape = self.xdomain.U_var_shape
        return np.zeros(U_var_shape, dtype=np.float_).flatten()

    F_0 = Float(1.0, auto_set=False, enter_set=True)
    '''Target value of a function (load).
    '''

    model_structure_changed = Event

    state_n = Property(Dict(Str, Array),
                       depends_on='model_structure_changed')
    '''Dictionary of state arrays.
    The entry names and shapes are defined by the material
    model.
    '''
    @cached_property
    def _get_state_n(self):
        xmodel_shape = self.xdomain.state_var_shape
        tmodel_shapes = self.model.state_var_shapes
        return {
            name: np.zeros(xmodel_shape + mats_sa_shape, dtype=np.float_)
            for name, mats_sa_shape
            in list(tmodel_shapes.items())
        }

    def init_state(self):
        '''Initialize state.
        '''
        self.t_n = 0.0
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
            **self.state_n
        )
        F_t = self.F_0 * self.t_n1

        return F_t - F, dF, F

    R = Property

    def _get_R(self):
        R, _, _ = self._corr_pred
        return R.flatten()

    dR = Property

    def _get_dR(self):
        _, dR, _ = self._corr_pred
        return dR

    F_k = Property

    def _get_F_k(self):
        _, _, F_k = self._corr_pred
        return F_k

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
            **self.state_n
        )
        self.hist.record_timestep(
            self, self.U_n, self.state_n
        )
