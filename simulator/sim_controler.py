'''
Created on Mar 18, 2020

@author: rch

What are the use cases of this skeleton?

The services provided by this basic process structure include

 #Model

 ##Instantiate the model
   The model is an image of reality at a particular time instant t_k.
   Its STATE is defined by the primary variables and by state variables
   It provides the derived variables at the current time instance in form
   of properties.
   For hierarchically defined model that can incorporate several loops
   over different subdomains with variables defined only on a fractions
   of the model the properties are gathered from the definition of the 
   the subdomains (subdomain - material model). Such heterogeneous
   domain must define the mapping between the shared and local geometry and 
   representation discretization.

 ##Model interface
   The model provides the Residuum function `F` that is calculated
   for the current STATE, Further, the model provides also the
   derivatives with respect to the primary state variables dR_U.
   These two methods are used by the TStep to implement the predictor
   corrector scheme. The TStep interprets and controls the state
   for the purpose of time stepping within the TLoop

 ##Model specialization
   Model configurability. A particular model can be implemented
   In a few steps exploiting the subclassing and configurability of the
   of the model.
   
 #Controller
   Interaction with the model
   The evaluation process is controlled by the SimController, 
   It can launch the calculation within the current thread
   without interaction / monitoring of the current calculation
   or in a separate thread. The former option is useful for debugging.
 - Time line mixin is used to define the time range and time stepping
   parameters. Its changes are controlled by the TLoop.
   In a monitoring mode it is directly linked with the visual object
   time (VOT) in the plot sheet that defines the time step to be visualized.

 # Hist
   The model itself has no history. History is recorded during the
   time stepping TStep by the time loop TLoop. Variables to be recorded
   are specified in the model. The history is used for the calculation
   of cumulative properties based on the recorded variables and for the
   implementation of plot functions showing these variables
   
 ##Configuration
   Hist is used for recording of the model states during the time
   stepping. The variables to be recorded are specified in the Model 
   configuration using the `record' list. The recording can be 
   either minimalistic - specifying only the primary variables U and state 
   variables. Any other derived variable can be calculated 
   using the property functions. The Model is however implemented
   as a state machine depending on a value at a single time instant.
   Should the time dimension be implicitly allowed? The answer is not.
   This would bring about the possibility to include time-space problems  
   or regularized time steps as well in the algorithm. TEST it.
   
   
 # Why not timeline
Why not simply include the history in the model implicitly and 
providing an interface for TStep, TCut

Then, not only a single time step but a history slice 
would be possible to use in the calculation of the properties. 
This view would completely separate the notion of fundamental
state or history, the trial steps and the step to be updated.
This fits into the notions of dynamic calculations.

This model would then provide access to U_n which is a time 
series already accepted and U_k is the time sequence that is 
being changed during the current iteration. Upon an increment
TStep would append a mapping U_l = f(U_k) to U_n.

The history might be either 
set explicitly from previous calculation or calculated within
the current time loop. 

The nice thing about this approach would be the fact that 
there is no copy of the model in the Hist object. 
The role of the Hist object would be to map the data either
into the file structure or visualization back ends.

A point that would be on the negative side is the memory 
consumption for large finite element calculations. There,
the U_tn steps should be stored on the disk and only the U_tk
should be in memory. This makes the postprocessing difficult.

So the most general storage strategy seems to be
to preserve the current concept with one primary and state 
variables being recorded by default. Additional variables 
can be registered for recording given properties.

Within this scheme, also the field variables as large-sized
arrays can be conveniently stored in the file store back end.

This leads to the design that requires subclassing of Hist
for cumulative variables and for further plotting methods

'''

from view.ui import BMCSTreeNode

import numpy as np
import traits.api as tr

from .tline_mixin import TLineMixIn


class ITLoop(tr.Interface):
    pass


class IHist(tr.Interface):
    pass


class ITStep(tr.Interface):
    pass


class IModel(tr.Interface):
    pass


@tr.provides(ITStep)
class TStep(tr.HasStrictTraits):

    model = tr.Instance(IModel)
    sim = tr.DelegatesTo('model')

    t_n = tr.Float(0.0)
    t_n1 = tr.Float(0.0)

    def init_state(self):
        self.t_n = 0
        self.t_n1 = 0
        self.model.init_state()
        self.linalg_sys.register_constraints(0, 1)
        for var, state in self.model.S.items():
            state[...] = 0

    trial_state_changed = tr.Event

    R_dR_dU = tr.Property(depends_on='trial_state_changed, t_n1')

    @tr.cached_property
    def _get_R_dR_dU(self):
        R = self.R
        dR_dU = self.dR_dU
        return R, dR_dU

    R = tr.Property(depends_on='trial_state_changed, t_n1')

    @tr.cached_property
    def _get_R(self):
        bc = self.model.bc
        R = self.model.F - self.t_n1
        return R

    R_norm = tr.Property(depends_on='trial_state_changed, t_n1')

    @tr.cached_property
    def _get_R_norm(self):
        R = self.R
        return np.sqrt(np.einsum('...i,...i', R, R))

    dR_dU = tr.Property(depends_on='trial_state_changed')

    @tr.cached_property
    def _get_dR_dU(self):
        return self.model.d_F_U

    linalg_sys = tr.Instance(LinAlgSys, ())

    def make_iter(self):
        R, dR_dU = self.R_dR_dU
        self.linalg_sys.A = dR_dU
        self.linalg_sys.b_0 = R
        self.linalg_sys.apply_constraints(
            self.step_flag, self.t_n, self.t_n1
        )

        d_U = self.linalg_sys.solve()
        #d_U = -R / dR_dU
        self.model.U_k += d_U
        self.trial_state_changed = True

    def make_incr(self, t_n1):
        self.model.U_n[...] = self.model.U_k
        self.t_n1 = t_n1


@tr.provides(ITLoop)
class TLoop(tr.HasTraits):

    tstep = tr.Instance(ITStep)
    sim = tr.DelegatesTo('tstep')
    tline = tr.Property

    def _get_tline(self):
        return self.sim.tline

    k_max = tr.Int(100, enter_set=True, auto_set=False)

    acc = tr.Float(1e-4, enter_set=True, auto_set=False)

    verbose = tr.Bool(False, enter_set=True, auto_set=False)

    paused = tr.Bool(False)

    restart = tr.Bool(True)

    user_wants_abort = tr.Property

    def _get_user_wants_abort(self):
        return self.restart or self.paused

    def init(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = self.tline.min
            self.tstep.init_state()
            self.restart = False

    def eval(self):
        t_n1 = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step

        if self.verbose:
            print('t:', end='')

        while t_n1 <= (t_max + 1e-8):
            if self.verbose:
                print('\t%5.2f' % t_n1, end='')
            k = 0
            self.tstep.t_n1 = t_n1
            while (k < self.k_max) and (not self.user_wants_abort):
                R_norm = self.tstep.R_norm
                if R_norm < self.acc:
                    if self.verbose:
                        print('(%g), ' % k, end='\n')
                    break
                try:
                    self.tstep.make_iter()
                except RuntimeError as e:
                    raise(e)
                k += 1
            else:  # handle unfinished iteration loop
                if k >= self.k_max:  # maximum number of restarts exceeded
                    # no success abort the simulation
                    self.restart = True
                    print('')
                    raise StopIteration('Warning: '
                                        'convergence not reached in %g iterations' % k)
                else:  # reduce the step size
                    dt /= 2
                    continue

            # accept the time step and record the state in history
            self.tstep.make_incr(t_n1)
            # update the line - launches notifiers to subscribers
            self.tline.val = min(t_n1, self.tline.max)
            # set a new target time
            t_n1 += dt
            self.tstep.t_n1 = t_n1
        return


@tr.provides(IHist)
class Hist(tr.HasStrictTraits):

    model = tr.Instance(IModel)
    tstep = tr.DelegatesTo('model')

    def init_state(self):
        pass


class SimControler(BMCSTreeNode, TLineMixIn):

    model = tr.Instance(IModel)
    hist = tr.DelegatesTo('model')
    tstep = tr.DelegatesTo('model')
    tloop = tr.DelegatesTo('model')


@tr.provides(IModel)
class Model(BMCSTreeNode):
    '''Contains the primary unknowns variables U_k
    '''
    tstep_type = tr.Type(TStep)
    tloop_type = tr.Type(TLoop)
    hist_type = tr.Type(Hist)
    sim_type = tr.Type(SimControler)

    tstep = tr.Property(depends_on='tstep_type')

    @tr.cached_property
    def _get_tstep(self):
        return self.tstep_type(model=self)

    tloop = tr.Property(depends_on='tloop_type')

    @tr.cached_property
    def _get_tloop(self):
        return self.tloop_type(tstep=self.tstep)

    hist = tr.Property(depends_on='hist_type')

    @tr.cached_property
    def _get_hist(self):
        return self.hist_type(model=self)

    sim = tr.Property(depends_on='sim_type')

    @tr.cached_property
    def _get_sim(self):
        return self.sim_type(model=self)

    bc = tr.List(tr.Callable)

    U_shape = tr.Tuple(1,)

    def init_state(self):
        self.U_k = np.zeros(self.U_shape, dtype=np.float)
        self.U_n = np.copy(self.U_n)
        self.hist.init_state()

    def get_plot_sheet(self):
        return

    U_k = tr.Array(np.float_, TRIAL_STATE=True)
    U_n = tr.Array(np.float_, FUND_STATE=True)

    S = tr.Dict(tr.Str, tr.Array(np.float), STATE=True)

    F = tr.Property(depends_on='+TRIAL_STATE,+INPUT')

    @tr.cached_property
    def _get_F(self):
        raise NotImplemented

    d_F_U = tr.Property(depends_on='+TRIAL_STATE,+INPUT')

    @tr.cached_property
    def _get_d_F_U(self):
        raise NotImplemented


class BCModel(Model):
    bc = [lambda t: t]
    '''List of vector functions that implement the mapping 
    F_\alpha = bar{F}(t) 
    and U_\alpha = \bar{U}_\alpha(t). 
    '''


class SinusModel(Model):

    #=========================================================================
    # Model implementation - F and Fprime
    #=========================================================================

    def _get_F(self):
        return np.sin(self.U_k)

    def _get_d_F_U(self):
        return [np.cos(self.U_k)]

    #=========================================================================
    # Derived variables
    #=========================================================================
    G = tr.Property

    def _get_G(self):
        return np.arctan(self.U_k)


if __name__ == '__main__':
    m = SinusModel()
    m.tloop.init()
    m.tloop.eval()
    print(m.U_k)
    pass
    # Configuration options:
    # Boundary conditions
    #
    # bc - dictionary of array based boundary conditions
    #
    # Add a plotting functionality - in fact this example rephrases
    # the situation from the current application. Now show the recording
    # functionality.
    #
    # define the recording of variables
    # define plotting functions
    # define save use case
    # define load use case
