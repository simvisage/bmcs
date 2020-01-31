#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jun 9, 2009 by: rchx

'''
Generic implementation of the time loop.
'''

import os
from threading import Thread
import time
from warnings import warn

from ibvpy.core.astrategy import AStrategyBase
from ibvpy.core.ibv_resource import IBVResource
from ibvpy.core.tline import TLine
from ibvpy.core.tstepper import TStepper
from traits.api import Array, Bool, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, \
    Delegate, on_trait_change, Button, Property, \
    cached_property, property_depends_on, \
    Directory
from traits.util.home_directory import get_home_directory
from traitsui.api import \
    Item, View, HGroup, \
    Group, Handler, RangeEditor, spring
from traitsui.menu import OKButton, CancelButton, \
    Action

import numpy as np


# import pymem
LOGGING_ON = False

if LOGGING_ON:
    import logging
    log = logging.getLogger('info')


class XTLine(HasTraits):

    '''
    Time line for the control parameter.

    This class sets the time-range of the computation - the start and stop time.
    val is the value of the current time.

    TODO - the info page including the number of load steps
    and estimated computation time.

    TODO - the slide bar is not read-only. How to include a real progress bar?
    '''
    min = Float(0.0)
    max = Float(1.0)
    step = Float(0.1)
    val = Float(0.0)
    time = Property

    @property_depends_on('val')
    def _get_time(self):
        return self.val

    def _set_time(self, value):
        warn('This should be read-only slider')
        self.val = value

    def _val_changed(self):
        if self.time_change_notifier:
            self.time_change_notifier(self.val)

    steps_to_go = Property
    '''Number of steps to go from the current state'''

    def _get_steps_to_go(self):
        return int((self.max - self.val) / self.step)

    time_change_notifier = Callable

    traits_view = View(HGroup(Item('min'),
                              spring,
                              Item('step', label='step size'),
                              spring,
                              Item('max')),
                       Item('time', editor=RangeEditor(low_name='min',
                                                       high_name='max',
                                                       auto_set=False,
                                                       enter_set=True,
                                                       ),
                            show_label=False
                            ),
                       resizable=True
                       )


class CompTimer(object):

    def __init__(self, name):
        self.name = name
        self.last_duration = 0.0
        self.duration = 0.0
        self.reset()

    def reset(self):
        self.start = time.clock()  # time of this process
        self.start_all = time.time()  # wall clock time

    def record(self):
        self.last_duration = self.duration
        self.duration = time.clock() - self.start
        self.overall_duration = time.time() - self.start_all

    def report(self):
        print("Timer", self.name)
        print("Computation Time : %8.2f sec" % self.overall_duration)
        print("Elapsed time     : %8.2f sec" % self.duration)


class TLoop(IBVResource):

    ''' Time loop management.

    This class implements the generic time-stepping scheme. It can be
    configured using an adaptive strategy to serve for wide class
    of time-stepping algorithms.
    '''
    # service specifiers - used to link the service to this object
    service_class = 'ibvpy.plugins.tloop_service.TLoopService'
    service_attrib = 'tloop'

    dir = Directory

    def _dir_default(self):

        # get the path to the simdb directory
        home_dir = get_home_directory()
        sim_data_dir = os.path.join(home_dir, 'simdb', 'simdata')
        if not os.path.exists(sim_data_dir):
            os.mkdir(sim_data_dir)

        # derive the name of the simdata directory from the main python module
        # name
        mod_base_name = os.path.basename(os.getcwd())
        mod_path = os.path.join(sim_data_dir, mod_base_name)
        if not os.path.exists(mod_path):
            os.mkdir(mod_path)

        return mod_path

    user_wants_abort = False

    name = Str('<unnamed>')

    tline = Instance(TLine)

    def _tline_default(self):
        return TLine(min=0.0, max=1.0, step=1.0)

    report_time_step = Callable

    # Convenience property to set the boundary condition list
    # in the time stepper through the time loop
    #
    bcond_list = Property

    def _get_bcond_list(self):
        return self.tstepper.bcond_list

    def _set_bcond_list(self, value):
        self.tstepper.bcond_list = value

    # Convenience property to set the response trace list
    # in the time stepper through the time loop
    #
    rtrace_list = Property

    def _get_rtrace_list(self):
        return self.tstepper.rtrace_list

    def _set_rtrace_list(self, value):
        self.tstepper.rtrace_list = value

    DT = Float

    KMAX = Int(40)
    RESETMAX = Int(10)
    tolerance = Float(1e-8)

    #
    #
    adap = Trait(AStrategyBase())
    tstepper = Trait(TStepper)

    def _tstepper_default(self):
        return TStepper()

    rtrace_mngr = Delegate('tstepper')

    # Supply the global resp-trace-evals. They can be accessed from
    # the time-stepper and included in views to the model.
    #
    rte_dict = Property(List, depends_on='tstepper:rte_dict')

    @cached_property
    def _get_rte_dict(self):
        return {'time': lambda sctx, U_k, *args, **kw: np.array([self.t_n1]),
                'mem_counter': lambda sctx, U_k, *args, **kw: self.mem_counter,
                'U_k': lambda sctx, U_k, *args, **kw: U_k,
                }

    sync_resp_tracing = Bool(False)
    _updated = Int(0)
    eval_timer = Instance(CompTimer)
    iter_timer = Instance(CompTimer)
    crpr_timer = Instance(CompTimer)
    solv_timer = Instance(CompTimer)
    rtrace_mngr_timer = Instance(CompTimer)

    def __init__(self, *args, **kwtraits):
        super(TLoop, self).__init__(**kwtraits)
        self._updated = 0

        # make algorithm accessible from the components
        #
        self.adap.tloop = self
        self.tstepper.tloop = self

        # Computation timer for individual steps.
        #
        self.eval_timer = CompTimer(name='Eval')
        self.iter_timer = CompTimer(name='Iter')
        self.crpr_timer = CompTimer(name='CorrPred')
        self.solv_timer = CompTimer(name='Solve')
        self.rtrace_mngr_timer = CompTimer(name='RTrace')

        # self.mem_counter = pymem.MemCounter()

    t_n = Float()

    def _t_n_default(self):
        return self.tline.min

    U_n = Array()

    # Time-loop control variable.
    # It must be within the interval < tline.min, t_n >
    #
    def reset(self):
        self.t_n = 0.0
        self._reset = True

    _reset = Bool(True)

    def setup(self):

        if self._reset == False:
            return

        self.k = 0
        self.n_reset = 0
        self.ttot_k = 0

        self.tstepper.setup()

        # Initialize the variables
        #
        self.R_k = self.tstepper.new_resp_var()
        self.U_k = self.tstepper.U_k
        self.d_U = self.tstepper.d_U
        self.U_n = self.tstepper.new_cntl_var()

        self.U_k[:] = 0.0
        self.d_U[:] = 0.0
        self.tot_k = 0.0
        self.ttot_k = 0.0
        self.norm = 1.0
        self._reset = False

    debug = Bool(False)

    def get_initial_state(self):
        self.setup()
        self.t_n1 = self.t_n
        return self.tstepper.eval(self.U_k,
                                  self.d_U,
                                  self.t_n,
                                  self.t_n1,
                                  'predictor')

    # Tolerance in the time variable to end the iteration.
    step_tolerance = Float(1e-8)

    paused = Bool(False)
    restart = Bool(True)

    def init(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = 0
            self.setup()
            self.restart = False

    # specify type of 'linalg.norm'
    # default 'ord = None' returns 2-norm
    #
    ord = Enum(None, np.inf)

    def eval(self, e=None):

        if not self.sync_resp_tracing:
            self.rtrace_mngr.start_timer()

        adap = self.adap
        tstepper = self.tstepper
        self.setup()

        self.t_n1 = self.t_n
        self.U_k[:] = self.U_n[:]
        self.d_U[:] = 0.0
        self.norm = 1

        if self.DT:
            self.d_t = self.DT
            warn('DEPRECATED: DT attribute of tloop should not be used'
                 'set the step attribute of TLine instead', DeprecationWarning)
        else:
            self.d_t = self.tline.step

        self.ls_counter = -1
        abort_tloop = False

        # Measure computation time

        self.eval_timer.reset()

        if self.verbose_load_step:
            print('======= Time range: %g -> %g =========' % \
                (self.t_n, self.tline.max))

        # Register the list of variables requrired by the adaptive
        # strategy in the time stepper to include their evaluation in
        # the corrector-predictor evaluation ts.extend_output_list(
        # adap.get_indicator_list() )
        #

        while (self.t_n1 - self.tline.max) <= self.step_tolerance and \
                not self.user_wants_abort:

            self.adap.begin_time_step(self.t_n1)

            self.iter_timer.reset()

            if LOGGING_ON:
                log.info('===================================')
                log.info('time t_n1 = %f', self.t_n1)

            self.ls_counter += 1
            self.report_load_step_start()

            self.k = 0
            step_flag = 'predictor'

            while self.k < self.KMAX:

                self.report_iteration()

                if self.debug:
                    print('U_k\n', self.U_k)
                    print('d_U\n', self.d_U)
                    print('t_n\n', self.t_n)
                    print('t_n1\n', self.t_n1)

                # Extract the matrix representation of the time t_n1
                # stepper for the given time and control variable u
                #
                self.crpr_timer.reset()
                try:
                    K, R, n_F_int = tstepper.eval(self.U_k,
                                                  self.d_U,
                                                  self.t_n,
                                                  self.t_n1,
                                                  step_flag,
                                                  )
                except np.linalg.LinAlgError as e:
                    # abort computation due to ultimate failure
                    # might be caused by material instability
                    print('Exception of equation solver, aborting')
                    abort_tloop = True
                    break

                if self.debug:
                    print('K\n', K)
                    print('R\n', R)

                # self.crpr_timer.record()
                # self.crpr_timer.report()
                self.rtrace_mngr_timer.reset()
                self.rtrace_mngr.record_iter(self.tstepper.sctx, self.U_k,
                                             t=self.t_n1)
                self.rtrace_mngr_timer.record()
                # self.rtrace_mngr_timer.report()

                if adap.ihandler_needed():
                    adap.ihandler_invoke()
                    self.d_t *= adap.ihandler_get_scale()
                    self.t_n1 = self.t_n + self.d_t
                    step_flag = 'predictor'
                    self.k = 0

                    if LOGGING_ON:
                        log.info("redo time step")
                    continue

                # the residuum must now learn about its constraints
                # so that the constrained equations are marked as fulfilled
                #
                K.apply_constraints(R)

                if self.debug:
                    print('constrained K\n', K)
                    print('constrained R\n', R)

                # default set to 2-norm, i.e
                # "norm = sqrt(sum(x_i**2))
                # set 'ord=np.inf' to switch norm to
                # "norm = max(abs(x_i))"
                #
                if n_F_int == 0.0:
                    n_F_int = 1.0
                self.norm = np.linalg.norm(R, ord=self.ord)  # / n_F_int

                if self.debug:
                    print('Norm:', self.norm)

                if self.norm < self.tolerance:  # convergence satisfied
                    self.n_reset = 0
                    if LOGGING_ON:
                        log.info(
                            "time step equilibrated in %d step(s)", self.k)
                    break  # update_switch -> on

                self.solv_timer.reset()
                self.d_U = K.solve()  # DG_k * d_U = r

                if self.debug:
                    print('d_U\n', self.d_U)

                self.solv_timer.record()
                # self.solv_timer.report()

                self.U_k += self.d_U

                self.k += 1
                self.tot_k += 1
                self.ttot_k += 1
                step_flag = 'corrector'

            else:  # handling nonconverged step

                if abort_tloop:
                    # ultimate failure exit the calculation
                    self.rtrace_mngr.record(self.tstepper.sctx, self.U_k)
                    break

                if LOGGING_ON:
                    log.info("no convergence reached - refinement in time")
                # adaptive strategy halving d_t
                self.n_reset = self.n_reset + 1
                # max number of resets achieved
                if self.n_reset > self.RESETMAX:

                    # Handle this situation with an exception with an
                    # associated dialog box and concise report of the
                    # iteration process
                    #
                    # @TODO raise the NoConvergence exception
                    if not self.sync_resp_tracing:
                        self.rtrace_mngr.stop_timer()
                        self.rtrace_mngr.timer_tick()
                    return

                self.d_t *= adap.fhandler_get_scale()
                self.t_n1 = self.t_n + self.d_t

                if LOGGING_ON:
                    log.info("redo time step with dt = %s",)
                continue

            self.iter_timer.record()
            # self.iter_timer.report()
            # self.mem_counter.record()

            if self.sync_resp_tracing:
                self.rtrace_mngr.timer_tick()

            if abort_tloop:
                # ultimate failure exit the calculation
                print('ABORT tloop with exception', e)
                try:
                    self.rtrace_mngr.record(self.tstepper.sctx, self.U_k)
                except:
                    pass
                break

            if adap.ehandler_needed():  # explicit adaptations
                if adap.ehandler_accept():  # accept equilibrium?
                    self.accept_time_step()  # register the state and response
                adap.ehandler_invoke()
            else:

                try:
                    self.accept_time_step()
                except np.linalg.LinAlgError:
                    print('Something wrong aborting agoin')
                    pass
                self.adap.end_time_step(self.t_n1)

                # Set new target time
                self.d_t *= adap.ehandler_get_scale()
                self.t_n = self.t_n1
                self.t_n1 = self.t_n + self.d_t
                # handle the case that due to numerical noice
                # t_n might exceed the max of the range.
                self.tline.val = min(self.t_n, self.tline.max)

        # Hack to get the state variables in the last time step.
        # This invokes the update state operator in the
        # material model. That would normaly await for the next
        # corrector/predictor step.
        #
        # @todo: How to handle the tracers that only return state variables?
        # 1) extra call in order to make the state update
        # 2) deferred call to the response trace evaluation (after the update).
        #
#        tstepper.eval( step_flag,
#                          self.U_k,
#                          self.d_U,
#                          self.t_n,
#                          self.t_n1 )
#        self.accept_time_step()

        # Report computation time
        #
        self.eval_timer.record()
        if self.verbose_time:
            self.eval_timer.report()

        if not self.sync_resp_tracing:
            self.rtrace_mngr.stop_timer()
            self.rtrace_mngr.timer_tick()

        self.rtrace_mngr.close()

        # return the displacement vector as convenience for tests
#        print 'U_k at the end of tloop.eval: U_k', self.U_k
        return self.U_k

    on_accept_time_step = Callable(lambda: None)

    def accept_time_step(self):
        '''
        After the equilibrium is found, first fill the responce tracers for
        equilibrium and then set the update state on (sets the global flag
        in  tstepper)
        '''
        self.U_n = np.copy(self.U_k)

        if LOGGING_ON:
            log.debug("u_n = \n%s", str(self.U_n))

        self.record_equilibrium()

        self.on_accept_time_step()

        self.tstepper.update_state(self.U_k)

    verbose_time = Bool(False)
    verbose_load_step = Bool(False)
    verbose_iteration = Bool(False)

    def record_equilibrium(self):
        '''Record current state in the state array.
        '''
        self.rtrace_mngr_timer.reset()
        self.rtrace_mngr.record_equilibrium(self.tstepper.sctx,
                                            self.U_k, t=self.t_n1)
        self.rtrace_mngr_timer.record()

    def report_load_step_start(self):
        if self.verbose_load_step:
            print('LS:%3d, Time: %.4f' % \
                (self.ls_counter, self.t_n1), end=' ')
            if self.verbose_iteration:
                print()  # just add a new line

    def report_load_step_end(self):
        if self.verbose_load_step and not self.verbose_iteration:
            # just a summary of iterations within the load step
            print(' Iter: %4d' % \
                (self.k,))

    def report_iteration(self):
        if self.verbose_iteration:
            print('\tIT: %3d, Norm: %3g' % \
                (self.k, self.norm))

    calculate = Button
    computation_thread = Instance(Thread)

    @on_trait_change('calculate')
    def teval(self):
        '''
        Eval in a thread.
        '''
        if self.computation_thread and self.computation_thread.isAlive():
            self.user_wants_abort = True
        else:
            self.user_wants_abort = False
            self.computation_thread = Thread(
                target=self.eval, name="computation",
            )
            self.computation_thread.daemon = True
            self.computation_thread.start()

    view = View(
        Group(
            Item('calculate', show_label=False),
            Item('tline', label='Time line', style='custom'),
            HGroup(
                Item('KMAX', label='Max. number of iterations'),
                Item('RESETMAX', label='Max. number of resets'),
                Item('tolerance', label='Tolerance on residual norm')
            ),
        ),
        resizable=True,
        scrollable=True,
        height=0.75, width=0.75,
        buttons=[OKButton, CancelButton]
    )


if LOGGING_ON:
    import logging.config
    logging.config.fileConfig("logging.conf")

if __name__ == '__main__':
    tl = TLoop()
    tl.configure_traits()
