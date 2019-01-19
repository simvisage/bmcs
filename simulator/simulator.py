r'''

Simulator implementation
========================

'''
from threading import Thread

from traits.api import \
    Instance, on_trait_change, Str, \
    Property, cached_property, Bool

from simulator.hist import Hist
from view.ui.bmcs_tree_node import BMCSRootNode

from .i_hist import IHist
from .i_model import IModel
from .i_tstep import ITStep
from .i_xdomain import IXDomain
from .tline import TLine
from .tloop import TLoop


class RunTimeLoopThread(Thread):
    r'''Thread launcher class used to issue a calculation.
    in an independent thread.
    '''

    def __init__(self, simulator, *args, **kw):
        super(RunTimeLoopThread, self).__init__(*args, **kw)
        self.daemon = True
        self.simulator = simulator

    def run(self):
        try:
            # start the calculation
            self.simulator.tloop()
        except Exception as e:
            self.simulator.running = False
            raise e  # re-raise exception
        self.simulator.running = False


class Simulator(BMCSRootNode):
    r'''Base class for simulators included in the BMCS Tool Suite.
    It implements the state dependencies within the simulation tree.
    It handles also the communication between the simulation and
    the user interface in several modes of interaction.
    '''
    title = Str

    desc = Str

    #=========================================================================
    # MODEL
    #=========================================================================
    model = Instance(IModel)
    r'''Model implementation.
    '''

    #=========================================================================
    # Spatial domain
    #=========================================================================
    xdomain = Instance(IXDomain)
    r'''Spatial domain represented by a finite element discretization.
    providing the kinematic mapping between the linear algebra (vector and
    matrix) and field representation of the primary variables.
    '''

    #=========================================================================
    # TIME LINE
    #=========================================================================
    tline = Instance(TLine)
    r'''Time line defining the time range, discretization and state,  
    '''

    def _tline_default(self):
        return TLine(
            time_change_notifier=self.time_changed,
            time_range_change_notifier=self.time_range_changed
        )

    def time_changed(self, time):
        if self.ui != None:
            self.ui.viz_sheet.time_changed(time)

    def time_range_changed(self, tmax):
        self.tline.max = tmax
        if self.ui != None:
            self.ui.viz_sheet.time_range_changed(tmax)

    def set_tmax(self, time):
        self.time_range_changed(time)

    #=========================================================================
    # TIME LOOP
    #=========================================================================
    tloop = Property(Instance(TLoop), depends_on='model,xdomain')
    r'''Time loop constructed based on the current model.
    '''
    @cached_property
    def _get_tloop(self):
        return self.model.tloop_type(tstep=self.tstep,
                                     hist=self.hist,
                                     tline=self.tline)

    tstep = Property(Instance(ITStep), depends_on='model,xdomain')
    r'''Class representing the time step and state
    '''
    @cached_property
    def _get_tstep(self):
        return self.model.tstep_type(model=self.model,
                                     xdomain=self.xdomain,
                                     hist=self.hist)

    def pause(self):
        self.tloop.paused = True
        self.join()

    def stop(self):
        self.tloop.restart = True
        self.join()

    #=========================================================================
    # HISTORY
    #=========================================================================
    hist = Instance(IHist)
    r'''History representation of the model response.
    '''

    def _hist_default(self):
        return Hist()

    #=========================================================================
    # COMPUTATION THREAD
    #=========================================================================
    run_thread = Instance(RunTimeLoopThread)
    running = Bool(False)

    def run(self):
        r'''Run a thread if it does not exist - do nothing otherwise
        '''
        if self.running:
            return
        self.running = True
        self.run_thread = RunTimeLoopThread(self)
        self.run_thread.start()

    def join(self):
        r'''Wait until the thread finishes
        '''
        self.run_thread.join()

    @on_trait_change('MAT,ALG,CS,GEO,BC,+BC')
    def signal_reset(self):
        '''Upon the change of the model parameters,
        signal the user interface that further calculation
        does not make sense.
        '''
        if self.ui:
            self.ui.stop()
