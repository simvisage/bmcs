r'''

Simulator implementation
========================

'''
from threading import Thread

from traits.api import \
    Instance, on_trait_change, Str, \
    Property, cached_property, Bool

from bmcs.simulator.hist import Hist
from view.ui.bmcs_tree_node import BMCSRootNode

from .i_hist import IHist
from .i_model import IModel
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
    tloop = Property(Instance(TLoop), depends_on='model')
    r'''Time loop constructed based on the current model.
    '''
    @cached_property
    def _get_tloop(self):
        return self.model.tloop_type(model=self.model,
                                     tline=self.tline,
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
    hist = Property(Instance(IHist), depends_on='model')
    r'''History representation of the model response.
    '''
    @cached_property
    def _get_hist(self):
        return Hist(model=self.model)

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
