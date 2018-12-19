r'''

Simulator implementation
========================

'''
from threading import Thread
from traits.api import \
    Instance, on_trait_change, Str, DelegatesTo, \
    Property, cached_property, Bool
from bmcs.simulator.hist import Hist
from view.ui.bmcs_tree_node import BMCSRootNode
from .i_hist import IHist
from .i_model import IModel
from .i_tloop import ITLoop
from .model import Model
from .tline import TLine


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
    '''Base class for simulators included in the BMCS Tool Suite.
    It implements the state dependencies within the simulation tree.
    It handles also the communication between the simulation and
    the user interface in several modes of interaction.
    '''
    title = Str

    desc = Str

    tline = Instance(TLine)

    def _tline_default(self):
        # assign the parameters for solver and loading_scenario
        t_max = 1.0  # self.loading_scenario.t_max
        d_t = 0.1  # self.loading_scenario.d_t
        return TLine(min=0.0, step=d_t, max=t_max,
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

    def init(self):
        if self.paused:
            self.tloop.paused = False
        if self.restart:
            self.tloop.restart = False
            self.tline.val = self.tline.min
            self.tline.max = 1
            self.model.init_state()
            self.hist.timesteps = []

    def pause(self):
        self.tloop.paused = True
        self.join()

    def stop(self):
        self.tloop.restart = True
        self.join()

    tloop = Property(Instance(ITLoop), depends_on='model')
    r'''Time loop constructed based on the current model.
    '''
    @cached_property
    def _get_tloop(self):
        return self.model.tloop_type(model=self.model,
                                     tline=self.tline,
                                     hist=self.hist)

    hist = Instance(IHist)
    r'''History representation of the model response.
    '''

    def _hist_default(self):
        return Hist()

    model = Instance(IModel)
    r'''Model implementation.
    '''

    def _model_default(self):
        return Model()

    run_thread = Instance(RunTimeLoopThread)
    running = Bool(False)

    paused = DelegatesTo('tloop')
    restart = DelegatesTo('tloop')

    def run(self):
        if self.running:
            return
        self.running = True
        self.run_thread = RunTimeLoopThread(self)
        self.run_thread.start()

    def join(self):
        '''Wait until the thread finishes
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
