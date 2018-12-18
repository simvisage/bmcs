'''
@author: rch
'''

from traits.api import \
    Instance, on_trait_change, Str, DelegatesTo
from bmcs.simulator.hist import Hist
from view.ui.bmcs_tree_node import \
    BMCSRootNode
from .i_hist import IHist
from .i_model import IModel
from .i_tloop import ITLoop
from .model import Model
from .tline import TLine
from .tloop import TLoop


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

    paused = DelegatesTo('tloop')
    restart = DelegatesTo('tloop')

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
        self.paused_state()
        self.tloop.paused = True

    def stop(self):
        self.stop_state()
        self.tloop.restart = True

    tloop = Instance(ITLoop)

    def _tloop_default(self):
        return TLoop(tline=self.tline,
                     model=self.model,
                     hist=self.hist)

    hist = Instance(IHist)

    def _hist_default(self):
        return Hist()

    model = Instance(IModel)

    def _model_default(self):
        return Model()

    def run(self):
        '''Run starts or resumes the simulation depending on the 
        pause or restart variables.
        '''
        self.init()
        if self.ui:
            # inform ui that the simulation is running in a thread
            self.ui.start_event = True
            self.ui.running = True
        try:
            # start the calculation
            self.eval()
        except Exception as e:
            if self.ui:
                self.ui.running = False
            raise e  # re-raise exception
        if self.ui:
            # cleanup ui and send the finish event
            self.ui.running = False
            self.ui.finish_event = True

    def init_state(self):
        '''Method called upon start event.
        '''
        pass

    def paused_state(self):
        '''Method called upon the pause event.
        '''
        pass

    def stop_state(self):
        '''Method called upon the stop event.
        '''
        pass

    def eval(self):
        '''Method called upon the run event
        must support the resume calculation.
        '''
        self.tloop.eval()

    @on_trait_change('MAT,ALG,CS,GEO,BC,+BC')
    def signal_reset(self):
        '''Upon the change of the model parameters,
        signal the user interface that further calculation
        does not make sense.
        '''
        if self.ui:
            self.ui.stop()
