'''
@author: rch
'''

from traits.api import \
    Instance, List, Bool, on_trait_change, Str, provides

from ibvpy.core.tline import TLine
import numpy as np
from simulator.api import ISimulator
from view.ui.bmcs_tree_node import \
    BMCSRootNode


@provides(ISimulator)
class BMCSModel(BMCSRootNode):
    '''Base class for models included in the iMCS Tool Suite.
    It implements the state dependencies within the model tree.
    It handles also the communication between the model and
    the user interface in several modes of interaction.

    The scenarios of usage are implemented in 
    bmcs_interaction_patterns
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
        if self._paused:
            self._paused = False
        if self._restart:
            self.tline.val = self.tline.min
            self.tline.max = 1
            self._restart = False
            self.init_state()
            self.timesteps = []

    def paused(self):
        self.paused_state()
        self._paused = True

    def stop(self):
        self.stop_state()
        self._restart = True

    _paused = Bool(False)
    _restart = Bool(True)

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
        raise NotImplemented

    def add_timestep(self, t):
        self.tline.val = min(t, self.tline.max)
        self.timesteps.append(t)

    timesteps = List()

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        x = np.array(self.timesteps, dtype=np.float_)
        idx = np.array(np.arange(len(x)), dtype=np.float_)
        t_idx = np.interp(vot, x, idx)
        return np.array(t_idx, np.int_)

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))

    @on_trait_change('MAT,ALG,CS,GEO,BC,+BC')
    def signal_reset(self):
        '''Upon the change of the model parameters,
        signal the user interface that further calculation
        does not make sense.
        '''
        if self.ui:
            self.ui.stop()
