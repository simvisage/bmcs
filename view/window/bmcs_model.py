'''

@author: rch
'''


from ibvpy.core.tline import TLine
from traits.api import \
    Instance, on_trait_change, Str
from view.ui.bmcs_tree_node import \
    BMCSRootNode


class BMCSModel(BMCSRootNode):
    '''Base class for models included in the iMCS Tool Suite.
    It implements the state dependencies within the model tree.
    It handles also the communication between the model and
    the user interface in several modes of interaction.

    The scenarios of usage are implemented in 
    bmcs_interaction_pattens
    '''
    tline = Instance(TLine)

    title = Str

    desc = Str

    def _tline_default(self):
        return TLine(min=0.0, step=0.1, max=0.0,
                     time_change_notifier=self.time_changed,
                     )

    def time_changed(self, time):
        if self.ui:
            self.ui.viz_sheet.time_changed(time)

    def time_range_changed(self, tmax):
        self.tline.max = tmax
        if self.ui:
            self.ui.viz_sheet.time_range_changed(tmax)

    def set_tmax(self, time):
        self.time_range_changed(time)

    def init(self):
        return

    def eval(self):
        return

    def paused(self):
        pass

    def stop(self):
        pass

    def run(self):
        self.init()
        if self.ui:
            self.ui.start_event = True
            self.ui.running = True
        try:
            self.eval()
        except Exception as e:
            if self.ui:
                self.ui.running = False
            raise
        if self.ui:
            self.ui.running = False
            self.ui.finish_event = True

    @on_trait_change('MAT,ALG,CS,GEO,BC,+BC')
    def signal_reset(self):
        '''Upon the change of the model parameters,
        signal the user interface that further calculation
        does not make sense.
        '''
        if self.ui:
            self.ui.stop()
