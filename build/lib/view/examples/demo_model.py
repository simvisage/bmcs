'''

Example demonstrating the interaction between model 
and Visualization sheet.

VizSheet time coincides with the Model time.
The model is required to deliver the time range 
by returning the values of time_low, time_high and time_step. 
By default, the model is expected to return 
the time range as (0,1,10). Updates of the model 
time range should be immediately reflected in the VizSheet. 
An update of the range can be done by the user or by the 
calculation changing the value of t within the model. 
The change of the visual time is reported to the VizXD 
adapters that are requested to change their state.

Provide a dummy model delivering the time range and demonstrate 
the case of interactive steering versus separate calculation 
and browsing steps.

The example should be provided in view.examples.time_control.py

Created on Mar 10, 2017

@author: rch
'''

import time

from traits.api import Instance, List,  HasStrictTraits,\
    Property, cached_property, Bool
from traitsui.api import View, Include, VGroup, UItem

from bmcs.time_functions.tfun_pwl_interactive import TFunPWLInteractive
from ibvpy.api import BCDof
from ibvpy.core.tline import TLine
import numpy as np
from view.window import BMCSWindow
from view.window.bmcs_window import BMCSModel

from .response_tracer import ResponseTracer


class TimeLoop(HasStrictTraits):

    tline = Instance(TLine)

    paused = Bool(False)
    restart = Bool(True)

    def init_loop(self):
        if self.paused:
            self.paused = False
        if self.restart:
            self.tline.val = 0
            self.restart = False

    def eval(self):
        '''this method is just called by the tloop_thread'''

        self.init_loop()
        t_min = self.tline.val
        t_max = self.tline.max
        n_steps = 5
        tarray = np.linspace(t_min, t_max, n_steps)
        for idx, t in enumerate(tarray):
            if self.restart or self.paused:
                break
            time.sleep(1)
            self.tline.val = t


class DemoModel(BMCSModel):
    '''Demo model of the BMCS Window

    Shows how the time control within an application of BMCS
    is realized.

    Run mode
    ========
    The model provides the methods for the Actions 
    Run, Pause, and Continue.

    During these actions, values are registered within
    the response tracers. The model can also send an update
    to the visual object time - so that the response tracers
    are asked to be updated.

    Sliding mode
    ============
    Once the calculation is finished, the slider in the VizSheet
    can be used to browse through the viz-adapters and visualize 
    the response for the current vot.

    Interactive mode
    ================
    Boundary condition can be constructed interactively by
    a boundary condition factory. The factory acts as a Run mode.
    '''
    node_name = 'demo model'

    tree_node_list = List

    def _tree_node_list_default(self):
        return [self.tline, self.rt, self.bc_dof]

    tloop = Property(Instance(TimeLoop))

    @cached_property
    def _get_tloop(self):
        return TimeLoop(tline=self.tline)

    def eval(self):
        self.tloop.eval()

    rt = Instance(ResponseTracer, ())

    bc_dof = Instance(BCDof)

    def _bc_dof_default(self):
        return BCDof(time_function=TFunPWLInteractive())

    tree_view = View(
        VGroup(
            Include('actions'),
            UItem('bc_dof@', height=500)
        )
    )


if __name__ == '__main__':
    model = DemoModel()
    model.run()

    tv = BMCSWindow(model=model)
    model.rt.add_viz2d('time_profile', 'response tracer #1')
    tv.finish_event = True
    tv.configure_traits()
