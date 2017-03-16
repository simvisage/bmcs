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

from ibvpy.api import BCDof
from pyface.api import ProgressDialog
from traits.api import Instance, List
from traitsui.api import View, Include
from view.ui import BMCSTreeNode
from view.window import BMCSWindow

import numpy as np
from response_tracer import ResponseTracer
from tfun_pwl_interactive import TFunPWLInteractive


class DemoModel(BMCSTreeNode):
    '''Demo model of the BMCS Window

    Shows how the time control within an application of BMCS
    is realized.

    Run mode
    ========
    The model provides the methods for the Actions 
    Run, Interrupt, and Continue.

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
        return [self.rt, self.bc, self.bc_dof]

    def run(self):
        print 'Model: recalculating'

        n_steps = 5

        # todo distinguish target time -- make a threaded
        # interaction with the TLoop
        #
        t_min, t_max = self.ui.get_vot_range()
        pd = ProgressDialog(title='simulation progress',
                                  message='running %d steps' % n_steps,
                                  min=0, max=n_steps,
                                  show_time=True,
                                  can_cancel=True)
        pd.open()
        tarray = np.linspace(t_min, t_max, n_steps)
        for idx, t in enumerate(tarray):
            print 't', t
            pd.update(idx)
            time.sleep(1)
            self.ui.vot = t
        pd.update(n_steps)

    def pause(self):
        pass

    def stop(self):
        pass

    bc = Instance(TFunPWLInteractive, ())

    rt = Instance(ResponseTracer, ())

    bc_dof = Instance(BCDof)

    def _bc_dof_default(self):
        return BCDof(time_function=TFunPWLInteractive())

    tree_view = View(
        Include('actions')
    )

if __name__ == '__main__':
    model = DemoModel()
    tv = BMCSWindow(model=model)
    model.bc.add_viz2d('time_function', 'boundary condition #1')
    model.rt.add_viz2d('time_profile', 'response tracer #1')
    tv.configure_traits()
