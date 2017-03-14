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

from traits.api import Instance, List
from traitsui.api import View, Include
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSTreeNode
from view.window import BMCSWindow

from boundary_condition import BoundaryCondition
from response_tracer import ResponseTracer


class DemoVizControl(Viz2D):

    def plot(self, ax, vot=0):
        print 'recalculate for', vot
        self.vis2d.eval(vot)


class DemoModel(BMCSTreeNode, Vis2D):
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
        return [self.rt, self.bc]

    def run(self):
        pass

    def interrupt(self):
        pass

    def continue_(self):
        pass

    def eval(self, vot):
        print 'recalculating for', vot

    viz2d_classes = {'control_viz': DemoVizControl}

    bc = Instance(BoundaryCondition, ())

    rt = Instance(ResponseTracer, ())

    tree_view = View(
        Include('actions')
    )

if __name__ == '__main__':

    model = DemoModel(node_name='demo')

    tv = BMCSWindow(root=model)
    model.add_viz2d('control_viz', 'time control')
    model.bc.add_viz2d('time_function', 'boundary condition #1')
    model.rt.add_viz2d('time_profile', 'response tracer #1')
    tv.configure_traits()
