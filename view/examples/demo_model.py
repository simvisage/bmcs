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

from ibvpy.api import BCDof, TLine
from traits.api import Instance, List, Bool
from traitsui.api import View, Include, VGroup, UItem
from view.ui import BMCSTreeNode
from view.window import BMCSWindow

import numpy as np
from response_tracer import ResponseTracer
from tfun_pwl_interactive import TFunPWLInteractive
from scratch.ui.tloop_thread import TLoopThread
from PyQt4.Qt import QProgressDialog, QSize, SIGNAL


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

    tline = Instance(TLine)
    '''Time range.
    '''

    def _tline_default(self):
        return TLine(min=0.0, step=0.1, max=0.0,
                     time_change_notifier=self.time_changed,
                     )

    def time_changed(self, time):
        self.ui.viz_sheet.time_changed(time)


    def time_range_changed(self, tmax):
        self.tline.max = tmax
        self.ui.viz_sheet.time_range_changed(tmax)

    def set_tmax(self, time):
        self.time_range_changed(time)
        
    tloop_thread = Instance(TLoopThread)
    
    tloop_running = Bool(False)
    
    def _tloop_running_changed(self):
        self.bc.enable_slider = not self.tloop_running 
        
    def run(self):
        print 'Model: recalculating'
        
        if self.tloop_running:
            return
        else:
            self.tloop_running = True
        
        n_steps = 5
        
        pd = QProgressDialog(windowTitle='simulation progress',
                             labelText='running %d steps' % n_steps,
                             minimum=0, maximum=n_steps,
                             autoClose=True,
                             size=QSize(250,100))

        self.tloop_thread = TLoopThread(model = self)
        self.tloop_thread.start()
        
        pd.open()
        pd.connect(self.tloop_thread, 
                   SIGNAL('updateProgressDialog'), 
                   lambda i: pd.setValue(i))
        
        
    def do_progress(self):
        '''this method is just called by the tloop_thread'''
        n_steps = 5
        
        # todo distinguish target time -- make a threaded
        # interaction with the TLoop

        t_min = self.tline.val
        t_max = self.tline.max
        tarray = np.linspace(t_min, t_max, n_steps)
        for idx, t in enumerate(tarray):
            print 't', t
            self.tloop_thread.emit(SIGNAL('updateProgressDialog'), idx)
            time.sleep(1)
            self.tline.val = t
            self.ui.viz_sheet.time_changed(t)
        self.tloop_thread.emit(SIGNAL('updateProgressDialog'), n_steps)
        
        self.tloop_running = False

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
        VGroup(
            Include('actions'),
            UItem('tline@')
        )
    )
 
if __name__ == '__main__':
    model = DemoModel()
    tv = BMCSWindow(model=model)
    model.bc.add_viz2d('time_function', 'boundary condition #1')
    model.rt.add_viz2d('time_profile', 'response tracer #1')
    tv.configure_traits()
